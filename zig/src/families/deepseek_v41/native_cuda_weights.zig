//! One lossless, contiguous device image of a rank's official text-only tensors.

const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("native_checkpoint.zig");
const payload = @import("native_payload.zig");
const DType = @import("pack.zig").DType;
const projection = @import("native_projection.zig");

pub const Tensor = struct {
    ptr: u64,
    len: usize,
    dtype: DType,
    shape: [8]usize,
    rank: u8,

    pub fn dims(t: *const Tensor) []const usize {
        return t.shape[0..t.rank];
    }
};

pub const DeviceProjection = struct {
    kind: projection.Kind,
    weight: Tensor,
    scale: ?Tensor,
    /// Rank-local logical output and input widths. MXFP4 storage packs two K values per byte.
    n: usize,
    k: usize,
};

pub const ExpertTables = struct {
    image: cuda.DeviceBuffer,
    w1: u64,
    s1: u64,
    w3: u64,
    s3: u64,
    w2: u64,
    s2: u64,
    kinds: u64,
    experts: usize,

    pub fn deinit(t: *ExpertTables) void {
        t.image.free();
        t.* = undefined;
    }
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    image: cuda.DeviceBuffer,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,
    /// Small runtime forms (norms, router bias, indexer projection); matrix payloads stay in `image` unchanged.
    forms: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ck: *const checkpoint.Checkpoint, rank: u32) !Weights {
        if (rank >= ck.world) return error.BadRank;
        const capacity: usize = @intCast(ck.summary.rankBytes());
        var out: Weights = .{ .gpa = gpa, .image = try cuda.DeviceBuffer.alloc(d, capacity), .bytes = 0 };
        errdefer out.deinit();

        const names = try gpa.alloc([]const u8, ck.pack.names.count());
        defer gpa.free(names);
        var count: usize = 0;
        var it = ck.pack.names.iterator();
        while (it.next()) |entry| {
            const p = try checkpoint.tensorPlan(entry.key_ptr.*, entry.value_ptr.*, ck.world);
            if (p.scope != .text) continue;
            names[count] = entry.key_ptr.*;
            count += 1;
        }
        std.mem.sort([]const u8, names[0..count], {}, struct {
            fn less(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.less);
        try out.tensors.ensureTotalCapacity(gpa, @intCast(count));
        var engram: ?EngramPending = null;
        defer if (engram) |pending| {
            gpa.free(pending.name);
            gpa.free(pending.values);
        };
        for (names[0..count]) |name| {
            const tensor = try payload.read(gpa, io, &ck.pack, name, rank, ck.world);
            defer gpa.free(tensor.bytes);
            if (tensor.bytes.len > capacity - out.bytes) return error.NativeCapacityMismatch;
            try out.image.upload(out.bytes, tensor.bytes);
            const key = try gpa.dupe(u8, name);
            errdefer gpa.free(key);
            out.tensors.putAssumeCapacity(key, .{
                .ptr = out.image.ptr + out.bytes,
                .len = tensor.bytes.len,
                .dtype = tensor.dtype,
                .shape = tensor.shape,
                .rank = tensor.rank,
            });
            try out.addRuntimeAlias(name, &tensor, out.image.ptr + out.bytes);
            try out.addEngramQk(name, &tensor, &engram);
            out.bytes += tensor.bytes.len;
        }
        if (engram != null) return error.IncompleteEngramQk;
        return out;
    }

    pub fn get(w: *const Weights, name: []const u8) ?Tensor {
        return w.tensors.get(name);
    }

    fn addRuntimeAlias(w: *Weights, official: []const u8, src: *const payload.Tensor, raw_ptr: u64) !void {
        var alias_buf: [256]u8 = undefined;
        const spec = runtimeAlias(official, &alias_buf) orelse return;
        if (w.tensors.contains(spec.name)) return error.DuplicateNativeAlias;
        var tensor: Tensor = .{ .ptr = raw_ptr, .len = src.bytes.len, .dtype = src.dtype, .shape = src.shape, .rank = src.rank };
        if (spec.widen) {
            const count = elementCount(src.dims()) catch return error.Overflow;
            const host = try w.gpa.alignedAlloc(u8, .@"16", 4 * count);
            defer w.gpa.free(host);
            for (0..count) |i| {
                const value = try scalarF32(src.dtype, src.bytes, i);
                std.mem.writeInt(u32, host[4 * i ..][0..4], @bitCast(value), .little);
            }
            var form = try cuda.DeviceBuffer.alloc(rawDriver(w), host.len);
            errdefer form.free();
            try form.upload(0, host);
            try w.forms.append(w.gpa, form);
            tensor.ptr = form.ptr;
            tensor.len = host.len;
            tensor.dtype = .f32;
        }
        try w.tensors.put(w.gpa, try w.gpa.dupe(u8, spec.name), tensor);
        if (spec.bf16_form) {
            const count = elementCount(src.dims()) catch return error.Overflow;
            const host = try w.gpa.alignedAlloc(u8, .@"16", 2 * count);
            defer w.gpa.free(host);
            for (0..count) |i|
                std.mem.writeInt(u16, host[2 * i ..][0..2], bf16Round(try scalarF32(src.dtype, src.bytes, i)), .little);
            var form = try cuda.DeviceBuffer.alloc(rawDriver(w), host.len);
            errdefer form.free();
            try form.upload(0, host);
            try w.forms.append(w.gpa, form);
            var role_buf: [288]u8 = undefined;
            const role = try std.fmt.bufPrint(&role_buf, "s.{s}16", .{spec.name});
            try w.tensors.put(w.gpa, try w.gpa.dupe(u8, role), .{
                .ptr = form.ptr,
                .len = host.len,
                .dtype = .bf16,
                .shape = src.shape,
                .rank = src.rank,
            });
        }
    }

    fn addEngramQk(w: *Weights, official: []const u8, src: *const payload.Tensor, pending: *?EngramPending) !void {
        var alias_buf: [256]u8 = undefined;
        const alias = engramAlias(official, &alias_buf) orelse return;
        const count = try elementCount(src.dims());
        const values = try w.gpa.alloc(f32, count);
        errdefer w.gpa.free(values);
        for (values, 0..) |*value, i| value.* = try scalarF32(src.dtype, src.bytes, i);
        if (pending.* == null) {
            pending.* = .{ .name = try w.gpa.dupe(u8, alias), .values = values, .shape = src.shape, .rank = src.rank };
            return;
        }
        var first = pending.*.?;
        pending.* = null;
        defer w.gpa.free(first.name);
        defer w.gpa.free(first.values);
        defer w.gpa.free(values);
        if (!std.mem.eql(u8, first.name, alias) or first.rank != src.rank or
            !std.mem.eql(usize, first.shape[0..first.rank], src.shape[0..src.rank]) or first.values.len != values.len)
            return error.MismatchedEngramQk;
        const host = try w.gpa.alignedAlloc(u8, .@"16", 4 * count);
        defer w.gpa.free(host);
        for (first.values, values, 0..) |q, k, i| std.mem.writeInt(u32, host[4 * i ..][0..4], @bitCast(q * k), .little);
        var form = try cuda.DeviceBuffer.alloc(rawDriver(w), host.len);
        errdefer form.free();
        try form.upload(0, host);
        try w.forms.append(w.gpa, form);
        try w.tensors.put(w.gpa, try w.gpa.dupe(u8, alias), .{
            .ptr = form.ptr,
            .len = host.len,
            .dtype = .f32,
            .shape = src.shape,
            .rank = src.rank,
        });
    }

    /// Resolve one rank-local official projection without copying or changing its encoding.
    pub fn projection(w: *const Weights, prefix: []const u8) !DeviceProjection {
        var name: [512]u8 = undefined;
        const weight = w.get(try std.fmt.bufPrint(&name, "{s}.weight", .{prefix})) orelse return error.MissingWeight;
        if (weight.rank != 2 or weight.shape[0] == 0 or weight.shape[1] == 0) return error.BadNativeProjection;
        if (weight.dtype == .bf16) return .{
            .kind = .bf16,
            .weight = weight,
            .scale = null,
            .n = weight.shape[0],
            .k = weight.shape[1],
        };

        const scale = w.get(try std.fmt.bufPrint(&name, "{s}.scale", .{prefix})) orelse return error.MissingScale;
        if (scale.dtype != .f8_e8m0 or scale.rank != 2) return error.BadNativeScale;
        if (weight.dtype == .f8_e4m3) {
            const n = weight.shape[0];
            const k = weight.shape[1];
            if (scale.shape[0] != divCeil(n, 32) or scale.shape[1] != divCeil(k, 32)) return error.BadNativeScale;
            return .{
                .kind = if (std.mem.endsWith(u8, prefix, ".attn.wo_a")) .wo_a else .mxfp8,
                .weight = weight,
                .scale = scale,
                .n = n,
                .k = k,
            };
        }
        if (weight.dtype == .i8 and isExpert(prefix)) {
            const n = weight.shape[0];
            const k = std.math.mul(usize, weight.shape[1], 2) catch return error.BadNativeProjection;
            if (k % 32 != 0 or scale.shape[0] != n or scale.shape[1] != k / 32) return error.BadNativeScale;
            return .{ .kind = .mxfp4, .weight = weight, .scale = scale, .n = n, .k = k };
        }
        return error.UnsupportedNativeProjection;
    }

    /// Pointer tables for one layer's routed MXFP4 experts followed by its MXFP8 shared expert.
    /// The table is derived only from already-loaded official tensors; payload bytes are not copied or converted.
    pub fn expertTables(w: *const Weights, d: *const cuda.Driver, layer: u32, routed: usize) !ExpertTables {
        if (routed == 0) return error.BadNativeMoe;
        const count = routed + 1;
        const table_bytes = std.math.mul(usize, count, 8) catch return error.Overflow;
        const table_stride = std.mem.alignForward(usize, table_bytes, 16);
        const kinds_at = 6 * table_stride;
        const total = std.mem.alignForward(usize, kinds_at + count, 16);
        const host = try w.gpa.alignedAlloc(u8, .@"16", total);
        defer w.gpa.free(host);
        @memset(host, 0);
        const projs = [_][]const u8{ "w1", "w3", "w2" };
        for (0..count) |expert| for (projs, 0..) |proj, p| {
            var prefix: [160]u8 = undefined;
            const name = if (expert < routed)
                try std.fmt.bufPrint(&prefix, "layers.{d}.ffn.experts.{d}.{s}", .{ layer, expert, proj })
            else
                try std.fmt.bufPrint(&prefix, "layers.{d}.ffn.shared_experts.{s}", .{ layer, proj });
            const matrix = try w.projection(name);
            if ((expert < routed and matrix.kind != .mxfp4) or (expert == routed and matrix.kind != .mxfp8))
                return error.BadNativeMoe;
            writePointer(host, (2 * p) * table_stride + 8 * expert, matrix.weight.ptr);
            writePointer(host, (2 * p + 1) * table_stride + 8 * expert, matrix.scale.?.ptr);
        };
        host[kinds_at + routed] = 1;
        var image = try cuda.DeviceBuffer.alloc(d, total);
        errdefer image.free();
        try image.upload(0, host);
        return .{
            .image = image,
            .w1 = image.ptr,
            .s1 = image.ptr + table_stride,
            .w3 = image.ptr + 2 * table_stride,
            .s3 = image.ptr + 3 * table_stride,
            .w2 = image.ptr + 4 * table_stride,
            .s2 = image.ptr + 5 * table_stride,
            .kinds = image.ptr + kinds_at,
            .experts = count,
        };
    }

    /// Bind one layer's persistent native-MoE pointer-table roles into a runner.
    pub fn bindExpertTables(w: *const Weights, runner: anytype, layer: u32, tables: *const ExpertTables) !void {
        _ = w;
        var names: [7][96]u8 = undefined;
        var external: [7]@TypeOf(runner.*).External = undefined;
        const suffixes = [_][]const u8{ "w1", "s1", "w3", "s3", "w2", "s2", "kinds" };
        const ptrs = [_]u64{ tables.w1, tables.s1, tables.w3, tables.s3, tables.w2, tables.s2, tables.kinds };
        for (&external, suffixes, ptrs, 0..) |*item, suffix, ptr, i| item.* = .{
            .role = try std.fmt.bufPrint(&names[i], "s.native.L{d}.moe.{s}", .{ layer, suffix }),
            .addr = ptr,
        };
        try runner.externals(&external);
    }

    pub fn deinit(w: *Weights) void {
        var keys = w.tensors.keyIterator();
        while (keys.next()) |key| w.gpa.free(key.*);
        w.tensors.deinit(w.gpa);
        for (w.forms.items) |*form| form.free();
        w.forms.deinit(w.gpa);
        w.image.free();
        w.* = undefined;
    }
};

const Alias = struct { name: []const u8, widen: bool, bf16_form: bool = false };
const EngramPending = struct { name: []u8, values: []f32, shape: [8]usize, rank: u8 };

fn runtimeAlias(name: []const u8, buf: []u8) ?Alias {
    if (std.mem.eql(u8, name, "embed.weight")) return .{ .name = "embed", .widen = false };
    if (std.mem.eql(u8, name, "norm.weight")) return .{ .name = "norm", .widen = true };
    if (!std.mem.startsWith(u8, name, "layers.")) return null;
    const rest0 = name["layers.".len..];
    const dot = std.mem.indexOfScalar(u8, rest0, '.') orelse return null;
    const layer = rest0[0..dot];
    const suffix = rest0[dot + 1 ..];
    const rules = [_]struct { from: []const u8, to: []const u8, widen: bool, bf16_form: bool = false }{
        .{ .from = "attn_norm.weight", .to = "attn_norm", .widen = true },
        .{ .from = "ffn_norm.weight", .to = "ffn_norm", .widen = true },
        .{ .from = "hc_attn_fn", .to = "hc_attn.fn", .widen = false, .bf16_form = true },
        .{ .from = "hc_attn_base", .to = "hc_attn.base", .widen = false },
        .{ .from = "hc_attn_scale", .to = "hc_attn.scale", .widen = false },
        .{ .from = "hc_ffn_fn", .to = "hc_ffn.fn", .widen = false, .bf16_form = true },
        .{ .from = "hc_ffn_base", .to = "hc_ffn.base", .widen = false },
        .{ .from = "hc_ffn_scale", .to = "hc_ffn.scale", .widen = false },
        .{ .from = "attn.q_norm.weight", .to = "attn.q_norm", .widen = true },
        .{ .from = "attn.kv_norm.weight", .to = "attn.kv_norm", .widen = true },
        .{ .from = "attn.attn_sink", .to = "attn.sink", .widen = false },
        .{ .from = "attn.compressor.norm.weight", .to = "attn.comp_norm", .widen = true },
        .{ .from = "attn.indexer.k_norm.weight", .to = "attn.ix_knorm", .widen = true },
        .{ .from = "attn.indexer.weights_proj.weight", .to = "attn.ix_wp", .widen = true },
        .{ .from = "ffn.gate.weight", .to = "moe.gate", .widen = false },
        .{ .from = "ffn.gate.bias", .to = "moe.bias", .widen = true },
    };
    for (rules) |rule| if (std.mem.eql(u8, suffix, rule.from)) return .{
        .name = std.fmt.bufPrint(buf, "L{s}.{s}", .{ layer, rule.to }) catch return null,
        .widen = rule.widen,
        .bf16_form = rule.bf16_form,
    };
    return null;
}

fn engramAlias(name: []const u8, buf: []u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, name, "layers.")) return null;
    const rest = name["layers.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const suffix = rest[dot + 1 ..];
    if (!std.mem.eql(u8, suffix, "engram.q_weight") and !std.mem.eql(u8, suffix, "engram.k_weight")) return null;
    return std.fmt.bufPrint(buf, "L{s}.engram.qk", .{rest[0..dot]}) catch null;
}

fn elementCount(dims: []const usize) !usize {
    var out: usize = 1;
    for (dims) |dim| out = std.math.mul(usize, out, dim) catch return error.Overflow;
    return out;
}

fn scalarF32(dtype: DType, bytes: []const u8, i: usize) !f32 {
    return switch (dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, bytes[4 * i ..][0..4], .little)),
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)))),
        else => error.UnsupportedNativeFormDType,
    };
}

fn bf16Round(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if (std.math.isNan(value)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

/// All allocations in a native image use one CUDA driver. DeviceBuffer keeps that driver.
fn rawDriver(w: *const Weights) *const cuda.Driver {
    return w.image.d;
}

fn divCeil(value: usize, divisor: usize) usize {
    return value / divisor + @intFromBool(value % divisor != 0);
}

fn isExpert(prefix: []const u8) bool {
    return (std.mem.startsWith(u8, prefix, "layers.") or std.mem.startsWith(u8, prefix, "mtp.")) and
        std.mem.indexOf(u8, prefix, ".ffn.experts.") != null and
        (std.mem.endsWith(u8, prefix, ".w1") or std.mem.endsWith(u8, prefix, ".w2") or std.mem.endsWith(u8, prefix, ".w3"));
}

fn writePointer(bytes: []u8, offset: usize, ptr: u64) void {
    std.mem.writeInt(u64, bytes[offset..][0..8], ptr, .little);
}

test "native all-resident tensor metadata keeps the official dtype and logical shape" {
    const t: Tensor = .{ .ptr = 1, .len = 16, .dtype = .bf16, .shape = .{ 2, 4, 1, 1, 1, 1, 1, 1 }, .rank = 2 };
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, t.dims());
}

test "native all-resident projections preserve local MXFP8 and packed MXFP4 geometry" {
    const a = std.testing.allocator;
    var w: Weights = .{ .gpa = a, .image = undefined, .bytes = 0 };
    defer w.tensors.deinit(a);
    try w.tensors.put(a, "layers.0.attn.wq_b.weight", .{ .ptr = 1, .len = 64 * 96, .dtype = .f8_e4m3, .shape = .{ 64, 96, 1, 1, 1, 1, 1, 1 }, .rank = 2 });
    try w.tensors.put(a, "layers.0.attn.wq_b.scale", .{ .ptr = 2, .len = 6, .dtype = .f8_e8m0, .shape = .{ 2, 3, 1, 1, 1, 1, 1, 1 }, .rank = 2 });
    try w.tensors.put(a, "layers.0.ffn.experts.7.w2.weight", .{ .ptr = 3, .len = 32 * 48, .dtype = .i8, .shape = .{ 32, 48, 1, 1, 1, 1, 1, 1 }, .rank = 2 });
    try w.tensors.put(a, "layers.0.ffn.experts.7.w2.scale", .{ .ptr = 4, .len = 96, .dtype = .f8_e8m0, .shape = .{ 32, 3, 1, 1, 1, 1, 1, 1 }, .rank = 2 });
    const dense = try w.projection("layers.0.attn.wq_b");
    try std.testing.expectEqual(projection.Kind.mxfp8, dense.kind);
    try std.testing.expectEqual(@as(usize, 64), dense.n);
    try std.testing.expectEqual(@as(usize, 96), dense.k);
    const expert = try w.projection("layers.0.ffn.experts.7.w2");
    try std.testing.expectEqual(projection.Kind.mxfp4, expert.kind);
    try std.testing.expectEqual(@as(usize, 96), expert.k);
}

test "official native tensors map to the established runtime roles" {
    var buf: [256]u8 = undefined;
    const cases = [_]struct { official: []const u8, role: []const u8, widen: bool }{
        .{ .official = "embed.weight", .role = "embed", .widen = false },
        .{ .official = "norm.weight", .role = "norm", .widen = true },
        .{ .official = "layers.14.attn_norm.weight", .role = "L14.attn_norm", .widen = true },
        .{ .official = "layers.14.hc_attn_fn", .role = "L14.hc_attn.fn", .widen = false },
        .{ .official = "layers.14.attn.indexer.weights_proj.weight", .role = "L14.attn.ix_wp", .widen = true },
        .{ .official = "layers.14.ffn.gate.weight", .role = "L14.moe.gate", .widen = false },
        .{ .official = "layers.14.ffn.gate.bias", .role = "L14.moe.bias", .widen = true },
    };
    for (cases) |case| {
        const got = runtimeAlias(case.official, &buf).?;
        try std.testing.expectEqualStrings(case.role, got.name);
        try std.testing.expectEqual(case.widen, got.widen);
    }
    try std.testing.expectEqualStrings("L14.engram.qk", engramAlias("layers.14.engram.k_weight", &buf).?);
    try std.testing.expectEqualStrings("L14.engram.qk", engramAlias("layers.14.engram.q_weight", &buf).?);
    try std.testing.expect(runtimeAlias("layers.14.hc_attn_fn", &buf).?.bf16_form);
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16Round(1.00390625));
}
