//! Header-only admission and TP planning for the official DeepSeek-V4.1 mixed safetensors.

const std = @import("std");
const core = @import("core");
const pack_module = @import("pack.zig");
const Pack = pack_module.Pack;
const Info = pack_module.Info;
const DType = core.safetensors.DType;
const Io = std.Io;

pub const Scope = enum { text, optional_draft, excluded_multimodal };
pub const Residency = enum { resident, cold_expert, cold_engram };
pub const Encoding = enum { bf16, f32, mxfp8_e4m3, mxfp4_e2m1_packed, ue8m0_scale };
pub const Split = enum { replicated, rows, col, row };

pub const TensorPlan = struct {
    scope: Scope,
    residency: Residency,
    encoding: Encoding,
    split: Split,
    axis: ?u8,
    rank_bytes: u64,
    read_spans: u64,

    pub fn contiguous(p: TensorPlan) bool {
        return p.read_spans == 1;
    }
};

pub const Summary = struct {
    tensors: u64 = 0,
    shards: u64 = 0,
    payload: u64 = 0,
    text: u64 = 0,
    optional_draft: u64 = 0,
    excluded_multimodal: u64 = 0,
    rank_resident: u64 = 0,
    rank_expert: u64 = 0,
    rank_engram: u64 = 0,
    replicated_spans: u64 = 0,
    rows_spans: u64 = 0,
    col_spans: u64 = 0,
    row_spans: u64 = 0,

    pub fn rankBytes(s: Summary) u64 {
        return s.rank_resident + s.rank_expert + s.rank_engram;
    }
};

pub const Checkpoint = struct {
    pack: Pack,
    summary: Summary,
    world: u32,

    pub fn open(gpa: std.mem.Allocator, io: Io, dir: []const u8, world: u32) !Checkpoint {
        if (world == 0) return error.BadWorld;
        try validateConfig(gpa, io, dir);
        var pack = try Pack.open(gpa, io, dir);
        errdefer pack.deinit();
        if (pack.missing != 0) return error.IncompleteCheckpoint;
        if (pack.unindexed != 0) return error.UnexpectedTensor;
        if (pack.misplaced != 0) return error.MisplacedTensor;
        if (pack.owner.count() != 0 and pack.owner.count() != pack.names.count()) return error.IncompleteCheckpoint;
        try validatePairs(&pack);
        const summary = try summarize(&pack, world);
        if (pack.index_total_size) |expected| if (expected != summary.payload) return error.CheckpointSizeMismatch;
        return .{ .summary = summary, .pack = pack, .world = world };
    }

    pub fn deinit(c: *Checkpoint) void {
        c.pack.deinit();
        c.* = undefined;
    }
};

pub fn inspect(gpa: std.mem.Allocator, io: Io, dir: []const u8, world: u32, out: *std.Io.Writer) !void {
    var checkpoint = try Checkpoint.open(gpa, io, dir, world);
    defer checkpoint.deinit();
    const s = checkpoint.summary;
    try out.print("checkpoint: deepseek_v41 official_mixed tensors={d} shards={d}\n", .{ s.tensors, s.shards });
    try out.print("payload: {d} bytes world={d}\n", .{ s.payload, world });
    try out.print("scope: text={d} optional_draft={d} excluded_multimodal={d}\n", .{
        s.text, s.optional_draft, s.excluded_multimodal,
    });
    try out.print("per-rank: resident={d} expert={d} engram={d} total={d}\n", .{
        s.rank_resident, s.rank_expert, s.rank_engram, s.rankBytes(),
    });
    try out.print("read-spans: replicated={d} rows={d} col={d} row={d}\n", .{
        s.replicated_spans, s.rows_spans, s.col_spans, s.row_spans,
    });
    try out.writeAll("validation: PASS\n");
}

/// Stream the deterministic per-tensor plan instead of materializing a second
/// 96k-entry report tree in memory.
pub fn writeJson(gpa: std.mem.Allocator, io: Io, dir: []const u8, world: u32, out: *std.Io.Writer) !void {
    var checkpoint = try Checkpoint.open(gpa, io, dir, world);
    defer checkpoint.deinit();
    const pack = &checkpoint.pack;
    const s = checkpoint.summary;
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    try names.ensureTotalCapacity(gpa, pack.names.count());
    var names_it = pack.names.iterator();
    while (names_it.next()) |entry| names.appendAssumeCapacity(entry.key_ptr.*);
    std.mem.sort([]const u8, names.items, {}, lessName);

    var json: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try json.beginObject();
    try json.objectField("checkpoint");
    try json.write(.{
        .model_type = "deepseek_v41",
        .format = "official_mixed",
        .tensors = s.tensors,
        .shards = s.shards,
        .header_payload_bytes = s.payload,
    });
    try json.objectField("world");
    try json.write(world);
    try json.objectField("per_rank_bytes");
    try json.write(.{ .resident = s.rank_resident, .cold_expert = s.rank_expert, .cold_engram = s.rank_engram, .total = s.rankBytes() });
    try json.objectField("read_spans_per_rank");
    try json.write(.{ .replicated = s.replicated_spans, .rows = s.rows_spans, .col = s.col_spans, .row = s.row_spans });
    try json.objectField("validation");
    try json.write(.{ .ok = true });
    try json.objectField("tensors");
    try json.beginArray();
    for (names.items) |name| {
        const info = pack.get(name).?;
        const item = try tensorPlan(name, info, world);
        try json.beginObject();
        try json.objectField("name");
        try json.write(name);
        try json.objectField("shard");
        try json.write(pack.files.items[info.file]);
        try json.objectField("dtype");
        try json.write(dtypeName(info.dtype));
        try json.objectField("shape");
        try json.write(info.dims());
        try json.objectField("scope");
        try json.write(@tagName(item.scope));
        try json.objectField("residency");
        try json.write(@tagName(item.residency));
        try json.objectField("encoding");
        try json.write(@tagName(item.encoding));
        try json.objectField("split");
        try json.write(@tagName(item.split));
        try json.objectField("axis");
        try json.write(item.axis);
        try json.objectField("rank_bytes");
        try json.write(item.rank_bytes);
        try json.objectField("read_spans");
        try json.write(item.read_spans);
        try json.objectField("contiguous");
        try json.write(item.contiguous());
        try json.endObject();
    }
    try json.endArray();
    try json.endObject();
    try out.writeByte('\n');
}

fn lessName(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

fn dtypeName(dtype: DType) []const u8 {
    return switch (dtype) {
        .bf16 => "BF16",
        .f32 => "F32",
        .i8 => "I8",
        .f8_e4m3 => "F8_E4M3",
        .f8_e8m0 => "F8_E8M0",
        else => @tagName(dtype),
    };
}

fn validateConfig(gpa: std.mem.Allocator, io: Io, dir: []const u8) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadConfig;
    const root = parsed.value.object;
    try expectString(root, "model_type", "deepseek_v41");
    const qv = root.get("quantization_config") orelse return error.MissingField;
    if (qv != .object) return error.BadConfig;
    const q = qv.object;
    try expectString(q, "quant_method", "fp8");
    try expectString(q, "activation_scheme", "dynamic");
    try expectString(q, "scale_fmt", "ue8m0");
    try expectString(q, "expert_dtype", "fp4");
    const block = q.get("weight_block_size") orelse return error.MissingField;
    if (block != .array or block.array.items.len != 2) return error.BadConfig;
    for (block.array.items) |item| if (item != .integer or item.integer != 32) return error.UnsupportedCheckpoint;
}

fn expectString(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const value = o.get(key) orelse return error.MissingField;
    if (value != .string or !std.mem.eql(u8, value.string, expected)) return error.UnsupportedCheckpoint;
}

fn summarize(pack: *const Pack, world: u32) !Summary {
    var out: Summary = .{ .tensors = pack.names.count(), .shards = pack.files.items.len };
    var it = pack.names.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const info = entry.value_ptr.*;
        out.payload = std.math.add(u64, out.payload, info.nbytes) catch return error.Overflow;
        const sc = scope(name);
        switch (sc) {
            .text => out.text += info.nbytes,
            .optional_draft => out.optional_draft += info.nbytes,
            .excluded_multimodal => out.excluded_multimodal += info.nbytes,
        }
        const item = try tensorPlan(name, info, world);
        switch (item.residency) {
            .resident => out.rank_resident += item.rank_bytes,
            .cold_expert => out.rank_expert += item.rank_bytes,
            .cold_engram => out.rank_engram += item.rank_bytes,
        }
        switch (item.split) {
            .replicated => out.replicated_spans += item.read_spans,
            .rows => out.rows_spans += item.read_spans,
            .col => out.col_spans += item.read_spans,
            .row => out.row_spans += item.read_spans,
        }
    }
    return out;
}

pub fn tensorPlan(name: []const u8, info: Info, world: u32) !TensorPlan {
    const rule = splitRule(name);
    const sc = scope(name);
    var rank_bytes: u64 = 0;
    if (sc == .text) {
        if (rule.axis) |axis| {
            const extent: u64 = @intCast(info.shape[axis]);
            if (extent == 0) return error.BadTensorGeometry;
            if (rule.split == .rows and isEngramTable(name)) {
                const part = std.math.divCeil(u64, extent, world) catch return error.BadWorld;
                rank_bytes = info.nbytes / extent * part;
            } else {
                if (extent % world != 0 or info.nbytes % world != 0) return error.SplitNotEven;
                rank_bytes = info.nbytes / world;
            }
        } else rank_bytes = info.nbytes;
    }
    return .{
        .scope = sc,
        .residency = residency(name),
        .encoding = try encoding(name, info.dtype),
        .split = rule.split,
        .axis = rule.axis,
        .rank_bytes = rank_bytes,
        // Text-only admission never reads optional draft or multimodal payloads.
        .read_spans = if (sc == .text) rule.spans(info) else 1,
    };
}

const Rule = struct {
    split: Split,
    axis: ?u8,

    fn spans(r: Rule, info: Info) u64 {
        return if (r.axis == 1 and info.rank == 2) info.shape[0] else 1;
    }
};

fn splitRule(name: []const u8) Rule {
    if (expertProjection(name)) |proj|
        return if (proj == .w2) .{ .split = .row, .axis = 1 } else .{ .split = .col, .axis = 0 };
    if (std.mem.eql(u8, name, "embed.weight") or std.mem.eql(u8, name, "head.weight") or isEngramTable(name))
        return .{ .split = .rows, .axis = 0 };
    const col = [_][]const u8{ ".attn.wq_b.", ".attn.wo_a.", ".ffn.shared_experts.w1.", ".ffn.shared_experts.w3." };
    for (col) |part| if (std.mem.indexOf(u8, name, part) != null) return .{ .split = .col, .axis = 0 };
    const row = [_][]const u8{ ".attn.wo_b.", ".ffn.shared_experts.w2." };
    for (row) |part| if (std.mem.indexOf(u8, name, part) != null) return .{ .split = .row, .axis = 1 };
    return .{ .split = .replicated, .axis = null };
}

fn scope(name: []const u8) Scope {
    if (std.mem.startsWith(u8, name, "vision.") or std.mem.startsWith(u8, name, "aligner.") or
        std.mem.eql(u8, name, "image_start") or std.mem.eql(u8, name, "image_end") or
        std.mem.eql(u8, name, "image_newline")) return .excluded_multimodal;
    return if (std.mem.startsWith(u8, name, "mtp.")) .optional_draft else .text;
}

fn residency(name: []const u8) Residency {
    if (expertProjection(name) != null) return .cold_expert;
    if (isEngramTable(name)) return .cold_engram;
    return .resident;
}

fn isEngramTable(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".engram.embed.") != null;
}

const ExpertProjection = enum { w1, w2, w3 };

fn expertProjection(name: []const u8) ?ExpertProjection {
    var rest = if (std.mem.startsWith(u8, name, "layers."))
        name["layers.".len..]
    else if (std.mem.startsWith(u8, name, "mtp."))
        name["mtp.".len..]
    else
        return null;
    const layer_end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    if (!decimal(rest[0..layer_end])) return null;
    rest = rest[layer_end + 1 ..];
    const expert_prefix = "ffn.experts.";
    if (!std.mem.startsWith(u8, rest, expert_prefix)) return null;
    rest = rest[expert_prefix.len..];
    const expert_end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    if (!decimal(rest[0..expert_end])) return null;
    rest = rest[expert_end..];
    inline for (.{ ExpertProjection.w1, .w2, .w3 }) |proj| {
        var buf: [16]u8 = undefined;
        const suffix = std.fmt.bufPrint(&buf, ".{t}.weight", .{proj}) catch unreachable;
        if (std.mem.eql(u8, rest, suffix)) return proj;
        const scale = std.fmt.bufPrint(&buf, ".{t}.scale", .{proj}) catch unreachable;
        if (std.mem.eql(u8, rest, scale)) return proj;
    }
    return null;
}

fn decimal(text: []const u8) bool {
    if (text.len == 0) return false;
    for (text) |c| if (c < '0' or c > '9') return false;
    return true;
}

fn encoding(name: []const u8, dtype: DType) !Encoding {
    if (expertProjection(name) != null and std.mem.endsWith(u8, name, ".weight")) {
        if (dtype != .i8) return error.BadExpertDType;
        return .mxfp4_e2m1_packed;
    }
    return switch (dtype) {
        .bf16 => .bf16,
        .f32 => .f32,
        .f8_e4m3 => .mxfp8_e4m3,
        .f8_e8m0 => .ue8m0_scale,
        else => error.UnsupportedOfficialDType,
    };
}

fn validatePairs(pack: *const Pack) !void {
    var it = pack.names.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const weight = entry.value_ptr.*;
        _ = try encoding(name, weight.dtype);
        if (!std.mem.endsWith(u8, name, ".weight") or
            (weight.dtype != .i8 and weight.dtype != .f8_e4m3)) continue;
        var buf: [512]u8 = undefined;
        const scale_name = try std.fmt.bufPrint(&buf, "{s}scale", .{name[0 .. name.len - "weight".len]});
        const scale = pack.get(scale_name) orelse return error.MissingScale;
        if (scale.dtype != .f8_e8m0 or weight.rank != 2 or scale.rank != 2) return error.BadScale;
        const logical_k = std.math.mul(usize, weight.shape[1], if (expertProjection(name) != null) 2 else 1) catch return error.Overflow;
        const expected_k = std.math.divCeil(usize, logical_k, 32) catch return error.BadScale;
        const expected_n = if (expertProjection(name) != null or isEngramTable(name))
            weight.shape[0]
        else
            std.math.divCeil(usize, weight.shape[0], 32) catch return error.BadScale;
        if (scale.shape[0] != expected_n or scale.shape[1] != expected_k) return error.BadScale;
    }
}

test "official mixed checkpoint: format, scope, TP split and fragmented w2 reads" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const shard = try @import("pack.zig").image(a, &.{
        .{ .name = "embed.weight", .dtype = "BF16", .shape = &.{ 8, 4 } },
        .{ .name = "vision.patch.weight", .dtype = "BF16", .shape = &.{ 4, 4 } },
        .{ .name = "mtp.0.norm.weight", .dtype = "BF16", .shape = &.{4} },
        .{ .name = "layers.6.ffn.experts.0.w1.weight", .dtype = "I8", .shape = &.{ 64, 32 } },
        .{ .name = "layers.6.ffn.experts.0.w1.scale", .dtype = "F8_E8M0", .shape = &.{ 64, 2 } },
        .{ .name = "layers.6.ffn.experts.0.w2.weight", .dtype = "I8", .shape = &.{ 32, 64 } },
        .{ .name = "layers.6.ffn.experts.0.w2.scale", .dtype = "F8_E8M0", .shape = &.{ 32, 4 } },
    });
    defer a.free(shard);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = shard });
    const config =
        \\{"model_type":"deepseek_v41","quantization_config":{"quant_method":"fp8",
        \\"activation_scheme":"dynamic","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var checkpoint = try Checkpoint.open(a, io, dir, 4);
    defer checkpoint.deinit();
    try std.testing.expectEqual(@as(u64, 7), checkpoint.summary.tensors);
    try std.testing.expectEqual(@as(u64, 16), checkpoint.summary.rank_resident);
    try std.testing.expectEqual(@as(u64, 1088), checkpoint.summary.rank_expert);
    try std.testing.expectEqual(@as(u64, 64), checkpoint.summary.row_spans);
    try std.testing.expectEqual(@as(u64, 32), checkpoint.summary.excluded_multimodal);
    try std.testing.expectEqual(@as(u64, 8), checkpoint.summary.optional_draft);

    var json_out: std.Io.Writer.Allocating = .init(a);
    defer json_out.deinit();
    try writeJson(a, io, dir, 4, &json_out.writer);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, json_out.written(), .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 4), root.get("world").?.integer);
    const tensors = root.get("tensors").?.array.items;
    try std.testing.expectEqual(@as(usize, 7), tensors.len);
    try std.testing.expectEqualStrings("embed.weight", tensors[0].object.get("name").?.string);
    const w2 = tensors[4].object;
    try std.testing.expectEqualStrings("row", w2.get("split").?.string);
    try std.testing.expectEqual(@as(i64, 32), w2.get("read_spans").?.integer);
    try std.testing.expect(!w2.get("contiguous").?.bool);
}

test "official mixed checkpoint refuses an MXFP4 scale with dense geometry" {
    var pack = Pack.init(std.testing.allocator);
    defer pack.deinit();
    const bytes = try @import("pack.zig").image(std.testing.allocator, &.{
        .{ .name = "layers.6.ffn.experts.0.w1.weight", .dtype = "I8", .shape = &.{ 64, 32 } },
        .{ .name = "layers.6.ffn.experts.0.w1.scale", .dtype = "F8_E8M0", .shape = &.{ 2, 2 } },
    });
    defer std.testing.allocator.free(bytes);
    try pack.addHeaderDump("bad.safetensors", bytes);
    try std.testing.expectError(error.BadScale, validatePairs(&pack));
}

test "official mixed checkpoint refuses an unclassified I8 tensor" {
    var pack = Pack.init(std.testing.allocator);
    defer pack.deinit();
    const bytes = try @import("pack.zig").image(std.testing.allocator, &.{.{
        .name = "layers.bad.ffn.experts.0.w1.weight",
        .dtype = "I8",
        .shape = &.{ 64, 32 },
    }});
    defer std.testing.allocator.free(bytes);
    try pack.addHeaderDump("bad.safetensors", bytes);
    try std.testing.expectError(error.UnsupportedOfficialDType, validatePairs(&pack));
}

test "text-only plan does not schedule optional draft row spans" {
    var shape: [core.safetensors.max_rank]usize = @splat(0);
    shape[0] = 5120;
    shape[1] = 1152;
    const item = try tensorPlan("mtp.0.ffn.experts.0.w2.weight", .{
        .file = 0,
        .dtype = .i8,
        .rank = 2,
        .shape = shape,
        .start = 0,
        .nbytes = 5120 * 1152,
    }, 4);
    try std.testing.expectEqual(Scope.optional_draft, item.scope);
    try std.testing.expectEqual(@as(u64, 0), item.rank_bytes);
    try std.testing.expectEqual(@as(u64, 1), item.read_spans);
    try std.testing.expect(item.contiguous());
}

test "official mixed checkpoint refuses an unindexed header tensor" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const shard = try @import("pack.zig").image(a, &.{.{ .name = "embed.weight", .dtype = "BF16", .shape = &.{ 8, 4 } }});
    defer a.free(shard);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = shard });
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"head.weight\":\"model.safetensors\"}}",
    });
    try writeTestConfig(io, &tmp.dir);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectError(error.UnexpectedTensor, Checkpoint.open(a, io, dir, 4));
}

test "official mixed checkpoint refuses an index total-size mismatch" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const shard = try @import("pack.zig").image(a, &.{.{ .name = "embed.weight", .dtype = "BF16", .shape = &.{ 8, 4 } }});
    defer a.free(shard);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = shard });
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"metadata\":{\"total_size\":65},\"weight_map\":{\"embed.weight\":\"model.safetensors\"}}",
    });
    try writeTestConfig(io, &tmp.dir);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectError(error.CheckpointSizeMismatch, Checkpoint.open(a, io, dir, 4));
}

test "official mixed checkpoint refuses tensors stored in the wrong indexed shard" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const first = try @import("pack.zig").image(a, &.{.{ .name = "head.weight", .dtype = "BF16", .shape = &.{ 8, 4 } }});
    defer a.free(first);
    const second = try @import("pack.zig").image(a, &.{.{ .name = "embed.weight", .dtype = "BF16", .shape = &.{ 8, 4 } }});
    defer a.free(second);
    try tmp.dir.writeFile(io, .{ .sub_path = "a.safetensors", .data = first });
    try tmp.dir.writeFile(io, .{ .sub_path = "b.safetensors", .data = second });
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"embed.weight\":\"a.safetensors\",\"head.weight\":\"b.safetensors\"}}",
    });
    try writeTestConfig(io, &tmp.dir);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectError(error.MisplacedTensor, Checkpoint.open(a, io, dir, 4));
}

test "official mixed checkpoint refuses a malformed shard header" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var header: [8]u8 = undefined;
    std.mem.writeInt(u64, &header, 64, .little);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &header });
    try writeTestConfig(io, &tmp.dir);
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectError(error.BadSafetensors, Checkpoint.open(a, io, dir, 4));
}

fn writeTestConfig(io: Io, dir: *Io.Dir) !void {
    try dir.writeFile(io, .{
        .sub_path = "config.json",
        .data =
        \\{"model_type":"deepseek_v41","quantization_config":{"quant_method":"fp8",
        \\"activation_scheme":"dynamic","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}}
        ,
    });
}
