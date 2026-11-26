//! Independent host arithmetic for the official DeepSeek-V4.1 mixed checkpoint.
//! These scalar FP64 paths qualify storage semantics before any CUDA kernel is selected.

const std = @import("std");
const core = @import("core");

pub const weight_block = 32;
pub const expert_group = 32;

pub const Bf16 = struct {
    n: usize,
    k: usize,
    bytes: []const u8,

    pub fn validate(m: Bf16) !void {
        if (m.n == 0 or m.k == 0) return error.BadMatrix;
        const elements = std.math.mul(usize, m.n, m.k) catch return error.BadMatrix;
        if (m.bytes.len != std.math.mul(usize, elements, 2) catch return error.BadMatrix) return error.BadMatrix;
    }

    fn value(m: Bf16, row: usize, col: usize) f64 {
        const at = 2 * (row * m.k + col);
        const bits = std.mem.readInt(u16, m.bytes[at..][0..2], .little);
        return @floatCast(fromBf16(bits));
    }
};

pub const Mxfp8 = struct {
    n: usize,
    k: usize,
    codes: []const u8,
    scales: []const u8,

    pub fn validate(m: Mxfp8) !void {
        if (m.n == 0 or m.k == 0) return error.BadMatrix;
        if (m.codes.len != std.math.mul(usize, m.n, m.k) catch return error.BadMatrix) return error.BadMatrix;
        const nb = divCeil(m.n, weight_block);
        const kb = divCeil(m.k, weight_block);
        if (m.scales.len != std.math.mul(usize, nb, kb) catch return error.BadMatrix) return error.BadMatrix;
    }

    fn value(m: Mxfp8, row: usize, col: usize) !f64 {
        const kb = divCeil(m.k, weight_block);
        const scale = try ue8m0(m.scales[row / weight_block * kb + col / weight_block]);
        const decoded = core.quant.fp8.e4m3(m.codes[row * m.k + col]);
        if (!std.math.isFinite(decoded)) return error.NonFiniteWeight;
        return decoded * scale;
    }
};

pub const Mxfp4 = struct {
    n: usize,
    k: usize,
    codes: []const u8,
    scales: []const u8,

    pub fn validate(m: Mxfp4) !void {
        if (m.n == 0 or m.k == 0 or m.k % expert_group != 0) return error.BadMatrix;
        const logical = std.math.mul(usize, m.n, m.k) catch return error.BadMatrix;
        if (m.codes.len != logical / 2 or m.scales.len != logical / expert_group) return error.BadMatrix;
    }

    fn value(m: Mxfp4, row: usize, col: usize) !f64 {
        const byte = m.codes[row * (m.k / 2) + col / 2];
        const nibble: u8 = if (col & 1 == 0) byte & 0xF else byte >> 4;
        const scale = try ue8m0(m.scales[row * (m.k / expert_group) + col / expert_group]);
        return core.quant.nvfp4.e2m1(nibble) * scale;
    }
};

pub const Matrix = union(enum) {
    bf16: Bf16,
    mxfp8: Mxfp8,
    mxfp4: Mxfp4,

    pub fn n(m: Matrix) usize {
        return switch (m) {
            inline else => |x| x.n,
        };
    }

    pub fn k(m: Matrix) usize {
        return switch (m) {
            inline else => |x| x.k,
        };
    }

    pub fn validate(m: Matrix) !void {
        switch (m) {
            inline else => |x| try x.validate(),
        }
    }

    pub fn value(m: Matrix, row: usize, col: usize) !f64 {
        return switch (m) {
            .bf16 => |x| x.value(row, col),
            .mxfp8 => |x| x.value(row, col),
            .mxfp4 => |x| x.value(row, col),
        };
    }

    /// `out = weight * input`, accumulated in FP64.
    pub fn matvec(m: Matrix, input: []const f64, out: []f64) !void {
        try m.matmulF64(input, 1, out);
    }

    /// Decode (`rows == 1`) and prompt rows, with FP64 products and accumulation.
    pub fn matmulF64(m: Matrix, input: []const f64, rows: usize, out: []f64) !void {
        try m.validate();
        if (rows == 0 or input.len != rows * m.k() or out.len != rows * m.n()) return error.BadMatrix;
        for (0..rows) |batch| for (0..m.n()) |row| {
            var sum: f64 = 0;
            for (0..m.k()) |col| sum += try m.value(row, col) * input[batch * m.k() + col];
            out[batch * m.n() + row] = sum;
        };
    }

    /// A second reference with explicit FP32 products and accumulation.
    pub fn matmulF32(m: Matrix, input: []const f32, rows: usize, out: []f32) !void {
        try m.validate();
        if (rows == 0 or input.len != rows * m.k() or out.len != rows * m.n()) return error.BadMatrix;
        for (0..rows) |batch| for (0..m.n()) |row| {
            var sum: f32 = 0;
            for (0..m.k()) |col| sum += @as(f32, @floatCast(try m.value(row, col))) * input[batch * m.k() + col];
            out[batch * m.n() + row] = sum;
        };
    }
};

/// The official `wo_a` is block diagonal over attention groups. Its stored MXFP8
/// rows are converted to BF16 before each group consumes only its own input heads.
pub const WoA = struct {
    weight: Mxfp8,
    groups: usize,
    rank: usize,

    pub fn matvec(w: WoA, input: []const f64, out: []f64) !void {
        try w.weight.validate();
        if (w.groups == 0 or w.rank == 0 or w.weight.n != w.groups * w.rank) return error.BadWoA;
        if (input.len != w.groups * w.weight.k or out.len != w.weight.n) return error.BadWoA;
        for (0..w.groups) |group| for (0..w.rank) |r| {
            const row = group * w.rank + r;
            var sum: f64 = 0;
            for (0..w.weight.k) |col| {
                const decoded = try w.weight.value(row, col);
                const bf = fromBf16(toBf16(@floatCast(decoded)));
                sum += @as(f64, bf) * input[group * w.weight.k + col];
            }
            out[row] = sum;
        };
    }
};

/// Official dynamic MXFP8 activation quantization. Input is interpreted as BF16;
/// each row-local block gets a power-of-two UE8M0 scale and the result is widened
/// back to BF16, matching the official inplace path used before `wo_a`.
pub fn dynamicMxfp8(input: []const f32, block: usize, dequant: []f32, scales: []u8) !void {
    if (input.len == 0 or block == 0 or dequant.len != input.len or scales.len != divCeil(input.len, block))
        return error.BadActivation;
    for (0..scales.len) |group| {
        const lo = group * block;
        const hi = @min(lo + block, input.len);
        var amax: f64 = 1e-4;
        for (input[lo..hi]) |x| {
            const bf: f64 = fromBf16(toBf16(x));
            if (!std.math.isFinite(bf)) return error.NonFiniteActivation;
            amax = @max(amax, @abs(bf));
        }
        const exponent: i32 = @intFromFloat(@ceil(std.math.log2(amax / 448.0)));
        if (exponent < -127 or exponent > 127) return error.ActivationScaleOutsideUE8M0;
        const code: u8 = @intCast(exponent + 127);
        scales[group] = code;
        const scale = try ue8m0(code);
        for (lo..hi) |i| {
            const bf: f64 = fromBf16(toBf16(input[i]));
            const q = core.quant.fp8.toE4m3(std.math.clamp(bf / scale, -448.0, 448.0));
            dequant[i] = fromBf16(toBf16(@floatCast(q * scale)));
        }
    }
}

pub fn ue8m0(code: u8) !f64 {
    if (code == 255) return error.ReservedUE8M0;
    return std.math.ldexp(@as(f64, 1), @as(i32, code) - 127);
}

pub fn fromBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn toBf16(value: f32) u16 {
    if (std.math.isNan(value)) return 0xFFFF;
    const bits: u32 = @bitCast(value);
    return @intCast((bits +% (0x7FFF + ((bits >> 16) & 1))) >> 16);
}

fn divCeil(value: usize, divisor: usize) usize {
    return value / divisor + @intFromBool(value % divisor != 0);
}

test "BF16 and MXFP8 references use their checkpoint geometries" {
    const bf_bits = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0x3f00, 0x0000, 0x3f80 };
    const bf: Matrix = .{ .bf16 = .{ .n = 2, .k = 4, .bytes = std.mem.sliceAsBytes(&bf_bits) } };
    var bf_out: [2]f64 = undefined;
    try bf.matvec(&.{ 1, 1, 1, 1 }, &bf_out);
    try std.testing.expectEqual(@as(f64, 10), bf_out[0]);
    try std.testing.expectEqual(@as(f64, 0.5), bf_out[1]);

    var codes: [33 * 33]u8 = @splat(0x38); // e4m3 1.0
    codes[0] = 0x40; // 2.0
    const mx: Matrix = .{ .mxfp8 = .{ .n = 33, .k = 33, .codes = &codes, .scales = &.{ 127, 128, 129, 130 } } };
    var mx_out: [33]f64 = undefined;
    var ones: [33]f64 = @splat(1);
    try mx.matvec(&ones, &mx_out);
    try std.testing.expectEqual(@as(f64, 35), mx_out[0]);
    try std.testing.expectEqual(@as(f64, 136), mx_out[32]);
}

test "decode and prompt rows have explicit FP64 and FP32 references" {
    const bits = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0x3f00, 0x0000, 0x3f80 };
    const matrix: Matrix = .{ .bf16 = .{ .n = 2, .k = 4, .bytes = std.mem.sliceAsBytes(&bits) } };
    const input64 = [_]f64{ 1, 1, 1, 1, 2, 0, -1, 0.5 };
    var out64: [4]f64 = undefined;
    try matrix.matmulF64(&input64, 2, &out64);
    try std.testing.expectEqualSlices(f64, &.{ 10, 0.5, 1, -1.5 }, &out64);
    const input32 = [_]f32{ 1, 1, 1, 1, 2, 0, -1, 0.5 };
    var out32: [4]f32 = undefined;
    try matrix.matmulF32(&input32, 2, &out32);
    try std.testing.expectEqualSlices(f32, &.{ 10, 0.5, 1, -1.5 }, &out32);
}

test "MXFP4 is low-nibble first with one UE8M0 scale per 32 logical K" {
    var codes: [32]u8 = @splat(0xA2); // +1, -1
    codes[0] = 0x42; // +1, +2
    const m: Matrix = .{ .mxfp4 = .{ .n = 2, .k = 32, .codes = &codes, .scales = &.{ 127, 128 } } };
    var out: [2]f64 = undefined;
    var ones: [32]f64 = @splat(1);
    try m.matvec(&ones, &out);
    try std.testing.expectEqual(@as(f64, 3), out[0]);
    try std.testing.expectEqual(@as(f64, 0), out[1]);
}

test "wo_a consumes only its own attention group's columns" {
    const weight: Mxfp8 = .{ .n = 2, .k = 2, .codes = &.{ 0x38, 0x40, 0x44, 0x48 }, .scales = &.{127} };
    var out: [2]f64 = undefined;
    try (WoA{ .weight = weight, .groups = 2, .rank = 1 }).matvec(&.{ 1, 10, 100, 1000 }, &out);
    try std.testing.expectEqual(@as(f64, 21), out[0]);
    try std.testing.expectEqual(@as(f64, 4300), out[1]);
}

test "dynamic activation quantization rounds scales to UE8M0 powers of two" {
    const input = [_]f32{ 448, -224, 1, 0, 224, -112, 0.5, -0.5 };
    var dequant: [input.len]f32 = undefined;
    var scales: [2]u8 = undefined;
    try dynamicMxfp8(&input, 4, &dequant, &scales);
    try std.testing.expectEqualSlices(u8, &.{ 127, 126 }, &scales);
    try std.testing.expectEqual(@as(f32, 448), dequant[0]);
    try std.testing.expectEqual(@as(f32, -224), dequant[1]);
    try std.testing.expectEqual(@as(f32, 224), dequant[4]);
    try std.testing.expectError(error.ReservedUE8M0, ue8m0(255));
}

const Fixture = struct {
    schema: []const u8,
    checkpoint: struct {
        model_type: []const u8,
        tensors: u64,
        shards: u64,
        config_sha256: []const u8,
        index_sha256: []const u8,
    },
    cases: []const Case,
};

const Case = struct {
    name: []const u8,
    kind: enum { bf16, mxfp8, mxfp4, wo_a },
    n: usize,
    k: usize,
    weight: []const u8,
    scale: []const u8,
    input: []const f64,
    expected: []const f64,
    payload_sha256: []const u8,
};

test "real official slices match the independent BF16, MXFP8, MXFP4 and wo_a references" {
    const a = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(Fixture, a, @embedFile("fixtures/native-reference-v1.json"), .{});
    defer parsed.deinit();
    const fixture = parsed.value;
    try std.testing.expectEqualStrings("tensorfold.deepseek_v41.native_reference.v1", fixture.schema);
    try std.testing.expectEqualStrings("deepseek_v41", fixture.checkpoint.model_type);
    try std.testing.expectEqual(@as(u64, 96_085), fixture.checkpoint.tensors);
    try std.testing.expectEqual(@as(u64, 48), fixture.checkpoint.shards);
    try std.testing.expectEqual(@as(usize, 4), fixture.cases.len);
    for (fixture.cases) |case| {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(case.weight);
        hash.update(case.scale);
        hash.update(std.mem.sliceAsBytes(case.input));
        const hex = std.fmt.bytesToHex(hash.finalResult(), .lower);
        try std.testing.expectEqualStrings(case.payload_sha256, &hex);

        const out = try a.alloc(f64, case.n);
        defer a.free(out);
        switch (case.kind) {
            .bf16 => try (Matrix{ .bf16 = .{ .n = case.n, .k = case.k, .bytes = case.weight } }).matvec(case.input, out),
            .mxfp8 => try (Matrix{ .mxfp8 = .{ .n = case.n, .k = case.k, .codes = case.weight, .scales = case.scale } }).matvec(case.input, out),
            .mxfp4 => try (Matrix{ .mxfp4 = .{ .n = case.n, .k = case.k, .codes = case.weight, .scales = case.scale } }).matvec(case.input, out),
            .wo_a => try (WoA{
                .weight = .{ .n = case.n, .k = case.k, .codes = case.weight, .scales = case.scale },
                .groups = 1,
                .rank = case.n,
            }).matvec(case.input, out),
        }
        try std.testing.expectEqual(case.expected.len, out.len);
        for (case.expected, out) |expected, actual|
            try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
}
