//! Projection dispatch from lossless official tensors to the correctness-first CUDA operators.

const weights = @import("native_cuda_weights.zig");

pub const Options = struct {
    /// Official dense layers use checkpoint-defined dynamic activation quantization.
    dynamic: bool = true,
    /// Logical output rank of one `wo_a` group. Required only for `wo_a`.
    wo_rank: usize = 0,
};

/// The current native kernels consume BF16 rows and produce FP32 rows. Epilogues deliberately remain outside
/// this function so a caller cannot silently change the checkpoint arithmetic between projections.
pub fn run(ops: anytype, p: weights.DeviceProjection, x: u64, out: u64, rows: usize, options: Options) !void {
    switch (p.kind) {
        .bf16 => try ops.bf16(x, p.weight.ptr, out, rows, p.n, p.k),
        .mxfp8 => try ops.mxfp8(x, p.weight.ptr, p.scale.?.ptr, out, rows, p.n, p.k, options.dynamic),
        .mxfp4 => try ops.mxfp4(x, p.weight.ptr, p.scale.?.ptr, out, rows, p.n, p.k),
        .wo_a => {
            if (options.wo_rank == 0 or p.n % options.wo_rank != 0) return error.BadNativeProjection;
            try ops.woA(x, p.weight.ptr, p.scale.?.ptr, out, rows, p.n / options.wo_rank, options.wo_rank, p.k);
        },
    }
}

const TestOps = struct {
    seen: ?*[3]usize = null,
    pub fn bf16(_: @This(), _: u64, _: u64, _: u64, _: usize, _: usize, _: usize) !void {}
    pub fn mxfp8(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: bool) !void {}
    pub fn mxfp4(m: @This(), _: u64, _: u64, _: u64, _: u64, rows: usize, n: usize, k: usize) !void {
        if (m.seen) |seen| seen.* = .{ rows, n, k };
    }
    pub fn woA(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: usize) !void {}
};

test "native projection dispatch keeps MXFP4 packed geometry explicit" {
    var seen: [3]usize = undefined;
    try run(TestOps{ .seen = &seen }, .{
        .kind = .mxfp4,
        .weight = .{ .ptr = 2, .len = 8, .dtype = .i8, .shape = .{ 4, 2, 1, 1, 1, 1, 1, 1 }, .rank = 2 },
        .scale = .{ .ptr = 3, .len = 4, .dtype = .f8_e8m0, .shape = .{ 4, 1, 1, 1, 1, 1, 1, 1 }, .rank = 2 },
        .n = 4,
        .k = 32,
    }, 1, 4, 2, .{});
    try @import("std").testing.expectEqual([3]usize{ 2, 4, 32 }, seen);
}

test "native wo_a dispatch derives only whole local groups" {
    const p: weights.DeviceProjection = .{
        .kind = .wo_a,
        .weight = .{ .ptr = 2, .len = 8, .dtype = .f8_e4m3, .shape = .{ 6, 32, 1, 1, 1, 1, 1, 1 }, .rank = 2 },
        .scale = .{ .ptr = 3, .len = 1, .dtype = .f8_e8m0, .shape = .{ 1, 1, 1, 1, 1, 1, 1, 1 }, .rank = 2 },
        .n = 6,
        .k = 32,
    };
    try @import("std").testing.expectError(error.BadNativeProjection, run(TestOps{}, p, 1, 4, 1, .{ .wo_rank = 4 }));
}
