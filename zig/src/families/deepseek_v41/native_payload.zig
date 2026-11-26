//! Lossless rank-local payload reads from the official safetensors.

const std = @import("std");
const checkpoint = @import("native_checkpoint.zig");
const pack_mod = @import("pack.zig");
const Pack = pack_mod.Pack;
const Info = pack_mod.Info;

pub const Tensor = struct {
    dtype: pack_mod.DType,
    shape: [8]usize = @splat(1),
    rank: u8,
    bytes: []align(16) u8,

    pub fn deinit(t: *Tensor, gpa: std.mem.Allocator) void {
        gpa.free(t.bytes);
        t.* = undefined;
    }

    pub fn dims(t: *const Tensor) []const usize {
        return t.shape[0..t.rank];
    }
};

pub const View = struct {
    offset: u64,
    len: u64,
    stride: u64 = 0,
    count: u64 = 1,
    axis: ?u8,
    shape: [8]usize = @splat(1),
    rank: u8,
};

pub fn view(name: []const u8, info: Info, rank: u32, world: u32) !View {
    if (world == 0 or rank >= world) return error.BadRank;
    const plan = try checkpoint.tensorPlan(name, info, world);
    if (plan.scope != .text) return error.TensorOutsideTextScope;
    var out: View = .{ .offset = 0, .len = info.nbytes, .axis = plan.axis, .rank = info.rank };
    @memcpy(out.shape[0..info.rank], info.dims());
    switch (plan.split) {
        .replicated => {},
        .rows, .col => {
            const axis = plan.axis orelse return error.BadTensorGeometry;
            if (axis != 0 or info.shape[0] == 0 or info.nbytes % info.shape[0] != 0) return error.BadTensorGeometry;
            const per_row = info.nbytes / info.shape[0];
            const base = info.shape[0] / world;
            const extra = info.shape[0] % world;
            if (plan.split == .col and extra != 0) return error.SplitNotEven;
            const rows = base + @intFromBool(rank < extra);
            const first = base * rank + @min(rank, extra);
            out.offset = per_row * first;
            out.len = per_row * rows;
            out.shape[0] = rows;
        },
        .row => {
            const axis = plan.axis orelse return error.BadTensorGeometry;
            if (axis != 1 or info.rank != 2 or info.shape[0] == 0 or info.shape[1] % world != 0)
                return error.BadTensorGeometry;
            const row_bytes = info.nbytes / info.shape[0];
            if (row_bytes % world != 0) return error.BadTensorGeometry;
            out.len = row_bytes / world;
            out.offset = out.len * rank;
            out.stride = row_bytes;
            out.count = info.shape[0];
            out.shape[1] /= world;
        },
    }
    const actual = out.len * out.count;
    if (actual != plan.rank_bytes and (plan.split != .rows or actual > plan.rank_bytes)) return error.NativePlanMismatch;
    return out;
}

pub fn read(gpa: std.mem.Allocator, io: std.Io, pack: *const Pack, name: []const u8, rank: u32, world: u32) !Tensor {
    const info = try pack.need(name);
    const v = try view(name, info, rank, world);
    const total = std.math.mul(u64, v.len, v.count) catch return error.Overflow;
    const bytes = try gpa.alignedAlloc(u8, .@"16", @intCast(total));
    errdefer gpa.free(bytes);
    try pack.read(io, info, .{ .offset = v.offset, .len = v.len, .stride = v.stride, .count = v.count }, bytes);
    return .{ .dtype = info.dtype, .shape = v.shape, .rank = v.rank, .bytes = bytes };
}

test "official rank views preserve replicated, axis-zero and row-split byte geometry" {
    const repl: Info = .{ .file = 0, .dtype = .bf16, .rank = 1, .shape = .{ 8, 1, 1, 1, 1 }, .start = 0, .nbytes = 16 };
    const rv = try view("model.norm.weight", repl, 2, 4);
    try std.testing.expectEqual(@as(u64, 16), rv.len);

    const col: Info = .{ .file = 0, .dtype = .f8_e4m3, .rank = 2, .shape = .{ 8, 16, 1, 1, 1 }, .start = 0, .nbytes = 128 };
    const cv = try view("layers.0.attn.wq_b.weight", col, 1, 4);
    try std.testing.expectEqualSlices(usize, &.{ 2, 16 }, cv.shape[0..2]);
    try std.testing.expectEqual(@as(u64, 32), cv.offset);
    try std.testing.expectEqual(@as(u64, 32), cv.len);

    const row: Info = .{ .file = 0, .dtype = .i8, .rank = 2, .shape = .{ 8, 16, 1, 1, 1 }, .start = 0, .nbytes = 128 };
    const wv = try view("layers.0.ffn.experts.0.w2.weight", row, 3, 4);
    try std.testing.expectEqualSlices(usize, &.{ 8, 4 }, wv.shape[0..2]);
    try std.testing.expectEqual(@as(u64, 12), wv.offset);
    try std.testing.expectEqual(@as(u64, 4), wv.len);
    try std.testing.expectEqual(@as(u64, 16), wv.stride);
    try std.testing.expectEqual(@as(u64, 8), wv.count);
}
