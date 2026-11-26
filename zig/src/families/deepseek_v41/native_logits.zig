//! Deterministic host merge for TP-sharded teacher-forced logit captures.

const std = @import("std");

pub const Candidate = struct {
    token: u32,
    logit: f32,
};

fn before(a: Candidate, b: Candidate) bool {
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.token < b.token;
}

/// Merge rank-major, row-major local top-k arrays; each offset is that rank's first global token id.
pub fn merge(
    out: []Candidate,
    values: []const f32,
    columns: []const i64,
    offsets: []const u32,
    rows: usize,
    k: usize,
) !void {
    if (rows == 0 or k < 2 or offsets.len == 0) return error.Shape;
    const rank_items = std.math.mul(usize, rows, k) catch return error.Shape;
    const all_items = std.math.mul(usize, rank_items, offsets.len) catch return error.Shape;
    if (out.len != rank_items or values.len != all_items or columns.len != all_items) return error.Shape;

    for (0..rows) |row| {
        const dst = out[row * k ..][0..k];
        @memset(dst, .{ .token = std.math.maxInt(u32), .logit = -std.math.inf(f32) });
        for (offsets, 0..) |offset, rank| {
            const base = rank * rank_items + row * k;
            for (0..k) |i| {
                if (!std.math.isFinite(values[base + i])) return error.NonFinite;
                const local = columns[base + i];
                if (local < 0) return error.BadColumn;
                const token64 = @as(u64, offset) + @as(u64, @intCast(local));
                if (token64 > std.math.maxInt(u32)) return error.BadColumn;
                const candidate: Candidate = .{ .token = @intCast(token64), .logit = values[base + i] };
                var at: usize = 0;
                while (at < k and !before(candidate, dst[at])) : (at += 1) {}
                if (at == k) continue;
                var j = k - 1;
                while (j > at) : (j -= 1) dst[j] = dst[j - 1];
                dst[at] = candidate;
            }
        }
    }
}

test "TP top-k merge is deterministic across ranks, rows, and ties" {
    const offsets = [_]u32{ 0, 100, 200, 300 };
    // Four ranks, two rows, two local candidates per row.
    const values = [_]f32{
        9, 7, 4, 3,
        8, 6, 5, 2,
        9, 8, 6, 1,
        7, 0, 6, 5,
    };
    const columns = [_]i64{
        5, 6, 1, 2,
        1, 2, 3, 4,
        1, 2, 7, 8,
        0, 1, 9, 10,
    };
    var out: [4]Candidate = undefined;
    try merge(&out, &values, &columns, &offsets, 2, 2);
    try std.testing.expectEqual(@as(u32, 5), out[0].token); // tie: lower global token wins
    try std.testing.expectEqual(@as(u32, 201), out[1].token);
    try std.testing.expectEqual(@as(u32, 207), out[2].token);
    try std.testing.expectEqual(@as(u32, 309), out[3].token);
}

test "TP top-k merge rejects malformed shapes and columns" {
    var out: [2]Candidate = undefined;
    try std.testing.expectError(error.Shape, merge(&out, &.{ 1, 0 }, &.{ 0, 1 }, &.{0}, 1, 1));
    try std.testing.expectError(error.BadColumn, merge(&out, &.{ 1, 0 }, &.{ -1, 1 }, &.{0}, 1, 2));
}
