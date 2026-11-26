//! Teacher-forced correctness capture over the production Forward.

const std = @import("std");
const Forward = @import("forward.zig").Forward;
const logits = @import("native_logits.zig");

pub const Capture = struct {
    candidates: []logits.Candidate,
    rows: usize,
    k: usize,

    pub fn deinit(c: Capture, allocator: std.mem.Allocator) void {
        allocator.free(c.candidates);
    }

    pub fn row(c: Capture, index: usize) []const logits.Candidate {
        return c.candidates[index * c.k ..][0..c.k];
    }
};

/// Capture each forced token's distribution, committing prior tokens as one-row decode windows after prefill.
pub fn capture(
    allocator: std.mem.Allocator,
    f: *Forward,
    prompt: []const u32,
    forced: []const u32,
    k: usize,
) !Capture {
    if (prompt.len == 0 or forced.len == 0 or k < 2) return error.Shape;
    const out = try allocator.alloc(logits.Candidate, forced.len * k);
    errdefer allocator.free(out);

    try f.prefill(prompt);
    try f.finishPrompt();
    if (f.last_rows == 0) return error.NoPromptLogits;
    try f.captureTopK(f.last_rows - 1, 1, k, out[0..k]);

    for (forced[0 .. forced.len - 1], 0..) |token, i| {
        const one = [_]u32{token};
        try f.window(&one);
        try f.captureTopK(0, 1, k, out[(i + 1) * k ..][0..k]);
        try f.keep(0);
    }
    return .{ .candidates = out, .rows = forced.len, .k = k };
}

test "capture row views preserve row-major candidates" {
    const a = std.testing.allocator;
    const items = try a.dupe(logits.Candidate, &.{
        .{ .token = 1, .logit = 3 },
        .{ .token = 2, .logit = 2 },
        .{ .token = 3, .logit = 1 },
        .{ .token = 4, .logit = 0 },
    });
    const c: Capture = .{ .candidates = items, .rows = 2, .k = 2 };
    defer c.deinit(a);
    try std.testing.expectEqual(@as(u32, 3), c.row(1)[0].token);
}
