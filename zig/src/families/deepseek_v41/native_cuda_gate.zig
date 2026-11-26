//! Host-only compile gate for the official mixed CUDA path.

const std = @import("std");
const dk = @import("dsv41_kernels");

test {
    inline for (.{
        @import("native_cuda_weights.zig"),
        @import("native_cuda_projection.zig"),
        @import("native_resident.zig"),
        @import("native_emit.zig"),
        @import("native_logits.zig"),
        @import("block_moe.zig"),
        @import("block_attn.zig"),
        @import("block_prefill.zig"),
        @import("block_prefill_attn.zig"),
        @import("block_prefill_moe.zig"),
        dk.native,
        dk.native_moe,
        dk.replay,
    }) |module| std.testing.refAllDecls(module);
}

test "official mixed decode emits native dense and MoE calls without EXL3 widths" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const block = @import("block.zig");
    const cfg: @import("config.zig").Config = .{};
    const widths: block.Widths = .{};
    const calls = try block.emit(a, &cfg, &widths, .{
        .native_mixed = true,
        .world = 4,
        .rank = 0,
        .expert_rows = 1,
        .l2pf_budget = 0,
    }, &.{0}, 1, 0, false);
    var dense: usize = 0;
    var moe: usize = 0;
    for (calls) |call| {
        if (std.mem.startsWith(u8, call.name, "tf_dsv41_native_mixed_v1.")) dense += 1;
        if (std.mem.startsWith(u8, call.name, "tf_dsv41_native_moe_v1.")) moe += 1;
        try std.testing.expect(!std.mem.startsWith(u8, call.name, "tensorfold_exl3_"));
        try std.testing.expect(!std.mem.startsWith(u8, call.name, "tf_dsv41_x3"));
    }
    try std.testing.expect(dense > 0);
    try std.testing.expectEqual(@as(usize, 2), moe);
}

test "official mixed prefill emits native dense and MoE calls without EXL3 widths" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const block = @import("block.zig");
    const block_prefill = @import("block_prefill.zig");
    const cfg: @import("config.zig").Config = .{};
    const widths: block.Widths = .{};
    const calls = try block_prefill.emitPrefill(a, &cfg, &widths, .{
        .native_mixed = true,
        .world = 4,
        .rank = 0,
        .expert_rows = 1,
        .l2pf_budget = 0,
        .limit = 4096,
        .rope_rows = 4160,
    }, &.{0}, 32, 0, true);
    var dense: usize = 0;
    var moe: usize = 0;
    for (calls) |call| {
        if (std.mem.startsWith(u8, call.name, "tf_dsv41_native_mixed_v1.")) dense += 1;
        if (std.mem.startsWith(u8, call.name, "tf_dsv41_native_moe_v1.")) moe += 1;
        try std.testing.expect(!std.mem.startsWith(u8, call.name, "tensorfold_exl3_"));
        try std.testing.expect(!std.mem.startsWith(u8, call.name, "tf_dsv41_x3"));
    }
    try std.testing.expect(dense > 0);
    try std.testing.expectEqual(@as(usize, 2), moe);
}

test "official mixed full backbone emits every layer without packed-weight calls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const block = @import("block.zig");
    const cfg: @import("config.zig").Config = .{};
    const widths: block.Widths = .{};
    var layers: [40]u32 = undefined;
    for (&layers, 0..) |*layer, i| layer.* = @intCast(i);
    const emitted = try block.emit(a, &cfg, &widths, .{
        .native_mixed = true,
        .world = 4,
        .rank = 0,
        .expert_rows = 1,
        .l2pf_budget = 0,
    }, &layers, 1, 0, false);
    var moe: usize = 0;
    for (emitted) |call| {
        if (std.mem.startsWith(u8, call.name, "tf_dsv41_native_moe_v1.")) moe += 1;
        try std.testing.expect(!std.mem.startsWith(u8, call.name, "tensorfold_exl3_"));
        try std.testing.expect(!std.mem.startsWith(u8, call.name, "tf_dsv41_x3"));
    }
    try std.testing.expectEqual(@as(usize, 80), moe);
}
