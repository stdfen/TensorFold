//! All-resident TP4 ownership for the official DeepSeek-V4.1-Flash-1017 checkpoint.
//!
//! This is the correctness bring-up owner. It preserves official matrix bytes, owns only
//! the small runtime forms and expert pointer tables beside them, and can bind an existing runner.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const checkpoint = @import("native_checkpoint.zig");
const native_weights = @import("native_cuda_weights.zig");

pub const Resident = struct {
    gpa: std.mem.Allocator,
    checkpoint: checkpoint.Checkpoint,
    weights: native_weights.Weights,
    tables: []native_weights.ExpertTables,

    /// Open one TP4 rank. Draft/MTP weights remain outside the text-only M3 scope.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, driver: *const cuda.Driver, dir: []const u8, rank: u32, cfg: *const Config) !Resident {
        const world: u32 = 4;
        if (rank >= world) return error.BadRank;
        if (cfg.layers == 0 or cfg.experts == 0 or cfg.shared_experts != 1) return error.UnsupportedNativeConfig;
        var ck = try checkpoint.Checkpoint.open(gpa, io, dir, world);
        errdefer ck.deinit();
        var weights = try native_weights.Weights.load(gpa, io, driver, &ck, rank);
        errdefer weights.deinit();
        const tables = try gpa.alloc(native_weights.ExpertTables, cfg.layers);
        var built: usize = 0;
        errdefer {
            for (tables[0..built]) |*table| table.deinit();
            gpa.free(tables);
        }
        while (built < tables.len) : (built += 1)
            tables[built] = try weights.expertTables(driver, @intCast(built), cfg.experts);
        return .{ .gpa = gpa, .checkpoint = ck, .weights = weights, .tables = tables };
    }

    /// Select the official weight backend and bind every persistent native-MoE role.
    pub fn bind(r: *Resident, runner: anytype) !void {
        runner.weights = null;
        runner.native_weights = &r.weights;
        for (r.tables, 0..) |*tables, layer| {
            try r.weights.bindExpertTables(runner, @intCast(layer), tables);
            inline for (.{ "hc_attn", "hc_ffn" }) |which| {
                var role_buf: [96]u8 = undefined;
                const role = try std.fmt.bufPrint(&role_buf, "s.L{d}.{s}.fn16", .{ layer, which });
                const form = r.weights.get(role) orelse return error.MissingNativeMhcForm;
                try runner.external(role, form.ptr);
            }
        }
    }

    pub fn deinit(r: *Resident) void {
        for (r.tables) |*tables| tables.deinit();
        r.gpa.free(r.tables);
        r.weights.deinit();
        r.checkpoint.deinit();
        r.* = undefined;
    }
};

test "native resident is pinned to text-only TP4 and all backbone expert tables" {
    const cfg: Config = .{};
    try std.testing.expectEqual(@as(u32, 40), cfg.layers);
    try std.testing.expectEqual(@as(u32, 384), cfg.experts);
    try std.testing.expectEqual(@as(u32, 1), cfg.shared_experts);
}
