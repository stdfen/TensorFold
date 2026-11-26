//! Projection metadata for the official mixed checkpoint, independent of EXL3.

const std = @import("std");
const pk = @import("pack.zig");

pub const Kind = enum { bf16, mxfp8, mxfp4, wo_a };

pub const Projection = struct {
    kind: Kind,
    weight: pk.Info,
    scale: ?pk.Info,
    n: usize,
    k: usize,
};

pub fn read(pack: *const pk.Pack, prefix: []const u8) !Projection {
    var buf: [512]u8 = undefined;
    const weight = try pack.need(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}));
    if (weight.rank != 2 or weight.shape[0] == 0 or weight.shape[1] == 0) return error.BadNativeProjection;
    if (weight.dtype == .bf16)
        return .{ .kind = .bf16, .weight = weight, .scale = null, .n = weight.shape[0], .k = weight.shape[1] };

    const scale = try pack.need(try std.fmt.bufPrint(&buf, "{s}.scale", .{prefix}));
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
    if (weight.dtype == .i8 and expertPrefix(prefix)) {
        const n = weight.shape[0];
        const k = std.math.mul(usize, weight.shape[1], 2) catch return error.BadNativeProjection;
        if (k % 32 != 0 or scale.shape[0] != n or scale.shape[1] != k / 32) return error.BadNativeScale;
        return .{ .kind = .mxfp4, .weight = weight, .scale = scale, .n = n, .k = k };
    }
    return error.UnsupportedNativeProjection;
}

fn expertPrefix(prefix: []const u8) bool {
    if (!(std.mem.startsWith(u8, prefix, "layers.") or std.mem.startsWith(u8, prefix, "mtp."))) return false;
    if (std.mem.indexOf(u8, prefix, ".ffn.experts.") == null) return false;
    return std.mem.endsWith(u8, prefix, ".w1") or std.mem.endsWith(u8, prefix, ".w2") or std.mem.endsWith(u8, prefix, ".w3");
}

fn divCeil(value: usize, divisor: usize) usize {
    return value / divisor + @intFromBool(value % divisor != 0);
}

test "official projection kinds stay separate from EXL3" {
    const a = std.testing.allocator;
    var pack = pk.Pack.init(a);
    defer pack.deinit();
    const bytes = try pk.image(a, &.{
        .{ .name = "embed.weight", .dtype = "BF16", .shape = &.{ 64, 32 } },
        .{ .name = "layers.0.attn.wq_a.weight", .dtype = "F8_E4M3", .shape = &.{ 64, 96 } },
        .{ .name = "layers.0.attn.wq_a.scale", .dtype = "F8_E8M0", .shape = &.{ 2, 3 } },
        .{ .name = "layers.0.attn.wo_a.weight", .dtype = "F8_E4M3", .shape = &.{ 64, 96 } },
        .{ .name = "layers.0.attn.wo_a.scale", .dtype = "F8_E8M0", .shape = &.{ 2, 3 } },
        .{ .name = "layers.0.ffn.experts.7.w2.weight", .dtype = "I8", .shape = &.{ 32, 48 } },
        .{ .name = "layers.0.ffn.experts.7.w2.scale", .dtype = "F8_E8M0", .shape = &.{ 32, 3 } },
    });
    defer a.free(bytes);
    try pack.addHeaderDump("native.safetensors", bytes);
    try std.testing.expectEqual(Kind.bf16, (try read(&pack, "embed")).kind);
    try std.testing.expectEqual(Kind.mxfp8, (try read(&pack, "layers.0.attn.wq_a")).kind);
    try std.testing.expectEqual(Kind.wo_a, (try read(&pack, "layers.0.attn.wo_a")).kind);
    const expert = try read(&pack, "layers.0.ffn.experts.7.w2");
    try std.testing.expectEqual(Kind.mxfp4, expert.kind);
    try std.testing.expectEqual(@as(usize, 96), expert.k);
}

test "official projection refuses dense-style MXFP4 scale geometry" {
    const a = std.testing.allocator;
    var pack = pk.Pack.init(a);
    defer pack.deinit();
    const bytes = try pk.image(a, &.{
        .{ .name = "layers.0.ffn.experts.0.w1.weight", .dtype = "I8", .shape = &.{ 64, 32 } },
        .{ .name = "layers.0.ffn.experts.0.w1.scale", .dtype = "F8_E8M0", .shape = &.{ 2, 2 } },
    });
    defer a.free(bytes);
    try pack.addHeaderDump("bad.safetensors", bytes);
    try std.testing.expectError(error.BadNativeScale, read(&pack, "layers.0.ffn.experts.0.w1"));
}
