//! Call emission for the official-mixed checkpoint path. Existing EXL3 emitters remain the default.

const block = @import("block.zig");
const calls = @import("block_calls.zig");

const E = block.Emitter;
const Arg = calls.Arg;

/// Translate the established emitter prefixes to the pinned official checkpoint names.
pub fn officialPrefix(e: *E, name: []const u8) ![]const u8 {
    if (@import("std").mem.eql(u8, name, "head") or @import("std").mem.eql(u8, name, "embed")) return e.a.dupe(u8, name);
    if (name.len < 3 or name[0] != 'L') return error.UnsupportedNativeProjection;
    const dot = @import("std").mem.indexOfScalar(u8, name, '.') orelse return error.UnsupportedNativeProjection;
    const layer = try @import("std").fmt.parseInt(u32, name[1..dot], 10);
    const rest = name[dot + 1 ..];
    if (@import("std").mem.startsWith(u8, rest, "attn.comp_")) {
        const tail = rest["attn.comp_".len..];
        return e.fmt("layers.{d}.attn.compressor.{s}", .{ layer, tail });
    }
    if (@import("std").mem.startsWith(u8, rest, "attn.ix_")) {
        const tail = rest["attn.ix_".len..];
        return e.fmt("layers.{d}.attn.indexer.{s}", .{ layer, tail });
    }
    return e.fmt("layers.{d}.{s}", .{ layer, rest });
}

fn cdiv(value: i64, divisor: i64) i64 {
    return @divFloor(value + divisor - 1, divisor);
}

pub fn mxfp8(e: *E, prefix: []const u8, x: Arg, y: Arg, k: i64, n: i64, dynamic: bool) !void {
    try e.ext("tf_dsv41_native_mixed_v1.mxfp8", &.{
        x,
        try e.weight("{s}.weight", .{prefix}, .u8, &.{ n, k }),
        try e.weight("{s}.scale", .{prefix}, .u8, &.{ cdiv(n, 32), cdiv(k, 32) }),
        y,
        .{ .b = dynamic },
    });
}

pub fn bf16(e: *E, prefix: []const u8, x: Arg, y: Arg, k: i64, n: i64) !void {
    try e.ext("tf_dsv41_native_mixed_v1.bf16", &.{
        x,
        try e.weight("{s}.weight", .{prefix}, .bf16, &.{ n, k }),
        try e.empty(.u8),
        y,
    });
}

pub fn woA(e: *E, prefix: []const u8, x: Arg, y: Arg, k: i64, n: i64, rank: i64) !void {
    if (rank <= 0 or @rem(n, rank) != 0) return error.BadNativeProjection;
    try e.ext("tf_dsv41_native_mixed_v1.wo_a", &.{
        x,
        try e.weight("{s}.weight", .{prefix}, .u8, &.{ n, k }),
        try e.weight("{s}.scale", .{prefix}, .u8, &.{ cdiv(n, 32), cdiv(k, 32) }),
        y,
        .{ .b = false },
        .{ .i = rank },
    });
}

fn woAGroup(e: *E, prefix: []const u8, group: i64, x: Arg, y: Arg, k: i64, n: i64) !void {
    const weight_name = try e.fmt("{s}.weight", .{prefix});
    const scale_name = try e.fmt("{s}.scale", .{prefix});
    const weight = try e.view(.{ .weight = weight_name }, .u8, &.{ n, k }, &.{ k, 1 }, group * n * k);
    const sn = cdiv(n, 32);
    const sk = cdiv(k, 32);
    const scale = try e.view(.{ .weight = scale_name }, .u8, &.{ sn, sk }, &.{ sk, 1 }, group * sn * sk);
    try e.ext("tf_dsv41_native_mixed_v1.wo_a", &.{ x, weight, scale, y, .{ .b = false }, .{ .i = n } });
}

pub fn projection(e: *E, name: []const u8, x: Arg, y: Arg, k: i64, n: i64) !void {
    const std = @import("std");
    const prefix = try officialPrefix(e, name);
    if (std.mem.indexOf(u8, name, ".attn.wo_a.")) |at| {
        const group = try std.fmt.parseInt(i64, name[at + ".attn.wo_a.".len ..], 10);
        const base = prefix[0 .. std.mem.lastIndexOfScalar(u8, prefix, '.').?];
        return woAGroup(e, base, group, x, y, k, n);
    }
    if (std.mem.eql(u8, name, "head") or std.mem.endsWith(u8, name, ".attn.ix_wk") or
        std.mem.endsWith(u8, name, ".attn.comp_wkv") or std.mem.endsWith(u8, name, ".attn.comp_wgate"))
        return bf16(e, prefix, x, y, k, n);
    return mxfp8(e, prefix, x, y, k, n, true);
}

/// Routed MXFP4 experts plus the final MXFP8 shared expert. The persistent pointer tables are bound at boot.
pub fn moe(e: *E, layer: u32, x: Arg, pick: Arg, route: Arg, out: Arg, rows: i64, slots: i64, routed: i64, hidden: i64, width: i64, limit: f64) !void {
    if (rows <= 0 or slots <= 0 or routed <= 0 or hidden <= 0 or width <= 0) return error.BadNativeMoe;
    const experts = routed + 1;
    const act = try e.buf("L.native.moe.act", .{}, .bf16, &.{ rows, slots, width });
    const table = struct {
        fn get(emitter: *E, L: u32, comptime name: []const u8, dt: calls.Dt, shape: []const i64) !Arg {
            return emitter.buf("s.native.L{d}.moe." ++ name, .{L}, dt, shape);
        }
    };
    const w1 = try table.get(e, layer, "w1", .i64, &.{experts});
    const s1 = try table.get(e, layer, "s1", .i64, &.{experts});
    const w3 = try table.get(e, layer, "w3", .i64, &.{experts});
    const s3 = try table.get(e, layer, "s3", .i64, &.{experts});
    const w2 = try table.get(e, layer, "w2", .i64, &.{experts});
    const s2 = try table.get(e, layer, "s2", .i64, &.{experts});
    const kinds = try table.get(e, layer, "kinds", .u8, &.{experts});
    try e.ext("tf_dsv41_native_moe_v1.gateup", &.{
        x, pick, w1, s1, w3, s3, kinds, act,
        .{ .i = rows }, .{ .i = slots }, .{ .i = experts }, .{ .i = hidden }, .{ .i = width }, .{ .f = limit },
    });
    try e.ext("tf_dsv41_native_moe_v1.down", &.{
        act, pick, route, w2, s2, kinds, out,
        .{ .i = rows }, .{ .i = slots }, .{ .i = experts }, .{ .i = hidden }, .{ .i = width },
    });
}
