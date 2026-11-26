//! Replays captured extension calls through the Zig launches, so block replay compares every .cu kernel's outputs.

const std = @import("std");
const cuda = @import("cuda");
const kernels = @import("kernels.zig");
const exl3 = @import("cuda").qlinear.block.experts;
const dense = @import("cuda").qlinear.block.dense;
const ops = @import("kernels_ops.zig");
const replay_experts = @import("kernels_replay_experts.zig");
const replay_ops = @import("kernels_replay_ops.zig");

const Kernels = kernels.Kernels;

pub const DType = enum { f16, bf16, f32, f64, i8, u8, i16, i32, i64, bool };
pub const Tensor = struct { ptr: u64, dtype: DType, shape: []const i64, stride: []const i64 }; // element strides, ptr already at storage_offset
pub const Arg = union(enum) { tensor: Tensor, int: i64, float: f64, boolean: bool, none, list: []const Arg };
pub const Kw = struct { name: []const u8, value: Arg };

/// Device bytes a call may use beyond its arguments (cleared per call by the caller).
pub const Scratch = struct {
    ptr: u64,
    len: usize,
    used: usize = 0,

    pub fn take(s: *Scratch, n: usize) !u64 {
        const at = std.mem.alignForward(usize, s.used, 256);
        if (at + n > s.len) return error.ScratchTooSmall;
        s.used = at + n;
        return s.ptr + at;
    }
};

pub const Error = error{ MissingArgument, WrongArgument, Unsupported };

/// The call's arguments by pybind position, or by keyword when passed so (`names` lists the binding's parameters).
pub const Call = struct {
    args: []const Arg,
    kwargs: []const Kw,
    names: []const []const u8,

    fn get(c: Call, i: usize) Error!Arg {
        if (i < c.args.len) return c.args[i];
        for (c.kwargs) |kw| if (std.mem.eql(u8, kw.name, c.names[i])) return kw.value;
        return error.MissingArgument;
    }

    pub fn opt(c: Call, i: usize) Error!?Arg {
        const a = c.get(i) catch |e| switch (e) {
            error.MissingArgument => return null,
            else => return e,
        };
        return if (a == .none) null else a;
    }

    pub fn tensor(c: Call, i: usize) Error!Tensor {
        return switch (try c.get(i)) {
            .tensor => |t| t,
            else => error.WrongArgument,
        };
    }

    /// An optional tensor's address (None or an empty tensor: 0, as the bindings pass nullptr).
    pub fn ptrOrNull(c: Call, i: usize) Error!u64 {
        const a = (try c.opt(i)) orelse return 0;
        return switch (a) {
            .tensor => |t| if (numel(t) == 0) 0 else t.ptr,
            else => error.WrongArgument,
        };
    }

    pub fn ptr(c: Call, i: usize) Error!u64 {
        return (try c.tensor(i)).ptr;
    }

    pub fn int(c: Call, i: usize) Error!i64 {
        return switch (try c.get(i)) {
            .int => |v| v,
            .boolean => |b| @intFromBool(b),
            else => error.WrongArgument,
        };
    }

    pub fn usz(c: Call, i: usize) Error!usize {
        const v = try c.int(i);
        if (v < 0) return error.WrongArgument;
        return @intCast(v);
    }

    pub fn u(c: Call, i: usize) Error!u32 {
        return @intCast(try c.usz(i));
    }

    pub fn float(c: Call, i: usize) Error!f64 {
        return switch (try c.get(i)) {
            .float => |v| v,
            .int => |v| @floatFromInt(v),
            else => error.WrongArgument,
        };
    }

    pub fn boolean(c: Call, i: usize, default: bool) Error!bool {
        const a = (try c.opt(i)) orelse return default;
        return switch (a) {
            .boolean => |b| b,
            .int => |v| v != 0,
            else => error.WrongArgument,
        };
    }

    pub fn list(c: Call, i: usize) Error![]const Arg {
        return switch (try c.get(i)) {
            .list => |l| l,
            else => error.WrongArgument,
        };
    }
};

pub fn numel(t: Tensor) i64 {
    var n: i64 = 1;
    for (t.shape) |d| n *= d;
    return n;
}

pub fn dim(t: Tensor, i: usize) Error!usize {
    if (i >= t.shape.len) return error.WrongArgument;
    return @intCast(t.shape[i]);
}

pub fn ioType(t: DType) Error!dense.DType {
    return switch (t) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        else => error.WrongArgument,
    };
}

pub fn input(t: DType) Error!exl3.Input {
    return switch (t) {
        .bf16 => .bf16,
        .f16 => .f16,
        else => error.WrongArgument,
    };
}

pub fn elem(t: DType) usize {
    return switch (t) {
        .i8, .u8, .bool => 1,
        .f16, .bf16, .i16 => 2,
        .f32, .i32 => 4,
        .f64, .i64 => 8,
    };
}

pub fn is(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// One captured extension call re-issued on `s`; false when there is no binding for it (reported as skipped).
pub fn call(k: *const Kernels, s: cuda.Stream, ext: []const u8, func: []const u8, args: []const Arg, kwargs: []const Kw, scratch: *Scratch) !bool {
    _ = scratch;
    if (is(ext, "tf_dsv41_kv_glue_v1") and is(func, "norm_store")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "w", "cs", "v", "s", "pos", "sl", "x_stride", "cs_stride", "v_stride", "s_stride", "inv_k", "eps", "ring", "rows" } };
        const x = try c.tensor(0);
        const rows = try c.boolean(14, false);
        if (x.dtype != .bf16 or try dim(x, 1) != 512 or (try c.tensor(1)).dtype != .f32 or
            (try c.tensor(2)).dtype != .f32 or (try c.tensor(5)).dtype != (if (rows) DType.i64 else DType.i32) or
            (rows and (try c.tensor(6)).dtype != .i64)) return error.WrongArgument;
        try k.kvStore(s).normStore(.{
            .x = x.ptr,
            .w = try c.ptr(1),
            .cs = try c.ptr(2),
            .v = try c.ptr(3),
            .s = try c.ptr(4),
            .pos = try c.ptr(5),
            .sl = try c.ptrOrNull(6),
            .x_stride = try c.int(7),
            .cs_stride = try c.int(8),
            .v_stride = try c.int(9),
            .s_stride = try c.int(10),
            .inv_k = @floatCast(try c.float(11)),
            .eps = @floatCast(try c.float(12)),
            .ring = @intCast(try c.int(13)),
            .rows = @intFromBool(rows),
        }, try dim(x, 0));
        return true;
    }
    if (is(ext, "tensorfold_exl3_linear_v4")) return linearExt(k.linears(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_native_mixed_v1")) return nativeExt(k.nativeMixed(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_native_moe_v1")) return nativeMoeExt(k.nativeMoe(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_x3seg_v3")) return segExt(k.linears(s), func, args, kwargs, false);
    if (is(ext, "tf_dsv41_dense3_v1")) return segExt(k.linears(s), func, args, kwargs, true);
    if (is(ext, "tensorfold_exl3_experts_v1")) return expertsExt(k.experts(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_x3ld_v1") or is(ext, "tf_dsv41_x3pf_v1")) return groupedExt(k.experts(s), ext, func, args, kwargs);
    if (is(ext, "tf_dsv41_x3gm_v1")) return x3gmExt(k.experts(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_x3ld_epi_v1")) return x3ldEpiExt(k.experts(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_attn_cuda_v1")) return attnExt(k.others(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_topk_b_v1") and is(func, "topk")) return topkCall(k.others(s), args, kwargs, true);
    if (is(ext, "tf_dsv41_mhc_cuda_v1") and is(func, "run")) return mhcRun(k.others(s), args, kwargs);
    if (is(ext, "tf_dsv41_mhc_cuda_v1") and is(func, "coef")) return mhcCoef(k.others(s), args, kwargs);
    if (is(ext, "tf_dsv41_mhc_pf_v1") and is(func, "run")) return mhcPfRun(k.others(s), args, kwargs);
    if (is(ext, "tf_dsv41_router_gemv_v2") and is(func, "route")) return route(k.others(s), args, kwargs);
    if (is(ext, "tf_dsv41_pfdense_v1")) return pfdenseExt(k.others(s), func, args, kwargs);
    if (is(ext, "tf_dsv41_l2pace_v1") and is(func, "paced")) return paced(k.others(s), args, kwargs);
    if (is(ext, "tf_dsv41_l2pf_v1") and is(func, "segments")) return segments(k.others(s), args, kwargs);
    return false; // engram_gate.gate_copy reads host-mapped words a capture cannot relocate: not replayed
}

fn nativeExt(native_ops: anytype, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "weight", "scale", "out", "dynamic", "wo_rank" } };
    const x = try c.tensor(0);
    const weight = try c.tensor(1);
    const out = try c.tensor(3);
    if (x.dtype != .bf16 or x.shape.len != 2 or weight.shape.len != 2 or (out.dtype != .f32 and out.dtype != .bf16) or out.shape.len != 2)
        return error.WrongArgument;
    const out_bf16 = out.dtype == .bf16;
    const rows = try dim(x, 0);
    const n = try dim(weight, 0);
    if (try dim(out, 0) != rows or try dim(out, 1) != n) return error.WrongArgument;
    if (is(func, "bf16")) {
        if (weight.dtype != .bf16 or try dim(x, 1) != try dim(weight, 1)) return error.WrongArgument;
        if (out_bf16) try native_ops.bf16To(x.ptr, weight.ptr, out.ptr, rows, n, try dim(weight, 1), true) else try native_ops.bf16(x.ptr, weight.ptr, out.ptr, rows, n, try dim(weight, 1));
        return true;
    }
    const scale = try c.tensor(2);
    if (scale.dtype != .u8 or scale.shape.len != 2) return error.WrongArgument;
    if (is(func, "mxfp8")) {
        const k = try dim(weight, 1);
        if (weight.dtype != .u8 or try dim(x, 1) != k or try dim(scale, 0) != divCeil(n, 32) or try dim(scale, 1) != divCeil(k, 32)) return error.WrongArgument;
        const dynamic = try c.boolean(4, true);
        if (out_bf16) try native_ops.mxfp8To(x.ptr, weight.ptr, scale.ptr, out.ptr, rows, n, k, dynamic, true) else try native_ops.mxfp8(x.ptr, weight.ptr, scale.ptr, out.ptr, rows, n, k, dynamic);
        return true;
    }
    if (is(func, "mxfp4")) {
        const k = 2 * try dim(weight, 1);
        if (weight.dtype != .i8 or try dim(x, 1) != k or try dim(scale, 0) != n or try dim(scale, 1) != k / 32) return error.WrongArgument;
        if (out_bf16) try native_ops.mxfp4To(x.ptr, weight.ptr, scale.ptr, out.ptr, rows, n, k, true) else try native_ops.mxfp4(x.ptr, weight.ptr, scale.ptr, out.ptr, rows, n, k);
        return true;
    }
    if (is(func, "wo_a")) {
        const k = try dim(weight, 1);
        const rank = try c.usz(5);
        if (weight.dtype != .u8 or rank == 0 or n % rank != 0 or try dim(x, 1) != k or try dim(scale, 0) != divCeil(n, 32) or try dim(scale, 1) != divCeil(k, 32)) return error.WrongArgument;
        if (out_bf16) try native_ops.woATo(x.ptr, weight.ptr, scale.ptr, out.ptr, rows, n / rank, rank, k, true) else try native_ops.woA(x.ptr, weight.ptr, scale.ptr, out.ptr, rows, n / rank, rank, k);
        return true;
    }
    return false;
}

fn divCeil(value: usize, divisor: usize) usize {
    return value / divisor + @intFromBool(value % divisor != 0);
}

fn nativeMoeExt(native_ops: anytype, func: []const u8, args: []const Arg, kwargs: []const Kw) !bool {
    if (is(func, "gateup")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "x", "pick", "w1", "s1", "w3", "s3", "kinds", "act", "rows", "slots", "experts", "hidden", "width", "limit" } };
        const x = try c.tensor(0);
        const pick = try c.tensor(1);
        const act = try c.tensor(7);
        if (x.dtype != .bf16 or pick.dtype != .i32 or act.dtype != .bf16) return error.WrongArgument;
        const p: @import("native_moe_kernels.zig").Ops.Plan = .{
            .rows = try c.usz(8), .slots = try c.usz(9), .experts = try c.usz(10), .hidden = try c.usz(11), .width = try c.usz(12),
        };
        try native_ops.gateup(x.ptr, pick.ptr, try c.ptr(2), try c.ptr(3), try c.ptr(4), try c.ptr(5), try c.ptr(6), act.ptr, p, @floatCast(try c.float(13)));
        return true;
    }
    if (is(func, "down")) {
        const c: Call = .{ .args = args, .kwargs = kwargs, .names = &.{ "act", "pick", "route", "w2", "s2", "kinds", "out", "rows", "slots", "experts", "hidden", "width" } };
        const act = try c.tensor(0);
        const pick = try c.tensor(1);
        const route_w = try c.tensor(2);
        const out = try c.tensor(6);
        if (act.dtype != .bf16 or pick.dtype != .i32 or route_w.dtype != .f32 or out.dtype != .f32) return error.WrongArgument;
        const p: @import("native_moe_kernels.zig").Ops.Plan = .{
            .rows = try c.usz(7), .slots = try c.usz(8), .experts = try c.usz(9), .hidden = try c.usz(10), .width = try c.usz(11),
        };
        try native_ops.down(act.ptr, pick.ptr, route_w.ptr, try c.ptr(3), try c.ptr(4), try c.ptr(5), out.ptr, p);
        return true;
    }
    return false;
}

test "official mixed replay dispatch validates packed MXFP4 geometry" {
    const Mock = struct {
        seen: *[3]usize,
        pub fn bf16(_: @This(), _: u64, _: u64, _: u64, _: usize, _: usize, _: usize) !void {}
        pub fn bf16To(_: @This(), _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: bool) !void {}
        pub fn mxfp8(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: bool) !void {}
        pub fn mxfp8To(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: bool, _: bool) !void {}
        pub fn mxfp4(m: @This(), _: u64, _: u64, _: u64, _: u64, rows: usize, n: usize, k: usize) !void {
            m.seen.* = .{ rows, n, k };
        }
        pub fn mxfp4To(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: bool) !void {}
        pub fn woA(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: usize) !void {}
        pub fn woATo(_: @This(), _: u64, _: u64, _: u64, _: u64, _: usize, _: usize, _: usize, _: usize, _: bool) !void {}
    };
    const x_shape = [_]i64{ 2, 32 };
    const w_shape = [_]i64{ 4, 16 };
    const s_shape = [_]i64{ 4, 1 };
    const o_shape = [_]i64{ 2, 4 };
    const stride = [_]i64{ 1, 1 };
    const args = [_]Arg{
        .{ .tensor = .{ .ptr = 1, .dtype = .bf16, .shape = &x_shape, .stride = &stride } },
        .{ .tensor = .{ .ptr = 2, .dtype = .i8, .shape = &w_shape, .stride = &stride } },
        .{ .tensor = .{ .ptr = 3, .dtype = .u8, .shape = &s_shape, .stride = &stride } },
        .{ .tensor = .{ .ptr = 4, .dtype = .f32, .shape = &o_shape, .stride = &stride } },
    };
    var seen: [3]usize = undefined;
    try std.testing.expect(try nativeExt(Mock{ .seen = &seen }, "mxfp4", &args, &.{}));
    try std.testing.expectEqual([3]usize{ 2, 4, 32 }, seen);
}

const linearExt = replay_experts.linearExt;
const segExt = replay_experts.segExt;
const expertsExt = replay_experts.expertsExt;
const groupedExt = replay_experts.groupedExt;
const x3gmExt = replay_experts.x3gmExt;
const x3ldEpiExt = replay_experts.x3ldEpiExt;
const attnExt = replay_ops.attnExt;
const topkCall = replay_ops.topkCall;
const mhcRun = replay_ops.mhcRun;
const mhcCoef = replay_ops.mhcCoef;
const mhcPfRun = replay_ops.mhcPfRun;
const route = replay_ops.route;
const pfdenseExt = replay_ops.pfdenseExt;
const paced = replay_ops.paced;
const segments = replay_ops.segments;

test "arguments by position and keyword" {
    const shape = [_]i64{ 2, 3 };
    const t: Tensor = .{ .ptr = 64, .dtype = .bf16, .shape = &shape, .stride = &.{ 3, 1 } };
    const c: Call = .{ .args = &.{ .{ .tensor = t }, .{ .int = 7 } }, .kwargs = &.{.{ .name = "pdl", .value = .{ .boolean = true } }}, .names = &.{ "x", "n", "z", "pdl" } };
    try std.testing.expectEqual(@as(u64, 64), try c.ptr(0));
    try std.testing.expectEqual(@as(usize, 7), try c.usz(1));
    try std.testing.expectEqual(@as(u64, 0), try c.ptrOrNull(2));
    try std.testing.expect(try c.boolean(3, false));
    try std.testing.expectError(error.WrongArgument, c.tensor(1));
}
