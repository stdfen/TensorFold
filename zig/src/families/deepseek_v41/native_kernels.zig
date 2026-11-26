//! CUDA launch bindings for correctness-first official mixed-checkpoint projections.

const cuda = @import("cuda");

pub const Functions = struct {
    mxfp8: cuda.Function,
    mxfp8_plain: cuda.Function,
    mxfp4: cuda.Function,
    bf16: cuda.Function,
    wo_a: cuda.Function,
    mxfp8_bf16: cuda.Function,
    mxfp8_plain_bf16: cuda.Function,
    mxfp4_bf16: cuda.Function,
    bf16_bf16: cuda.Function,
    wo_a_bf16: cuda.Function,

    pub fn resolve(module: cuda.Module) !Functions {
        return .{
            .mxfp8 = try module.function("tf_dsv41_native_mxfp8"),
            .mxfp8_plain = try module.function("tf_dsv41_native_mxfp8_plain"),
            .mxfp4 = try module.function("tf_dsv41_native_mxfp4"),
            .bf16 = try module.function("tf_dsv41_native_bf16"),
            .wo_a = try module.function("tf_dsv41_native_wo_a"),
            .mxfp8_bf16 = try module.function("tf_dsv41_native_mxfp8_bf16"),
            .mxfp8_plain_bf16 = try module.function("tf_dsv41_native_mxfp8_plain_bf16"),
            .mxfp4_bf16 = try module.function("tf_dsv41_native_mxfp4_bf16"),
            .bf16_bf16 = try module.function("tf_dsv41_native_bf16_bf16"),
            .wo_a_bf16 = try module.function("tf_dsv41_native_wo_a_bf16"),
        };
    }
};

pub const Ops = struct {
    f: *const Functions,
    stream: cuda.Stream,

    fn go(o: Ops, f: cuda.Function, x: u64, weight: u64, scales: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0 or n == 0 or k == 0) return error.BadNativeProjection;
        var args: cuda.Args = .{};
        args.add(x);
        args.add(weight);
        args.add(scales);
        args.add(out);
        args.add(@as(i32, @intCast(rows)));
        args.add(@as(i32, @intCast(n)));
        args.add(@as(i32, @intCast(k)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * n + 127) / 128) }, .block = .{ .x = 128 } }, o.stream, &args);
    }

    pub fn mxfp8(o: Ops, x: u64, weight: u64, scales: u64, out: u64, rows: usize, n: usize, k: usize, dynamic: bool) !void {
        try o.mxfp8To(x, weight, scales, out, rows, n, k, dynamic, false);
    }

    pub fn mxfp8To(o: Ops, x: u64, weight: u64, scales: u64, out: u64, rows: usize, n: usize, k: usize, dynamic: bool, out_bf16: bool) !void {
        const f = if (out_bf16)
            (if (dynamic) o.f.mxfp8_bf16 else o.f.mxfp8_plain_bf16)
        else
            (if (dynamic) o.f.mxfp8 else o.f.mxfp8_plain);
        try o.go(f, x, weight, scales, out, rows, n, k);
    }

    pub fn mxfp4(o: Ops, x: u64, weight: u64, scales: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        try o.mxfp4To(x, weight, scales, out, rows, n, k, false);
    }

    pub fn mxfp4To(o: Ops, x: u64, weight: u64, scales: u64, out: u64, rows: usize, n: usize, k: usize, out_bf16: bool) !void {
        if (k % 32 != 0) return error.BadNativeProjection;
        try o.go(if (out_bf16) o.f.mxfp4_bf16 else o.f.mxfp4, x, weight, scales, out, rows, n, k);
    }

    pub fn bf16(o: Ops, x: u64, weight: u64, out: u64, rows: usize, n: usize, k: usize) !void {
        try o.bf16To(x, weight, out, rows, n, k, false);
    }

    pub fn bf16To(o: Ops, x: u64, weight: u64, out: u64, rows: usize, n: usize, k: usize, out_bf16: bool) !void {
        try o.go(if (out_bf16) o.f.bf16_bf16 else o.f.bf16, x, weight, 0, out, rows, n, k);
    }

    pub fn woA(o: Ops, x: u64, weight: u64, scales: u64, out: u64, rows: usize, groups: usize, rank: usize, k: usize) !void {
        try o.woATo(x, weight, scales, out, rows, groups, rank, k, false);
    }

    pub fn woATo(o: Ops, x: u64, weight: u64, scales: u64, out: u64, rows: usize, groups: usize, rank: usize, k: usize, out_bf16: bool) !void {
        if (groups == 0 or rank == 0) return error.BadNativeProjection;
        var args: cuda.Args = .{};
        for ([_]u64{ x, weight, scales, out }) |v| args.add(v);
        for ([_]usize{ rows, groups, rank, k }) |v| args.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(if (out_bf16) o.f.wo_a_bf16 else o.f.wo_a, .{ .grid = .{ .x = @intCast((rows * groups * rank + 127) / 128) }, .block = .{ .x = 128 } }, o.stream, &args);
    }
};

test "native mixed launch geometry refuses invalid official shapes before CUDA" {
    const fake: Ops = undefined;
    try @import("std").testing.expectError(error.BadNativeProjection, fake.mxfp4(0, 0, 0, 0, 1, 1, 31));
    try @import("std").testing.expectError(error.BadNativeProjection, fake.woA(0, 0, 0, 0, 1, 0, 1, 32));
}
