//! Launch bindings for the correctness-first official routed/shared MoE.

const cuda = @import("cuda");

pub const Functions = struct {
    gateup: cuda.Function,
    down: cuda.Function,

    pub fn resolve(module: cuda.Module) !Functions {
        return .{
            .gateup = try module.function("tf_dsv41_native_moe_gateup"),
            .down = try module.function("tf_dsv41_native_moe_down"),
        };
    }
};

pub const Ops = struct {
    f: *const Functions,
    stream: cuda.Stream,

    pub const Plan = struct { rows: usize, slots: usize, experts: usize, hidden: usize, width: usize };

    pub fn gateup(o: Ops, x: u64, pick: u64, w1: u64, s1: u64, w3: u64, s3: u64, kinds: u64, act: u64, p: Plan, limit: f32) !void {
        try validate(p);
        var args: cuda.Args = .{};
        for ([_]u64{ x, pick, w1, s1, w3, s3, kinds, act }) |v| args.add(v);
        for ([_]usize{ p.rows, p.slots, p.experts, p.hidden, p.width }) |v| args.add(@as(i32, @intCast(v)));
        args.add(limit);
        const count = p.rows * p.slots * p.width;
        try cuda.launch.launch(o.f.gateup, .{ .grid = .{ .x = @intCast((count + 127) / 128) }, .block = .{ .x = 128 } }, o.stream, &args);
    }

    pub fn down(o: Ops, act: u64, pick: u64, route: u64, w2: u64, s2: u64, kinds: u64, out: u64, p: Plan) !void {
        try validate(p);
        var args: cuda.Args = .{};
        for ([_]u64{ act, pick, route, w2, s2, kinds, out }) |v| args.add(v);
        for ([_]usize{ p.rows, p.slots, p.experts, p.hidden, p.width }) |v| args.add(@as(i32, @intCast(v)));
        const count = p.rows * p.hidden;
        try cuda.launch.launch(o.f.down, .{ .grid = .{ .x = @intCast((count + 127) / 128) }, .block = .{ .x = 128 } }, o.stream, &args);
    }

    fn validate(p: Plan) !void {
        if (p.rows == 0 or p.slots == 0 or p.experts == 0 or p.hidden == 0 or p.width == 0 or p.hidden % 32 != 0 or p.width % 32 != 0)
            return error.BadNativeMoe;
    }
};

test "native MoE launch plan refuses non-block official geometry before CUDA" {
    const fake: Ops = undefined;
    try @import("std").testing.expectError(error.BadNativeMoe, fake.down(0, 0, 0, 0, 0, 0, 0, .{ .rows = 1, .slots = 7, .experts = 385, .hidden = 5120, .width = 31 }));
}
