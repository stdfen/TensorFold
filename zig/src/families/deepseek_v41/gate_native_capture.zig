//! `tf-dsv41 native-capture MODEL ASSETS REQUEST OUT`: official TP4 teacher-forced top-k capture.

const std = @import("std");
const model_mod = @import("cuda_model.zig");
const tf = @import("native_teacher_forced.zig");
const Candidate = @import("native_logits.zig").Candidate;

const Request = struct {
    name: []const u8,
    prompt_tokens: []const u32,
    forced_tokens: []const u32,
    top_k: usize = 16,
};

const Row = struct { position: usize, forced_token: u32, candidates: []const Candidate };
const Case = struct {
    name: []const u8,
    phase: []const u8,
    prompt_tokens: []const u32,
    forced_tokens: []const u32,
    rows: []const Row,
};

fn digest(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![64]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
    var raw: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &raw, .{});
    return std.fmt.bytesToHex(raw, .lower);
}

pub fn main(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, assets: []const u8, request_path: []const u8, out_path: []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const request_bytes = try std.Io.Dir.cwd().readFileAlloc(io, request_path, a, .limited(1 << 24));
    const parsed = try std.json.parseFromSlice(Request, a, request_bytes, .{ .ignore_unknown_fields = false });
    const request = parsed.value;
    if (request.name.len == 0 or request.prompt_tokens.len == 0 or request.forced_tokens.len < 2 or request.top_k < 2 or request.top_k > 64)
        return error.BadCaptureRequest;

    const m = try model_mod.Model.open(gpa, io, model_dir, assets, .{
        .drafts = false,
        .own_prefill = true,
        .native_official = true,
    });
    defer m.close();
    if (m.tcfg.rank != 0) {
        try m.f.follow();
        return 0;
    }

    const captured = try tf.capture(a, &m.f, request.prompt_tokens, request.forced_tokens, request.top_k);
    errdefer m.f.stop() catch {};
    const rows = try a.alloc(Row, request.forced_tokens.len);
    for (rows, request.forced_tokens, 0..) |*row, forced, i| row.* = .{
        .position = request.prompt_tokens.len + i,
        .forced_token = forced,
        .candidates = captured.candidates[i * request.top_k ..][0..request.top_k],
    };
    const prompt_name = try std.fmt.allocPrint(a, "{s}/prompt", .{request.name});
    const decode_name = try std.fmt.allocPrint(a, "{s}/decode", .{request.name});
    const cases = [_]Case{
        .{ .name = prompt_name, .phase = "prompt", .prompt_tokens = request.prompt_tokens, .forced_tokens = request.forced_tokens[0..1], .rows = rows[0..1] },
        .{ .name = decode_name, .phase = "decode", .prompt_tokens = request.prompt_tokens, .forced_tokens = request.forced_tokens[1..], .rows = rows[1..] },
    };
    const config_sha = try digest(a, io, model_dir, "config.json");
    const index_sha = try digest(a, io, model_dir, "model.safetensors.index.json");
    const document = .{
        .schema = "tensorfold.deepseek_v41.teacher_forced.v1",
        .checkpoint = .{ .config_sha256 = config_sha[0..], .index_sha256 = index_sha[0..] },
        .runtime = .{ .backend = "tensorfold-native-official", .world = m.tcfg.world },
        .settings = .{ .world = m.tcfg.world, .context = m.f.opts.limit, .greedy = true, .drafts = false, .engram = true },
        .cases = cases[0..],
    };
    const json = try std.json.Stringify.valueAlloc(a, document, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = json });
    try m.f.stop();
    return 0;
}
