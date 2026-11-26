//! Host-only DeepSeek-V4.1 official-checkpoint admission tool.
//! This keeps M1 verification independent of CUDA and the served engine build.

const std = @import("std");
const native = @import("src/families/deepseek_v41/native_checkpoint.zig");
const payload = @import("src/families/deepseek_v41/native_payload.zig");

const usage =
    \\usage: tf-dsv41-native inspect-native MODEL
    \\       tf-dsv41-native plan-native MODEL --world N [--json OUT]
    \\       tf-dsv41-native payload-native MODEL TENSOR --rank R --world N
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 3 and std.mem.eql(u8, args[1], "inspect-native"))
        return run(init, args[2], 1, null);
    if ((args.len == 5 or args.len == 7) and
        std.mem.eql(u8, args[1], "plan-native") and
        std.mem.eql(u8, args[3], "--world") and
        (args.len == 5 or std.mem.eql(u8, args[5], "--json")))
    {
        const world = std.fmt.parseInt(u32, args[4], 10) catch return usageError();
        return run(init, args[2], world, if (args.len == 7) args[6] else null);
    }
    if (args.len == 8 and std.mem.eql(u8, args[1], "payload-native") and
        std.mem.eql(u8, args[4], "--rank") and std.mem.eql(u8, args[6], "--world"))
    {
        const rank = std.fmt.parseInt(u32, args[5], 10) catch return usageError();
        const world = std.fmt.parseInt(u32, args[7], 10) catch return usageError();
        return runPayload(init, args[2], args[3], rank, world);
    }
    return usageError();
}

fn runPayload(init: std.process.Init, dir: []const u8, name: []const u8, rank: u32, world: u32) u8 {
    if (world == 0 or rank >= world) return usageError();
    var checkpoint = native.Checkpoint.open(init.gpa, init.io, dir, world) catch |err| {
        std.debug.print("FAIL tf-dsv41-native payload admission: {t}\n", .{err});
        return 1;
    };
    defer checkpoint.deinit();
    var tensor = payload.read(init.gpa, init.io, &checkpoint.pack, name, rank, world) catch |err| {
        std.debug.print("FAIL tf-dsv41-native payload read: {t}\n", .{err});
        return 1;
    };
    defer init.gpa.free(tensor.bytes);
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(tensor.bytes, &sha, .{});
    const hex = std.fmt.bytesToHex(sha, .lower);
    std.debug.print("payload: {s} rank {d}/{d} {t} {any} {d} bytes sha256 {s}\n", .{
        name, rank, world, tensor.dtype, tensor.dims(), tensor.bytes.len, &hex,
    });
    return 0;
}

fn usageError() u8 {
    std.debug.print("{s}", .{usage});
    return 2;
}

fn run(init: std.process.Init, dir: []const u8, world: u32, json_path: ?[]const u8) u8 {
    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buf);
    native.inspect(init.gpa, init.io, dir, world, &stdout.interface) catch |err| {
        std.debug.print("FAIL tf-dsv41-native: {t}\n", .{err});
        return 1;
    };
    stdout.interface.flush() catch return 1;
    if (json_path) |path| {
        var file = std.Io.Dir.cwd().createFile(init.io, path, .{}) catch |err| {
            std.debug.print("FAIL tf-dsv41-native --json: {t}\n", .{err});
            return 1;
        };
        defer file.close(init.io);
        var json_buf: [64 << 10]u8 = undefined;
        var json_out = file.writerStreaming(init.io, &json_buf);
        native.writeJson(init.gpa, init.io, dir, world, &json_out.interface) catch |err| {
            std.debug.print("FAIL tf-dsv41-native --json: {t}\n", .{err});
            return 1;
        };
        json_out.interface.flush() catch return 1;
    }
    return 0;
}
