//! Cross-compile gate for DeepSeek-V4.1 official-mixed production boot on Linux AArch64.

const std = @import("std");
const dsv41 = @import("zig/build/dsv41_cuda.zig");

pub fn build(b: *std.Build) void {
    const target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .gnu });
    const module = dsv41.module(b, target, "native_model_gate.zig", false);
    const tests = b.addTest(.{ .root_module = module });
    b.default_step.dependOn(&tests.step);
    b.step("test", "Cross-compile the official-mixed production model gate for Linux AArch64").dependOn(&tests.step);
    @import("zig/build/sys.zig").wire(b);
}
