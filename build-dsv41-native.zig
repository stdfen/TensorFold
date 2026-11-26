//! Isolated DeepSeek-V4.1 official-mixed compile gate.
//!
//! This intentionally avoids configuring unrelated model families, so native-path failures remain attributable.

const std = @import("std");
const dsv41 = @import("zig/build/dsv41_cuda.zig");

pub fn build(b: *std.Build) void {
    const module = dsv41.module(b, b.graph.host, "native_cuda_gate.zig", false);
    const run = b.addRunArtifact(b.addTest(.{ .root_module = module }));
    b.default_step.dependOn(&run.step);
    b.step("test", "Compile and run the isolated official-mixed host contract gate").dependOn(&run.step);
    @import("zig/build/sys.zig").wire(b);
}
