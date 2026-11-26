//! Linux compile gate for the production official-mixed model and serving entry point.

const std = @import("std");

test {
    inline for (.{
        @import("cuda_model.zig"),
        @import("serve_engine.zig"),
        @import("native_resident.zig"),
        @import("gate_native_capture.zig"),
    }) |module| std.testing.refAllDecls(module);
}

test "official mixed serving switch is opt-in" {
    const options: @import("cuda_model.zig").Options = .{};
    try std.testing.expect(!options.native_official);
}
