//! Minimal CUDA surface for compiling the DeepSeek official-weight container without a driver.

pub const Driver = opaque {};

pub const DeviceBuffer = struct {
    ptr: u64 = 0,
    len: usize = 0,

    pub fn alloc(_: *const Driver, len: usize) !DeviceBuffer {
        return .{ .len = len };
    }

    pub fn upload(self: DeviceBuffer, offset: usize, bytes: []const u8) !void {
        if (offset > self.len or bytes.len > self.len - offset) return error.Invalid;
    }

    pub fn free(self: *DeviceBuffer) void {
        self.* = undefined;
    }
};
