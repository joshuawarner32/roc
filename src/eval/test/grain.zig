const std = @import("std");

const ValueIdx = struct { index: usize };

pub const Value = union(enum) {
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    i128: i128,
    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    u128: u128,
    float: f64,
    bool: bool,
    str: []const u8,
    list: struct {
        elements: []Value,
    },
    record: struct {
        fields: []Field,
    },
    tag: struct {
        name: []const u8,
        arguments: []Value,
    },
    closure: struct {
        function: *Func,
        captured_variables: *Scope,
    },

    fn ty(self: *const Value) Ty {
        return switch (self.*) {
            .i8 => Ty{ .i8 = {} },
            .i16 => Ty{ .i16 = {} },
            .i32 => Ty{ .i32 = {} },
            .i64 => Ty{ .i64 = {} },
            .i128 => Ty{ .i128 = {} },
            .u8 => Ty{ .u8 = {} },
            .u16 => Ty{ .u16 = {} },
            .u32 => Ty{ .u32 = {} },
            .u64 => Ty{ .u64 = {} },
            .u128 => Ty{ .u128 = {} },
            .float => Ty{ .float = {} },
            .bool => Ty{ .bool = {} },
            .str => Ty{ .str = {} },
            .list => @panic("list type inference not implemented"),
            .record => @panic("record type inference not implemented"),
            .tag => @panic("tag type inference not implemented"),
            .closure => @panic("closure type inference not implemented"),
        };
    }

    pub fn format(self: Value, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        var buffer = std.ArrayList(u8).init(std.heap.page_allocator);
        defer buffer.deinit();
        try self.formatWithIndent(&buffer, 0, 80);
        try writer.writeAll(buffer.items);
    }

    pub fn formatWithIndent(self: Value, buffer: *std.ArrayList(u8), indent: ?u32, max_width: u32) std.fmt.AllocPrintError!void {
        switch (self) {
            .i8 => |i| try buffer.writer().print("{}i8", .{i}),
            .i16 => |i| try buffer.writer().print("{}i16", .{i}),
            .i32 => |i| try buffer.writer().print("{}i32", .{i}),
            .i64 => |i| try buffer.writer().print("{}i64", .{i}),
            .i128 => |i| try buffer.writer().print("{}i128", .{i}),
            .u8 => |i| try buffer.writer().print("{}u8", .{i}),
            .u16 => |i| try buffer.writer().print("{}u16", .{i}),
            .u32 => |i| try buffer.writer().print("{}u32", .{i}),
            .u64 => |i| try buffer.writer().print("{}u64", .{i}),
            .u128 => |i| try buffer.writer().print("{}u128", .{i}),
            .float => |f| try buffer.writer().print("{d}", .{f}),
            .bool => |b| try buffer.writer().print("{}", .{b}),
            .str => |s| try buffer.writer().print("\"{s}\"", .{s}),
            .list => |lst| {
                if (indent == null) {
                    try buffer.writer().print("List(", .{});
                    for (lst.elements, 0..) |element, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try element.formatWithIndent(buffer, null, max_width);
                    }
                    try buffer.writer().print(")", .{});
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("List(", .{});
                for (lst.elements, 0..) |element, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try element.formatWithIndent(buffer, null, max_width);
                }
                try buffer.writer().print(")", .{});

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("List(\n", .{});
                for (lst.elements, 0..) |element, i| {
                    if (i > 0) try buffer.writer().print(",\n", .{});
                    try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                    try element.formatWithIndent(buffer, current_indent + 4, max_width);
                }
                try buffer.writer().print("\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent);
                try buffer.writer().print(")", .{});
            },
            .record => |rec| {
                if (indent == null) {
                    try buffer.writer().print("{{", .{});
                    for (rec.fields, 0..) |field, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try buffer.writer().print("{s}: ", .{field.name});
                        try field.value.formatWithIndent(buffer, null, max_width);
                    }
                    try buffer.writer().print("}}", .{});
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("{{", .{});
                for (rec.fields, 0..) |field, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try buffer.writer().print("{s}: ", .{field.name});
                    try field.value.formatWithIndent(buffer, null, max_width);
                }
                try buffer.writer().print("}}", .{});

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("{{\n", .{});
                for (rec.fields, 0..) |field, i| {
                    if (i > 0) try buffer.writer().print(",\n", .{});
                    try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                    try buffer.writer().print("{s}: ", .{field.name});
                    try field.value.formatWithIndent(buffer, current_indent + 4, max_width);
                }
                try buffer.writer().print("\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent);
                try buffer.writer().print("}}", .{});
            },
            .tag => |tag| {
                if (indent == null) {
                    try buffer.writer().print("{s}(", .{tag.name});
                    for (tag.arguments, 0..) |arg, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try arg.formatWithIndent(buffer, null, max_width);
                    }
                    try buffer.writer().print(")", .{});
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("{s}(", .{tag.name});
                for (tag.arguments, 0..) |arg, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try arg.formatWithIndent(buffer, null, max_width);
                }
                try buffer.writer().print(")", .{});

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("{s}(\n", .{tag.name});
                for (tag.arguments, 0..) |arg, i| {
                    if (i > 0) try buffer.writer().print(",\n", .{});
                    try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                    try arg.formatWithIndent(buffer, current_indent + 4, max_width);
                }
                try buffer.writer().print("\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent);
                try buffer.writer().print(")", .{});
            },
            .closure => |closure| {
                if (indent == null) {
                    try buffer.writer().print("|", .{});
                    for (closure.function.parameter_types, 0..) |param, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try buffer.writer().print("{}", .{param});
                    }
                    try buffer.writer().print("| ", .{});
                    try closure.function.body.formatWithIndent(buffer, null, max_width);
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("|", .{});
                for (closure.function.parameter_types, 0..) |param, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try buffer.writer().print("{}", .{param});
                }
                try buffer.writer().print("| ", .{});
                try closure.function.body.formatWithIndent(buffer, null, max_width);

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("|", .{});
                for (closure.function.parameter_types, 0..) |param, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try buffer.writer().print("{}", .{param});
                }
                try buffer.writer().print("| ", .{});
                try closure.function.body.formatWithIndent(buffer, current_indent, max_width);
            },
        }
    }

    pub fn generateRandom(allocator: std.mem.Allocator, random: std.Random, typ: *const Ty) !Value {
        return switch (typ.*) {
            .i8 => Value{ .i8 = random.int(i8) },
            .i16 => Value{ .i16 = random.int(i16) },
            .i32 => Value{ .i32 = random.int(i32) },
            .i64 => Value{ .i64 = random.int(i64) },
            .i128 => Value{ .i128 = random.int(i128) },
            .u8 => Value{ .u8 = random.int(u8) },
            .u16 => Value{ .u16 = random.int(u16) },
            .u32 => Value{ .u32 = random.int(u32) },
            .u64 => Value{ .u64 = random.int(u64) },
            .u128 => Value{ .u128 = random.int(u128) },
            .float => Value{ .float = random.float(f64) },
            .bool => Value{ .bool = random.boolean() },
            .str => blk: {
                const length = random.int(u32) % 10 + 1; // Random string length between 1 and 10
                var buffer = allocator.alloc(u8, length) catch return error.OutOfMemory;
                for (0..length) |i| {
                    buffer[i] = @as(u8, random.int(u8) % 26 + 97); // Random lowercase letters
                }
                break :blk Value{ .str = buffer }; // Don't free the buffer, it's owned by the Value
            },
            .list => {
                @panic("Generating list literals is not supported - generate an Expr instead");
            },
            .record => {
                @panic("Generating record literals is not supported - generate an Expr instead");
            },
            .tag => {
                @panic("Generating tag literals is not supported - generate an Expr instead");
            },
            .function => {
                @panic("Generating function literals is not supported - generate an Expr instead");
            },
        };
    }

    pub fn add(self: Value, other: Value) !Value {
        return switch (self) {
            .i8 => |a| switch (other) {
                .i8 => |b| Value{ .i8 = a +% b },
                else => error.TypeMismatch,
            },
            .i16 => |a| switch (other) {
                .i16 => |b| Value{ .i16 = a +% b },
                else => error.TypeMismatch,
            },
            .i32 => |a| switch (other) {
                .i32 => |b| Value{ .i32 = a +% b },
                else => error.TypeMismatch,
            },
            .i64 => |a| switch (other) {
                .i64 => |b| Value{ .i64 = a +% b },
                else => error.TypeMismatch,
            },
            .i128 => |a| switch (other) {
                .i128 => |b| Value{ .i128 = a +% b },
                else => error.TypeMismatch,
            },
            .u8 => |a| switch (other) {
                .u8 => |b| Value{ .u8 = a +% b },
                else => error.TypeMismatch,
            },
            .u16 => |a| switch (other) {
                .u16 => |b| Value{ .u16 = a +% b },
                else => error.TypeMismatch,
            },
            .u32 => |a| switch (other) {
                .u32 => |b| Value{ .u32 = a +% b },
                else => error.TypeMismatch,
            },
            .u64 => |a| switch (other) {
                .u64 => |b| Value{ .u64 = a +% b },
                else => error.TypeMismatch,
            },
            .u128 => |a| switch (other) {
                .u128 => |b| Value{ .u128 = a +% b },
                else => error.TypeMismatch,
            },
            .float => |a| switch (other) {
                .float => |b| Value{ .float = a + b },
                else => error.TypeMismatch,
            },
            else => error.TypeMismatch,
        };
    }

    pub fn subtract(self: Value, other: Value) !Value {
        return switch (self) {
            .i8 => |a| switch (other) {
                .i8 => |b| Value{ .i8 = a -% b },
                else => error.TypeMismatch,
            },
            .i16 => |a| switch (other) {
                .i16 => |b| Value{ .i16 = a -% b },
                else => error.TypeMismatch,
            },
            .i32 => |a| switch (other) {
                .i32 => |b| Value{ .i32 = a -% b },
                else => error.TypeMismatch,
            },
            .i64 => |a| switch (other) {
                .i64 => |b| Value{ .i64 = a -% b },
                else => error.TypeMismatch,
            },
            .i128 => |a| switch (other) {
                .i128 => |b| Value{ .i128 = a -% b },
                else => error.TypeMismatch,
            },
            .u8 => |a| switch (other) {
                .u8 => |b| Value{ .u8 = a -% b },
                else => error.TypeMismatch,
            },
            .u16 => |a| switch (other) {
                .u16 => |b| Value{ .u16 = a -% b },
                else => error.TypeMismatch,
            },
            .u32 => |a| switch (other) {
                .u32 => |b| Value{ .u32 = a -% b },
                else => error.TypeMismatch,
            },
            .u64 => |a| switch (other) {
                .u64 => |b| Value{ .u64 = a -% b },
                else => error.TypeMismatch,
            },
            .u128 => |a| switch (other) {
                .u128 => |b| Value{ .u128 = a -% b },
                else => error.TypeMismatch,
            },
            .float => |a| switch (other) {
                .float => |b| Value{ .float = a - b },
                else => error.TypeMismatch,
            },
            else => error.TypeMismatch,
        };
    }

    pub fn multiply(self: Value, other: Value) !Value {
        return switch (self) {
            .i8 => |a| switch (other) {
                .i8 => |b| Value{ .i8 = a *% b },
                else => error.TypeMismatch,
            },
            .i16 => |a| switch (other) {
                .i16 => |b| Value{ .i16 = a *% b },
                else => error.TypeMismatch,
            },
            .i32 => |a| switch (other) {
                .i32 => |b| Value{ .i32 = a *% b },
                else => error.TypeMismatch,
            },
            .i64 => |a| switch (other) {
                .i64 => |b| Value{ .i64 = a *% b },
                else => error.TypeMismatch,
            },
            .i128 => |a| switch (other) {
                .i128 => |b| Value{ .i128 = a *% b },
                else => error.TypeMismatch,
            },
            .u8 => |a| switch (other) {
                .u8 => |b| Value{ .u8 = a *% b },
                else => error.TypeMismatch,
            },
            .u16 => |a| switch (other) {
                .u16 => |b| Value{ .u16 = a *% b },
                else => error.TypeMismatch,
            },
            .u32 => |a| switch (other) {
                .u32 => |b| Value{ .u32 = a *% b },
                else => error.TypeMismatch,
            },
            .u64 => |a| switch (other) {
                .u64 => |b| Value{ .u64 = a *% b },
                else => error.TypeMismatch,
            },
            .u128 => |a| switch (other) {
                .u128 => |b| Value{ .u128 = a *% b },
                else => error.TypeMismatch,
            },
            .float => |a| switch (other) {
                .float => |b| Value{ .float = a * b },
                else => error.TypeMismatch,
            },
            else => error.TypeMismatch,
        };
    }

    pub fn divide(self: Value, other: Value) !Value {
        return switch (self) {
            .i8 => |a| switch (other) {
                .i8 => |b| if (b == 0) error.DivideByZero else Value{ .i8 = @divTrunc(a, b) },
                else => error.TypeMismatch,
            },
            .i16 => |a| switch (other) {
                .i16 => |b| if (b == 0) error.DivideByZero else Value{ .i16 = @divTrunc(a, b) },
                else => error.TypeMismatch,
            },
            .i32 => |a| switch (other) {
                .i32 => |b| if (b == 0) error.DivideByZero else Value{ .i32 = @divTrunc(a, b) },
                else => error.TypeMismatch,
            },
            .i64 => |a| switch (other) {
                .i64 => |b| if (b == 0) error.DivideByZero else Value{ .i64 = @divTrunc(a, b) },
                else => error.TypeMismatch,
            },
            .i128 => |a| switch (other) {
                .i128 => |b| if (b == 0) error.DivideByZero else Value{ .i128 = @divTrunc(a, b) },
                else => error.TypeMismatch,
            },
            .u8 => |a| switch (other) {
                .u8 => |b| if (b == 0) error.DivideByZero else Value{ .u8 = a / b },
                else => error.TypeMismatch,
            },
            .u16 => |a| switch (other) {
                .u16 => |b| if (b == 0) error.DivideByZero else Value{ .u16 = a / b },
                else => error.TypeMismatch,
            },
            .u32 => |a| switch (other) {
                .u32 => |b| if (b == 0) error.DivideByZero else Value{ .u32 = a / b },
                else => error.TypeMismatch,
            },
            .u64 => |a| switch (other) {
                .u64 => |b| if (b == 0) error.DivideByZero else Value{ .u64 = a / b },
                else => error.TypeMismatch,
            },
            .u128 => |a| switch (other) {
                .u128 => |b| if (b == 0) error.DivideByZero else Value{ .u128 = a / b },
                else => error.TypeMismatch,
            },
            .float => |a| switch (other) {
                .float => |b| if (b == 0.0) error.DivideByZero else Value{ .float = a / b },
                else => error.TypeMismatch,
            },
            else => error.TypeMismatch,
        };
    }

    pub fn equals(self: Value, other: Value) !Value {
        const result = switch (self) {
            .i8 => |a| switch (other) {
                .i8 => |b| a == b,
                else => false,
            },
            .i16 => |a| switch (other) {
                .i16 => |b| a == b,
                else => false,
            },
            .i32 => |a| switch (other) {
                .i32 => |b| a == b,
                else => false,
            },
            .i64 => |a| switch (other) {
                .i64 => |b| a == b,
                else => false,
            },
            .i128 => |a| switch (other) {
                .i128 => |b| a == b,
                else => false,
            },
            .u8 => |a| switch (other) {
                .u8 => |b| a == b,
                else => false,
            },
            .u16 => |a| switch (other) {
                .u16 => |b| a == b,
                else => false,
            },
            .u32 => |a| switch (other) {
                .u32 => |b| a == b,
                else => false,
            },
            .u64 => |a| switch (other) {
                .u64 => |b| a == b,
                else => false,
            },
            .u128 => |a| switch (other) {
                .u128 => |b| a == b,
                else => false,
            },
            .float => |a| switch (other) {
                .float => |b| a == b,
                else => false,
            },
            .bool => |a| switch (other) {
                .bool => |b| a == b,
                else => false,
            },
            .str => |a| switch (other) {
                .str => |b| std.mem.eql(u8, a, b),
                else => false,
            },
            else => false, // Lists, records, tags, closures not comparable for now
        };
        return Value{ .bool = result };
    }

    pub fn notEquals(self: Value, other: Value) !Value {
        const eq_result = try self.equals(other);
        return Value{ .bool = !eq_result.bool };
    }
    pub fn lessThan(self: Value, other: Value) !Value {
        const result = switch (self) {
            .i8 => |a| switch (other) {
                .i8 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .i16 => |a| switch (other) {
                .i16 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .i32 => |a| switch (other) {
                .i32 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .i64 => |a| switch (other) {
                .i64 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .i128 => |a| switch (other) {
                .i128 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u8 => |a| switch (other) {
                .u8 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u16 => |a| switch (other) {
                .u16 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u32 => |a| switch (other) {
                .u32 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u64 => |a| switch (other) {
                .u64 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u128 => |a| switch (other) {
                .u128 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .float => |a| switch (other) {
                .float => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            else => {
                std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                return error.TypeMismatch;
            },
        };
        return Value{ .bool = result };
    }

    pub fn greaterThan(self: Value, other: Value) !Value {
        const result = switch (self) {
            .i8 => |a| switch (other) {
                .i8 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .i16 => |a| switch (other) {
                .i16 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .i32 => |a| switch (other) {
                .i32 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .i64 => |a| switch (other) {
                .i64 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .i128 => |a| switch (other) {
                .i128 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u8 => |a| switch (other) {
                .u8 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u16 => |a| switch (other) {
                .u16 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u32 => |a| switch (other) {
                .u32 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u64 => |a| switch (other) {
                .u64 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .u128 => |a| switch (other) {
                .u128 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            .float => |a| switch (other) {
                .float => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    return error.TypeMismatch;
                },
            },
            else => {
                std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                return error.TypeMismatch;
            },
        };
        return Value{ .bool = result };
    }

    pub fn andValue(self: Value, other: Value) !Value {
        return switch (self) {
            .bool => |a| switch (other) {
                .bool => |b| Value{ .bool = a and b },
                else => {
                    std.debug.print("Type mismatch in and: {s}\n", .{self.ty()});
                    return error.TypeMismatch;
                },
            },
            else => {
                std.debug.print("Type mismatch in and: {s}\n", .{self.ty()});
                return error.TypeMismatch;
            },
        };
    }

    pub fn orValue(self: Value, other: Value) !Value {
        return switch (self) {
            .bool => |a| switch (other) {
                .bool => |b| Value{ .bool = a or b },
                else => {
                    std.debug.print("Type mismatch in or: {s}\n", .{self.ty()});
                    return error.TypeMismatch;
                },
            },
            else => {
                std.debug.print("Type mismatch in or: {s}\n", .{self.ty()});
                return error.TypeMismatch;
            },
        };
    }

    pub fn negate(self: Value) !Value {
        return switch (self) {
            .i8 => |a| Value{ .i8 = -%a },
            .i16 => |a| Value{ .i16 = -%a },
            .i32 => |a| Value{ .i32 = -%a },
            .i64 => |a| Value{ .i64 = -%a },
            .i128 => |a| Value{ .i128 = -%a },
            .float => |a| Value{ .float = -a },
            else => {
                std.debug.print("Type mismatch in negate: {s}\n", .{self.ty()});
                return error.TypeMismatch;
            },
        };
    }

    pub fn not(self: Value) !Value {
        return switch (self) {
            .bool => |a| Value{ .bool = !a },
            else => {
                std.debug.print("Type mismatch in not: {s}\n", .{self.ty()});
                return error.TypeMismatch;
            },
        };
    }
};

const Field = struct {
    name: []const u8,
    value: Value,
};

const FieldExpr = struct {
    name: []const u8,
    value: Expr,
};

const TyIdx = struct { index: usize };

const Var = struct { name: []const u8 };

const FunctionTy = struct {
    parameter_types: []*const Ty,
    return_type: *const Ty,
};

pub const Ty = union(enum) {
    i8,
    i16,
    i32,
    i64,
    i128,
    u8,
    u16,
    u32,
    u64,
    u128,
    float,
    bool,
    str,
    list: *const Ty,
    record: struct {
        fields: []const FieldTy,
    },
    tag: struct {
        name: []const u8,
        arguments: []*const Ty,
    },
    function: FunctionTy,

    pub fn equals(self: *const Ty, other: *const Ty) bool {
        if (self == other) return true;
        return switch (self.*) {
            .i8 => |_| other.* == .i8,
            .i16 => |_| other.* == .i16,
            .i32 => |_| other.* == .i32,
            .i64 => |_| other.* == .i64,
            .i128 => |_| other.* == .i128,
            .u8 => |_| other.* == .u8,
            .u16 => |_| other.* == .u16,
            .u32 => |_| other.* == .u32,
            .u64 => |_| other.* == .u64,
            .u128 => |_| other.* == .u128,
            .float => |_| other.* == .float,
            .bool => |_| other.* == .bool,
            .str => |_| other.* == .str,
            .list => |lst| other.* == .list and lst == other.list,
            .record => |rec| {
                if (other.* != .record) return false;
                if (rec.fields.len != other.record.fields.len) return false;
                for (rec.fields, other.record.fields) |field, other_field| {
                    if (!field.eql(other_field)) return false;
                }
                return true;
            },
            .tag => |tag| {
                if (other.* != .tag) return false;
                if (!std.mem.eql(u8, tag.name, other.tag.name)) {
                    return false;
                }
                if (tag.arguments.len != other.tag.arguments.len) {
                    return false;
                }
                for (tag.arguments, other.tag.arguments) |arg, other_arg| {
                    if (!arg.equals(other_arg)) {
                        return false;
                    }
                }
                return true;
            },
            .function => |func| {
                if (other.* != .function) return false;
                if (func.parameter_types.len != other.function.parameter_types.len) return false;
                if (!func.return_type.equals(other.function.return_type)) return false;
                for (func.parameter_types, other.function.parameter_types) |param, other_param| {
                    if (!param.equals(other_param)) return false;
                }
                return true;
            },
        };
    }

    pub fn format(self: Ty, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        switch (self) {
            .i8 => try writer.print("i8", .{}),
            .i16 => try writer.print("i16", .{}),
            .i32 => try writer.print("i32", .{}),
            .i64 => try writer.print("i64", .{}),
            .i128 => try writer.print("i128", .{}),
            .u8 => try writer.print("u8", .{}),
            .u16 => try writer.print("u16", .{}),
            .u32 => try writer.print("u32", .{}),
            .u64 => try writer.print("u64", .{}),
            .u128 => try writer.print("u128", .{}),
            .float => try writer.print("float", .{}),
            .bool => try writer.print("bool", .{}),
            .str => try writer.print("str", .{}),
            .list => |inner_ty| try writer.print("List({})", .{inner_ty.*}),
            .record => try writer.print("<record>", .{}),
            .tag => try writer.print("<tag>", .{}),
            .function => |func| {
                try writer.print("(", .{});
                for (func.parameter_types, 0..) |param, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{param});
                }
                try writer.print(" -> {})", .{func.return_type.*});
            },
        }
    }
};

const FieldTy = struct {
    name: []const u8,
    ty: *const Ty,

    pub fn eql(self: FieldTy, other: FieldTy) bool {
        return std.mem.eql(u8, self.name, other.name) and self.ty.equals(other.ty);
    }
};

const ExprIdx = struct { index: usize };

// For tracking bound variables during generation
const BoundVar = struct {
    name: []const u8,
    ty: Ty,
    is_pending_function: bool, // true if this is a function that needs body generation
};

// For tracking function parameters during generation
const Parameter = struct {
    name: []const u8,
    ty: Ty,
};

// Variable info for scope tracking
const VarInfo = struct {
    name: []const u8,
    ty: Ty,
};

// Scope tracking during expression generation
const GeneratingScope = struct {
    // Function parameters (in order)
    parameters: []const Parameter,
    // Assigned variables in body (in reverse order)
    assignments: std.ArrayList(BoundVar),
    // Parent scope (linked list)
    parent: ?*const GeneratingScope,

    fn init(allocator: std.mem.Allocator, parameters: []const Parameter, parent: ?*const GeneratingScope) GeneratingScope {
        return GeneratingScope{
            .parameters = parameters,
            .assignments = std.ArrayList(BoundVar).init(allocator),
            .parent = parent,
        };
    }

    fn deinit(self: *GeneratingScope) void {
        self.assignments.deinit();
    }

    // Find a variable by name in this scope or parent scopes
    fn findVar(self: *const GeneratingScope, name: []const u8) ?struct { ty: Ty, is_param: bool } {
        // Check parameters first
        for (self.parameters) |param| {
            if (std.mem.eql(u8, param.name, name)) {
                return .{ .ty = param.ty, .is_param = true };
            }
        }

        // Check assignments (most recent first)
        for (self.assignments.items) |assignment| {
            if (std.mem.eql(u8, assignment.name, name)) {
                return .{ .ty = assignment.ty, .is_param = false };
            }
        }

        // Check parent scope
        if (self.parent) |parent| {
            return parent.findVar(name);
        }

        return null;
    }

    // Get all available variables (parameters + assignments from all scopes)
    fn getAllVars(self: *const GeneratingScope, allocator: std.mem.Allocator) ![]VarInfo {
        var vars = std.ArrayList(VarInfo).init(allocator);

        // Add parameters
        for (self.parameters) |param| {
            try vars.append(.{ .name = param.name, .ty = param.ty });
        }

        // Add assignments (reverse order, so most recent first)
        for (self.assignments.items) |assignment| {
            if (!assignment.is_pending_function) { // Don't include pending functions as variables
                try vars.append(.{ .name = assignment.name, .ty = assignment.ty });
            }
        }

        // Add from parent scope
        if (self.parent) |parent| {
            const parent_vars = try parent.getAllVars(allocator);
            defer allocator.free(parent_vars);
            for (parent_vars) |parent_var| {
                try vars.append(parent_var);
            }
        }

        return vars.toOwnedSlice();
    }
};

const BinaryOp = enum {
    add,
    subtract,
    multiply,
    divide,
    equals,
    not_equals,
    less_than,
    greater_than,
    and_op,
    or_op,

    pub fn format(self: BinaryOp, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        const op_str = switch (self) {
            .add => "+",
            .subtract => "-",
            .multiply => "*",
            .divide => "/",
            .equals => "==",
            .not_equals => "!=",
            .less_than => "<",
            .greater_than => ">",
            .and_op => "and",
            .or_op => "or",
        };
        try writer.print("{s}", .{op_str});
    }
};

const UnaryOp = enum {
    negate,
    not,

    pub fn format(self: UnaryOp, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        const op_str = switch (self) {
            .negate => "-",
            .not => "!",
        };
        try writer.print("{s}", .{op_str});
    }
};

const GenError = error{ OutOfMemory, UnsupportedUnaryType };

const TbdReplaceError = error{OutOfMemory};

const ModificationIterator = struct {
    allocator: std.mem.Allocator,
    interp: *Interp,
    original_expr: *Expr,
    current_modifications: std.ArrayList(ModificationAttempt),
    child_iterators: std.ArrayList(*ModificationIterator),
    current_child_index: usize,
    self_simplified: bool,

    const ModificationAttempt = struct {
        path: []usize, // Path to the expression to modify
        replacement: Expr, // What to replace it with
    };

    pub fn init(allocator: std.mem.Allocator, interp: *Interp, expr: *Expr) !*ModificationIterator {
        const iterator = try allocator.create(ModificationIterator);
        iterator.* = ModificationIterator{
            .allocator = allocator,
            .interp = interp,
            .original_expr = expr,
            .current_modifications = std.ArrayList(ModificationAttempt).init(allocator),
            .child_iterators = std.ArrayList(*ModificationIterator).init(allocator),
            .current_child_index = 0,
            .self_simplified = false,
        };
        return iterator;
    }

    pub fn deinit(self: *ModificationIterator) void {
        for (self.child_iterators.items) |child| {
            child.deinit();
        }
        self.child_iterators.deinit();
        self.current_modifications.deinit();
        self.allocator.destroy(self);
    }

    pub fn next(self: *ModificationIterator) !?Expr {
        // First, try to simplify self (breadth-first)
        if (!self.self_simplified) {
            self.self_simplified = true;
            if (try self.getSimplestReplacement()) |simple| {
                return simple;
            }
        }

        // Then, try modifications from child expressions
        if (self.child_iterators.items.len == 0) {
            try self.initializeChildIterators();
        }

        // Try getting next modification from current child
        while (self.current_child_index < self.child_iterators.items.len) {
            const child_iterator = self.child_iterators.items[self.current_child_index];
            if (try child_iterator.next()) |child_modification| {
                // Apply this child modification to a copy of our expression
                var modified_expr = try self.copyExpr(self.original_expr.*);
                try self.applyChildModification(&modified_expr, self.current_child_index, child_modification);
                return modified_expr;
            } else {
                // This child is exhausted, move to next
                self.current_child_index += 1;
            }
        }

        // All children exhausted
        return null;
    }

    fn getSimplestReplacement(self: *ModificationIterator) !?Expr {
        // Check if the expression is already the simplest possible
        if (self.isAlreadySimplest(self.original_expr.*)) {
            return null; // No simpler replacement available
        }

        // Try to replace the entire expression with the simplest possible expression of the same type
        const expr_type = try self.inferType(self.original_expr.*);
        const simple_expr = try self.makeSimplestExpr(expr_type);
        return simple_expr;
    }

    fn makeSimplestExpr(self: *ModificationIterator, ty: Ty) !Expr {
        const value_ptr = try self.allocator.create(Value);
        value_ptr.* = switch (ty) {
            .i8 => Value{ .i8 = 0 },
            .i16 => Value{ .i16 = 0 },
            .i32 => Value{ .i32 = 0 },
            .i64 => Value{ .i64 = 0 },
            .i128 => Value{ .i128 = 0 },
            .u8 => Value{ .u8 = 0 },
            .u16 => Value{ .u16 = 0 },
            .u32 => Value{ .u32 = 0 },
            .u64 => Value{ .u64 = 0 },
            .u128 => Value{ .u128 = 0 },
            .float => Value{ .float = 0.0 },
            .bool => Value{ .bool = false },
            .str => Value{ .str = "" },
            .list => return Expr{ .list_literal = .{ .elements = &[_]Expr{} } },
            .record => return Expr{ .record_literal = .{ .fields = &[_]FieldExpr{} } },
            .tag => |tag_ty| return Expr{ .tag_literal = .{ .name = tag_ty.name, .arguments = &[_]Expr{} } },
            .function => {
                // Create simplest function: || 0 (or appropriate return type)
                const return_value_ptr = try self.allocator.create(Value);
                return_value_ptr.* = switch (ty.function.return_type.*) {
                    .i32 => Value{ .i32 = 0 },
                    .bool => Value{ .bool = false },
                    .str => Value{ .str = "" },
                    else => Value{ .i32 = 0 }, // fallback
                };
                const return_expr = Expr{ .literal = return_value_ptr };
                const block_ptr = try self.allocator.create(Block);
                block_ptr.* = Block{ .statements = &[_]Stmt{}, .return_expr = return_expr };
                const func_ptr = try self.allocator.create(Func);
                func_ptr.* = Func{ .parameter_types = &[_]*const Ty{}, .body = block_ptr };
                return Expr{ .function = func_ptr };
            },
        };
        return Expr{ .literal = value_ptr };
    }

    fn initializeChildIterators(self: *ModificationIterator) !void {
        switch (self.original_expr.*) {
            .binary => |bin| {
                if (!self.isAlreadySimplest(bin.left.*)) {
                    const left_iter = try ModificationIterator.init(self.allocator, self.interp, bin.left);
                    try self.child_iterators.append(left_iter);
                }
                if (!self.isAlreadySimplest(bin.right.*)) {
                    const right_iter = try ModificationIterator.init(self.allocator, self.interp, bin.right);
                    try self.child_iterators.append(right_iter);
                }
            },
            .unary => |un| {
                if (!self.isAlreadySimplest(un.operand.*)) {
                    const operand_iter = try ModificationIterator.init(self.allocator, self.interp, un.operand);
                    try self.child_iterators.append(operand_iter);
                }
            },
            .application => |app| {
                if (!self.isAlreadySimplest(app.function.*)) {
                    const func_iter = try ModificationIterator.init(self.allocator, self.interp, app.function);
                    try self.child_iterators.append(func_iter);
                }
                for (app.arguments) |arg| {
                    if (!self.isAlreadySimplest(arg.*)) {
                        const arg_iter = try ModificationIterator.init(self.allocator, self.interp, arg);
                        try self.child_iterators.append(arg_iter);
                    }
                }
            },
            .list_literal => |lst| {
                for (lst.elements) |*element| {
                    if (!self.isAlreadySimplest(element.*)) {
                        const elem_iter = try ModificationIterator.init(self.allocator, self.interp, element);
                        try self.child_iterators.append(elem_iter);
                    }
                }
            },
            .record_literal => |rec| {
                for (rec.fields) |*field| {
                    if (!self.isAlreadySimplest(field.value)) {
                        const field_iter = try ModificationIterator.init(self.allocator, self.interp, &field.value);
                        try self.child_iterators.append(field_iter);
                    }
                }
            },
            .tag_literal => |tag| {
                for (tag.arguments) |*arg| {
                    if (!self.isAlreadySimplest(arg.*)) {
                        const arg_iter = try ModificationIterator.init(self.allocator, self.interp, arg);
                        try self.child_iterators.append(arg_iter);
                    }
                }
            },
            .if_expression => |if_expr| {
                if (!self.isAlreadySimplest(if_expr.condition.*)) {
                    const cond_iter = try ModificationIterator.init(self.allocator, self.interp, if_expr.condition);
                    try self.child_iterators.append(cond_iter);
                }
                // Note: We'd need to add iterators for blocks too, but for simplicity we'll skip them for now
            },
            .field_access => |field| {
                if (!self.isAlreadySimplest(field.record.*)) {
                    const record_iter = try ModificationIterator.init(self.allocator, self.interp, field.record);
                    try self.child_iterators.append(record_iter);
                }
            },
            // Leaf nodes have no children
            .literal, .variable, .tbd => {},
            .function => {}, // Skip function bodies for now
        }
    }

    fn applyChildModification(self: *ModificationIterator, target: *Expr, child_index: usize, modification: Expr) !void {
        _ = self;
        var current_child: usize = 0;
        switch (target.*) {
            .binary => |*bin| {
                if (current_child == child_index) {
                    bin.left.* = modification;
                    return;
                }
                current_child += 1;
                if (current_child == child_index) {
                    bin.right.* = modification;
                    return;
                }
                current_child += 1;
            },
            .unary => |*un| {
                if (current_child == child_index) {
                    un.operand.* = modification;
                    return;
                }
                current_child += 1;
            },
            .application => |*app| {
                if (current_child == child_index) {
                    app.function.* = modification;
                    return;
                }
                current_child += 1;
                for (app.arguments) |arg| {
                    if (current_child == child_index) {
                        arg.* = modification;
                        return;
                    }
                    current_child += 1;
                }
            },
            .list_literal => |*lst| {
                for (lst.elements) |*element| {
                    if (current_child == child_index) {
                        element.* = modification;
                        return;
                    }
                    current_child += 1;
                }
            },
            .record_literal => |*rec| {
                for (rec.fields) |*field| {
                    if (current_child == child_index) {
                        field.value = modification;
                        return;
                    }
                    current_child += 1;
                }
            },
            .tag_literal => |*tag| {
                for (tag.arguments) |*arg| {
                    if (current_child == child_index) {
                        arg.* = modification;
                        return;
                    }
                    current_child += 1;
                }
            },
            .if_expression => |*if_expr| {
                if (current_child == child_index) {
                    if_expr.condition.* = modification;
                    return;
                }
                current_child += 1;
            },
            .field_access => |*field| {
                if (current_child == child_index) {
                    field.record.* = modification;
                    return;
                }
                current_child += 1;
            },
            else => {},
        }
    }

    fn copyExpr(self: *ModificationIterator, expr: Expr) !Expr {
        _ = self;
        // For simplicity, we'll do a shallow copy and rely on the fact that
        // we're only modifying pointers, not the underlying data
        return expr;
    }

    fn isAlreadySimplest(self: *ModificationIterator, expr: Expr) bool {
        _ = self;
        return switch (expr) {
            // These are already the simplest possible for their types
            .literal => |val| switch (val.*) {
                .i8 => |i| i == 0,
                .i16 => |i| i == 0,
                .i32 => |i| i == 0,
                .i64 => |i| i == 0,
                .i128 => |i| i == 0,
                .u8 => |i| i == 0,
                .u16 => |i| i == 0,
                .u32 => |i| i == 0,
                .u64 => |i| i == 0,
                .u128 => |i| i == 0,
                .float => |f| f == 0.0,
                .bool => |b| b == false,
                .str => |s| s.len == 0,
                .list => |lst| lst.elements.len == 0,
                .record => |rec| rec.fields.len == 0,
                .tag => |tag| tag.arguments.len == 0,
                else => false,
            },
            .list_literal => |lst| lst.elements.len == 0,
            .record_literal => |rec| rec.fields.len == 0,
            .tag_literal => |tag| tag.arguments.len == 0,
            // These can potentially be simplified further
            .binary, .unary, .application, .if_expression, .field_access => false,
            // Variables and functions are leaf nodes but might not be simplest
            .variable, .function, .tbd => false,
        };
    }

    fn inferType(self: *ModificationIterator, expr: Expr) !Ty {
        // Simple type inference - in a real implementation this would be more sophisticated
        return switch (expr) {
            .literal => |val| val.ty(),
            .tbd => |ty| ty.*,
            .binary => |bin| switch (bin.op) {
                .add, .subtract, .multiply, .divide => try self.inferType(bin.left.*),
                .equals, .not_equals, .less_than, .greater_than, .and_op, .or_op => Ty{ .bool = {} },
            },
            .unary => |un| switch (un.op) {
                .negate => try self.inferType(un.operand.*),
                .not => Ty{ .bool = {} },
            },
            else => Ty{ .i32 = {} }, // Default fallback
        };
    }
};

/// Minimize an expression to the smallest form that still satisfies the given predicate
pub fn minimize(allocator: std.mem.Allocator, interp: *Interp, expr: *Expr, predicate: fn (*Expr) bool) !Expr {
    var current_best = expr.*;
    var iterator = try ModificationIterator.init(allocator, interp, expr);
    defer iterator.deinit();

    // Try each modification from the iterator
    while (try iterator.next()) |candidate| {
        var candidate_copy = candidate;
        if (predicate(&candidate_copy)) {
            // This simpler expression still satisfies the predicate
            current_best = candidate;
            // Restart iteration with the new smaller expression
            iterator.deinit();
            iterator = try ModificationIterator.init(allocator, interp, &current_best);
        }
    }

    return current_best;
}

/// Minimize an expression to the smallest form that still satisfies the given predicate with context
pub fn minimizeWithContext(allocator: std.mem.Allocator, interp: *Interp, expr: *Expr, context: anytype, predicate: fn (*Expr, @TypeOf(context)) bool) !Expr {
    var current_best = expr.*;
    var iterator = try ModificationIterator.init(allocator, interp, expr);
    defer iterator.deinit();

    // Try each modification from the iterator
    while (try iterator.next()) |candidate| {
        var candidate_copy = candidate;
        if (predicate(&candidate_copy, context)) {
            // This simpler expression still satisfies the predicate
            current_best = candidate;
            // Restart iteration with the new smaller expression
            iterator.deinit();
            iterator = try ModificationIterator.init(allocator, interp, &current_best);
        }
    }

    return current_best;
}

pub const Expr = union(enum) {
    tbd: *const Ty,
    literal: *Value,
    variable: Var,
    function: *Func,
    application: struct {
        function: *Expr,
        arguments: []*Expr,
    },
    list_literal: struct {
        elements: []Expr,
    },
    record_literal: struct {
        fields: []FieldExpr,
    },
    tag_literal: struct {
        name: []const u8,
        arguments: []Expr,
    },
    binary: struct {
        op: BinaryOp,
        left: *Expr,
        right: *Expr,
    },
    unary: struct {
        op: UnaryOp,
        operand: *Expr,
    },
    field_access: struct {
        record: *Expr,
        field_name: []const u8,
    },
    if_expression: struct {
        condition: *Expr,
        then_branch: *Block,
        else_branch: *Block,
    },

    pub fn format(self: Expr, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        var buffer = std.ArrayList(u8).init(std.heap.page_allocator);
        defer buffer.deinit();
        try self.formatWithIndent(&buffer, 0, 80);
        try writer.writeAll(buffer.items);
    }

    pub fn formatWithIndent(self: Expr, buffer: *std.ArrayList(u8), indent: ?u32, max_width: u32) std.fmt.AllocPrintError!void {
        switch (self) {
            .tbd => |ty| {
                try buffer.writer().print("<tbd: {}>", .{ty.*});
            },
            .literal => |val| try val.formatWithIndent(buffer, indent, max_width),
            .variable => |v| try buffer.writer().print("{s}", .{v.name}),
            .function => |f| {
                if (indent == null) {
                    try buffer.writer().print("|", .{});
                    for (f.parameter_types, 0..) |param, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try buffer.writer().print("{}", .{param});
                    }
                    try buffer.writer().print("| ", .{});
                    try f.body.formatWithIndent(buffer, null, max_width);
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("|", .{});
                for (f.parameter_types, 0..) |param, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try buffer.writer().print("{}", .{param});
                }
                try buffer.writer().print("| ", .{});
                try f.body.formatWithIndent(buffer, null, max_width);

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("|", .{});
                for (f.parameter_types, 0..) |param, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try buffer.writer().print("{}", .{param});
                }
                try buffer.writer().print("| ", .{});
                try f.body.formatWithIndent(buffer, current_indent, max_width);
            },
            .application => |app| {
                if (indent == null) {
                    try app.function.formatWithIndent(buffer, null, max_width);
                    try buffer.writer().print("(", .{});
                    for (app.arguments, 0..) |arg, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try arg.formatWithIndent(buffer, null, max_width);
                    }
                    try buffer.writer().print(")", .{});
                    return;
                }

                const start_pos = buffer.items.len;
                try app.function.formatWithIndent(buffer, null, max_width);
                try buffer.writer().print("(", .{});
                for (app.arguments, 0..) |arg, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try arg.formatWithIndent(buffer, null, max_width);
                }
                try buffer.writer().print(")", .{});

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try app.function.formatWithIndent(buffer, current_indent, max_width);
                try buffer.writer().print("(\n", .{});
                for (app.arguments, 0..) |arg, i| {
                    if (i > 0) try buffer.writer().print(",\n", .{});
                    try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                    try arg.formatWithIndent(buffer, current_indent + 4, max_width);
                }
                try buffer.writer().print("\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent);
                try buffer.writer().print(")", .{});
            },
            .list_literal => |lst| {
                if (indent == null) {
                    try buffer.writer().print("[", .{});
                    for (lst.elements, 0..) |element, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try element.formatWithIndent(buffer, null, max_width);
                    }
                    try buffer.writer().print("]", .{});
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("[", .{});
                for (lst.elements, 0..) |element, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try element.formatWithIndent(buffer, null, max_width);
                }
                try buffer.writer().print("]", .{});

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("[\n", .{});
                for (lst.elements, 0..) |element, i| {
                    if (i > 0) try buffer.writer().print(",\n", .{});
                    try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                    try element.formatWithIndent(buffer, current_indent + 4, max_width);
                }
                try buffer.writer().print("\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent);
                try buffer.writer().print("]", .{});
            },
            .record_literal => |rec| {
                if (indent == null) {
                    try buffer.writer().print("{{", .{});
                    for (rec.fields, 0..) |field, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try buffer.writer().print("{s}: ", .{field.name});
                        try field.value.formatWithIndent(buffer, null, max_width);
                    }
                    try buffer.writer().print("}}", .{});
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("{{", .{});
                for (rec.fields, 0..) |field, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try buffer.writer().print("{s}: ", .{field.name});
                    try field.value.formatWithIndent(buffer, null, max_width);
                }
                try buffer.writer().print("}}", .{});

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("{{\n", .{});
                for (rec.fields, 0..) |field, i| {
                    if (i > 0) try buffer.writer().print(",\n", .{});
                    try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                    try buffer.writer().print("{s}: ", .{field.name});
                    try field.value.formatWithIndent(buffer, current_indent + 4, max_width);
                }
                try buffer.writer().print("\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent);
                try buffer.writer().print("}}", .{});
            },
            .tag_literal => |tag| {
                if (indent == null) {
                    try buffer.writer().print("{s}(", .{tag.name});
                    for (tag.arguments, 0..) |arg, i| {
                        if (i > 0) try buffer.writer().print(", ", .{});
                        try arg.formatWithIndent(buffer, null, max_width);
                    }
                    try buffer.writer().print(")", .{});
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("{s}(", .{tag.name});
                for (tag.arguments, 0..) |arg, i| {
                    if (i > 0) try buffer.writer().print(", ", .{});
                    try arg.formatWithIndent(buffer, null, max_width);
                }
                try buffer.writer().print(")", .{});

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("{s}(\n", .{tag.name});
                for (tag.arguments, 0..) |arg, i| {
                    if (i > 0) try buffer.writer().print(",\n", .{});
                    try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                    try arg.formatWithIndent(buffer, current_indent + 4, max_width);
                }
                try buffer.writer().print("\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent);
                try buffer.writer().print(")", .{});
            },
            .binary => |bin| {
                try buffer.writer().print("(", .{});
                try bin.left.formatWithIndent(buffer, indent, max_width);
                try buffer.writer().print(" {} ", .{bin.op});
                try bin.right.formatWithIndent(buffer, indent, max_width);
                try buffer.writer().print(")", .{});
            },
            .unary => |un| {
                try buffer.writer().print("(", .{});
                try buffer.writer().print("{}", .{un.op});
                try un.operand.formatWithIndent(buffer, indent, max_width);
                try buffer.writer().print(")", .{});
            },
            .field_access => |field| {
                try field.record.formatWithIndent(buffer, indent, max_width);
                try buffer.writer().print(".{s}", .{field.field_name});
            },
            .if_expression => |if_expr| {
                if (indent == null) {
                    try buffer.writer().print("if (", .{});
                    try if_expr.condition.formatWithIndent(buffer, null, max_width);
                    try buffer.writer().print(") ", .{});
                    try if_expr.then_branch.formatWithIndent(buffer, null, max_width);
                    try buffer.writer().print(" else ", .{});
                    try if_expr.else_branch.formatWithIndent(buffer, null, max_width);
                    return;
                }

                const start_pos = buffer.items.len;
                try buffer.writer().print("if (", .{});
                try if_expr.condition.formatWithIndent(buffer, null, max_width);
                try buffer.writer().print(") ", .{});
                try if_expr.then_branch.formatWithIndent(buffer, null, max_width);
                try buffer.writer().print(" else ", .{});
                try if_expr.else_branch.formatWithIndent(buffer, null, max_width);

                if (buffer.items.len - start_pos <= max_width) {
                    return;
                }

                buffer.shrinkRetainingCapacity(start_pos);
                const current_indent = indent.?;
                try buffer.writer().print("if (", .{});
                try if_expr.condition.formatWithIndent(buffer, current_indent + 4, max_width);
                try buffer.writer().print(")\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                try if_expr.then_branch.formatWithIndent(buffer, current_indent + 4, max_width);
                try buffer.writer().print("\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent);
                try buffer.writer().print("else\n", .{});
                try buffer.writer().writeByteNTimes(' ', current_indent + 4);
                try if_expr.else_branch.formatWithIndent(buffer, current_indent + 4, max_width);
            },
        }
    }

    pub fn generateRandom(allocator: std.mem.Allocator, random: *const std.Random, typ: *const Ty) GenError!Expr {
        // Create an empty root scope for simple generation
        var root_scope = GeneratingScope.init(allocator, &[_]Parameter{}, null);
        defer root_scope.deinit();
        return generateRandomWithScopeAndDepth(allocator, random, typ, &root_scope, 0);
    }

    pub fn generateRandomBlock(allocator: std.mem.Allocator, random: *const std.Random, typ: *const Ty) !*Block {
        // Create an empty root scope for block generation
        var root_scope = GeneratingScope.init(allocator, &[_]Parameter{}, null);
        defer root_scope.deinit();
        return generateBlockWithScope(allocator, random, typ, &root_scope, 0);
    }

    /// Replace all tbd expressions with simple concrete expressions of the appropriate type
    pub fn replaceTbdExpressions(self: *Expr, interp: *Interp) TbdReplaceError!void {
        switch (self.*) {
            .tbd => |ty| {
                // Replace this tbd with a simple expression of the same type
                self.* = try interp.makeSimpleExpr(ty);
                // Recursively clean the new expression too
                try self.replaceTbdExpressions(interp);
            },
            .literal => {}, // Nothing to clean
            .variable => {}, // Nothing to clean
            .function => |f| {
                try f.body.replaceTbdExpressions(interp);
            },
            .application => |app| {
                try app.function.replaceTbdExpressions(interp);
                for (app.arguments) |arg| {
                    try arg.replaceTbdExpressions(interp);
                }
            },
            .list_literal => |lst| {
                for (lst.elements) |*element| {
                    try element.replaceTbdExpressions(interp);
                }
            },
            .record_literal => |rec| {
                for (rec.fields) |*field| {
                    try field.value.replaceTbdExpressions(interp);
                }
            },
            .tag_literal => |tag| {
                for (tag.arguments) |*arg| {
                    try arg.replaceTbdExpressions(interp);
                }
            },
            .binary => |bin| {
                try bin.left.replaceTbdExpressions(interp);
                try bin.right.replaceTbdExpressions(interp);
            },
            .unary => |un| {
                try un.operand.replaceTbdExpressions(interp);
            },
            .field_access => |field| {
                try field.record.replaceTbdExpressions(interp);
            },
            .if_expression => |if_expr| {
                try if_expr.condition.replaceTbdExpressions(interp);
                try if_expr.then_branch.replaceTbdExpressions(interp);
                try if_expr.else_branch.replaceTbdExpressions(interp);
            },
        }
    }

    fn generateSimpleRandom(allocator: std.mem.Allocator, random: *const std.Random, typ: *const Ty, scope: *GeneratingScope, depth: u32) GenError!Expr {
        if (typ.* == .function) {
            return generateFunction(allocator, random, &typ.function, scope, depth);
        } else if (typ.* == .list) {
            // Generate a list literal
            return try Expr.generateList(allocator, random, typ.list);
        } else if (typ.* == .record) {
            // Generate a record literal
            return try Expr.generateRecord(allocator, random, typ.record.fields);
        } else if (typ.* == .tag) {
            // Generate a tag literal
            return try Expr.generateTag(allocator, random, typ.tag.name, typ.tag.arguments);
        } else {
            // Generate a literal value
            const value = try Value.generateRandom(allocator, random, typ);
            const value_ptr = try allocator.create(Value);
            value_ptr.* = value;
            return Expr{ .literal = value_ptr };
        }
    }

    fn generateRandomWithScopeAndDepth(
        allocator: std.mem.Allocator,
        random: *const std.Random,
        typ: *const Ty,
        scope: *GeneratingScope,
        depth: u32,
    ) GenError!Expr {
        // Prevent infinite recursion by limiting depth or for function types
        if (depth > 3 and typ.* != .function and typ.* != .list and typ.* != .record and typ.* != .tag) {
            // Generate a literal value
            const value = try Value.generateRandom(allocator, random, typ);
            const value_ptr = try allocator.create(Value);
            value_ptr.* = value;
            return Expr{ .literal = value_ptr };
        }

        // Get available variables in scope
        const available_vars = try scope.getAllVars(allocator);
        defer allocator.free(available_vars);

        // Filter variables that match the desired type
        var matching_vars = std.ArrayList([]const u8).init(allocator);
        defer matching_vars.deinit();
        for (available_vars) |var_info| {
            if (var_info.ty.equals(typ)) {
                try matching_vars.append(var_info.name);
            }
        }

        // Decide what kind of expression to generate
        // More options if we have variables available
        const base_options: u32 = 6; // literal, application, binary, unary, field_access, if
        const has_vars = matching_vars.items.len > 0;
        const total_options: u32 = if (has_vars) base_options + 1 else base_options;

        const which = random.int(u32) % total_options;
        switch (which) {
            0 => {
                return try generateSimpleRandom(allocator, random, typ, scope, depth);
            },
            1 => {
                // Generate an application that resolves to the given type
                return try generateApplication(allocator, random, typ, scope, depth);
            },
            2 => {
                // Generate a binary expression that resolves to the given type
                return try generateBinaryExpressionWithScope(allocator, random, typ, scope, depth);
            },
            3 => {
                // Generate a unary expression that resolves to the given type
                const res = generateUnaryExpression(allocator, random, typ, scope, depth) catch |err| {
                    if (err == error.UnsupportedUnaryType) {
                        return try generateSimpleRandom(allocator, random, typ, scope, depth);
                    } else {
                        return err; // Propagate other errors
                    }
                };

                return res;
            },
            4 => {
                // Generate a field access expression that resolves to the given type
                return try generateFieldAccess(allocator, random, typ, scope, depth);
            },
            5 => {
                // Generate an if expression that resolves to the given type
                return try generateIfExpressionWithScope(allocator, random, typ, scope, depth);
            },
            6 => {
                // Generate a variable reference (only if we have matching variables)
                if (matching_vars.items.len > 0) {
                    const chosen_var_name = matching_vars.items[random.int(usize) % matching_vars.items.len];
                    const v = Var{ .name = chosen_var_name };
                    return Expr{ .variable = v };
                } else {
                    // Fallback to literal
                    const value = try Value.generateRandom(allocator, random, typ);
                    const value_ptr = try allocator.create(Value);
                    value_ptr.* = value;
                    return Expr{ .literal = value_ptr };
                }
            },
            else => unreachable,
        }
    }

    fn generateApplication(
        allocator: std.mem.Allocator,
        random: *const std.Random,
        return_type: *const Ty,
        scope: *GeneratingScope,
        depth: u32,
    ) GenError!Expr {
        // Generate random argument types - for simplicity, use 0-2 arguments
        const num_args = random.int(u32) % 3;
        var arg_types = try allocator.alloc(Ty, num_args);

        // For "don't care" argument types, generate random basic types
        for (0..num_args) |i| {
            const arg_type_choice = random.int(u32) % 13;
            arg_types[i] = switch (arg_type_choice) {
                0 => Ty{ .i8 = {} },
                1 => Ty{ .i16 = {} },
                2 => Ty{ .i32 = {} },
                3 => Ty{ .i64 = {} },
                4 => Ty{ .i128 = {} },
                5 => Ty{ .u8 = {} },
                6 => Ty{ .u16 = {} },
                7 => Ty{ .u32 = {} },
                8 => Ty{ .u64 = {} },
                9 => Ty{ .u128 = {} },
                10 => Ty{ .float = {} },
                11 => Ty{ .bool = {} },
                12 => Ty{ .str = {} },
                else => unreachable,
            };
        }

        // Create the function type that will return our desired type
        const return_type_ptr = try allocator.create(Ty);
        return_type_ptr.* = return_type.*;

        const func_type = Ty{ .function = .{
            .parameter_types = arg_types,
            .return_type = return_type_ptr,
        } };

        // Generate a function expression that has this type (always a literal closure)
        const func_expr = try Expr.generateRandomWithScopeAndDepth(allocator, random, &func_type, scope, depth + 1);
        const func_expr_ptr = try allocator.create(Expr);
        func_expr_ptr.* = func_expr;

        // Generate arguments of the appropriate types (always literals)
        var arguments = try allocator.alloc(Expr, num_args);
        for (0..num_args) |i| {
            arguments[i] = try Expr.generateRandomWithScopeAndDepth(allocator, random, &arg_types[i], scope, depth + 1);
        }

        return Expr{ .application = .{
            .function = func_expr_ptr,
            .arguments = arguments,
        } };
    }

    fn generateList(allocator: std.mem.Allocator, random: *const std.Random, element_ty: *const Ty) GenError!Expr {
        var elements = std.ArrayList(Expr).init(allocator);
        errdefer elements.deinit();
        const count = random.int(u32) % 5 + 1; // Random list size between 1 and 5
        for (0..count) |_| {
            const element_value = try generateRandom(allocator, random, element_ty);
            try elements.append(element_value);
        }
        return Expr{ .list_literal = .{ .elements = try elements.toOwnedSlice() } };
    }

    fn generateRecord(allocator: std.mem.Allocator, random: *const std.Random, record_fields: []const FieldTy) GenError!Expr {
        var fields = std.ArrayList(FieldExpr).init(allocator);
        errdefer fields.deinit();
        for (record_fields) |field_ty| {
            const field_value = try generateRandom(allocator, random, &field_ty.ty);
            try fields.append(FieldExpr{ .name = field_ty.name, .value = field_value });
        }
        return Expr{ .record_literal = .{ .fields = try fields.toOwnedSlice() } };
    }

    fn generateTag(allocator: std.mem.Allocator, random: *const std.Random, name: []const u8, argument_tys: []const Ty) GenError!Expr {
        var arguments = std.ArrayList(Expr).init(allocator);
        errdefer arguments.deinit();
        const count = random.int(u32) % 3; // Random number of arguments between 0 and 2
        for (0..count) |_| {
            const arg_value = try generateRandom(allocator, random, &argument_tys[0]); // Use first argument type for simplicity
            try arguments.append(arg_value);
        }
        return Expr{ .tag_literal = .{ .name = name, .arguments = try arguments.toOwnedSlice() } };
    }

    fn generateBinaryExpression(allocator: std.mem.Allocator, random: *const std.Random, result_type: *const Ty, depth: u32) GenError!Expr {
        _ = depth; // Unused for now
        // Choose a binary operator based on the desired result type
        const ops_for_type = switch (result_type.*) {
            .i8, .i16, .i32, .i64, .i128, .u8, .u16, .u32, .u64, .u128 => &[_]BinaryOp{ .add, .subtract, .multiply, .divide },
            .bool => &[_]BinaryOp{ .equals, .not_equals, .less_than, .greater_than, .and_op, .or_op },
            else => &[_]BinaryOp{.equals}, // Default to equality for other types
        };

        const op = ops_for_type[random.int(usize) % ops_for_type.len];

        // Generate operand types based on the operator
        const operand_type = switch (op) {
            .add, .subtract, .multiply, .divide, .less_than, .greater_than => result_type.*, // Use the same type as result for arithmetic
            .and_op, .or_op => Ty{ .bool = {} },
            .equals, .not_equals => blk: {
                // For equality, use a random basic type
                const type_choice = random.int(u32) % 13;
                break :blk switch (type_choice) {
                    0 => Ty{ .i8 = {} },
                    1 => Ty{ .i16 = {} },
                    2 => Ty{ .i32 = {} },
                    3 => Ty{ .i64 = {} },
                    4 => Ty{ .i128 = {} },
                    5 => Ty{ .u8 = {} },
                    6 => Ty{ .u16 = {} },
                    7 => Ty{ .u32 = {} },
                    8 => Ty{ .u64 = {} },
                    9 => Ty{ .u128 = {} },
                    10 => Ty{ .float = {} },
                    11 => Ty{ .bool = {} },
                    12 => Ty{ .str = {} },
                    else => unreachable,
                };
            },
        };

        // Generate left and right operands (use literals to avoid deep recursion)
        const left_value = try Value.generateRandom(allocator, random, &operand_type);
        const left_value_ptr = try allocator.create(Value);
        left_value_ptr.* = left_value;
        const left_expr_ptr = try allocator.create(Expr);
        left_expr_ptr.* = Expr{ .literal = left_value_ptr };

        const right_value = try Value.generateRandom(allocator, random, &operand_type);
        const right_value_ptr = try allocator.create(Value);
        right_value_ptr.* = right_value;
        const right_expr_ptr = try allocator.create(Expr);
        right_expr_ptr.* = Expr{ .literal = right_value_ptr };

        return Expr{ .binary = .{
            .op = op,
            .left = left_expr_ptr,
            .right = right_expr_ptr,
        } };
    }

    fn generateUnaryExpression(allocator: std.mem.Allocator, random: *const std.Random, result_type: *const Ty, scope: *GeneratingScope, depth: u32) GenError!Expr {
        _ = depth; // Unused for now
        _ = scope; // Unused for now

        // Choose a unary operator based on the desired result type
        const op = switch (result_type.*) {
            .i8, .i16, .i32, .i64, .i128, .u8, .u16, .u32, .u64, .u128 => UnaryOp.negate,
            .bool => UnaryOp.not,
            else => return error.UnsupportedUnaryType,
        };

        // The operand type should match the result type for these operators
        const operand_type = result_type.*;

        // Generate operand (use literal to avoid deep recursion)
        const operand_value = try Value.generateRandom(allocator, random, &operand_type);
        const operand_value_ptr = try allocator.create(Value);
        operand_value_ptr.* = operand_value;
        const operand_expr_ptr = try allocator.create(Expr);
        operand_expr_ptr.* = Expr{ .literal = operand_value_ptr };

        return Expr{ .unary = .{
            .op = op,
            .operand = operand_expr_ptr,
        } };
    }

    fn generateFieldAccess(allocator: std.mem.Allocator, random: *const std.Random, field_type: *const Ty, scope: *GeneratingScope, depth: u32) GenError!Expr {
        // Generate a record with a field of the desired type
        const field_name = "field"; // Simple field name for now
        const field_ty = FieldTy{ .name = field_name, .ty = field_type.* };
        const record_fields = try allocator.alloc(FieldTy, 1);
        record_fields[0] = field_ty;

        const record_type = Ty{ .record = .{ .fields = record_fields } };

        const record_expr = try Expr.generateRandomWithScopeAndDepth(allocator, random, &record_type, scope, depth + 1);
        const record_expr_ptr = try allocator.create(Expr);
        record_expr_ptr.* = record_expr;

        return Expr{ .field_access = .{
            .record = record_expr_ptr,
            .field_name = field_name,
        } };
    }

    fn generateFunction(allocator: std.mem.Allocator, random: *const std.Random, func_ty: *const FunctionTy, scope: *GeneratingScope, depth: u32) GenError!Expr {
        _ = scope; // Unused for now
        _ = depth; // Unused for now

        // Generate a closure with the specified function type
        const func_ptr = try allocator.create(Func);

        // Copy parameter types
        var param_types = try allocator.alloc(Ty, func_ty.parameter_types.len);
        for (func_ty.parameter_types, 0..) |param_ty, i| {
            param_types[i] = param_ty;
        }

        // Create parameters for the function scope
        var parameters = try allocator.alloc(Parameter, func_ty.parameter_types.len);
        for (func_ty.parameter_types, 0..) |param_ty, i| {
            const param_name = try std.fmt.allocPrint(allocator, "arg{}", .{i});
            parameters[i] = Parameter{ .name = param_name, .ty = param_ty };
        }

        // Create function scope with parameters
        var func_scope = GeneratingScope.init(allocator, parameters, null);
        defer func_scope.deinit();

        // Generate a block with assignments and return expression
        const block_ptr = try Expr.generateBlockWithScope(allocator, random, func_ty.return_type, &func_scope, 0);

        func_ptr.* = Func{
            .parameter_types = param_types,
            .body = block_ptr,
        };

        return Expr{ .function = func_ptr };
    }

    fn generateBinaryExpressionWithScope(allocator: std.mem.Allocator, random: *const std.Random, result_type: *const Ty, scope: *GeneratingScope, depth: u32) GenError!Expr {
        _ = scope; // TODO: use scope for operand generation
        return generateBinaryExpression(allocator, random, result_type, depth);
    }

    fn generateIfExpressionWithScope(allocator: std.mem.Allocator, random: *const std.Random, result_type: *const Ty, scope: *GeneratingScope, depth: u32) GenError!Expr {
        // Generate a condition (always a boolean literal for simplicity)
        const condition_value = Value{ .bool = random.boolean() };
        const condition_value_ptr = try allocator.create(Value);
        condition_value_ptr.* = condition_value;
        const condition_expr_ptr = try allocator.create(Expr);
        condition_expr_ptr.* = Expr{ .literal = condition_value_ptr };

        // Generate then branch using scope-aware block generation
        const then_block_ptr = try generateBlockWithScope(allocator, random, result_type, scope, depth + 1);

        // Generate else branch using scope-aware block generation
        const else_block_ptr = try generateBlockWithScope(allocator, random, result_type, scope, depth + 1);

        return Expr{ .if_expression = .{
            .condition = condition_expr_ptr,
            .then_branch = then_block_ptr,
            .else_branch = else_block_ptr,
        } };
    }

    // Generate a Block with assignments and return expression
    fn generateBlockWithScope(allocator: std.mem.Allocator, random: *const std.Random, return_type: *const Ty, scope: *GeneratingScope, depth: u32) GenError!*Block {
        // Only generate assignments if we're not too deep (to avoid infinite recursion)
        const num_assignments = if (depth > 2) 0 else random.int(u32) % 3;

        var statements = std.ArrayList(Stmt).init(allocator);
        var local_scope = GeneratingScope.init(allocator, &[_]Parameter{}, scope);
        defer local_scope.deinit();

        // Generate assignments
        for (0..num_assignments) |i| {
            // Generate a random variable name
            const var_name = try std.fmt.allocPrint(allocator, "var{}", .{i});

            // Generate a random type for this variable
            const var_type_choice = random.int(u32) % 13;
            const var_type = switch (var_type_choice) {
                0 => Ty{ .i8 = {} },
                1 => Ty{ .i16 = {} },
                2 => Ty{ .i32 = {} },
                3 => Ty{ .i64 = {} },
                4 => Ty{ .i128 = {} },
                5 => Ty{ .u8 = {} },
                6 => Ty{ .u16 = {} },
                7 => Ty{ .u32 = {} },
                8 => Ty{ .u64 = {} },
                9 => Ty{ .u128 = {} },
                10 => Ty{ .float = {} },
                11 => Ty{ .bool = {} },
                12 => Ty{ .str = {} },
                else => unreachable,
            };

            // Generate an expression for the assignment value (use literal to avoid deep recursion)
            const assignment_value = try Value.generateRandom(allocator, random, &var_type);
            const assignment_value_ptr = try allocator.create(Value);
            assignment_value_ptr.* = assignment_value;
            const assignment_expr = Expr{ .literal = assignment_value_ptr };

            // Add this variable to our local scope
            try local_scope.assignments.append(BoundVar{
                .name = var_name,
                .ty = var_type,
                .is_pending_function = false,
            });

            // Create the assignment statement
            try statements.append(Stmt{
                .assignment = .{
                    .variable_name = var_name,
                    .value = assignment_expr,
                },
            });
        }

        // Generate return expression using the local scope (which now has assignments)
        const return_expr = try generateRandomWithScopeAndDepth(allocator, random, return_type, &local_scope, depth + 1);

        const block_ptr = try allocator.create(Block);
        block_ptr.* = Block{
            .statements = try statements.toOwnedSlice(),
            .return_expr = return_expr,
        };

        return block_ptr;
    }

    fn generateIfExpression(allocator: std.mem.Allocator, random: *const std.Random, result_type: *const Ty, depth: u32) GenError!Expr {
        _ = depth; // Unused for now

        // Generate a condition (always a boolean literal for simplicity)
        const condition_value = Value{ .bool = random.boolean() };
        const condition_value_ptr = try allocator.create(Value);
        condition_value_ptr.* = condition_value;
        const condition_expr_ptr = try allocator.create(Expr);
        condition_expr_ptr.* = Expr{ .literal = condition_value_ptr };

        // Generate then branch (a block with a literal of the desired type)
        const then_value = try Value.generateRandom(allocator, random, result_type);
        const then_value_ptr = try allocator.create(Value);
        then_value_ptr.* = then_value;
        const then_expr = Expr{ .literal = then_value_ptr };
        const then_block_ptr = try allocator.create(Block);
        then_block_ptr.* = Block{
            .statements = &[_]Stmt{}, // No statements for simplicity
            .return_expr = then_expr,
        };

        // Generate else branch (a block with a literal of the desired type)
        const else_value = try Value.generateRandom(allocator, random, result_type);
        const else_value_ptr = try allocator.create(Value);
        else_value_ptr.* = else_value;
        const else_expr = Expr{ .literal = else_value_ptr };
        const else_block_ptr = try allocator.create(Block);
        else_block_ptr.* = Block{
            .statements = &[_]Stmt{}, // No statements for simplicity
            .return_expr = else_expr,
        };

        return Expr{ .if_expression = .{
            .condition = condition_expr_ptr,
            .then_branch = then_block_ptr,
            .else_branch = else_block_ptr,
        } };
    }
};

const Func = struct {
    parameter_types: []*const Ty,
    body: *Block,
};

const Block = struct {
    statements: []Stmt,
    return_expr: Expr,

    pub fn format(self: Block, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        var buffer = std.ArrayList(u8).init(std.heap.page_allocator);
        defer buffer.deinit();
        try self.formatWithIndent(&buffer, 0, 80);
        try writer.writeAll(buffer.items);
    }

    pub fn formatWithIndent(self: Block, buffer: *std.ArrayList(u8), indent: ?u32, max_width: u32) std.fmt.AllocPrintError!void {
        if (indent == null) {
            try buffer.writer().print("{{", .{});
            for (self.statements, 0..) |stmt, i| {
                if (i > 0) try buffer.writer().print("; ", .{});
                try stmt.formatWithIndent(buffer, null, max_width);
            }
            if (self.statements.len > 0) try buffer.writer().print("; ", .{});
            try self.return_expr.formatWithIndent(buffer, null, max_width);
            try buffer.writer().print("}}", .{});
            return;
        }

        const start_pos = buffer.items.len;
        try buffer.writer().print("{{", .{});
        for (self.statements, 0..) |stmt, i| {
            if (i > 0) try buffer.writer().print("; ", .{});
            try stmt.formatWithIndent(buffer, null, max_width);
        }
        if (self.statements.len > 0) try buffer.writer().print("; ", .{});
        try self.return_expr.formatWithIndent(buffer, null, max_width);
        try buffer.writer().print("}}", .{});

        if (buffer.items.len - start_pos <= max_width) {
            return;
        }

        buffer.shrinkRetainingCapacity(start_pos);
        const current_indent = indent.?;
        try buffer.writer().print("{{\n", .{});
        for (self.statements) |stmt| {
            try buffer.writer().writeByteNTimes(' ', current_indent + 4);
            try stmt.formatWithIndent(buffer, current_indent + 4, max_width);
            try buffer.writer().print("\n", .{});
        }
        try buffer.writer().writeByteNTimes(' ', current_indent + 4);
        try self.return_expr.formatWithIndent(buffer, current_indent + 4, max_width);
        try buffer.writer().print("\n", .{});
        try buffer.writer().writeByteNTimes(' ', current_indent);
        try buffer.writer().print("}}", .{});
    }

    /// Replace all tbd expressions in this block with simple concrete expressions
    pub fn replaceTbdExpressions(self: *Block, interp: *Interp) TbdReplaceError!void {
        for (self.statements) |*stmt| {
            try stmt.replaceTbdExpressions(interp);
        }
        try self.return_expr.replaceTbdExpressions(interp);
    }
};

const Stmt = union(enum) {
    assignment: struct {
        variable_name: []const u8,
        value: Expr,
    },
    expression: Expr,
    return_statement: Expr,

    pub fn format(self: Stmt, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        var buffer = std.ArrayList(u8).init(std.heap.page_allocator);
        defer buffer.deinit();
        try self.formatWithIndent(&buffer, 0, 80);
        try writer.writeAll(buffer.items);
    }

    pub fn formatWithIndent(self: Stmt, buffer: *std.ArrayList(u8), indent: ?u32, max_width: u32) std.fmt.AllocPrintError!void {
        switch (self) {
            .assignment => |assign| {
                try buffer.writer().print("{s} = ", .{assign.variable_name});
                try assign.value.formatWithIndent(buffer, indent, max_width);
            },
            .expression => |expr| try expr.formatWithIndent(buffer, indent, max_width),
            .return_statement => |ret| {
                try buffer.writer().print("return ", .{});
                try ret.formatWithIndent(buffer, indent, max_width);
            },
        }
    }

    /// Replace all tbd expressions in this statement with simple concrete expressions
    pub fn replaceTbdExpressions(self: *Stmt, interp: *Interp) TbdReplaceError!void {
        switch (self.*) {
            .assignment => |*assign| {
                try assign.value.replaceTbdExpressions(interp);
            },
            .expression => |*expr| {
                try expr.replaceTbdExpressions(interp);
            },
            .return_statement => |*ret| {
                try ret.replaceTbdExpressions(interp);
            },
        }
    }
};

const Module = struct {
    statements: []const Stmt,
};

const ScopeItem = struct {
    name: []const u8,
    ty: *const Ty,
    value: Value,
};

pub const Scope = struct {
    values: std.ArrayList(ScopeItem),
    parent: ?*const Scope,

    pub fn init(allocator: std.mem.Allocator) Scope {
        return Scope{
            .values = std.ArrayList(ScopeItem).init(allocator),
            .parent = null,
        };
    }

    fn findRandomItemWithType(self: *const Scope, random: std.Random, ty: *const Ty) ?ScopeItem {
        // Use reservoir sampling to find a random value of the given type
        var found_value: ?ScopeItem = null;
        var count: u32 = 0;

        var scope_ptr: ?*const Scope = self;
        while (scope_ptr) |scope| {
            for (scope.values.items) |item| {
                if (item.ty.equals(ty)) {
                    count += 1;
                    if (random.int(u32) % count == 0) {
                        found_value = item;
                    }
                }
            }
            scope_ptr = scope.parent;
        }

        return found_value;
    }

    fn get(self: *const Scope, v: Var) Value {
        for (self.values.items) |item| {
            if (std.mem.eql(u8, item.name, v.name)) {
                return item.value;
            }
        }
        @panic("Variable not found in scope");
    }
};

const InterpError = error{
    OutOfGas,
    StackOverflow,
    UnificationFailed,
    ContradictoryConstraint,
    TypeMismatch,
    VariableNotFound,
    TbdExpressionNotImplemented,
    ApplicationNotImplemented,
    IfExpressionNotImplemented,
    UnsupportedUnaryType,
    OutOfMemory,
    DivideByZero,
};

pub const Interp = struct {
    allocator: std.mem.Allocator,
    gas: u64,
    random: std.Random,

    pub fn init(allocator: std.mem.Allocator, gas: u64, random: std.Random) Interp {
        return Interp{
            .allocator = allocator,
            .gas = gas,
            .random = random,
        };
    }

    fn makeSimpleExpr(self: *Interp, ty: *const Ty) !Expr {
        switch (ty.*) {
            .function => |fty| {
                var body = try self.allocator.create(Block);
                body.statements = &[_]Stmt{};
                body.return_expr = Expr{ .tbd = fty.return_type };
                var func = try self.allocator.create(Func);
                func.parameter_types = fty.parameter_types;
                func.body = body;
                return Expr{ .function = func };
            },
            .list => |el_ty| {
                const len = self.random.int(u32) % 6;
                const elements = try self.allocator.alloc(Expr, len);
                for (0..len) |i| {
                    elements[i] = .{ .tbd = el_ty };
                }
                return Expr{ .list_literal = .{ .elements = elements } };
            },
            .record => |fields| {
                const field_exprs = try self.allocator.alloc(FieldExpr, fields.fields.len);
                for (fields.fields, 0..) |*field, i| {
                    field_exprs[i] = FieldExpr{ .name = field.name, .value = .{ .tbd = field.ty } };
                }
                return Expr{ .record_literal = .{ .fields = field_exprs } };
            },
            .tag => |tag| {
                const arg_exprs = try self.allocator.alloc(Expr, tag.arguments.len);
                for (tag.arguments, 0..) |arg_ty, i| {
                    arg_exprs[i] = .{ .tbd = arg_ty };
                }
                return Expr{ .tag_literal = .{ .name = tag.name, .arguments = arg_exprs } };
            },
            else => {
                const value = try Value.generateRandom(self.allocator, self.random, ty);
                const value_ptr = try self.allocator.create(Value);
                value_ptr.* = value;
                return Expr{ .literal = value_ptr };
            },
        }
    }

    fn makeTrivialExpr(self: *Interp, ty: *const Ty) !*Expr {
        const expr_ptr = try self.allocator.create(Expr);
        expr_ptr.* = Expr{ .tbd = ty };
        return expr_ptr;
    }

    fn makeTrivialBlock(self: *Interp, ty: *const Ty) !*Block {
        const block_ptr = try self.allocator.create(Block);
        block_ptr.* = Block{
            .statements = &[_]Stmt{},
            .return_expr = Expr{ .tbd = ty }, // Default to i32 for simplicity
        };
        return block_ptr;
    }

    fn makeBoolTy(self: *Interp) !*const Ty {
        const ty_ptr = try self.allocator.create(Ty);
        ty_ptr.* = Ty{ .bool = {} };
        return ty_ptr;
    }

    fn makeNumTy(self: *Interp) !*const Ty {
        const ty_ptr = try self.allocator.create(Ty);
        // Make a random type!
        const type_choice = self.random.intRangeAtMost(u8, 0, 10);
        ty_ptr.* = switch (type_choice) {
            0 => Ty{ .i8 = {} },
            1 => Ty{ .i16 = {} },
            2 => Ty{ .i32 = {} },
            3 => Ty{ .i64 = {} },
            4 => Ty{ .i128 = {} },
            5 => Ty{ .u8 = {} },
            6 => Ty{ .u16 = {} },
            7 => Ty{ .u32 = {} },
            8 => Ty{ .u64 = {} },
            9 => Ty{ .u128 = {} },
            10 => Ty{ .float = {} },
            else => unreachable,
        };
        return ty_ptr;
    }

    fn makeFuncTy(self: *Interp, return_ty: *const Ty) !struct { *const Ty, []const *const Ty } {
        const ty_ptr = try self.allocator.create(Ty);
        // Generate a function type with random parameters
        const num_params = self.random.int(u32) % 3; // 0-2 parameters
        var param_types = try self.allocator.alloc(*const Ty, num_params);
        for (0..num_params) |i| {
            param_types[i] = try self.makeAnyTy();
        }
        ty_ptr.* = Ty{ .function = .{
            .parameter_types = param_types,
            .return_type = return_ty,
        } };
        return .{ ty_ptr, param_types };
    }

    fn makeAnyTy(self: *Interp) !*const Ty {
        const ty_ptr = try self.allocator.create(Ty);
        // Make a random type!
        const type_choice = self.random.intRangeAtMost(u8, 0, 16);
        ty_ptr.* = blk: switch (type_choice) {
            0 => Ty{ .i8 = {} },
            1 => Ty{ .i16 = {} },
            2 => Ty{ .i32 = {} },
            3 => Ty{ .i64 = {} },
            4 => Ty{ .i128 = {} },
            5 => Ty{ .u8 = {} },
            6 => Ty{ .u16 = {} },
            7 => Ty{ .u32 = {} },
            8 => Ty{ .u64 = {} },
            9 => Ty{ .u128 = {} },
            10 => Ty{ .float = {} },
            11 => Ty{ .bool = {} },
            12 => Ty{ .str = {} },
            13 => {
                // Generate a function type with random parameters
                const num_params = self.random.int(u32) % 3;
                var param_types = try self.allocator.alloc(*const Ty, num_params);
                for (0..num_params) |i| {
                    param_types[i] = try self.makeAnyTy();
                }
                break :blk Ty{ .function = .{
                    .parameter_types = param_types,
                    .return_type = try self.makeAnyTy(),
                } };
            },
            14 => {
                // Generate a list type with a random element type
                const element_type = try self.makeAnyTy();
                break :blk Ty{ .list = element_type };
            },
            15 => {
                // Generate a record type with random fields
                const num_fields = self.random.int(u32) % 3 + 1; // At least one field
                var fields = try self.allocator.alloc(FieldTy, num_fields);
                for (0..num_fields) |i| {
                    const field_name = try std.fmt.allocPrint(self.allocator, "field{}", .{i});
                    fields[i] = FieldTy{ .name = field_name, .ty = try self.makeAnyTy() };
                }
                break :blk Ty{ .record = .{ .fields = fields } };
            },
            16 => {
                // Generate a tag type with a random name and arguments
                const tag_name = try std.fmt.allocPrint(self.allocator, "Tag{}", .{self.random.int(u32)});
                const num_args = self.random.int(u32) % 3; // 0-2 arguments
                var arguments = try self.allocator.alloc(*const Ty, num_args);
                for (0..num_args) |i| {
                    arguments[i] = try self.makeAnyTy();
                }
                break :blk Ty{ .tag = .{ .name = tag_name, .arguments = arguments } };
            },
            else => unreachable,
        };
        return ty_ptr;
    }

    /// Evaluate the program and return the final value.
    pub fn eval(self: *Interp, scope: *Scope, expr: *Expr, stack_depth: u32) InterpError!Value {
        if (self.gas == 0) {
            return error.OutOfGas;
        }
        if (stack_depth > 1000) {
            return error.StackOverflow; // Prevent stack overflow
        }

        var retry_count: u32 = 0;
        while (true) {
            switch (expr.*) {
                .tbd => |t| {
                    // We need to generate a random expression!
                    while (true) {
                        if (stack_depth > 20) {
                            // If we're too deep, just return a trivial expression
                            expr.* = try self.makeSimpleExpr(t);
                            break;
                        }
                        const rand_num = self.random.intRangeAtMost(i32, 0, 5);
                        switch (rand_num) {
                            0 => {
                                expr.* = try self.makeSimpleExpr(t);
                                break;
                            },
                            1 => {
                                // Check if there's anything in scope that matches the type
                                if (scope.findRandomItemWithType(self.random, t)) |var_info| {
                                    expr.* = Expr{ .variable = Var{ .name = var_info.name } };
                                    break;
                                }
                            },
                            2 => {
                                const condition = try self.allocator.create(Expr);
                                condition.* = Expr{ .tbd = try self.makeBoolTy() };
                                // Generate an if expression
                                expr.* = Expr{
                                    .if_expression = .{
                                        .condition = condition,
                                        .then_branch = try self.makeTrivialBlock(t),
                                        .else_branch = try self.makeTrivialBlock(t),
                                    },
                                };
                                break;
                            },
                            3 => {
                                // Generate a binop
                                switch (t.*) {
                                    .function, .list, .record, .tag, .str => continue,
                                    .i8, .i16, .i32, .i64, .i128, .u8, .u16, .u32, .u64, .u128, .float => {
                                        const op = switch (self.random.int(u32) % 4) {
                                            0 => BinaryOp.add,
                                            1 => BinaryOp.subtract,
                                            2 => BinaryOp.multiply,
                                            3 => BinaryOp.divide,
                                            else => unreachable,
                                        };
                                        expr.* = Expr{
                                            .binary = .{
                                                .op = op,
                                                .left = try self.makeTrivialExpr(t),
                                                .right = try self.makeTrivialExpr(t),
                                            },
                                        };
                                        break;
                                    },
                                    .bool => {
                                        const op = switch (self.random.int(u32) % 6) {
                                            0 => BinaryOp.equals,
                                            1 => BinaryOp.not_equals,
                                            2 => BinaryOp.less_than,
                                            3 => BinaryOp.greater_than,
                                            4 => BinaryOp.and_op,
                                            5 => BinaryOp.or_op,
                                            else => unreachable,
                                        };
                                        const ty = switch (op) {
                                            .equals, .not_equals => try self.makeAnyTy(),
                                            .less_than, .greater_than => try self.makeNumTy(),
                                            .and_op, .or_op => try self.makeBoolTy(),
                                            else => unreachable,
                                        };
                                        expr.* = Expr{
                                            .binary = .{
                                                .op = op,
                                                .left = try self.makeTrivialExpr(ty),
                                                .right = try self.makeTrivialExpr(ty),
                                            },
                                        };
                                        break;
                                    },
                                }
                                break;
                            },
                            4 => {
                                // Generate a unary expression
                                switch (t.*) {
                                    .i8, .i16, .i32, .i64, .i128, .float => {
                                        const op = UnaryOp.negate;
                                        expr.* = Expr{
                                            .unary = .{
                                                .op = op,
                                                .operand = try self.makeTrivialExpr(t),
                                            },
                                        };
                                        break;
                                    },
                                    .bool => {
                                        const op = UnaryOp.not;
                                        expr.* = Expr{
                                            .unary = .{
                                                .op = op,
                                                .operand = try self.makeTrivialExpr(t),
                                            },
                                        };
                                        break;
                                    },
                                    else => continue, // Unsupported unary type
                                }
                            },
                            5 => {
                                // Generate an application
                                const func_ty, const param_types = try self.makeFuncTy(t);
                                const func_expr = try self.makeTrivialExpr(func_ty);
                                const args = try self.allocator.alloc(*Expr, param_types.len);
                                for (param_types, 0..) |param_ty, i| {
                                    args[i] = try self.makeTrivialExpr(param_ty);
                                }
                                expr.* = Expr{
                                    .application = .{
                                        .function = func_expr,
                                        .arguments = args,
                                    },
                                };
                                break;
                            },
                            else => @panic("boo"),
                        }
                    }
                    const result = self.eval(scope, expr, stack_depth + 1) catch |err| {
                        if (retry_count < 100 and (err == error.StackOverflow or err == error.DivideByZero)) {
                            retry_count += 1;
                            continue; // Retry evaluation
                        }
                        return err;
                    };
                    return result;
                },
                .literal => |lit| {
                    return lit.*;
                },
                .variable => |v| {
                    return scope.get(v);
                },
                .function => |func| {
                    return Value{ .closure = .{
                        .function = func,
                        .captured_variables = scope,
                    } };
                },
                .list_literal => |list_lit| {
                    var elements = std.ArrayList(Value).init(self.allocator);
                    defer elements.deinit();

                    for (list_lit.elements) |*element_expr| {
                        // For simplicity, use a generic type for list elements
                        const element_value = try self.eval(scope, element_expr, stack_depth + 1);
                        try elements.append(element_value);
                    }

                    return Value{ .list = .{ .elements = try elements.toOwnedSlice() } };
                },
                .binary => |bin| {
                    const left_value = try self.eval(scope, bin.left, stack_depth + 1);
                    const right_value = try self.eval(scope, bin.right, stack_depth + 1);
                    return try switch (bin.op) {
                        .add => left_value.add(right_value),
                        .subtract => left_value.subtract(right_value),
                        .multiply => left_value.multiply(right_value),
                        .divide => left_value.divide(right_value),
                        .equals => left_value.equals(right_value),
                        .not_equals => left_value.notEquals(right_value),
                        .less_than => left_value.lessThan(right_value),
                        .greater_than => left_value.greaterThan(right_value),
                        .and_op => left_value.andValue(right_value),
                        .or_op => left_value.orValue(right_value),
                    };
                },
                .unary => |un| {
                    const operand_value = try self.eval(scope, un.operand, stack_depth + 1);
                    return try switch (un.op) {
                        .negate => operand_value.negate(),
                        .not => operand_value.not(),
                    };
                },
                .record_literal => |rec| {
                    var fields = std.ArrayList(Field).init(self.allocator);
                    defer fields.deinit();

                    for (rec.fields) |*field| {
                        const field_value = try self.eval(scope, &field.value, stack_depth + 1);
                        try fields.append(Field{ .name = field.name, .value = field_value });
                    }

                    return Value{ .record = .{ .fields = try fields.toOwnedSlice() } };
                },
                .tag_literal => |tag| {
                    var arguments = std.ArrayList(Value).init(self.allocator);
                    defer arguments.deinit();
                    for (tag.arguments) |*arg_expr| {
                        const arg_value = try self.eval(scope, arg_expr, stack_depth + 1);
                        try arguments.append(arg_value);
                    }
                    return Value{ .tag = .{
                        .name = tag.name,
                        .arguments = try arguments.toOwnedSlice(),
                    } };
                },
                .field_access => |fa| {
                    // Evaluate the record expression first
                    const record_value = try self.eval(scope, fa.record, stack_depth + 1);
                    // Check if it's a record
                    if (record_value != .record) {
                        return error.TypeMismatch;
                    }
                    // Now access the field
                    const record = record_value.record;
                    for (record.fields) |field| {
                        if (std.mem.eql(u8, field.name, fa.field_name)) {
                            return field.value;
                        }
                    }
                    return error.VariableNotFound; // Field not found
                },
                .application => |app| {
                    // Evaluate the function expression
                    const func_value = try self.eval(scope, app.function, stack_depth + 1);

                    // It should be a closure
                    if (func_value != .closure) {
                        std.debug.print("Expected closure, got: {}\n", .{func_value});
                        std.debug.print("Function expression: {}\n", .{expr});
                        return error.TypeMismatch;
                    }

                    const closure = func_value.closure;

                    // Check argument count
                    if (app.arguments.len != closure.function.parameter_types.len) {
                        return error.TypeMismatch;
                    }

                    // Evaluate arguments
                    var arg_values = try self.allocator.alloc(Value, app.arguments.len);
                    defer self.allocator.free(arg_values);

                    for (app.arguments, 0..) |arg_expr, i| {
                        arg_values[i] = try self.eval(scope, arg_expr, stack_depth + 1);
                    }

                    // Create new scope for function execution
                    var func_scope = Scope.init(self.allocator);

                    // Set parent to captured variables
                    func_scope.parent = closure.captured_variables;

                    const func_scope_ptr = try self.allocator.create(Scope);
                    func_scope_ptr.* = func_scope;

                    // Add parameters to scope
                    for (arg_values, 0..) |arg_value, i| {
                        const param_name = try std.fmt.allocPrint(self.allocator, "arg{}", .{i});
                        defer self.allocator.free(param_name);

                        try func_scope.values.append(ScopeItem{
                            .name = param_name,
                            .ty = closure.function.parameter_types[i],
                            .value = arg_value,
                        });
                    }

                    // Execute function body
                    return self.evalBlock(func_scope_ptr, closure.function.body, stack_depth + 1);
                },
                .if_expression => |if_expr| {
                    // Evaluate condition
                    const condition_value = try self.eval(scope, if_expr.condition, stack_depth + 1);

                    const should_take_then_branch = switch (condition_value) {
                        .bool => |b| b,
                        else => return error.TypeMismatch,
                    };

                    if (should_take_then_branch) {
                        return self.evalBlock(scope, if_expr.then_branch, stack_depth + 1);
                    } else {
                        return self.evalBlock(scope, if_expr.else_branch, stack_depth + 1);
                    }
                },
            }

            break;
        }
    }

    fn evalBlock(self: *Interp, scope: *Scope, block: *Block, stack_depth: u32) InterpError!Value {
        // For simplicity, just evaluate the return expression
        // In a full implementation, you'd need to handle statements
        return self.eval(scope, &block.return_expr, stack_depth + 1);
    }
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .{};
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const count = if (args.len > 1)
        std.fmt.parseInt(u32, args[1], 10) catch 1
    else
        1;

    const int_ty = Ty{ .u8 = {} };

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = prng.random();

    for (0..count) |_| {
        var interp = Interp.init(allocator, 100, random);

        var value = Expr{ .tbd = &int_ty };

        var scope = Scope.init(allocator);

        const res = interp.eval(&scope, &value, 0) catch |err| {
            std.debug.print("Error evaluating expression: {}\n", .{err});
            std.debug.print("Final expr: {}\n\n", .{value});
            if (err == error.DivideByZero or err == error.StackOverflow or err == error.OutOfGas) {
                continue;
            } else {
                return err;
            }
        };
        std.debug.print("Result: {}\n", .{res});
        std.debug.print("Final expr: {}\n\n", .{value});
    }
}
