const std = @import("std");
const repl = @import("repl/eval.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const count = if (args.len > 1)
        std.fmt.parseInt(u32, args[1], 10) catch 10
    else
        10;

    const seed = if (args.len > 2)
        std.fmt.parseInt(u64, args[2], 10) catch @as(u64, @intCast(std.time.timestamp()))
    else
        @as(u64, @intCast(std.time.timestamp()));

    std.debug.print("Running grain fuzzer with {} iterations, seed {}\n", .{ count, seed });

    // Initialize REPL
    var repl_instance = try repl.Repl.init(allocator);
    defer repl_instance.deinit();

    // Initialize random number generator
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    var successes: u32 = 0;
    var failures: u32 = 0;

    for (0..count) |i| {
        std.debug.print("\n--- Test {} ---\n", .{i + 1});

        // Generate a random expression and evaluate it with our grain evaluator
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const result = try generateAndTestExpression(arena.allocator(), random, &repl_instance);

        if (result) {
            successes += 1;
            std.debug.print("✓ PASS\n", .{});
        } else {
            failures += 1;
            std.debug.print("✗ FAIL\n", .{});
        }
    }

    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("Successes: {}\n", .{successes});
    std.debug.print("Failures: {}\n", .{failures});
    std.debug.print("Success rate: {d:.1}%\n", .{@as(f64, @floatFromInt(successes)) / @as(f64, @floatFromInt(count)) * 100.0});

    if (failures > 0) {
        std.process.exit(1);
    }
}

fn evalExprInRepl(allocator: std.mem.Allocator, expr: *Expr) ![]const u8 {
    var repl_instance = try repl.Repl.init(allocator);
    defer repl_instance.deinit();

    const roc_expr = try expressionToRocSyntax(allocator, expr);
    defer allocator.free(roc_expr);

    const repl_output = repl_instance.step(roc_expr) catch |err| {
        std.debug.print("REPL evaluation failed: {}\n", .{err});
        return error.ReplError;
    };

    return repl_output;
}

fn evalExprInGrain(allocator: std.mem.Allocator, expr: *Expr) ![]const u8 {
    var random = std.Random.DefaultPrng.init(0);
    var interp = Interp.init(allocator, 1000, random.random());
    var scope = Scope.init(allocator);
    defer scope.values.deinit();

    const value = interp.eval(&scope, expr, 0) catch |err| {
        std.debug.print("Grain evaluation failed: {}\n", .{err});
        return error.GrainEvalError;
    };

    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    try value.formatWithIndent(&buf, 0, 80);
    const result_str = buf.items;
    return try allocator.dupe(u8, result_str);
}

const CheckResult = union(enum) {
    match,
    mismatch,
    err: []const u8,
};

fn doCheck(allocator: std.mem.Allocator, expr: *Expr) !CheckResult {
    const grain = try evalExprInGrain(allocator, expr);
    const repl_output = try evalExprInRepl(allocator, expr);
    if (!std.mem.eql(u8, grain, repl_output)) {
        if (std.mem.indexOf(u8, repl_output, "error") != null) {
            return .{ .err = repl_output };
        }
        return .mismatch;
    }
    return .match;
}

fn generateAndTestExpression(allocator: std.mem.Allocator, random: std.Random, repl_instance: *repl.Repl) !bool {
    // Generate a random type (start with simple types)
    const type_choice = random.intRangeAtMost(i32, 0, 11);
    const target_type = switch (type_choice) {
        0 => Ty{ .i8 = {} },
        1 => Ty{ .i16 = {} },
        2 => Ty{ .i32 = {} },
        3 => Ty{ .i64 = {} },
        4 => Ty{ .u8 = {} },
        5 => Ty{ .u16 = {} },
        6 => Ty{ .u32 = {} },
        7 => Ty{ .u64 = {} },
        8 => Ty{ .float = {} },
        9 => Ty{ .bool = {} },
        10 => Ty{ .str = {} },
        else => unreachable,
    };

    // Generate a random expression of this type
    var expr = Expr{ .tbd = &target_type };

    // Initialize grain interpreter
    var interp = Interp.init(allocator, 1000, random);
    var scope = Scope.init(allocator);
    defer scope.values.deinit();

    // Evaluate the expression with grain to get the expected result
    _ = interp.eval(&scope, &expr, 0) catch |err| {
        std.debug.print("Grain evaluation failed: {}\n", .{err});
        return false;
    };

    // Replace all remaining tbd expressions with simple concrete expressions
    // This is safe because these tbd expressions were never reached during evaluation
    expr.replaceTbdExpressions(&interp) catch |err| {
        std.debug.print("Failed to replace tbd expressions: {}\n", .{err});
        return false;
    };

    const grain = try evalExprInGrain(allocator, &expr);
    const initial_repl_output = try evalExprInRepl(allocator, &expr);
    if (std.mem.eql(u8, grain, initial_repl_output)) {
        std.debug.print("Expression matches in both evaluations: {}\n", .{expr});
        std.debug.print("Output: {s}\n", .{initial_repl_output});
        return true; // No issue found, the expression matches in both evaluations
    }
    var error_msg: ?[]const u8 = null;
    if (std.mem.indexOf(u8, initial_repl_output, "error") != null) {
        error_msg = initial_repl_output;
    }

    // Create a context for minimization
    const MinContext = struct {
        allocator: std.mem.Allocator,
        repl_instance: *repl.Repl,
        error_msg: ?[]const u8,
    };

    const min_context = MinContext{
        .allocator = allocator,
        .repl_instance = repl_instance,
        .error_msg = error_msg,
    };

    // Define the predicate: an expression reproduces the issue if it fails in the same way
    const ReproducePredicate = struct {
        pub fn call(test_expr: *Expr, context: MinContext) bool {
            std.debug.print("Trying expression to reproduce issue: {}\n", .{test_expr});
            const res = doCheck(context.allocator, test_expr) catch |err| {
                std.debug.print("Check failed: {}\n", .{err});
                return false; // Skip this test case if check fails
            };
            switch (res) {
                .match => return false, // This expression does not reproduce the issue
                .mismatch => return context.error_msg == null, // This expression reproduces the issue
                .err => {
                    std.debug.print("Error during check: {s}\n", .{res.err});
                    if (context.error_msg) |expected_err_msg| {
                        return std.mem.eql(u8, res.err, expected_err_msg);
                    } else {
                        return false; // If no error message was expected, this is not a reproducing case
                    }
                },
            }
        }
    };

    // Minimize the expression to find the smallest reproducing case
    std.debug.print("Minimizing expression...\n", .{});
    const minimized_expr = minimizeWithContext(allocator, &interp, &expr, min_context, ReproducePredicate.call) catch |err| {
        std.debug.print("Minimization failed: {}\n", .{err});
        return false; // Skip this test case if minimization fails
    };

    // Use the minimized expression instead of the original
    expr = minimized_expr;

    std.debug.print("Minimized expression: {}\n", .{expr});

    const grain_result = try evalExprInGrain(allocator, &expr);
    const repl_output = try evalExprInRepl(allocator, &expr);
    if (!std.mem.eql(u8, grain, repl_output)) {
        std.debug.print("Mismatch after minimization:\nGrain: {s}\nREPL: {s}\n", .{ grain_result, repl_output });
        return false;
    }
    std.debug.print("Expression passed check: {}\n", .{expr});
    return true;
}

fn expressionToRocSyntax(allocator: std.mem.Allocator, expr: *Expr) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();

    try expr.formatWithIndent(&buf, 0, 30);

    // Convert grain syntax to Roc syntax
    return try allocator.dupe(u8, buf.items);
}

const ValueIdx = struct { index: usize };

pub const Value = union(enum) {
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    // i128: i128,
    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    // u128: u128,
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
            // .i128 => Ty{ .i128 = {} },
            .u8 => Ty{ .u8 = {} },
            .u16 => Ty{ .u16 = {} },
            .u32 => Ty{ .u32 = {} },
            .u64 => Ty{ .u64 = {} },
            // .u128 => Ty{ .u128 = {} },
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
        var buffer = std.ArrayList(u8).init(std.heap.c_allocator);
        defer buffer.deinit();
        try self.formatWithIndent(&buffer, 0, 80);
        try writer.writeAll(buffer.items);
    }

    pub fn formatWithIndent(self: Value, buffer: *std.ArrayList(u8), indent: ?u32, max_width: u32) std.fmt.AllocPrintError!void {
        switch (self) {
            .i8 => |i| try buffer.writer().print("{}", .{i}),
            .i16 => |i| try buffer.writer().print("{}", .{i}),
            .i32 => |i| try buffer.writer().print("{}", .{i}),
            .i64 => |i| try buffer.writer().print("{}", .{i}),
            // .i128 => |i| try buffer.writer().print("{}", .{i}),
            .u8 => |i| try buffer.writer().print("{}", .{i}),
            .u16 => |i| try buffer.writer().print("{}", .{i}),
            .u32 => |i| try buffer.writer().print("{}", .{i}),
            .u64 => |i| try buffer.writer().print("{}", .{i}),
            // .u128 => |i| try buffer.writer().print("{}", .{i}),
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

    pub fn structuralEquals(a: Value, b: Value) bool {
        if (@intFromEnum(a) != @intFromEnum(b)) return false;
        return switch (a) {
            .i8 => |ai| b.i8 == ai,
            .i16 => |ai| b.i16 == ai,
            .i32 => |ai| b.i32 == ai,
            .i64 => |ai| b.i64 == ai,
            // .i128 => |ai| b.i128 == ai,
            .u8 => |ai| b.u8 == ai,
            .u16 => |ai| b.u16 == ai,
            .u32 => |ai| b.u32 == ai,
            .u64 => |ai| b.u64 == ai,
            // .u128 => |ai| b.u128 == ai,
            .float => |af| b.float == af,
            .bool => |ab| b.bool == ab,
            .str => |as| std.mem.eql(u8, b.str, as),
            .list => |al| blk: {
                const bl = b.list;
                if (al.elements.len != bl.elements.len) break :blk false;
                for (al.elements, bl.elements) |ae, be| {
                    if (!ae.structuralEquals(be)) break :blk false;
                }
                break :blk true;
            },
            .record => |ar| blk: {
                const br = b.record;
                if (ar.fields.len != br.fields.len) break :blk false;
                for (ar.fields, br.fields) |af, bf| {
                    if (!std.mem.eql(u8, af.name, bf.name)) break :blk false;
                    if (!af.value.structuralEquals(bf.value)) break :blk false;
                }
                break :blk true;
            },
            .tag => |at| blk: {
                const bt = b.tag;
                if (!std.mem.eql(u8, at.name, bt.name)) break :blk false;
                if (at.arguments.len != bt.arguments.len) break :blk false;
                for (at.arguments, bt.arguments) |ae, be| {
                    if (!ae.structuralEquals(be)) break :blk false;
                }
                break :blk true;
            },
            .closure => |ac| blk: {
                const bc = b.closure;
                // Compare function pointer and captured variables pointer
                if (ac.function != bc.function) break :blk false;
                if (ac.captured_variables != bc.captured_variables) break :blk false;
                break :blk true;
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
            // .i128 => |a| switch (other) {
            //     .i128 => |b| Value{ .i128 = a +% b },
            //     else => error.TypeMismatch,
            // },
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
            // .u128 => |a| switch (other) {
            //     .u128 => |b| Value{ .u128 = a +% b },
            //     else => error.TypeMismatch,
            // },
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
            // .i128 => |a| switch (other) {
            //     .i128 => |b| Value{ .i128 = a -% b },
            //     else => error.TypeMismatch,
            // },
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
            // .u128 => |a| switch (other) {
            //     .u128 => |b| Value{ .u128 = a -% b },
            //     else => error.TypeMismatch,
            // },
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
            // .i128 => |a| switch (other) {
            //     .i128 => |b| Value{ .i128 = a *% b },
            //     else => error.TypeMismatch,
            // },
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
            // .u128 => |a| switch (other) {
            //     .u128 => |b| Value{ .u128 = a *% b },
            //     else => error.TypeMismatch,
            // },
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
            // .i128 => |a| switch (other) {
            //     .i128 => |b| if (b == 0) error.DivideByZero else Value{ .i128 = @divTrunc(a, b) },
            //     else => error.TypeMismatch,
            // },
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
            // .u128 => |a| switch (other) {
            //     .u128 => |b| if (b == 0) error.DivideByZero else Value{ .u128 = a / b },
            //     else => error.TypeMismatch,
            // },
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
            // .i128 => |a| switch (other) {
            //     .i128 => |b| a == b,
            //     else => false,
            // },
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
            // .u128 => |a| switch (other) {
            //     .u128 => |b| a == b,
            //     else => false,
            // },
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
                    @panic("Type mismatch");
                },
            },
            .i16 => |a| switch (other) {
                .i16 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .i32 => |a| switch (other) {
                .i32 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .i64 => |a| switch (other) {
                .i64 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            // .i128 => |a| switch (other) {
            //     .i128 => |b| a < b,
            //     else => {
            //         std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
            //         @panic("Type mismatch");
            //     },
            // },
            .u8 => |a| switch (other) {
                .u8 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .u16 => |a| switch (other) {
                .u16 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .u32 => |a| switch (other) {
                .u32 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .u64 => |a| switch (other) {
                .u64 => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            // .u128 => |a| switch (other) {
            //     .u128 => |b| a < b,
            //     else => {
            //         std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
            //         @panic("Type mismatch");
            //     },
            // },
            .float => |a| switch (other) {
                .float => |b| a < b,
                else => {
                    std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            else => {
                std.debug.print("Type mismatch in lessThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                @panic("Type mismatch");
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
                    @panic("Type mismatch");
                },
            },
            .i16 => |a| switch (other) {
                .i16 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .i32 => |a| switch (other) {
                .i32 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .i64 => |a| switch (other) {
                .i64 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            // .i128 => |a| switch (other) {
            //     .i128 => |b| a > b,
            //     else => {
            //         std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
            //         @panic("Type mismatch");
            //     },
            // },
            .u8 => |a| switch (other) {
                .u8 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .u16 => |a| switch (other) {
                .u16 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .u32 => |a| switch (other) {
                .u32 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            .u64 => |a| switch (other) {
                .u64 => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            // .u128 => |a| switch (other) {
            //     .u128 => |b| a > b,
            //     else => {
            //         std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
            //         @panic("Type mismatch");
            //     },
            // },
            .float => |a| switch (other) {
                .float => |b| a > b,
                else => {
                    std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                    @panic("Type mismatch");
                },
            },
            else => {
                std.debug.print("Type mismatch in greaterThan: {s} vs {s}\n", .{ self.ty(), other.ty() });
                @panic("Type mismatch");
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
                    @panic("Type mismatch");
                },
            },
            else => {
                std.debug.print("Type mismatch in and: {s}\n", .{self.ty()});
                @panic("Type mismatch");
            },
        };
    }

    pub fn orValue(self: Value, other: Value) !Value {
        return switch (self) {
            .bool => |a| switch (other) {
                .bool => |b| Value{ .bool = a or b },
                else => {
                    std.debug.print("Type mismatch in or: {s}\n", .{self.ty()});
                    @panic("Type mismatch");
                },
            },
            else => {
                std.debug.print("Type mismatch in or: {s}\n", .{self.ty()});
                @panic("Type mismatch");
            },
        };
    }

    pub fn negate(self: Value) !Value {
        return switch (self) {
            .i8 => |a| Value{ .i8 = -%a },
            .i16 => |a| Value{ .i16 = -%a },
            .i32 => |a| Value{ .i32 = -%a },
            .i64 => |a| Value{ .i64 = -%a },
            // .i128 => |a| Value{ .i128 = -%a },
            .float => |a| Value{ .float = -a },
            else => {
                std.debug.print("Type mismatch in negate: {s}\n", .{self.ty()});
                @panic("Type mismatch");
            },
        };
    }

    pub fn not(self: Value) !Value {
        return switch (self) {
            .bool => |a| Value{ .bool = !a },
            else => {
                std.debug.print("Type mismatch in not: {s}\n", .{self.ty()});
                @panic("Type mismatch");
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
    value: *Expr,
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
    // i128,
    u8,
    u16,
    u32,
    u64,
    // u128,
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
            // .i128 => |_| other.* == .i128,
            .u8 => |_| other.* == .u8,
            .u16 => |_| other.* == .u16,
            .u32 => |_| other.* == .u32,
            .u64 => |_| other.* == .u64,
            // .u128 => |_| other.* == .u128,
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
            // .i128 => try writer.print("i128", .{}),
            .u8 => try writer.print("u8", .{}),
            .u16 => try writer.print("u16", .{}),
            .u32 => try writer.print("u32", .{}),
            .u64 => try writer.print("u64", .{}),
            // .u128 => try writer.print("u128", .{}),
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

const TbdReplaceError = error{OutOfMemory};

const ModificationIterator = struct {
    allocator: std.mem.Allocator,
    interp: *Interp,
    original_expr: *Expr,
    current_modifications: std.ArrayList(ModificationAttempt),
    child_iterators: std.ArrayList(*ModificationIterator),
    current_child_index: usize,
    self_simplified: bool,
    self_child_index: u32, // Index of this iterator in its parent's child list

    const ModificationAttempt = struct {
        path: []usize, // Path to the expression to modify
        replacement: Expr, // What to replace it with
    };

    pub fn init(allocator: std.mem.Allocator, interp: *Interp, expr: *Expr, self_child_index: u32) !*ModificationIterator {
        const iterator = try allocator.create(ModificationIterator);
        iterator.* = ModificationIterator{
            .allocator = allocator,
            .interp = interp,
            .original_expr = expr,
            .current_modifications = std.ArrayList(ModificationAttempt).init(allocator),
            .child_iterators = std.ArrayList(*ModificationIterator).init(allocator),
            .current_child_index = 0,
            .self_simplified = false,
            .self_child_index = self_child_index,
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
                std.debug.print("Applying child modification: {}\nModified: {}\n", .{ child_iterator.original_expr, child_modification });
                if (child_iterator.original_expr.structuralEquals(&child_modification)) {
                    std.debug.print("Child modification is structurally equal to original expression, this should not happen: {s}\n", .{child_iterator.original_expr});
                    @panic("Child modification is structurally equal to original expression, this should not happen");
                }
                // Apply this child modification to a copy of our expression
                var modified_expr = try self.copyExpr(self.original_expr.*);
                try self.applyChildModification(&modified_expr, child_iterator.self_child_index, child_modification);
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
        if (self.isAlreadySimplest(self.original_expr)) {
            return null; // No simpler replacement available
        }

        // Try to replace the entire expression with the simplest possible expression of the same type
        const expr_type = try self.inferType(self.original_expr);
        const simple_expr = try self.makeSimplestExpr(expr_type);
        if (simple_expr.structuralEquals(self.original_expr)) {
            return null; // No change, already simplest
        }
        return simple_expr;
    }

    fn makeSimplestExpr(self: *ModificationIterator, ty: *const Ty) !Expr {
        const value_ptr = try self.allocator.create(Value);
        value_ptr.* = switch (ty.*) {
            .i8 => Value{ .i8 = 0 },
            .i16 => Value{ .i16 = 0 },
            .i32 => Value{ .i32 = 0 },
            .i64 => Value{ .i64 = 0 },
            // .i128 => Value{ .i128 = 0 },
            .u8 => Value{ .u8 = 0 },
            .u16 => Value{ .u16 = 0 },
            .u32 => Value{ .u32 = 0 },
            .u64 => Value{ .u64 = 0 },
            // .u128 => Value{ .u128 = 0 },
            .float => Value{ .float = 0.0 },
            .bool => Value{ .bool = false },
            .str => Value{ .str = "" },
            .list => |element_ty| {
                return Expr{
                    .list_literal = .{
                        .elements = &[_]*Expr{},
                        .element_ty = element_ty,
                    },
                };
            },
            .record => |record_ty| {
                // Create a record with all fields initialized to simplest values
                const fields = try self.allocator.alloc(FieldExpr, record_ty.fields.len);
                for (record_ty.fields, 0..) |field, i| {
                    const field_value = try self.interp.makeExpr(try self.makeSimplestExpr(field.ty));
                    fields[i] = FieldExpr{ .name = field.name, .value = field_value };
                }
                return Expr{ .record_literal = .{ .fields = fields } };
            },
            .tag => |tag_ty| {
                // Create a tag with all arguments initialized to simplest values
                const args = try self.allocator.alloc(*Expr, tag_ty.arguments.len);
                for (tag_ty.arguments, 0..) |arg_ty, i| {
                    const arg_expr = try self.allocator.create(Expr);
                    arg_expr.* = try self.makeSimplestExpr(arg_ty);
                    args[i] = arg_expr;
                }
                return Expr{ .tag_literal = .{ .name = tag_ty.name, .arguments = args } };
            },
            .function => |func_ty| {
                // Create a function with the same parameters but with a body that returns a simplest value
                const return_expr = try self.makeSimplestExpr(func_ty.return_type);
                return Expr{ .function = .{
                    .body = try self.interp.makeExpr(return_expr),
                    .parameter_types = func_ty.parameter_types,
                } };
            },
        };
        return Expr{ .literal = value_ptr };
    }

    fn initializeChildIterators(self: *ModificationIterator) !void {
        switch (self.original_expr.*) {
            .binary => |bin| {
                if (!self.isAlreadySimplest(bin.left)) {
                    const left_iter = try ModificationIterator.init(self.allocator, self.interp, bin.left, 0);
                    try self.child_iterators.append(left_iter);
                }
                if (!self.isAlreadySimplest(bin.right)) {
                    const right_iter = try ModificationIterator.init(self.allocator, self.interp, bin.right, 1);
                    try self.child_iterators.append(right_iter);
                }
            },
            .unary => |un| {
                if (!self.isAlreadySimplest(un.operand)) {
                    const operand_iter = try ModificationIterator.init(self.allocator, self.interp, un.operand, 0);
                    try self.child_iterators.append(operand_iter);
                }
            },
            .application => |app| {
                if (!self.isAlreadySimplest(app.function)) {
                    const func_iter = try ModificationIterator.init(self.allocator, self.interp, app.function, 0);
                    try self.child_iterators.append(func_iter);
                }
                for (app.arguments, 1..) |arg, i| {
                    if (!self.isAlreadySimplest(arg)) {
                        const arg_iter = try ModificationIterator.init(self.allocator, self.interp, arg, @intCast(i));
                        try self.child_iterators.append(arg_iter);
                    }
                }
            },
            .list_literal => |lst| {
                for (lst.elements, 0..) |element, i| {
                    if (!self.isAlreadySimplest(element)) {
                        const elem_iter = try ModificationIterator.init(self.allocator, self.interp, element, @intCast(i));
                        try self.child_iterators.append(elem_iter);
                    }
                }
            },
            .record_literal => |rec| {
                for (rec.fields, 0..) |*field, i| {
                    if (!self.isAlreadySimplest(field.value)) {
                        const field_iter = try ModificationIterator.init(self.allocator, self.interp, field.value, @intCast(i));
                        try self.child_iterators.append(field_iter);
                    }
                }
            },
            .tag_literal => |tag| {
                for (tag.arguments, 0..) |arg, i| {
                    if (!self.isAlreadySimplest(arg)) {
                        const arg_iter = try ModificationIterator.init(self.allocator, self.interp, arg, @intCast(i));
                        try self.child_iterators.append(arg_iter);
                    }
                }
            },
            .if_expression => |if_expr| {
                if (!self.isAlreadySimplest(if_expr.condition)) {
                    const cond_iter = try ModificationIterator.init(self.allocator, self.interp, if_expr.condition, 0);
                    try self.child_iterators.append(cond_iter);
                }
                // Note: We'd need to add iterators for blocks too, but for simplicity we'll skip them for now
            },
            .field_access => |field| {
                if (!self.isAlreadySimplest(field.record)) {
                    const record_iter = try ModificationIterator.init(self.allocator, self.interp, field.record, 0);
                    try self.child_iterators.append(record_iter);
                }
            },
            // Leaf nodes have no children
            .literal, .variable, .tbd => {},
            .function => |func| {
                // Simplify the body of the function
                if (!self.isAlreadySimplest(func.body)) {
                    const body_iter = try ModificationIterator.init(self.allocator, self.interp, func.body, 0);
                    try self.child_iterators.append(body_iter);
                }
            },
            .block => |block| {
                // For blocks, we can simplify each expression in the block
                for (block.statements, 0..) |stmt, i| {
                    switch (stmt) {
                        .assignment => |assign| {
                            if (!self.isAlreadySimplest(assign.value)) {
                                const expr_iter = try ModificationIterator.init(self.allocator, self.interp, assign.value, @intCast(i));
                                try self.child_iterators.append(expr_iter);
                            }
                        },
                        .return_statement => |expr| {
                            if (!self.isAlreadySimplest(expr)) {
                                const expr_iter = try ModificationIterator.init(self.allocator, self.interp, expr, @intCast(i));
                                try self.child_iterators.append(expr_iter);
                            }
                        },
                    }
                }
                if (!self.isAlreadySimplest(block.return_expr)) {
                    const expr_iter = try ModificationIterator.init(self.allocator, self.interp, block.return_expr, @intCast(block.statements.len));
                    try self.child_iterators.append(expr_iter);
                }
            },
        }
    }

    fn applyChildModification(self: *ModificationIterator, target: *Expr, child_index: usize, modification: Expr) !void {
        var current_child: usize = 0;
        const modification_ptr = try self.interp.allocator.create(Expr);
        modification_ptr.* = modification;
        switch (target.*) {
            .binary => |*bin| {
                if (current_child == child_index) {
                    std.debug.assert(!bin.left.structuralEquals(modification_ptr));
                    bin.left = modification_ptr;
                    return;
                }
                current_child += 1;
                if (current_child == child_index) {
                    std.debug.assert(!bin.right.structuralEquals(modification_ptr));
                    bin.right = modification_ptr;
                    return;
                }
                current_child += 1;
            },
            .unary => |*un| {
                if (current_child == child_index) {
                    std.debug.assert(!un.operand.structuralEquals(modification_ptr));
                    un.operand = modification_ptr;
                    return;
                }
                current_child += 1;
            },
            .application => |*app| {
                if (current_child == child_index) {
                    std.debug.assert(!app.function.structuralEquals(modification_ptr));
                    app.function = modification_ptr;
                    return;
                }
                current_child += 1;
                const new_args = try self.allocator.alloc(*Expr, app.arguments.len);
                for (app.arguments, 0..) |arg, i| {
                    if (current_child == child_index) {
                        std.debug.assert(!arg.structuralEquals(modification_ptr));
                        new_args[i] = modification_ptr;
                    } else {
                        new_args[i] = arg;
                    }
                    current_child += 1;
                }
                app.arguments = new_args;
                return;
            },
            .list_literal => |*lst| {
                // Clone the list to avoid modifying the original
                const new_elements = try self.allocator.alloc(*Expr, lst.elements.len);
                for (lst.elements, 0..) |element, i| {
                    if (current_child == child_index) {
                        std.debug.assert(!element.structuralEquals(modification_ptr));
                        new_elements[i] = modification_ptr;
                    } else {
                        new_elements[i] = element;
                    }
                    current_child += 1;
                }
                lst.elements = new_elements;
                return;
            },
            .record_literal => |*rec| {
                const new_fields = try self.allocator.alloc(FieldExpr, rec.fields.len);
                for (rec.fields, 0..) |field, i| {
                    if (current_child == child_index) {
                        std.debug.assert(!field.value.structuralEquals(modification_ptr));
                        new_fields[i] = FieldExpr{ .name = field.name, .value = modification_ptr };
                    } else {
                        new_fields[i] = field;
                    }
                    current_child += 1;
                }
                rec.fields = new_fields;
                return;
            },
            .tag_literal => |*tag| {
                const new_args = try self.allocator.alloc(*Expr, tag.arguments.len);
                for (tag.arguments, 0..) |arg, i| {
                    if (current_child == child_index) {
                        std.debug.assert(!arg.structuralEquals(modification_ptr));
                        new_args[i] = modification_ptr;
                    } else {
                        new_args[i] = arg;
                    }
                    current_child += 1;
                }
                tag.arguments = new_args;
                return;
            },
            .if_expression => |*if_expr| {
                if (current_child == child_index) {
                    std.debug.assert(!if_expr.condition.structuralEquals(modification_ptr));
                    if_expr.condition = modification_ptr;
                    return;
                }
                current_child += 1;
            },
            .field_access => |*field| {
                if (current_child == child_index) {
                    std.debug.assert(!field.record.structuralEquals(modification_ptr));
                    field.record = modification_ptr;
                    return;
                }
                current_child += 1;
            },
            .function => |*func| {
                if (current_child == child_index) {
                    std.debug.assert(!func.body.structuralEquals(modification_ptr));
                    func.body = modification_ptr;
                    return;
                }
                current_child += 1;
            },
            .block => |*block| {
                // For blocks, we need to check each statement
                const new_statements = try self.allocator.alloc(Stmt, block.statements.len);
                for (block.statements, 0..) |stmt, i| {
                    switch (stmt) {
                        .assignment => |assign| {
                            if (current_child == child_index) {
                                std.debug.assert(!assign.value.structuralEquals(modification_ptr));
                                new_statements[i] = Stmt{
                                    .assignment = .{
                                        .variable_name = assign.variable_name,
                                        .value = modification_ptr,
                                    },
                                };
                            } else {
                                new_statements[i] = Stmt{
                                    .assignment = assign,
                                };
                            }
                            current_child += 1;
                        },
                        .return_statement => |ret| {
                            if (current_child == child_index) {
                                std.debug.assert(!ret.structuralEquals(modification_ptr));
                                new_statements[i] = Stmt{
                                    .return_statement = modification_ptr,
                                };
                            } else {
                                new_statements[i] = Stmt{
                                    .return_statement = ret,
                                };
                            }
                            current_child += 1;
                        },
                    }
                }
                block.statements = new_statements;
                if (current_child == child_index) {
                    std.debug.assert(!block.*.return_expr.structuralEquals(modification_ptr));
                    block.return_expr = modification_ptr;
                }
                return;
            },
            else => std.debug.panic("unhandled {s} expression type in applyChildModification", .{@tagName(target.*)}),
        }
        @panic("Child index out of bounds in applyChildModification");
    }

    fn copyExpr(self: *ModificationIterator, expr: Expr) !Expr {
        _ = self;
        // For simplicity, we'll do a shallow copy and rely on the fact that
        // we're only modifying pointers, not the underlying data
        return expr;
    }

    fn isAlreadySimplest(self: *ModificationIterator, expr: *const Expr) bool {
        _ = self;
        return switch (expr.*) {
            // These are already the simplest possible for their types
            .literal => |val| switch (val.*) {
                .i8 => |i| i == 0,
                .i16 => |i| i == 0,
                .i32 => |i| i == 0,
                .i64 => |i| i == 0,
                // .i128 => |i| i == 0,
                .u8 => |i| i == 0,
                .u16 => |i| i == 0,
                .u32 => |i| i == 0,
                .u64 => |i| i == 0,
                // .u128 => |i| i == 0,
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
            .binary, .unary, .application, .if_expression, .field_access, .block => false,
            // Variables and functions are leaf nodes but might not be simplest
            .variable, .function, .tbd => false,
        };
    }

    fn inferType(self: *ModificationIterator, expr: *Expr) !*const Ty {
        // Simple type inference - in a real implementation this would be more sophisticated
        return switch (expr.*) {
            .literal => |val| {
                const value_ty_ptr = try self.allocator.create(Ty);
                value_ty_ptr.* = val.ty();
                return value_ty_ptr;
            },
            .binary => |bin| switch (bin.op) {
                .add, .subtract, .multiply, .divide => try self.inferType(bin.left),
                .equals, .not_equals, .less_than, .greater_than, .and_op, .or_op => self.interp.makeBoolTy(),
            },
            .unary => |un| switch (un.op) {
                .negate => try self.inferType(un.operand),
                .not => self.interp.makeBoolTy(),
            },
            .tbd => |ty| ty,
            .variable => |v| v.ty,
            .function => |f| {
                const return_type = try self.inferType(f.body);
                const function_ty_ptr = try self.allocator.create(Ty);
                function_ty_ptr.* = Ty{ .function = FunctionTy{
                    .parameter_types = f.parameter_types,
                    .return_type = return_type,
                } };
                return function_ty_ptr;
            },
            .application => |app| {
                const func_ty = try self.inferType(app.function);
                return switch (func_ty.*) {
                    .function => |f| f.return_type,
                    else => std.debug.panic("Cannot apply non-function type: {}", .{func_ty}),
                };
            },
            .list_literal => |list| {
                const list_ty_ptr = try self.allocator.create(Ty);
                list_ty_ptr.* = .{ .list = list.element_ty };
                return list_ty_ptr;
            },
            .record_literal => |record| {
                var field_types = try self.allocator.alloc(FieldTy, record.fields.len);
                for (record.fields, 0..) |field, i| {
                    const field_ty = try self.inferType(field.value);
                    field_types[i] = FieldTy{
                        .name = field.name,
                        .ty = field_ty,
                    };
                }
                const record_ty_ptr = try self.allocator.create(Ty);
                record_ty_ptr.* = Ty{ .record = .{ .fields = field_types } };
                return record_ty_ptr;
            },
            .tag_literal => |tag| {
                var arg_types = try self.allocator.alloc(*const Ty, tag.arguments.len);
                for (tag.arguments, 0..) |arg, i| {
                    const arg_ty = try self.inferType(arg);
                    arg_types[i] = arg_ty;
                }
                // return Ty{ .tag = .{ .name = tag.name, .arguments = arg_types } };
                const tag_ty_ptr = try self.allocator.create(Ty);
                tag_ty_ptr.* = Ty{ .tag = .{
                    .name = tag.name,
                    .arguments = arg_types,
                } };
                return tag_ty_ptr;
            },
            .field_access => |access| {
                const record_ty = try self.inferType(access.record);
                return switch (record_ty.*) {
                    .record => |record| {
                        for (record.fields) |field| {
                            if (std.mem.eql(u8, field.name, access.field_name)) {
                                return field.ty;
                            }
                        }
                        std.debug.panic("Field '{s}' not found in record", .{access.field_name});
                    },
                    else => std.debug.panic("Cannot access field on non-record type: {}", .{record_ty}),
                };
            },
            .if_expression => |if_expr| try self.inferType(if_expr.then_branch),
            .block => |block| try self.inferType(block.return_expr),
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
            std.debug.print("Found simpler expression: {s}\n", .{candidate_copy});
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
    var iterator = try ModificationIterator.init(allocator, interp, expr, 0);
    defer iterator.deinit();

    // Try each modification from the iterator
    while (try iterator.next()) |candidate| {
        if (current_best.structuralEquals(&candidate)) {
            std.debug.print("Candidate is structurally equal to current best: {s}\n", .{candidate});
            @panic("Candidate is structurally equal to current best, this should not happen");
        }
        var candidate_copy = candidate;
        if (predicate(&candidate_copy, context)) {
            std.debug.print("Found simpler expression: {s}\n", .{candidate_copy});
            // This simpler expression still satisfies the predicate
            current_best = candidate;
            // Restart iteration with the new smaller expression
            iterator.deinit();
            iterator = try ModificationIterator.init(allocator, interp, &current_best, 0);
        }
    }

    return current_best;
}

pub const Expr = union(enum) {
    tbd: *const Ty,
    literal: *Value,
    variable: struct {
        variable: Var,
        ty: *const Ty,
    },
    function: Func,
    application: struct {
        function: *Expr,
        arguments: []*Expr,
    },
    list_literal: struct {
        element_ty: *const Ty,
        elements: []*Expr,
    },
    record_literal: struct {
        fields: []FieldExpr,
    },
    tag_literal: struct {
        name: []const u8,
        arguments: []*Expr,
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
        then_branch: *Expr,
        else_branch: *Expr,
    },
    block: Block,

    pub fn format(self: Expr, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        var buffer = std.ArrayList(u8).init(std.heap.page_allocator);
        defer buffer.deinit();
        try self.formatWithIndent(&buffer, 0, 80);
        try writer.writeAll(buffer.items);
    }

    /// Returns true if two expressions are structurally equal (deep equality).
    pub fn structuralEquals(a: *const Expr, b: *const Expr) bool {
        if (a == b) return true; // Same pointer
        if (@intFromEnum(a.*) != @intFromEnum(b.*)) return false;
        return switch (a.*) {
            .tbd => |aty| switch (b.*) {
                .tbd => |bty| aty.equals(bty),
                else => false,
            },
            .literal => |aval| switch (b.*) {
                .literal => |bval| aval.structuralEquals(bval.*),
                else => false,
            },
            .variable => |av| switch (b.*) {
                .variable => |bv| std.mem.eql(u8, av.variable.name, bv.variable.name) and av.ty.equals(bv.ty),
                else => false,
            },
            .function => |af| switch (b.*) {
                .function => |bf| {
                    // Compare parameter types
                    if (af.parameter_types.len != bf.parameter_types.len) return false;
                    for (af.parameter_types, bf.parameter_types) |ap, bp| {
                        if (!ap.equals(bp)) return false;
                    }
                    // Compare bodies
                    return af.body.structuralEquals(bf.body);
                },
                else => false,
            },
            .application => |aa| switch (b.*) {
                .application => |ba| {
                    if (!aa.function.structuralEquals(ba.function)) return false;
                    if (aa.arguments.len != ba.arguments.len) return false;
                    for (aa.arguments, ba.arguments) |arg1, arg2| {
                        if (!arg1.structuralEquals(arg2)) return false;
                    }
                    return true;
                },
                else => false,
            },
            .list_literal => |al| switch (b.*) {
                .list_literal => |bl| {
                    if (!al.element_ty.equals(bl.element_ty)) return false;
                    if (al.elements.len != bl.elements.len) return false;
                    for (al.elements, bl.elements) |e1, e2| {
                        if (!e1.structuralEquals(e2)) return false;
                    }
                    return true;
                },
                else => false,
            },
            .record_literal => |ar| switch (b.*) {
                .record_literal => |br| {
                    if (ar.fields.len != br.fields.len) return false;
                    for (ar.fields, br.fields) |f1, f2| {
                        if (!std.mem.eql(u8, f1.name, f2.name)) return false;
                        if (!f1.value.structuralEquals(f2.value)) return false;
                    }
                    return true;
                },
                else => false,
            },
            .tag_literal => |at| switch (b.*) {
                .tag_literal => |bt| {
                    if (!std.mem.eql(u8, at.name, bt.name)) return false;
                    if (at.arguments.len != bt.arguments.len) return false;
                    for (at.arguments, bt.arguments) |a1, a2| {
                        if (!a1.structuralEquals(a2)) return false;
                    }
                    return true;
                },
                else => false,
            },
            .binary => |ab| switch (b.*) {
                .binary => |bb| {
                    return ab.op == bb.op and ab.left.structuralEquals(bb.left) and ab.right.structuralEquals(bb.right);
                },
                else => false,
            },
            .unary => |au| switch (b.*) {
                .unary => |bu| {
                    return au.op == bu.op and au.operand.structuralEquals(bu.operand);
                },
                else => false,
            },
            .field_access => |af| switch (b.*) {
                .field_access => |bf| {
                    return std.mem.eql(u8, af.field_name, bf.field_name) and af.record.structuralEquals(bf.record);
                },
                else => false,
            },
            .if_expression => |ai| switch (b.*) {
                .if_expression => |bi| {
                    return ai.condition.structuralEquals(bi.condition) and ai.then_branch.structuralEquals(bi.then_branch) and ai.else_branch.structuralEquals(bi.else_branch);
                },
                else => false,
            },
            .block => |ab| switch (b.*) {
                .block => |bb| {
                    if (ab.statements.len != bb.statements.len) return false;
                    for (ab.statements, bb.statements) |s1, s2| {
                        if (!s1.structuralEquals(s2)) return false;
                    }
                    return ab.return_expr.structuralEquals(bb.return_expr);
                },
                else => false,
            },
        };
    }

    pub fn formatWithIndent(self: Expr, buffer: *std.ArrayList(u8), indent: ?u32, max_width: u32) std.fmt.AllocPrintError!void {
        switch (self) {
            .tbd => |ty| {
                try buffer.writer().print("<tbd: {}>", .{ty.*});
            },
            .literal => |val| try val.formatWithIndent(buffer, indent, max_width),
            .variable => |v| try buffer.writer().print("{s}", .{v.variable.name}),
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
            .block => |block| {
                try block.formatWithIndent(buffer, indent, max_width);
            },
        }
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
                    try element.*.replaceTbdExpressions(interp);
                }
            },
            .record_literal => |rec| {
                for (rec.fields) |*field| {
                    try field.value.replaceTbdExpressions(interp);
                }
            },
            .tag_literal => |tag| {
                for (tag.arguments) |*arg| {
                    try arg.*.replaceTbdExpressions(interp);
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
            .block => |*block| {
                try block.replaceTbdExpressions(interp);
            },
        }
    }
};

const Func = struct {
    parameter_types: []*const Ty,
    body: *Expr,
};

const Block = struct {
    statements: []Stmt,
    return_expr: *Expr,

    /// Returns true if two blocks are structurally equal.
    pub fn structuralEquals(self: *const Block, other: *const Block) bool {
        if (self.statements.len != other.statements.len) return false;
        for (self.statements, other.statements) |s1, s2| {
            if (!s1.structuralEquals(s2)) return false;
        }
        return self.return_expr.structuralEquals(other.return_expr);
    }

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
        value: *Expr,
    },
    return_statement: *Expr,

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
            .return_statement => |ret| {
                try ret.replaceTbdExpressions(interp);
            },
        }
    }

    /// Returns true if two statements are structurally equal.
    pub fn structuralEquals(self: Stmt, other: Stmt) bool {
        if (@intFromEnum(self) != @intFromEnum(other)) return false;
        return switch (self) {
            .assignment => |a1| switch (other) {
                .assignment => |a2| std.mem.eql(u8, a1.variable_name, a2.variable_name) and a1.value.structuralEquals(a2.value),
                else => false,
            },
            .return_statement => |r1| switch (other) {
                .return_statement => |r2| r1.structuralEquals(r2),
                else => false,
            },
        };
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
    TypeMismatch,
    VariableNotFound,
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

    fn makeExpr(self: *Interp, expr: Expr) !*Expr {
        const expr_ptr = try self.allocator.create(Expr);
        expr_ptr.* = expr;
        return expr_ptr;
    }

    fn makeSimpleExpr(self: *Interp, ty: *const Ty) !Expr {
        switch (ty.*) {
            .function => |fty| {
                const body = try self.makeExpr(Expr{ .tbd = fty.return_type });
                return Expr{ .function = .{
                    .parameter_types = fty.parameter_types,
                    .body = body,
                } };
            },
            .list => |el_ty| {
                const len = self.random.int(u32) % 6;
                const elements = try self.allocator.alloc(*Expr, len);
                for (0..len) |i| {
                    const element = try self.allocator.create(Expr);
                    element.* = .{ .tbd = el_ty };
                    elements[i] = element;
                }
                return Expr{ .list_literal = .{ .elements = elements, .element_ty = el_ty } };
            },
            .record => |fields| {
                const field_exprs = try self.allocator.alloc(FieldExpr, fields.fields.len);
                for (fields.fields, 0..) |*field, i| {
                    field_exprs[i] = FieldExpr{ .name = field.name, .value = try self.makeExpr(.{ .tbd = field.ty }) };
                }
                return Expr{ .record_literal = .{ .fields = field_exprs } };
            },
            .tag => |tag| {
                const arg_exprs = try self.allocator.alloc(*Expr, tag.arguments.len);
                for (tag.arguments, 0..) |arg_ty, i| {
                    const arg_expr = try self.allocator.create(Expr);
                    arg_expr.* = .{ .tbd = arg_ty };
                    arg_exprs[i] = arg_expr;
                }
                return Expr{ .tag_literal = .{ .name = tag.name, .arguments = arg_exprs } };
            },
            .i8 => return self.makeLiteral(Value{ .i8 = self.random.int(i8) }),
            .i16 => return self.makeLiteral(Value{ .i16 = self.random.int(i16) }),
            .i32 => return self.makeLiteral(Value{ .i32 = self.random.int(i32) }),
            .i64 => return self.makeLiteral(Value{ .i64 = self.random.int(i64) }),
            // .i128 => return self.makeLiteral(Value{ .i128 = self.random.int(i128) }),
            .u8 => return self.makeLiteral(Value{ .u8 = self.random.int(u8) }),
            .u16 => return self.makeLiteral(Value{ .u16 = self.random.int(u16) }),
            .u32 => return self.makeLiteral(Value{ .u32 = self.random.int(u32) }),
            .u64 => return self.makeLiteral(Value{ .u64 = self.random.int(u64) }),
            // .u128 => return self.makeLiteral(Value{ .u128 = self.random.int(u128) }),
            .float => return self.makeLiteral(Value{ .float = self.random.float(f64) }),
            .bool => return self.makeLiteral(Value{ .bool = self.random.boolean() }),
            .str => {
                const length = self.random.int(u32) % 10 + 1; // Random string length between 1 and 10
                var buffer = try self.allocator.alloc(u8, length);
                for (0..length) |i| {
                    buffer[i] = @as(u8, self.random.int(u8) % 26 + 97); // Random lowercase letters
                }
                return self.makeLiteral(Value{ .str = buffer }); // Don't free the buffer, it's owned by the Value
            },
        }
    }

    fn makeLiteral(self: *Interp, value: Value) !Expr {
        const value_ptr = try self.allocator.create(Value);
        value_ptr.* = value;
        return Expr{ .literal = value_ptr };
    }

    fn makeTrivialExpr(self: *Interp, ty: *const Ty) !*Expr {
        const expr_ptr = try self.allocator.create(Expr);
        expr_ptr.* = Expr{ .tbd = ty };
        return expr_ptr;
    }

    fn makeBoolTy(self: *Interp) !*const Ty {
        const ty_ptr = try self.allocator.create(Ty);
        ty_ptr.* = Ty{ .bool = {} };
        return ty_ptr;
    }

    fn makeNumTy(self: *Interp) !*const Ty {
        const ty_ptr = try self.allocator.create(Ty);
        // Make a random type!
        const type_choice = self.random.intRangeAtMost(u8, 0, 8);
        ty_ptr.* = switch (type_choice) {
            0 => Ty{ .i8 = {} },
            1 => Ty{ .i16 = {} },
            2 => Ty{ .i32 = {} },
            3 => Ty{ .i64 = {} },
            4 => Ty{ .u8 = {} },
            5 => Ty{ .u16 = {} },
            6 => Ty{ .u32 = {} },
            7 => Ty{ .u64 = {} },
            8 => Ty{ .float = {} },
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
        const type_choice = self.random.intRangeAtMost(u8, 0, 14);
        ty_ptr.* = blk: switch (type_choice) {
            0 => Ty{ .i8 = {} },
            1 => Ty{ .i16 = {} },
            2 => Ty{ .i32 = {} },
            3 => Ty{ .i64 = {} },
            4 => Ty{ .u8 = {} },
            5 => Ty{ .u16 = {} },
            6 => Ty{ .u32 = {} },
            7 => Ty{ .u64 = {} },
            8 => Ty{ .float = {} },
            9 => Ty{ .bool = {} },
            10 => Ty{ .str = {} },
            11 => {
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
            12 => {
                // Generate a list type with a random element type
                const element_type = try self.makeAnyTy();
                break :blk Ty{ .list = element_type };
            },
            13 => {
                // Generate a record type with random fields
                const num_fields = self.random.int(u32) % 3 + 1; // At least one field
                var fields = try self.allocator.alloc(FieldTy, num_fields);
                for (0..num_fields) |i| {
                    const field_name = try std.fmt.allocPrint(self.allocator, "field{}", .{i});
                    fields[i] = FieldTy{ .name = field_name, .ty = try self.makeAnyTy() };
                }
                break :blk Ty{ .record = .{ .fields = fields } };
            },
            14 => {
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
                                    expr.* = Expr{ .variable = .{ .variable = Var{ .name = var_info.name }, .ty = var_info.ty } };
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
                                        .then_branch = try self.makeTrivialExpr(t),
                                        .else_branch = try self.makeTrivialExpr(t),
                                    },
                                };
                                break;
                            },
                            3 => {
                                // Generate a binop
                                switch (t.*) {
                                    .function, .list, .record, .tag, .str => continue,
                                    .i8, .i16, .i32, .i64, .u8, .u16, .u32, .u64, .float => {
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
                                    .i8, .i16, .i32, .i64, .float => {
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
                    const prev_gas = self.gas;
                    const result = self.eval(scope, expr, stack_depth + 1) catch |err| {
                        if (retry_count < 100 and (err == error.StackOverflow or err == error.DivideByZero)) {
                            self.gas = prev_gas; // Reset gas
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
                    return scope.get(v.variable);
                },
                .function => |*func| {
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
                        const element_value = try self.eval(scope, element_expr.*, stack_depth + 1);
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
                        const field_value = try self.eval(scope, field.value, stack_depth + 1);
                        try fields.append(Field{ .name = field.name, .value = field_value });
                    }

                    return Value{ .record = .{ .fields = try fields.toOwnedSlice() } };
                },
                .tag_literal => |tag| {
                    var arguments = std.ArrayList(Value).init(self.allocator);
                    defer arguments.deinit();
                    for (tag.arguments) |arg_expr| {
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
                        std.debug.print("Expected record, got: {}\n", .{record_value});
                        std.debug.print("Field access expression: {}\n", .{expr});
                        @panic("Type mismatch");
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
                        @panic("Type mismatch");
                    }

                    const closure = func_value.closure;

                    // Check argument count
                    if (app.arguments.len != closure.function.parameter_types.len) {
                        std.debug.print("Function '{}' expects {} arguments, got {}\n", .{ closure.function, closure.function.parameter_types.len, app.arguments.len });
                        std.debug.print("Function expression: {}\n", .{expr});
                        @panic("Type mismatch");
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
                    return self.eval(func_scope_ptr, closure.function.body, @intCast(stack_depth + 1));
                },
                .if_expression => |if_expr| {
                    // Evaluate condition
                    const condition_value = try self.eval(scope, if_expr.condition, stack_depth + 1);

                    const should_take_then_branch = switch (condition_value) {
                        .bool => |b| b,
                        else => {
                            std.debug.print("Expected boolean condition, got: {}\n", .{condition_value});
                            std.debug.print("If expression: {}\n", .{expr});
                            @panic("Type mismatch");
                        },
                    };

                    if (should_take_then_branch) {
                        return self.eval(scope, if_expr.then_branch, stack_depth + 1);
                    } else {
                        return self.eval(scope, if_expr.else_branch, stack_depth + 1);
                    }
                },
                .block => |*block| {
                    // Evaluate the block
                    return self.evalBlock(scope, block, stack_depth + 1);
                },
            }

            break;
        }
    }

    fn evalBlock(self: *Interp, scope: *Scope, block: *Block, stack_depth: u32) InterpError!Value {
        // For simplicity, just evaluate the return expression
        // In a full implementation, you'd need to handle statements
        return self.eval(scope, block.return_expr, stack_depth + 1);
    }
};
