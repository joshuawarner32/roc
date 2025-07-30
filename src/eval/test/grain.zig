const std = @import("std");

const ValueIdx = struct { index: usize };

const Value = union(enum) {
    int: i128,
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
        captured_variables: Scope,
    },

    fn ty(self: *const Value) Ty {
        return switch (self.*) {
            .int => Ty{ .int = {} },
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

        switch (self) {
            .int => |i| try writer.print("{}", .{i}),
            .i8 => |i| try writer.print("{}i8", .{i}),
            .i16 => |i| try writer.print("{}i16", .{i}),
            .i32 => |i| try writer.print("{}i32", .{i}),
            .i64 => |i| try writer.print("{}i64", .{i}),
            .i128 => |i| try writer.print("{}i128", .{i}),
            .u8 => |i| try writer.print("{}u8", .{i}),
            .u16 => |i| try writer.print("{}u16", .{i}),
            .u32 => |i| try writer.print("{}u32", .{i}),
            .u64 => |i| try writer.print("{}u64", .{i}),
            .u128 => |i| try writer.print("{}u128", .{i}),
            .float => |f| try writer.print("{d}", .{f}),
            .bool => |b| try writer.print("{}", .{b}),
            .str => |s| try writer.print("\"{s}\"", .{s}),
            .list => |lst| {
                try writer.print("List(", .{});
                for (lst.elements, 0..) |element, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{element});
                }
                try writer.print(")", .{});
            },
            .record => |rec| {
                try writer.print("{{", .{});
                for (rec.fields, 0..) |field, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{s}: {}", .{ field.name, field.value });
                }
                try writer.print("}}", .{});
            },
            .tag => |tag| {
                try writer.print("{s}(", .{tag.name});
                for (tag.arguments, 0..) |arg, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{arg});
                }
                try writer.print(")", .{});
            },
            .closure => |closure| {
                try writer.print("|", .{});
                for (closure.function.parameter_types, 0..) |param, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{param});
                }
                try writer.print("| {}", .{closure.function.body});
            },
        }
    }

    pub fn generateRandom(allocator: std.mem.Allocator, random: *const std.Random, typ: *const Ty) !Value {
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
            .list => |element_ty| {
                var elements = std.ArrayList(Value).init(allocator);
                errdefer elements.deinit();
                const count = random.int(u32) % 5 + 1; // Random list size between 1 and 5
                for (0..count) |_| {
                    const element_value = try generateRandom(allocator, random, element_ty);
                    try elements.append(element_value);
                }
                return Value{ .list = .{ .elements = try elements.toOwnedSlice() } };
            },
            .record => |record_ty| {
                var fields = std.ArrayList(Field).init(allocator);
                errdefer fields.deinit();
                for (record_ty.fields) |field_ty| {
                    const field_value = try generateRandom(allocator, random, &field_ty.ty);
                    try fields.append(Field{ .name = field_ty.name, .value = field_value });
                }
                return Value{ .record = .{ .fields = try fields.toOwnedSlice() } };
            },
            .tag => |tag_ty| {
                var arguments = std.ArrayList(Value).init(allocator);
                errdefer arguments.deinit();
                const count = random.int(u32) % 3; // Random number of arguments between 0 and 2
                for (0..count) |_| {
                    const arg_value = try generateRandom(allocator, random, &tag_ty.arguments[0]); // Use first argument type for simplicity
                    try arguments.append(arg_value);
                }
                return Value{ .tag = .{ .name = tag_ty.name, .arguments = try arguments.toOwnedSlice() } };
            },
            .function => |func_ty| {
                // Generate a closure with the specified function type
                const func_ptr = try allocator.create(Func);

                // Copy parameter types
                var param_types = try allocator.alloc(Ty, func_ty.parameter_types.len);
                for (func_ty.parameter_types, 0..) |param_ty, i| {
                    param_types[i] = param_ty;
                }

                // Create a simple block that returns a literal value of the return type
                const return_value = try Value.generateRandom(allocator, random, func_ty.return_type);
                const return_value_ptr = try allocator.create(Value);
                return_value_ptr.* = return_value;
                const return_expr = Expr{ .literal = return_value_ptr };
                const block_ptr = try allocator.create(Block);
                block_ptr.* = Block{
                    .statements = &[_]Stmt{}, // No statements for simplicity
                    .return_expr = return_expr,
                };

                func_ptr.* = Func{
                    .parameter_types = param_types,
                    .body = block_ptr,
                };

                return Value{ .closure = .{
                    .function = func_ptr,
                    .captured_variables = Scope{ .values = std.ArrayList(Value).init(allocator) },
                } };
            },
        };
    }
};

const Field = struct {
    name: []const u8,
    value: Value,
};

const TyIdx = struct { index: usize };

const Var = struct { index: usize };

const Ty = union(enum) {
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
        arguments: []const Ty,
    },
    function: struct {
        parameter_types: []const Ty,
        return_type: *const Ty,
    },

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
                    if (!arg.equals(&other_arg)) {
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
                    if (!param.equals(&other_param)) return false;
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
    ty: Ty,

    pub fn eql(self: FieldTy, other: FieldTy) bool {
        return std.mem.eql(u8, self.name, other.name) and self.ty.equals(&other.ty);
    }
};

const ExprIdx = struct { index: usize };

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
            .and_op => "&&",
            .or_op => "||",
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

const Expr = union(enum) {
    literal: *Value,
    variable: Var,
    function: *Func,
    application: struct {
        function: *Expr,
        arguments: []Expr,
    },
    list_literal: struct {
        elements: []Expr,
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

        switch (self) {
            .literal => |val| try writer.print("{}", .{val.*}),
            .variable => |v| try writer.print("var[{}]", .{v.index}),
            .function => |f| {
                try writer.print("|", .{});
                for (f.parameter_types, 0..) |param, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{param});
                }
                try writer.print("| {}", .{f.body});
            },
            .application => |app| {
                try writer.print("{}(", .{app.function.*});
                for (app.arguments, 0..) |arg, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{arg});
                }
                try writer.print(")", .{});
            },
            .list_literal => |lst| {
                try writer.print("[", .{});
                for (lst.elements, 0..) |element, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{element});
                }
                try writer.print("]", .{});
            },
            .binary => |bin| {
                try writer.print("({} {} {})", .{ bin.left.*, bin.op, bin.right.* });
            },
            .unary => |un| {
                try writer.print("({}{})", .{ un.op, un.operand.* });
            },
            .field_access => |field| {
                try writer.print("{}.{s}", .{ field.record.*, field.field_name });
            },
            .if_expression => |if_expr| {
                try writer.print("if ({}) {} else {}", .{ if_expr.condition.*, if_expr.then_branch, if_expr.else_branch });
            },
        }
    }

    pub fn generateRandom(allocator: std.mem.Allocator, random: *const std.Random, typ: *const Ty) !Expr {
        return generateRandomWithDepth(allocator, random, typ, 0);
    }

    fn generateRandomWithDepth(allocator: std.mem.Allocator, random: *const std.Random, typ: *const Ty, depth: u32) !Expr {
        // Prevent infinite recursion by limiting depth or for function types
        if (depth > 3 or typ.* == .function) {
            // Generate a literal value
            const value = try Value.generateRandom(allocator, random, typ);
            const value_ptr = try allocator.create(Value);
            value_ptr.* = value;
            return Expr{ .literal = value_ptr };
        }

        // there are lots of ways we can construct an expr!
        const which = random.int(u32) % 6;
        switch (which) {
            0 => {
                // Generate a literal value
                const value = try Value.generateRandom(allocator, random, typ);
                const value_ptr = try allocator.create(Value);
                value_ptr.* = value;
                return Expr{ .literal = value_ptr };
            },
            1 => {
                // Generate an application that resolves to the given type
                return try generateApplicationWithDepth(allocator, random, typ);
            },
            2 => {
                // Generate a binary expression that resolves to the given type
                return try generateBinaryExpression(allocator, random, typ, depth);
            },
            3 => {
                // Generate a unary expression that resolves to the given type
                return try generateUnaryExpression(allocator, random, typ, depth);
            },
            4 => {
                // Generate a field access expression that resolves to the given type
                return try generateFieldAccess(allocator, random, typ, depth);
            },
            5 => {
                // Generate an if expression that resolves to the given type
                return try generateIfExpression(allocator, random, typ, depth);
            },
            else => unreachable,
        }
    }

    fn generateApplicationWithDepth(allocator: std.mem.Allocator, random: *const std.Random, return_type: *const Ty) !Expr {
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
        const func_value = try Value.generateRandom(allocator, random, &func_type);
        const func_value_ptr = try allocator.create(Value);
        func_value_ptr.* = func_value;
        const func_expr = Expr{ .literal = func_value_ptr };
        const func_expr_ptr = try allocator.create(Expr);
        func_expr_ptr.* = func_expr;

        // Generate arguments of the appropriate types (always literals)
        var arguments = try allocator.alloc(Expr, num_args);
        for (0..num_args) |i| {
            const arg_value = try Value.generateRandom(allocator, random, &arg_types[i]);
            const arg_value_ptr = try allocator.create(Value);
            arg_value_ptr.* = arg_value;
            arguments[i] = Expr{ .literal = arg_value_ptr };
        }

        return Expr{ .application = .{
            .function = func_expr_ptr,
            .arguments = arguments,
        } };
    }

    fn generateBinaryExpression(allocator: std.mem.Allocator, random: *const std.Random, result_type: *const Ty, depth: u32) !Expr {
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
                const type_choice = random.int(u32) % 14; // Now we have 14 basic types
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

    fn generateUnaryExpression(allocator: std.mem.Allocator, random: *const std.Random, result_type: *const Ty, depth: u32) !Expr {
        _ = depth; // Unused for now

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

    fn generateFieldAccess(allocator: std.mem.Allocator, random: *const std.Random, field_type: *const Ty, depth: u32) !Expr {
        _ = depth; // Unused for now

        // Generate a record with a field of the desired type
        const field_name = "field"; // Simple field name for now
        const field_ty = FieldTy{ .name = field_name, .ty = field_type.* };
        const record_fields = try allocator.alloc(FieldTy, 1);
        record_fields[0] = field_ty;

        const record_type = Ty{ .record = .{ .fields = record_fields } };

        // Generate a record value (use literal to avoid deep recursion)
        const record_value = try Value.generateRandom(allocator, random, &record_type);
        const record_value_ptr = try allocator.create(Value);
        record_value_ptr.* = record_value;
        const record_expr_ptr = try allocator.create(Expr);
        record_expr_ptr.* = Expr{ .literal = record_value_ptr };

        return Expr{ .field_access = .{
            .record = record_expr_ptr,
            .field_name = field_name,
        } };
    }

    fn generateIfExpression(allocator: std.mem.Allocator, random: *const std.Random, result_type: *const Ty, depth: u32) !Expr {
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
    parameter_types: []Ty,
    body: *const Block,
};

const Block = struct {
    statements: []const Stmt,
    return_expr: Expr,

    pub fn format(self: Block, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("{{", .{});
        for (self.statements, 0..) |stmt, i| {
            if (i > 0) try writer.print("\n", .{});
            try stmt.format(fmt, options, writer);
        }
        try writer.print("{}", .{self.return_expr});
        try writer.print("}}", .{});
    }
};

const Stmt = union(enum) {
    assignment: struct {
        value: Expr,
    },
    expression: Expr,
    return_statement: Expr,

    pub fn format(self: Stmt, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        switch (self) {
            .assignment => |assign| try writer.print("{s} = {}", .{ "var", assign.value }),
            .expression => |expr| try writer.print("{}", .{expr}),
            .return_statement => |ret| try writer.print("return {}", .{ret}),
        }
    }
};

const Module = struct {
    statements: []const Stmt,
};

const Scope = struct {
    values: std.ArrayList(Value),

    fn get(self: *const Scope, v: Var) !Value {
        if (v.index >= self.values.items.len) {
            return error.VariableNotFound;
        }
        return self.values.items[v.index];
    }
};

const Interp = struct {
    gas: u64,

    /// Evaluate the program and return the final value.
    pub fn eval(self: *Interp, scope: *Scope, ty: *const Ty, expr: *const Expr) !Value {
        if (self.gas == 0) {
            return error.OutOfGas;
        }

        switch (expr.*) {
            .literal => |lit| {
                switch (lit.*) {
                    else => {
                        const actual_ty = lit.ty();
                        if (!actual_ty.equals(ty)) {
                            return error.TypeMismatch;
                        }
                        return lit.*;
                    },
                }
            },
            .variable => |index| {
                return scope.get(index);
            },
            .function => |func| {
                return Value{ .closure = .{
                    .function = func,
                    .captured_variables = scope.*,
                } };
            },
            .list_literal => |list_lit| {
                var elements = std.ArrayList(Value).init(self.context.values.allocator);
                defer elements.deinit();

                for (list_lit.elements) |element_expr| {
                    // For simplicity, use a generic type for list elements
                    const element_value = try self.eval(scope, &Ty{ .u8 = {} }, &element_expr);
                    try elements.append(element_value);
                }

                return Value{ .list = .{ .elements = try elements.toOwnedSlice() } };
            },
            .binary => |bin| {
                _ = bin;
                @panic("todo");
            },
            .unary => |un| {
                _ = un;
                @panic("todo");
            },
            .field_access => |_| {
                @panic("todo");
            },
            .application => |app| {
                _ = app;
                @panic("todo");
            },
            .if_expression => |if_expr| {
                // Evaluate condition
                const condition_value = try self.eval(scope, &Ty{ .bool = {} }, if_expr.condition);

                const should_take_then_branch = switch (condition_value) {
                    .bool => |b| b,
                    else => return error.TypeMismatch,
                };

                if (should_take_then_branch) {
                    return self.evalBlock(scope, ty, if_expr.then_branch);
                } else {
                    return self.evalBlock(scope, ty, if_expr.else_branch);
                }
            },
        }
    }

    fn evalBlock(self: *Interp, scope: *const Scope, ty: *const Ty, block: *const Block) error{ OutOfGas, UnificationFailed, ContradictoryConstraint, TypeMismatch, VariableNotFound, TbdExpressionNotImplemented, ApplicationNotImplemented, IfExpressionNotImplemented, OutOfMemory }!Value {
        // For simplicity, just evaluate the return expression
        // In a full implementation, you'd need to handle statements
        return self.eval(scope, ty, &block.return_expr);
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

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = prng.random();

    for (0..count) |_| {
        const expr = try Expr.generateRandom(allocator, &random, &Ty{ .u8 = {} });
        std.debug.print("Generated expression: {}\n", .{expr});
    }
}
