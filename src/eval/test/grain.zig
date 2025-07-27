const std = @import("std");

const Value = union(enum) {
    /// Represents a value that is not yet generated. Pointer to where we should fill in the value.
    tbd: *Expr,

    int: i128,
    float: f64,
    bool: bool,
    str: []const u8,
    list: struct {
        ty: *const Ty, // Must be a List type
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
        captured_variables: CapturedScope,
    },

    fn ty(self: *const Value) Ty {
        return switch (self.*) {
            .tbd => @panic("todo"),
            .int => Ty{ .int = {} },
            .float => Ty{ .float = {} },
            .bool => Ty{ .bool = {} },
            .str => Ty{ .str = {} },
            .list => |lst| lst.ty.*,
            else => @panic("todo"),
        };
    }
};

const Field = struct {
    name: []const u8,
    value: Value,
};

const Var = struct { index: usize };

const Ty = union(enum) {
    int,
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

    pub fn equals(self: *const Ty, other: *const Ty) bool {
        if (self == other) return true;
        return switch (self.*) {
            .int => |_| other.* == .int,
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
        };
    }
};

const FieldTy = struct {
    name: []const u8,
    ty: Ty,

    pub fn eql(self: FieldTy, other: FieldTy) bool {
        return std.mem.eql(u8, self.name, other.name) and self.ty.equals(&other.ty);
    }
};

const Expr = union(enum) {
    tbd: *Ty,
    literal: *Value,
    variable: Var,
    function: *Func,
    application: struct {
        function: *Expr,
        arguments: []Expr,
    },
    if_expression: struct {
        condition: *Expr,
        then_branch: *Block,
        else_branch: *Block,
    },
};

const Func = struct {
    parameters: []const []const u8,
    body: *const Block,
};

const Block = struct {
    statements: []const Stmt,
    return_expr: Expr,
};

const Stmt = union(enum) {
    assignment: struct {
        variable: []const u8,
        value: Expr,
    },
    expression: Expr,
    return_statement: Expr,
};

const Module = struct {
    statements: []const Stmt,
};

const CapturedScope = struct {
    values: []const Value,

    fn get(self: *const CapturedScope, v: Var) !Value {
        if (v.index >= self.values.len) {
            return error.VariableNotFound;
        }
        return self.values[v.index];
    }
};

const ActiveScope = struct {
    variables: std.ArrayList(Value),
};

const Interp = struct {
    gas: u64,

    /// Evaluate the program and return the final value.
    pub fn eval(self: *const Interp, scope: *const CapturedScope, ty: *const Ty, expr: *const Expr) !Value {
        if (self.gas == 0) {
            return error.OutOfGas;
        }

        switch (expr.*) {
            .tbd => |tbd_ty| {
                _ = tbd_ty; // We need to evaluate the expression to fill in the value.
                return error.TbdExpressionNotImplemented;
            },
            .literal => |lit| {
                const actual_ty = lit.ty();
                if (!actual_ty.equals(ty)) {
                    return error.TypeMismatch;
                }
                return lit.*;
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
            .application => |app| {
                _ = app; // We need to evaluate the function and arguments.
                return error.ApplicationNotImplemented;
            },
            .if_expression => |if_expr| {
                _ = if_expr; // We need to evaluate the condition and branches.
                return error.IfExpressionNotImplemented;
            },
        }
    }
};

pub fn main() !void {
    // const gpa = std.heap.DebugAllocator(.{});
    // const allocator = gpa.allocator();

    var interp = Interp{ .gas = 1000 };

    // Example usage
    const scope = CapturedScope{ .values = &.{.{ .int = 42 }} };
    const ty = Ty{ .int = {} };
    const expr = Expr{ .variable = Var{ .index = 0 } };

    const result = try interp.eval(&scope, &ty, &expr);
    std.debug.print("Result: {}\n", .{result});
}
