const std = @import("std");

const InequalityConstraint = struct {
    a: ValueIdx,
    b: ValueIdx,
};

const UnionFind = struct {
    parent: std.ArrayList(usize),
    rank: std.ArrayList(u32),

    pub fn init(allocator: std.mem.Allocator) UnionFind {
        return UnionFind{
            .parent = std.ArrayList(usize).init(allocator),
            .rank = std.ArrayList(u32).init(allocator),
        };
    }

    pub fn deinit(self: *UnionFind) void {
        self.parent.deinit();
        self.rank.deinit();
    }

    pub fn makeSet(self: *UnionFind) !usize {
        const id = self.parent.items.len;
        try self.parent.append(id);
        try self.rank.append(0);
        return id;
    }

    pub fn find(self: *UnionFind, x: usize) usize {
        if (self.parent.items[x] != x) {
            self.parent.items[x] = self.find(self.parent.items[x]); // Path compression
        }
        return self.parent.items[x];
    }

    pub fn unite(self: *UnionFind, x: usize, y: usize) void {
        const root_x = self.find(x);
        const root_y = self.find(y);

        if (root_x == root_y) return;

        // Union by rank
        if (self.rank.items[root_x] < self.rank.items[root_y]) {
            self.parent.items[root_x] = root_y;
        } else if (self.rank.items[root_x] > self.rank.items[root_y]) {
            self.parent.items[root_y] = root_x;
        } else {
            self.parent.items[root_y] = root_x;
            self.rank.items[root_x] += 1;
        }
    }
};

const Context = struct {
    values: std.ArrayList(Value),
    types: std.ArrayList(Ty),
    exprs: std.ArrayList(Expr),
    value_uf: UnionFind,
    inequality_constraints: std.ArrayList(InequalityConstraint),
    random: std.Random,

    pub fn init(allocator: std.mem.Allocator, random: std.Random) Context {
        return Context{
            .values = std.ArrayList(Value).init(allocator),
            .types = std.ArrayList(Ty).init(allocator),
            .exprs = std.ArrayList(Expr).init(allocator),
            .value_uf = UnionFind.init(allocator),
            .inequality_constraints = std.ArrayList(InequalityConstraint).init(allocator),
            .random = random,
        };
    }

    pub fn deinit(self: *Context) void {
        self.values.deinit();
        self.types.deinit();
        self.exprs.deinit();
        self.value_uf.deinit();
        self.inequality_constraints.deinit();
    }

    pub fn newValue(self: *Context) !ValueIdx {
        const idx = self.values.items.len;
        _ = try self.value_uf.makeSet();
        try self.values.append(Value{ .tbd = ValueIdx{ .index = idx } });
        return ValueIdx{ .index = idx };
    }

    pub fn newType(self: *Context) !Ty {
        const idx = try self.types.append(Ty{ .tbd = TyIdx{ .index = self.types.items.len } });
        return self.types.items[idx];
    }

    pub fn newExpr(self: *Context) !Expr {
        const idx = try self.exprs.append(Expr{ .tbd = ExprIdx{ .index = self.exprs.items.len } });
        return self.exprs.items[idx];
    }

    pub fn unifyValue(self: *Context, a: ValueIdx, b: ValueIdx) !void {
        const root_a = self.value_uf.find(a.index);
        const root_b = self.value_uf.find(b.index);
        
        if (root_a == root_b) return; // Already unified

        const value_a = &self.values.items[root_a];
        const value_b = &self.values.items[root_b];

        switch (value_a.*) {
            .tbd => {
                switch (value_b.*) {
                    .tbd => {
                        // Both are TBD, just unite them
                        self.value_uf.unite(root_a, root_b);
                    },
                    else => {
                        // A is TBD, B is concrete - unite and use B's value
                        self.value_uf.unite(root_a, root_b);
                        const final_root = self.value_uf.find(root_a);
                        if (final_root == root_a) {
                            value_a.* = value_b.*;
                        }
                    },
                }
            },
            else => {
                switch (value_b.*) {
                    .tbd => {
                        // A is concrete, B is TBD - unite and use A's value
                        self.value_uf.unite(root_a, root_b);
                        const final_root = self.value_uf.find(root_a);
                        if (final_root == root_b) {
                            value_b.* = value_a.*;
                        }
                    },
                    else => {
                        // Both are concrete - try to unify them structurally
                        try self.unifyStructurally(value_a.*, value_b.*);
                        self.value_uf.unite(root_a, root_b);
                        // After unifying, make sure one of the values is updated to reflect the unified result
                        const final_root = self.value_uf.find(root_a);
                        if (final_root == root_a) {
                            // Keep value_a as is
                        } else {
                            // Update value_a to match value_b
                            value_a.* = value_b.*;
                        }
                    },
                }
            },
        }
    }

    fn unifyStructurally(self: *Context, a: Value, b: Value) error{UnificationFailed, ContradictoryConstraint, OutOfMemory}!void {
        switch (a) {
            .tbd => {}, // TBD can unify with anything
            .int => |int_a| switch (b) {
                .tbd => {}, // TBD can unify with anything
                .int => |int_b| {
                    if (int_a != int_b) return error.UnificationFailed;
                },
                else => return error.UnificationFailed,
            },
            .list => |list_a| switch (b) {
                .tbd => {}, // TBD can unify with anything
                .list => |list_b| {
                    if (list_a.elements.len != list_b.elements.len) {
                        return error.UnificationFailed;
                    }
                    for (list_a.elements, list_b.elements) |elem_a, elem_b| {
                        const elem_a_idx = try self.storeValueForUnification(elem_a);
                        const elem_b_idx = try self.storeValueForUnification(elem_b);
                        try self.unify(elem_a_idx, elem_b_idx);
                    }
                },
                else => return error.UnificationFailed,
            },
            else => {
                if (!self.valuesEqual(a, b)) {
                    return error.UnificationFailed;
                }
            },
        }
    }

    fn storeValueForUnification(self: *Context, value: Value) !ValueIdx {
        switch (value) {
            .tbd => |tbd_idx| {
                // For TBD values, return the existing index
                return tbd_idx;
            },
            else => {
                const idx = self.values.items.len;
                _ = try self.value_uf.makeSet();
                try self.values.append(value);
                return ValueIdx{ .index = idx };
            },
        }
    }

    fn valuesEqual(self: *Context, a: Value, b: Value) bool {
        return switch (a) {
            .tbd => false, // TBD values are never equal to concrete values
            .int => |int_a| switch (b) {
                .int => |int_b| int_a == int_b,
                else => false,
            },
            .float => |float_a| switch (b) {
                .float => |float_b| float_a == float_b,
                else => false,
            },
            .bool => |bool_a| switch (b) {
                .bool => |bool_b| bool_a == bool_b,
                else => false,
            },
            .str => |str_a| switch (b) {
                .str => |str_b| std.mem.eql(u8, str_a, str_b),
                else => false,
            },
            .list => |list_a| switch (b) {
                .list => |list_b| {
                    if (list_a.elements.len != list_b.elements.len) return false;
                    for (list_a.elements, list_b.elements) |elem_a, elem_b| {
                        // If any element is TBD, we can't determine equality concretely
                        if (elem_a == .tbd or elem_b == .tbd) return false; // Return false to indicate "not concretely equal"
                        if (!self.valuesEqual(elem_a, elem_b)) return false;
                    }
                    return true;
                },
                else => false,
            },
            else => @panic("TODO: implement equality for other value types"),
        };
    }

    pub fn unify(self: *Context, a: ValueIdx, b: ValueIdx) !void {
        return self.unifyValue(a, b);
    }

    pub fn getValue(self: *Context, idx: ValueIdx) *Value {
        const root = self.value_uf.find(idx.index);
        return &self.values.items[root];
    }

    pub fn createTbdValue(self: *Context) !*Value {
        const idx = try self.newValue();
        return &self.values.items[idx.index];
    }

    pub fn resolveValue(self: *Context, value: Value) Value {
        switch (value) {
            .tbd => |tbd_idx| {
                const root = self.value_uf.find(tbd_idx.index);
                const resolved = &self.values.items[root];
                if (resolved.* == .tbd) {
                    return value; // Still unresolved
                }
                return self.resolveValue(resolved.*);
            },
            .list => |lst| {
                var resolved_elements = std.ArrayList(Value).init(self.values.allocator);
                defer resolved_elements.deinit();
                
                for (lst.elements) |element| {
                    resolved_elements.append(self.resolveValue(element)) catch return value;
                }
                
                return Value{ .list = .{ .elements = resolved_elements.toOwnedSlice() catch return value } };
            },
            else => return value,
        }
    }

    pub fn addInequalityConstraint(self: *Context, a: ValueIdx, b: ValueIdx) !void {
        const root_a = self.value_uf.find(a.index);
        const root_b = self.value_uf.find(b.index);
        
        if (root_a == root_b) {
            return error.ContradictoryConstraint; // Can't be both equal and unequal
        }
        
        try self.inequality_constraints.append(InequalityConstraint{
            .a = ValueIdx{ .index = root_a },
            .b = ValueIdx{ .index = root_b },
        });
    }

    fn violatesInequalityConstraints(self: *Context, a: ValueIdx, b: ValueIdx) bool {
        const root_a = self.value_uf.find(a.index);
        const root_b = self.value_uf.find(b.index);
        
        for (self.inequality_constraints.items) |constraint| {
            const constraint_a = self.value_uf.find(constraint.a.index);
            const constraint_b = self.value_uf.find(constraint.b.index);
            
            if ((constraint_a == root_a and constraint_b == root_b) or
                (constraint_a == root_b and constraint_b == root_a)) {
                return true;
            }
        }
        return false;
    }

    const EqualityResult = enum {
        True,
        False,
        Undetermined,
    };

    pub fn checkEquality(self: *Context, a: ValueIdx, b: ValueIdx) !EqualityResult {
        const root_a = self.value_uf.find(a.index);
        const root_b = self.value_uf.find(b.index);
        
        // If they're already unified, they're equal
        if (root_a == root_b) return .True;
        
        // Check if there's an inequality constraint preventing equality
        if (self.violatesInequalityConstraints(ValueIdx{ .index = root_a }, ValueIdx{ .index = root_b })) {
            return .False;
        }
        
        const value_a = &self.values.items[root_a];
        const value_b = &self.values.items[root_b];
        
        // If both are concrete, compare them directly but check for nested TBDs
        if (value_a.* != .tbd and value_b.* != .tbd) {
            if (self.containsTbd(value_a.*) or self.containsTbd(value_b.*)) {
                // Contains TBD values, so equality is undetermined
                return .Undetermined;
            }
            return if (self.valuesEqual(value_a.*, value_b.*)) .True else .False;
        }
        
        // At least one is TBD, so equality is undetermined
        return .Undetermined;
    }

    fn containsTbd(self: *Context, value: Value) bool {
        return switch (value) {
            .tbd => true,
            .list => |lst| {
                for (lst.elements) |element| {
                    if (self.containsTbd(element)) return true;
                }
                return false;
            },
            else => false,
        };
    }

    pub fn makeRandomEqualityChoice(self: *Context, a: ValueIdx, b: ValueIdx) !bool {
        const equality_result = try self.checkEquality(a, b);
        return switch (equality_result) {
            .True => true,
            .False => false,
            .Undetermined => {
                // Make random choice and enforce the consequences
                const choice = self.random.boolean();
                if (choice) {
                    // Choose equality - unify the values
                    try self.unify(a, b);
                } else {
                    // Choose inequality - add constraint
                    try self.addInequalityConstraint(a, b);
                }
                return choice;
            },
        };
    }

    pub fn resolveExpr(self: *Context, expr: Expr, allocator: std.mem.Allocator) !Expr {
        return switch (expr) {
            .literal => |val| {
                const resolved_val = try allocator.create(Value);
                resolved_val.* = self.resolveValue(val.*);
                const resolved_expr = try allocator.create(Expr);
                resolved_expr.* = Expr{ .literal = resolved_val };
                return resolved_expr.*;
            },
            .list_literal => |lst| {
                var resolved_elements = std.ArrayList(Expr).init(allocator);
                defer resolved_elements.deinit();
                
                for (lst.elements) |element| {
                    const resolved_element = try self.resolveExpr(element, allocator);
                    try resolved_elements.append(resolved_element);
                }
                
                const resolved_expr = try allocator.create(Expr);
                resolved_expr.* = Expr{ .list_literal = .{ .elements = try resolved_elements.toOwnedSlice() } };
                return resolved_expr.*;
            },
            .equality => |eq| {
                const resolved_left = try self.resolveExpr(eq.left.*, allocator);
                const resolved_right = try self.resolveExpr(eq.right.*, allocator);
                
                const left_ptr = try allocator.create(Expr);
                left_ptr.* = resolved_left;
                const right_ptr = try allocator.create(Expr);
                right_ptr.* = resolved_right;
                
                const resolved_expr = try allocator.create(Expr);
                resolved_expr.* = Expr{ .equality = .{ .left = left_ptr, .right = right_ptr } };
                return resolved_expr.*;
            },
            .if_expression => |if_expr| {
                const resolved_condition = try self.resolveExpr(if_expr.condition.*, allocator);
                const condition_ptr = try allocator.create(Expr);
                condition_ptr.* = resolved_condition;
                
                const resolved_expr = try allocator.create(Expr);
                resolved_expr.* = Expr{ .if_expression = .{
                    .condition = condition_ptr,
                    .then_branch = if_expr.then_branch,
                    .else_branch = if_expr.else_branch,
                } };
                return resolved_expr.*;
            },
            else => return expr,
        };
    }
};

const ValueIdx = struct { index: usize };

const Value = union(enum) {
    /// Represents a value that is not yet generated. Pointer to where we should fill in the value.
    tbd: ValueIdx,

    int: i128,
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
        captured_variables: CapturedScope,
    },

    fn ty(self: *const Value) Ty {
        return switch (self.*) {
            .tbd => @panic("todo"),
            .int => Ty{ .int = {} },
            .float => Ty{ .float = {} },
            .bool => Ty{ .bool = {} },
            .str => Ty{ .str = {} },
            .list => Ty{ .list = undefined }, // Simplified for now
            else => @panic("todo"),
        };
    }

    pub fn format(self: Value, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        
        switch (self) {
            .tbd => |tbd_idx| try writer.print("<tbd:{}>", .{tbd_idx.index}),
            .int => |i| try writer.print("{}", .{i}),
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
            else => try writer.print("<unknown>", .{}),
        }
    }
};

const Field = struct {
    name: []const u8,
    value: Value,
};

const TyIdx = struct { index: usize };

const Var = struct { index: usize };

const Ty = union(enum) {
    tbd: TyIdx,

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
            .tbd => |tbd_ty| {
                _ = tbd_ty; // We need to evaluate the type to fill in the value.
                @panic("todo");
            },
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

const ExprIdx = struct { index: usize };

const Expr = union(enum) {
    tbd: ExprIdx,
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
    equality: struct {
        left: *Expr,
        right: *Expr,
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
            .tbd => |tbd_idx| try writer.print("<tbd:{}>", .{tbd_idx.index}),
            .literal => |val| try writer.print("{}", .{val.*}),
            .variable => |v| try writer.print("var[{}]", .{v.index}),
            .function => try writer.print("<function>", .{}),
            .application => try writer.print("<application>", .{}),
            .list_literal => |lst| {
                try writer.print("List(", .{});
                for (lst.elements, 0..) |element, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{element});
                }
                try writer.print(")", .{});
            },
            .equality => |eq| {
                try writer.print("({} == {})", .{ eq.left.*, eq.right.* });
            },
            .if_expression => |if_expr| {
                try writer.print("if ({}) {{ {} }} else {{ {} }}", .{ 
                    if_expr.condition.*, 
                    if_expr.then_branch.return_expr,
                    if_expr.else_branch.return_expr 
                });
            },
        }
    }
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
    context: *Context,
    gas: u64,

    /// Evaluate the program and return the final value.
    pub fn eval(self: *Interp, scope: *const CapturedScope, ty: *const Ty, expr: *const Expr) !Value {
        if (self.gas == 0) {
            return error.OutOfGas;
        }

        switch (expr.*) {
            .tbd => |tbd_ty| {
                _ = tbd_ty; // We need to evaluate the expression to fill in the value.
                return error.TbdExpressionNotImplemented;
            },
            .literal => |lit| {
                switch (lit.*) {
                    .tbd => return lit.*, // Skip type checking for TBD values
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
                    const element_value = try self.eval(scope, &Ty{ .int = {} }, &element_expr);
                    try elements.append(element_value);
                }
                
                return Value{ .list = .{ .elements = try elements.toOwnedSlice() } };
            },
            .equality => |eq| {
                // Evaluate both sides to get ValueIdx references
                const left_value = try self.eval(scope, &Ty{ .list = undefined }, eq.left);
                const right_value = try self.eval(scope, &Ty{ .list = undefined }, eq.right);
                
                // For now, assume we can store values in the context and get their indices
                // This is a simplification - in practice you'd need a way to map Values to ValueIdx
                const left_idx = try self.storeValue(left_value);
                const right_idx = try self.storeValue(right_value);
                
                // Make random choice about equality
                const equality_result = try self.context.makeRandomEqualityChoice(left_idx, right_idx);
                return Value{ .bool = equality_result };
            },
            .application => |app| {
                _ = app; // We need to evaluate the function and arguments.
                return error.ApplicationNotImplemented;
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

    fn storeValue(self: *Interp, value: Value) !ValueIdx {
        // Check if this value is already a TBD that exists in our context
        switch (value) {
            .tbd => |tbd_idx| {
                // Return the existing ValueIdx for this TBD
                return tbd_idx;
            },
            else => {
                const idx = self.context.values.items.len;
                _ = try self.context.value_uf.makeSet();
                try self.context.values.append(value);
                return ValueIdx{ .index = idx };
            },
        }
    }

    fn evalBlock(self: *Interp, scope: *const CapturedScope, ty: *const Ty, block: *const Block) error{OutOfGas, UnificationFailed, ContradictoryConstraint, TypeMismatch, VariableNotFound, TbdExpressionNotImplemented, ApplicationNotImplemented, IfExpressionNotImplemented, OutOfMemory}!Value {
        // For simplicity, just evaluate the return expression
        // In a full implementation, you'd need to handle statements
        return self.eval(scope, ty, &block.return_expr);
    }
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .{};
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(2);
    const random = prng.random();

    var context = Context.init(allocator, random);
    defer context.deinit();

    var interp = Interp{
        .context = &context,
        .gas = 1000,
    };

    // Create the program: if (List(1, <tbd:123>) == List(<tbd:456>, 2)) { 1 } else { 2 }
    
    // Create TBD values
    const tbd_123 = try context.createTbdValue();
    const tbd_456 = try context.createTbdValue();
    
    // Create literal values
    const one_val = try allocator.create(Value);
    one_val.* = Value{ .int = 1 };
    const one_literal = try allocator.create(Expr);
    one_literal.* = Expr{ .literal = one_val };
    
    const two_val = try allocator.create(Value);
    two_val.* = Value{ .int = 2 };
    const two_literal = try allocator.create(Expr);
    two_literal.* = Expr{ .literal = two_val };
    
    const tbd_123_val = try allocator.create(Value);
    tbd_123_val.* = tbd_123.*;
    const tbd_123_literal = try allocator.create(Expr);
    tbd_123_literal.* = Expr{ .literal = tbd_123_val };
    
    const tbd_456_val = try allocator.create(Value);
    tbd_456_val.* = tbd_456.*;
    const tbd_456_literal = try allocator.create(Expr);
    tbd_456_literal.* = Expr{ .literal = tbd_456_val };
    
    // Create list expressions
    const left_list_elements = try allocator.alloc(Expr, 2);
    left_list_elements[0] = one_literal.*;
    left_list_elements[1] = tbd_123_literal.*;
    const left_list = try allocator.create(Expr);
    left_list.* = Expr{ .list_literal = .{ .elements = left_list_elements } };
    
    const right_list_elements = try allocator.alloc(Expr, 2);
    right_list_elements[0] = tbd_456_literal.*;
    right_list_elements[1] = two_literal.*;
    const right_list = try allocator.create(Expr);
    right_list.* = Expr{ .list_literal = .{ .elements = right_list_elements } };
    
    // Create equality expression
    const equality = try allocator.create(Expr);
    equality.* = Expr{ .equality = .{ .left = left_list, .right = right_list } };
    
    // Create then/else values
    const then_val = try allocator.create(Value);
    then_val.* = Value{ .int = 1 };
    const then_literal = try allocator.create(Expr);
    then_literal.* = Expr{ .literal = then_val };
    
    const else_val = try allocator.create(Value);
    else_val.* = Value{ .int = 2 };
    const else_literal = try allocator.create(Expr);
    else_literal.* = Expr{ .literal = else_val };
    
    // Create blocks
    const then_block = try allocator.create(Block);
    then_block.* = Block{ .statements = &.{}, .return_expr = then_literal.* };
    const else_block = try allocator.create(Block);
    else_block.* = Block{ .statements = &.{}, .return_expr = else_literal.* };
    
    // Create if expression
    const if_expr = Expr{ .if_expression = .{
        .condition = equality,
        .then_branch = then_block,
        .else_branch = else_block,
    } };
    
    std.debug.print("BEFORE execution:\n{}\n\n", .{if_expr});
    
    // Execute the program
    const scope = CapturedScope{ .values = &.{} };
    const ty = Ty{ .int = {} };
    const result = try interp.eval(&scope, &ty, &if_expr);
    
    std.debug.print("Result: {}\n", .{result});
    
    // Show resolved expression based on the result
    if (result.int == 1) {
        // If result was 1, the lists were unified, so show the resolved expression
        const resolved_expr = try context.resolveExpr(if_expr, allocator);
        std.debug.print("AFTER execution (resolved):\n{}\n", .{resolved_expr});
    } else {
        // If result was 2, the lists were not equal, so show original
        std.debug.print("AFTER execution (original):\n{}\n", .{if_expr});
    }
}
