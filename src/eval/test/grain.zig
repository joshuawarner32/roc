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
                        // Both are concrete - check if they're equal
                        if (!self.valuesEqual(value_a.*, value_b.*)) {
                            return error.UnificationFailed;
                        }
                        self.value_uf.unite(root_a, root_b);
                    },
                }
            },
        }
    }

    fn valuesEqual(self: *Context, a: Value, b: Value) bool {
        _ = self;
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
        
        // If both are concrete, compare them directly
        if (value_a.* != .tbd and value_b.* != .tbd) {
            return if (self.valuesEqual(value_a.*, value_b.*)) .True else .False;
        }
        
        // At least one is TBD, so equality is undetermined
        return .Undetermined;
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
    equality: struct {
        left: *Expr,
        right: *Expr,
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
            .equality => |eq| {
                // Evaluate both sides to get ValueIdx references
                const left_value = try self.eval(scope, ty, eq.left);
                const right_value = try self.eval(scope, ty, eq.right);
                
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
        const idx = self.context.values.items.len;
        _ = try self.context.value_uf.makeSet();
        try self.context.values.append(value);
        return ValueIdx{ .index = idx };
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

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    var context = Context.init(allocator, random);
    defer context.deinit();

    var interp = Interp{
        .context = &context,
        .gas = 1000,
    };

    // Example usage
    const scope = CapturedScope{ .values = &.{.{ .int = 42 }} };
    const ty = Ty{ .int = {} };
    const expr = Expr{ .variable = Var{ .index = 0 } };

    const result = try interp.eval(&scope, &ty, &expr);
    std.debug.print("Result: {}\n", .{result});

    // Test random equality choice with TBD values
    std.debug.print("Testing random equality choice:\n", .{});
    
    // Make several random choices to see the behavior
    for (0..5) |i| {
        var test_context = Context.init(allocator, random);
        defer test_context.deinit();
        
        const test_tbd1 = try test_context.newValue();
        const test_tbd2 = try test_context.newValue();
        
        const choice = try test_context.makeRandomEqualityChoice(test_tbd1, test_tbd2);
        std.debug.print("Choice {}: {} ", .{i, choice});
        
        if (choice) {
            // Values should be unified
            const root1 = test_context.value_uf.find(test_tbd1.index);
            const root2 = test_context.value_uf.find(test_tbd2.index);
            std.debug.print("(unified: roots {} == {})\n", .{root1, root2});
        } else {
            // Inequality constraint should be added
            std.debug.print("(inequality constraint added, {} constraints total)\n", .{test_context.inequality_constraints.items.len});
        }
    }
}
