const std = @import("std");

const InequalityConstraint = struct {
    a: ValueIdx,
    b: ValueIdx,
};

const UnionFind = struct {
    const Item = struct {
        parent: usize,
        rank: u32,
    };

    items: std.MultiArrayList(Item),

    pub fn init(allocator: std.mem.Allocator) UnionFind {
        return UnionFind{
            .items = std.MultiArrayList(Item).init(allocator),
        };
    }

    pub fn deinit(self: *UnionFind) void {
        self.items.deinit();
    }

    pub fn makeSet(self: *UnionFind) !usize {
        const id = self.items.len;
        try self.items.append(.{
            .parent = id,
            .rank = 0,
        });
        return id;
    }

    pub fn find(self: *UnionFind, x: usize) usize {
        var parent_slice = self.items.items(.parent);
        if (parent_slice[x] != x) {
            parent_slice[x] = self.find(parent_slice[x]); // Path compression
        }
        return parent_slice[x];
    }

    pub fn unite(self: *UnionFind, x: usize, y: usize) void {
        var parent_slice = self.items.items(.parent);
        var rank_slice = self.items.items(.rank);

        const root_x = self.find(x);
        const root_y = self.find(y);

        if (root_x == root_y) return;

        // Union by rank
        if (rank_slice[root_x] < rank_slice[root_y]) {
            parent_slice[root_x] = root_y;
        } else if (rank_slice[root_x] > rank_slice[root_y]) {
            parent_slice[root_y] = root_x;
        } else {
            parent_slice[root_y] = root_x;
            rank_slice[root_x] += 1;
        }
    }
};

const Context = struct {
    const ValueInfo = struct {
        resolved: Value,
        expr: ExprIdx,
        ty: Ty,
    };

    const ExprInfo = struct {
        resolved: Expr,
        ty: Ty,
    };

    values: std.ArrayList(ValueInfo),
    exprs: std.ArrayList(ExprInfo),
    types: std.ArrayList(Ty),

    value_uf: UnionFind,
    type_uf: UnionFind,

    inequality_constraints: std.ArrayList(InequalityConstraint),

    random: std.Random,

    pub fn init(allocator: std.mem.Allocator, random: std.Random) Context {
        return Context{
            .values = std.ArrayList(ValueInfo).init(allocator),
            .exprs = std.ArrayList(ExprInfo).init(allocator),
            .types = std.ArrayList(Ty).init(allocator),

            .value_uf = UnionFind.init(allocator),
            .type_uf = UnionFind.init(allocator),

            .inequality_constraints = std.ArrayList(InequalityConstraint).init(allocator),
            .expr_type_constraints = std.ArrayList(?Ty).init(allocator),
            .random = random,
        };
    }

    pub fn deinit(self: *Context) void {
        self.values.deinit();
        self.exprs.deinit();
        self.types.deinit();
        self.value_uf.deinit();
        self.type_uf.deinit();
        self.inequality_constraints.deinit();
        self.expr_type_constraints.deinit();
    }

    pub fn newValue(self: *Context) !*const Value {
        const idx = self.values.items.len;
        _ = try self.value_uf.makeSet();
        try self.values.append(Value{ .tbd = ValueIdx{ .index = idx } });
        return ValueIdx{ .index = idx };
    }

    pub fn newValueOfType(self: *Context, ty: *const Ty) !ValueIdx

    pub fn newType(self: *Context) !Ty {
        const idx = try self.types.append(Ty{ .tbd = TyIdx{ .index = self.types.items.len } });
        return self.types.items[idx];
    }

    pub fn newExpr(self: *Context) !*Expr {
        const idx = self.exprs.items.len;
        try self.exprs.append(Expr{ .tbd = ExprIdx{ .index = idx } });
        try self.expr_type_constraints.append(null); // Initially no constraint
        return &self.exprs.items[idx];
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

    fn unifyStructurally(self: *Context, a: Value, b: Value) error{ UnificationFailed, ContradictoryConstraint, OutOfMemory }!void {
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
                (constraint_a == root_b and constraint_b == root_a))
            {
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
            .tbd => |tbd_idx| {
                // Check if this TBD expression has been replaced with a generated expression
                const context_expr = &self.exprs.items[tbd_idx.index];
                if (context_expr.* != .tbd) {
                    return self.resolveExpr(context_expr.*, allocator);
                }
                // Still TBD, return as is
                return expr;
            },
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

    pub fn generateRandomExpressions(self: *Context) !void {
        // Step 2: Generate random expressions for all TBD expressions with type constraints
        for (self.exprs.items, 0..) |*expr, i| {
            if (expr.* == .tbd and self.expr_type_constraints.items[i] != null) {
                const constraint_ty = self.expr_type_constraints.items[i].?;
                const generated_expr = try self.generateRandomExpr(&constraint_ty);
                expr.* = generated_expr;
            }
        }
    }

    fn generateRandomExpr(self: *Context, ty: *const Ty) !Expr {
        const max_depth = 3;
        return self.generateRandomExprWithDepth(ty, max_depth);
    }

    fn generateRandomExprWithDepth(self: *Context, ty: *const Ty, depth: u32) error{ OutOfMemory, UnificationFailed, ContradictoryConstraint }!Expr {
        if (depth == 0) {
            return self.generateLiteralExpr(ty);
        }

        const expr_types = [_]u32{ 0, 1, 2, 3, 4 }; // literal, equality, if, list, tbd
        const choice = self.random.uintLessThan(u32, expr_types.len);

        return switch (choice) {
            0 => self.generateLiteralExpr(ty),
            1 => self.generateEqualityExpr(ty, depth),
            2 => self.generateIfExpr(ty, depth),
            3 => self.generateListExpr(ty, depth),
            4 => self.generateTbdExpr(ty),
            else => unreachable,
        };
    }

    fn generateLiteralExpr(self: *Context, ty: *const Ty) !Expr {
        return switch (ty.*) {
            .int => {
                const random_int = self.random.intRangeAtMost(i128, -100, 100);
                const int_val = try self.values.allocator.create(Value);
                int_val.* = Value{ .int = random_int };
                return Expr{ .literal = int_val };
            },
            .bool => {
                const random_bool = self.random.boolean();
                const bool_val = try self.values.allocator.create(Value);
                bool_val.* = Value{ .bool = random_bool };
                return Expr{ .literal = bool_val };
            },
            .str => {
                const strings = [_][]const u8{ "hello", "world", "foo", "bar", "test" };
                const random_str = strings[self.random.uintLessThan(usize, strings.len)];
                const str_val = try self.values.allocator.create(Value);
                str_val.* = Value{ .str = random_str };
                return Expr{ .literal = str_val };
            },
            .list => |list_ty| {
                const list_length = self.random.uintLessThan(usize, 3) + 1;
                const elements = try self.values.allocator.alloc(Expr, list_length);
                for (elements) |*element| {
                    element.* = try self.generateLiteralExpr(list_ty);
                }
                return Expr{ .list_literal = .{ .elements = elements } };
            },
            .tbd => {
                const random_types = [_]Ty{ Ty{ .int = {} }, Ty{ .bool = {} }, Ty{ .str = {} } };
                const random_ty = &random_types[self.random.uintLessThan(usize, random_types.len)];
                return self.generateLiteralExpr(random_ty);
            },
            else => {
                return self.generateLiteralExpr(&Ty{ .int = {} });
            },
        };
    }

    fn generateEqualityExpr(self: *Context, ty: *const Ty, depth: u32) !Expr {
        if (ty.* != .bool) {
            return self.generateLiteralExpr(ty);
        }

        const comparison_types = [_]Ty{ Ty{ .int = {} }, Ty{ .bool = {} }, Ty{ .str = {} } };
        const comp_ty = &comparison_types[self.random.uintLessThan(usize, comparison_types.len)];

        const left_expr = try self.values.allocator.create(Expr);
        const right_expr = try self.values.allocator.create(Expr);

        left_expr.* = try self.generateRandomExprWithDepth(comp_ty, depth - 1);
        right_expr.* = try self.generateRandomExprWithDepth(comp_ty, depth - 1);

        return Expr{ .equality = .{ .left = left_expr, .right = right_expr } };
    }

    fn generateIfExpr(self: *Context, ty: *const Ty, depth: u32) !Expr {
        const condition_expr = try self.values.allocator.create(Expr);
        condition_expr.* = try self.generateRandomExprWithDepth(&Ty{ .bool = {} }, depth - 1);

        const then_block = try self.values.allocator.create(Block);
        const else_block = try self.values.allocator.create(Block);

        then_block.* = Block{
            .statements = &.{},
            .return_expr = try self.generateRandomExprWithDepth(ty, depth - 1),
        };

        else_block.* = Block{
            .statements = &.{},
            .return_expr = try self.generateRandomExprWithDepth(ty, depth - 1),
        };

        return Expr{ .if_expression = .{
            .condition = condition_expr,
            .then_branch = then_block,
            .else_branch = else_block,
        } };
    }

    fn generateListExpr(self: *Context, ty: *const Ty, depth: u32) !Expr {
        const list_ty = switch (ty.*) {
            .list => |inner_ty| inner_ty,
            else => &Ty{ .int = {} },
        };

        const list_length = self.random.uintLessThan(usize, 3) + 1;
        const elements = try self.values.allocator.alloc(Expr, list_length);

        for (elements) |*element| {
            element.* = try self.generateRandomExprWithDepth(list_ty, depth - 1);
        }

        return Expr{ .list_literal = .{ .elements = elements } };
    }

    fn generateTbdExpr(self: *Context, ty: *const Ty) !Expr {
        const tbd_expr_idx = ExprIdx{ .index = self.exprs.items.len };
        try self.exprs.append(Expr{ .tbd = tbd_expr_idx });
        try self.expr_type_constraints.append(ty.*);
        return Expr{ .tbd = tbd_expr_idx };
    }

    pub fn printConstraints(self: *Context) void {
        std.debug.print("=== Recorded Type Constraints ===\n", .{});
        for (self.expr_type_constraints.items, 0..) |constraint, i| {
            if (constraint != null) {
                std.debug.print("TBD expr {} must have type: {}\n", .{ i, constraint.? });
            }
        }
        std.debug.print("\n", .{});
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

    pub fn format(self: Ty, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;

        switch (self) {
            .tbd => |tbd_idx| try writer.print("<tbd:{}>", .{tbd_idx.index}),
            .int => try writer.print("int", .{}),
            .float => try writer.print("float", .{}),
            .bool => try writer.print("bool", .{}),
            .str => try writer.print("str", .{}),
            .list => |inner_ty| try writer.print("List({})", .{inner_ty.*}),
            .record => try writer.print("<record>", .{}),
            .tag => try writer.print("<tag>", .{}),
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
            .function => |f| {
                try writer.print("|", .{});
                for (f.parameter_types, 0..) |param, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{param});
                }
                try writer.print("| {}", .{f.body});
            },
            .application => try writer.print("<application>", .{}),
            .list_literal => |lst| {
                try writer.print("[", .{});
                for (lst.elements, 0..) |element, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{element});
                }
                try writer.print("]", .{});
            },
            .equality => |eq| {
                try writer.print("({} == {})", .{ eq.left.*, eq.right.* });
            },
            .if_expression => |if_expr| {
                try writer.print("if ({}) {} else {}", .{ if_expr.condition.*, if_expr.then_branch, if_expr.else_branch });
            },
        }
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
            .tbd => |tbd_expr_idx| {
                // Step 1: Record the type constraint for this TBD expression
                self.context.expr_type_constraints.items[tbd_expr_idx.index] = ty.*;

                // Return a TBD value that indicates this expression needs to be resolved
                const tbd_value = try self.context.createTbdValue();
                return tbd_value.*;
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

    fn evalBlock(self: *Interp, scope: *const CapturedScope, ty: *const Ty, block: *const Block) error{ OutOfGas, UnificationFailed, ContradictoryConstraint, TypeMismatch, VariableNotFound, TbdExpressionNotImplemented, ApplicationNotImplemented, IfExpressionNotImplemented, OutOfMemory }!Value {
        // For simplicity, just evaluate the return expression
        // In a full implementation, you'd need to handle statements
        return self.eval(scope, ty, &block.return_expr);
    }
};

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .{};
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    var context = Context.init(allocator, random);
    defer context.deinit();

    std.debug.print("=== Testing Enhanced Random Expression Generation ===\n", .{});

    // Test different types with enhanced generation
    const test_types = [_]Ty{
        Ty{ .int = {} },
        Ty{ .bool = {} },
        Ty{ .str = {} },
        Ty{ .list = &Ty{ .int = {} } },
    };

    for (test_types) |ty| {
        std.debug.print("\n--- Generating expressions for type: {} ---\n", .{ty});

        var j: u32 = 0;
        while (j < 3) : (j += 1) {
            const expr = try context.generateRandomExpr(&ty);
            std.debug.print("  Example {}: {}\n", .{ j + 1, expr });
        }
    }

    // Test the iterative resolution process
    std.debug.print("\n=== Testing Iterative TBD Resolution ===\n", .{});

    var interp = Interp{
        .context = &context,
        .gas = 1000,
    };

    // Create a complex expression with nested TBDs that will create more TBDs
    const complex_expr = try context.generateRandomExpr(&Ty{ .bool = {} });
    std.debug.print("Generated complex expression: {}\n", .{complex_expr});

    // Check if any new TBDs were created and resolve them iteratively
    var iterations: u32 = 0;
    const max_iterations = 5;

    while (iterations < max_iterations) : (iterations += 1) {
        std.debug.print("\n--- Iteration {} ---\n", .{iterations + 1});

        // Check for unresolved TBDs
        var has_unresolved = false;
        for (context.exprs.items, 0..) |expr, idx| {
            if (expr == .tbd and context.expr_type_constraints.items[idx] != null) {
                has_unresolved = true;
                break;
            }
        }

        if (!has_unresolved) {
            std.debug.print("No more unresolved TBDs found!\n", .{});
            break;
        }

        std.debug.print("Resolving TBDs...\n", .{});
        context.printConstraints();
        try context.generateRandomExpressions();
    }

    // Show final state
    std.debug.print("\n=== Final Results ===\n", .{});
    std.debug.print("Total expressions generated: {}\n", .{context.exprs.items.len});
    std.debug.print("Final expressions in context:\n", .{});
    for (context.exprs.items, 0..) |expr, idx| {
        std.debug.print("  [{}]: {}\n", .{ idx, expr });
    }

    // Test evaluation of a simple generated expression
    std.debug.print("\n=== Testing Evaluation ===\n", .{});
    const simple_expr = try context.generateRandomExpr(&Ty{ .int = {} });
    std.debug.print("Generated for evaluation: {}\n", .{simple_expr});

    const scope = CapturedScope{ .values = &.{} };
    const result = try interp.eval(&scope, &Ty{ .int = {} }, &simple_expr);
    std.debug.print("Evaluation result: {}\n", .{result});
}
