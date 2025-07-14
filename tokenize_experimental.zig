const std = @import("std");

// SIMD vector type for processing 16 bytes at a time
const SimdVector = @Vector(16, u8);

// Character classification constants
const TokenClass = enum(u8) {
    uppercase_letter = 'A',
    lowercase_letter = 'a', 
    digit = '0',
    whitespace = '_',
    other = '*',
};

// Lookup table structure for ARM NEON tbl instruction
const LookupTable = struct {
    v0: SimdVector,
    v1: SimdVector,
    v2: SimdVector,
    v3: SimdVector,
};

// SIMD tokenizer state and operations
const SimdTokenizer = struct {
    table_low: LookupTable,   // For bytes 0-63
    table_high: LookupTable,  // For bytes 64-127
    
    const Self = @This();
    
    pub fn init() Self {
        return Self{
            .table_low = buildLookupTable(0),
            .table_high = buildLookupTable(64),
        };
    }
    
    pub fn classifyChunk(self: *const Self, chunk: [16]u8) SimdVector {
        const input_vec: SimdVector = chunk;
        
        // Mask for ASCII range (0..127)
        const ascii_mask: SimdVector = @splat(0x7F);
        const indices = input_vec & ascii_mask;
        
        // Select table: 0 if index < 64, 1 if index >= 64
        const table_select_bit: SimdVector = @splat(0x40);
        const use_high_table = (indices & table_select_bit) >> @as(SimdVector, @splat(6));
        
        // Indices for table lookup (0..63 range)
        const index_mask: SimdVector = @splat(0x3F);
        const table_indices = indices & index_mask;
        
        // Lookup in both tables
        const low_result = tableLookup(self.table_low, table_indices);
        const high_result = tableLookup(self.table_high, table_indices);
        
        // Select result from the correct table
        const select_mask: SimdVector = use_high_table * @as(SimdVector, @splat(0xFF));
        var result: SimdVector = (low_result & ~select_mask) | (high_result & select_mask);
        
        // Handle non-ASCII bytes (>=128) - classify as 'other'
        const non_ascii_bit: SimdVector = @splat(0x80);
        const is_non_ascii = (input_vec & non_ascii_bit) >> @as(SimdVector, @splat(7));
        const other_class: SimdVector = @splat(@intFromEnum(TokenClass.other));
        const non_ascii_mask: SimdVector = is_non_ascii * @as(SimdVector, @splat(0xFF));
        result = (result & ~non_ascii_mask) | (other_class & non_ascii_mask);
        
        return result;
    }
};

fn buildLookupTable(start_byte: u8) LookupTable {
    var table: [64]u8 = undefined;
    for (table, 0..) |_, i| {
        const byte_value = start_byte + @as(u8, @intCast(i));
        table[i] = @intFromEnum(classifyByte(byte_value));
    }
    return LookupTable{
        .v0 = table[0..16].*,
        .v1 = table[16..32].*,
        .v2 = table[32..48].*,
        .v3 = table[48..64].*,
    };
}

fn classifyByte(byte: u8) TokenClass {
    return switch (byte) {
        'A'...'Z' => TokenClass.uppercase_letter,
        'a'...'z' => TokenClass.lowercase_letter,
        '0'...'9' => TokenClass.digit,
        ' ', '\t', '\n', '\r' => TokenClass.whitespace,
        else => TokenClass.other,
    };
}

fn tableLookup(table: LookupTable, indices: SimdVector) SimdVector {
    var result: SimdVector = undefined;
    asm volatile (
        \\ tbl  %[result].16b, {%[v0].16b, %[v1].16b, %[v2].16b, %[v3].16b}, %[indices].16b
        : [result] "=w" (result),
        : [v0] "w" (table.v0),
          [v1] "w" (table.v1),
          [v2] "w" (table.v2),
          [v3] "w" (table.v3),
          [indices] "w" (indices),
    );
    return result;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <string>\n", .{args[0]});
        return;
    }
    
    const input = args[1];
    const tokenizer = SimdTokenizer.init();
    
    processInput(&tokenizer, input);
}

fn processInput(tokenizer: *const SimdTokenizer, input: []const u8) void {
    var i: usize = 0;
    while (i < input.len) : (i += 16) {
        const chunk_len = @min(16, input.len - i);
        var chunk: [16]u8 = [_]u8{0} ** 16;
        std.mem.copyForwards(u8, chunk[0..chunk_len], input[i..][0..chunk_len]);

        const classification = tokenizer.classifyChunk(chunk);
        
        printAlignedResults(chunk[0..chunk_len], @as(*const [16]u8, &classification)[0..chunk_len]);
    }
}

fn printAlignedResults(input: []const u8, classification: []const u8) void {
    std.debug.print("{s}\n", .{input});
    std.debug.print("{s}\n", .{classification});
}
