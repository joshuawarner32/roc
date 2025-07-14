const std = @import("std");

// SIMD vector type for processing 16 bytes at a time
const NeonChunk = @Vector(16, u8);
const Block = struct {
    chunks: [4]NeonChunk,
};

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
    v0: NeonChunk,
    v1: NeonChunk,
    v2: NeonChunk,
    v3: NeonChunk,
};

// SIMD tokenizer for processing 64-byte aligned blocks
const Tokenizer = struct {
    table_low: LookupTable, // For bytes 0-63
    table_high: LookupTable, // For bytes 64-127

    const Self = @This();
    const BLOCK_SIZE = 64;
    const CHUNK_SIZE = 16;

    pub fn init() Self {
        return Self{
            .table_low = buildLookupTable(0),
            .table_high = buildLookupTable(64),
        };
    }

    fn classifyChunk(self: *const Self, chunk: [CHUNK_SIZE]u8) NeonChunk {
        const input_vec: NeonChunk = chunk;

        // Mask for ASCII range (0..127)
        const ascii_mask: NeonChunk = @splat(0x7F);
        const indices = input_vec & ascii_mask;

        // Select table: 0 if index < 64, 1 if index >= 64
        const table_select_bit: NeonChunk = @splat(0x40);
        const use_high_table = (indices & table_select_bit) >> @as(NeonChunk, @splat(6));

        // Indices for table lookup (0..63 range)
        const index_mask: NeonChunk = @splat(0x3F);
        const table_indices = indices & index_mask;

        // Lookup in both tables
        const low_result = tableLookup(self.table_low, table_indices);
        const high_result = tableLookup(self.table_high, table_indices);

        // Select result from the correct table
        const select_mask: NeonChunk = use_high_table * @as(NeonChunk, @splat(0xFF));
        var result: NeonChunk = (low_result & ~select_mask) | (high_result & select_mask);

        // Handle non-ASCII bytes (>=128) - classify as 'other'
        const non_ascii_bit: NeonChunk = @splat(0x80);
        const is_non_ascii = (input_vec & non_ascii_bit) >> @as(NeonChunk, @splat(7));
        const other_class: NeonChunk = @splat(@intFromEnum(TokenClass.other));
        const non_ascii_mask: NeonChunk = is_non_ascii * @as(NeonChunk, @splat(0xFF));
        result = (result & ~non_ascii_mask) | (other_class & non_ascii_mask);

        return result;
    }

    pub fn classifyBlock(self: *const Self, input_block: *const Block) Block {
        var result_block: Block = undefined;

        // Process each chunk in the block
        for (input_block.chunks, 0..) |chunk, i| {
            const chunk_bytes: [CHUNK_SIZE]u8 = @bitCast(chunk);
            result_block.chunks[i] = self.classifyChunk(chunk_bytes);
        }

        return result_block;
    }

    pub fn processBlocks(self: *const Self, blocks: []align(BLOCK_SIZE) const Block) void {
        for (blocks, 0..) |*block, block_idx| {
            const classification = self.classifyBlock(block);
            printBlockResults(block, &classification, block_idx);
        }
    }
};

fn printBlockResults(input_block: *const Block, classification_block: *const Block, block_idx: usize) void {
    std.debug.print("Block {} (64 bytes):\n", .{block_idx});
    
    // Convert blocks to byte arrays for printing
    const input_bytes: *const [64]u8 = @ptrCast(input_block);
    const class_bytes: *const [64]u8 = @ptrCast(classification_block);
    
    std.debug.print("Input: {s}\n", .{input_bytes});
    std.debug.print("Class: {s}\n", .{class_bytes});
    std.debug.print("\n", .{});
}

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

fn tableLookup(table: LookupTable, indices: NeonChunk) NeonChunk {
    var result: NeonChunk = undefined;
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

fn loadFileAsBlocks(allocator: std.mem.Allocator, file_path: []const u8) ![]align(64) Block {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();

    // Round up to nearest 64-byte boundary
    const aligned_size = (file_size + 63) & ~@as(usize, 63);
    const num_blocks = aligned_size / 64;

    // Allocate 64-byte aligned memory as blocks
    const blocks = try allocator.alignedAlloc(Block, 64, num_blocks);

    // Read file data into byte representation
    const bytes_ptr: [*]u8 = @ptrCast(blocks.ptr);
    const bytes_read = try file.readAll(bytes_ptr[0..file_size]);
    if (bytes_read != file_size) {
        return error.IncompleteRead;
    }

    // Zero-pad the remaining bytes
    if (aligned_size > file_size) {
        @memset(bytes_ptr[file_size..aligned_size], 0);
    }

    return blocks;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <file_path>\n", .{args[0]});
        return;
    }

    const file_path = args[1];
    const blocks = loadFileAsBlocks(allocator, file_path) catch |err| {
        std.debug.print("Error loading file '{s}': {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(blocks);

    const tokenizer = Tokenizer.init();
    tokenizer.processBlocks(blocks);
}
