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

    pub fn generateIdentifierMask(classification_block: *const Block) u64 {
        // Create iota vector for bit positions within each byte
        const iota: NeonChunk = .{ 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 };

        var chunk_masks: [4]u16 = undefined;

        for (classification_block.chunks, 0..) |chunk, chunk_idx| {
            // Create identifier mask: 1 for letters/digits, 0 for others
            const upper_mask = chunk == @as(NeonChunk, @splat(@intFromEnum(TokenClass.uppercase_letter)));
            const lower_mask = chunk == @as(NeonChunk, @splat(@intFromEnum(TokenClass.lowercase_letter)));
            const digit_mask = chunk == @as(NeonChunk, @splat(@intFromEnum(TokenClass.digit)));

            // Combine masks using select operations
            const temp_mask = @select(bool, upper_mask, @as(@Vector(16, bool), @splat(true)), lower_mask);
            const letter_mask = @select(bool, temp_mask, @as(@Vector(16, bool), @splat(true)), digit_mask);

            // Convert boolean mask to 0/1 values
            const ident_vec: NeonChunk = @select(u8, letter_mask, @as(NeonChunk, @splat(1)), @as(NeonChunk, @splat(0)));

            // Shift left by iota positions within each byte
            const shifted = ident_vec << @as(@Vector(16, u3), @truncate(iota));

            // Shuffle to interleave: lanes 0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15
            const shuffle_indices: @Vector(16, i32) = .{ 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };
            const shuffled = @shuffle(u8, shifted, undefined, shuffle_indices);

            chunk_masks[chunk_idx] = vectorToU16Mask(shuffled);
        }

        // Merge four 16-bit results into 64-bit mask
        const mask: u64 = (@as(u64, chunk_masks[0])) |
            (@as(u64, chunk_masks[1]) << 16) |
            (@as(u64, chunk_masks[2]) << 32) |
            (@as(u64, chunk_masks[3]) << 48);

        return mask;
    }

    pub fn processBlocks(self: *const Self, blocks: []align(BLOCK_SIZE) const Block) void {
        for (blocks, 0..) |*block, block_idx| {
            const classification = self.classifyBlock(block);
            const identifier_mask = Tokenizer.generateIdentifierMask(&classification);
            printBlockResults(block, &classification, identifier_mask, block_idx);
        }
    }
};

fn printBlockResults(input_block: *const Block, classification_block: *const Block, identifier_mask: u64, block_idx: usize) void {
    std.debug.print("Block {} (64 bytes):\n", .{block_idx});

    // Convert blocks to byte arrays for printing
    const input_bytes: *const [64]u8 = @ptrCast(input_block);
    const class_bytes: *const [64]u8 = @ptrCast(classification_block);

    std.debug.print("Input: {s}\n", .{input_bytes});
    std.debug.print("Class: {s}\n", .{class_bytes});

    // Print identifier mask as little-endian bits aligned with input
    std.debug.print("Ident: ", .{});
    for (0..64) |i| {
        const bit = (identifier_mask >> @intCast(i)) & 1;
        std.debug.print("{}", .{bit});
    }
    std.debug.print("\n\n", .{});
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

fn vectorToU16Mask(vec: NeonChunk) u16 {
    // Reinterpret as u16 vector and use addv for horizontal sum
    const v: @Vector(8, u16) = @bitCast(vec);
    var result: u32 = undefined;
    asm volatile (
        \\ addv h0, %[v].8h
        \\ umov w0, v0.h[0]
        \\ mov %[result], x0
        : [result] "=r" (result),
        : [v] "w" (v),
        : "v0", "w0", "x0"
    );
    return @intCast(result);
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

fn enableRawMode() void {
    var termios = std.posix.tcgetattr(std.io.getStdIn().handle) catch return;
    const flags: u64 = 0x00000008 | 0x00000100; // ECHO | ICANON
    var lflag_value = @as(u64, @bitCast(termios.lflag));
    lflag_value &= ~flags;
    termios.lflag = @bitCast(lflag_value);
    _ = std.posix.tcsetattr(std.io.getStdIn().handle, std.posix.TCSA.NOW, termios) catch return;
}

fn disableRawMode() void {
    var termios = std.posix.tcgetattr(std.io.getStdIn().handle) catch return;
    const flags: u64 = 0x00000008 | 0x00000100; // ECHO | ICANON
    var lflag_value = @as(u64, @bitCast(termios.lflag));
    lflag_value |= flags;
    termios.lflag = @bitCast(lflag_value);
    _ = std.posix.tcsetattr(std.io.getStdIn().handle, std.posix.TCSA.NOW, termios) catch return;
}

fn processAndDisplay(tokenizer: *const Tokenizer, input_buffer: *const [64]u8) void {
    const block: *const Block = @ptrCast(@alignCast(input_buffer));
    const classification = tokenizer.classifyBlock(block);
    const identifier_mask = Tokenizer.generateIdentifierMask(&classification);

    // Clear screen and move cursor to top
    std.debug.print("\x1b[2J\x1b[H", .{});
    std.debug.print("Interactive Mode - Type characters (Ctrl+C to exit)\n\n", .{});
    printBlockResults(block, &classification, identifier_mask, 0);
}

fn runInteractiveMode(_: std.mem.Allocator) !void {
    const tokenizer = Tokenizer.init();
    var input_buffer: [64]u8 align(64) = [_]u8{0} ** 64;
    var cursor_pos: usize = 0;

    enableRawMode();
    defer disableRawMode();

    // Initial display
    processAndDisplay(&tokenizer, &input_buffer);

    while (true) {
        var char: [1]u8 = undefined;
        const bytes_read = try std.io.getStdIn().reader().read(&char);
        if (bytes_read == 0) break;

        const c = char[0];

        // Handle Ctrl+C
        if (c == 3) break;

        // Handle backspace
        if (c == 127 or c == 8) {
            if (cursor_pos > 0) {
                cursor_pos -= 1;
                input_buffer[cursor_pos] = 0;
                processAndDisplay(&tokenizer, &input_buffer);
            }
            continue;
        }

        // Handle enter (do nothing)
        if (c == '\n' or c == '\r') {
            continue;
        }

        // Handle printable characters
        if (c >= 32 and c <= 126 and cursor_pos < 64) {
            input_buffer[cursor_pos] = c;
            cursor_pos += 1;
            processAndDisplay(&tokenizer, &input_buffer);
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: {s} <file_path> OR {s} --interactive\n", .{ args[0], args[0] });
        return;
    }

    if (std.mem.eql(u8, args[1], "--interactive")) {
        try runInteractiveMode(allocator);
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
