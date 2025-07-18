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
    hash = '#',
    newline = '\n',
    quote = '"',
    lonely_symbol = '@',
    combining_symbol = '~',
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

    pub const TokenMasks = struct {
        lowercase: u64,
        uppercase: u64,
        digit: u64,
        lonely_symbol: u64,
        combining_symbol: u64,
    };

    pub fn generateTokenMasks(classification_block: *const Block) TokenMasks {
        var lower_masks: [4]u16 = undefined;
        var upper_masks: [4]u16 = undefined;
        var digit_masks: [4]u16 = undefined;
        var lonely_symbol_masks: [4]u16 = undefined;
        var combining_symbol_masks: [4]u16 = undefined;

        var quote: bool = false;
        var comment: bool = false;

        for (classification_block.chunks, 0..) |chunk, chunk_idx| {
            // Create separate masks for each token type
            var upper_mask = eqMask(chunk, .uppercase_letter);
            var lower_mask = eqMask(chunk, .lowercase_letter);
            var digit_mask = eqMask(chunk, .digit);
            var lonely_symbol_mask = eqMask(chunk, .lonely_symbol);
            var combining_symbol_mask = eqMask(chunk, .combining_symbol);

            const hash_mask = chunk == @as(NeonChunk, @splat(@intFromEnum(TokenClass.hash)));
            const quote_mask = chunk == @as(NeonChunk, @splat(@intFromEnum(TokenClass.quote)));

            var non_comment_mask: NeonChunk = @splat(0xff);
            var string_mask: NeonChunk = @splat(0);

            if (quote or comment or anyTrue(hash_mask) or anyTrue(quote_mask)) {
                // serial loop to handle comments and quotes
                for (0..CHUNK_SIZE) |i| {
                    const byte = chunk[i];
                    if (byte == @intFromEnum(TokenClass.hash) and !quote) {
                        comment = true; // Start comment
                    } else if (byte == @intFromEnum(TokenClass.newline) and comment) {
                        comment = false; // End comment
                    } else if (byte == @intFromEnum(TokenClass.quote) and !comment) {
                        quote = !quote; // Toggle quote state
                    }
                    if (quote) {
                        non_comment_mask[i] = 1; // Ignore this byte in masks
                        string_mask[i] = 1;
                    }
                    if (comment) {
                        non_comment_mask[i] = 0; // Ignore this byte in masks
                    }
                }
            }

            upper_mask = upper_mask & non_comment_mask;
            lower_mask = lower_mask & non_comment_mask;
            digit_mask = digit_mask & non_comment_mask;
            lonely_symbol_mask = lonely_symbol_mask & non_comment_mask;
            combining_symbol_mask = combining_symbol_mask & non_comment_mask;

            upper_masks[chunk_idx] = vectorToU16Mask(upper_mask);
            lower_masks[chunk_idx] = vectorToU16Mask(lower_mask);
            digit_masks[chunk_idx] = vectorToU16Mask(digit_mask);
            lonely_symbol_masks[chunk_idx] = vectorToU16Mask(lonely_symbol_mask);
            combining_symbol_masks[chunk_idx] = vectorToU16Mask(combining_symbol_mask);
        }

        // Merge four 16-bit results into 64-bit masks for each type
        const uppercase_mask = combine(upper_masks);
        const lowercase_mask = combine(lower_masks);
        const digit_mask = combine(digit_masks);
        const lonely_symbol_mask = combine(lonely_symbol_masks);
        const combining_symbol_mask = combine(combining_symbol_masks);

        return TokenMasks{
            .lowercase = lowercase_mask,
            .uppercase = uppercase_mask,
            .digit = digit_mask,
            .lonely_symbol = lonely_symbol_mask,
            .combining_symbol = combining_symbol_mask,
        };
    }

    pub const DebugMasks = struct {
        identifier_mask: u64,
        int_mask: u64,
        combining_symbol_mask: u64,
        token_start_mask: u64,
    };

    const StartMask = struct {
        start: u64,
        mask: u64,
    };

    fn flood(starts: u64, conts: u64) StartMask {
        const other = ~starts & ~conts;
        const shifted_other = (other << 1) | 1;
        const real_starts = shifted_other & starts;

        // Use bit-carrying behavior of add to flood fill identifiers
        // Adding start to cont causes cascading carries
        // that propagate through consecutive identifier characters
        const mask = ~(real_starts +% conts) & conts;
        const start = mask & real_starts;
        return .{ .start = start, .mask = mask };
    }

    pub fn generateIdentifierMask(token_masks: TokenMasks) DebugMasks {

        // Create identifier begin and continue masks
        const ident_begin = token_masks.uppercase | token_masks.lowercase;
        const ident_continue = ident_begin | token_masks.digit;

        const identifiers = flood(ident_begin, ident_continue);
        const digits = token_masks.digit & ~identifiers.mask;
        const ints = flood(digits, digits);
        const combining_symbols = flood(token_masks.combining_symbol, token_masks.combining_symbol);

        const token_start_mask = identifiers.start | token_masks.lonely_symbol | ints.start | combining_symbols.start;

        return DebugMasks{
            .identifier_mask = identifiers.mask,
            .int_mask = ints.mask,
            .combining_symbol_mask = combining_symbols.mask,
            .token_start_mask = token_start_mask,
        };
    }

    pub fn processBlocks(self: *const Self, blocks: []align(BLOCK_SIZE) const Block) void {
        for (blocks, 0..) |*block, block_idx| {
            const classification = self.classifyBlock(block);
            const token_masks = Tokenizer.generateTokenMasks(&classification);
            const debug_masks = Tokenizer.generateIdentifierMask(token_masks);
            printBlockResults(block, &classification, token_masks, debug_masks, block_idx);
        }
    }

    pub fn processBlocksBenchmark(self: *const Self, blocks: []align(BLOCK_SIZE) const Block) u64 {
        var accumulated_token_starts: u64 = 0;
        for (blocks) |*block| {
            const classification = self.classifyBlock(block);
            const token_masks = Tokenizer.generateTokenMasks(&classification);
            const debug_masks = Tokenizer.generateIdentifierMask(token_masks);
            accumulated_token_starts ^= debug_masks.token_start_mask;
        }
        return accumulated_token_starts;
    }

    pub const BenchmarkTiming = struct {
        classification_time: u64,
        token_mask_time: u64,
        debug_mask_time: u64,
        total_time: u64,
        accumulated_result: u64,
    };

    pub fn processBlocksBenchmarkTimed(self: *const Self, blocks: []align(BLOCK_SIZE) const Block) BenchmarkTiming {
        var accumulated_token_starts: u64 = 0;
        var total_classification_time: u64 = 0;
        var total_token_mask_time: u64 = 0;
        var total_debug_mask_time: u64 = 0;
        
        const start_total = std.time.nanoTimestamp();
        
        for (blocks) |*block| {
            const start_classify = std.time.nanoTimestamp();
            const classification = self.classifyBlock(block);
            const end_classify = std.time.nanoTimestamp();
            
            const start_token_masks = std.time.nanoTimestamp();
            const token_masks = Tokenizer.generateTokenMasks(&classification);
            const end_token_masks = std.time.nanoTimestamp();
            
            const start_debug_masks = std.time.nanoTimestamp();
            const debug_masks = Tokenizer.generateIdentifierMask(token_masks);
            const end_debug_masks = std.time.nanoTimestamp();
            
            total_classification_time += @intCast(end_classify - start_classify);
            total_token_mask_time += @intCast(end_token_masks - start_token_masks);
            total_debug_mask_time += @intCast(end_debug_masks - start_debug_masks);
            
            accumulated_token_starts ^= debug_masks.token_start_mask;
        }
        
        const end_total = std.time.nanoTimestamp();
        
        return BenchmarkTiming{
            .classification_time = total_classification_time,
            .token_mask_time = total_token_mask_time,
            .debug_mask_time = total_debug_mask_time,
            .total_time = @intCast(end_total - start_total),
            .accumulated_result = accumulated_token_starts,
        };
    }
};

fn combine(masks: [4]u16) u64 {
    return (@as(u64, masks[0])) |
        (@as(u64, masks[1]) << 16) |
        (@as(u64, masks[2]) << 32) |
        (@as(u64, masks[3]) << 48);
}

fn eqMask(chunk: NeonChunk, cls: TokenClass) NeonChunk {
    const eq = chunk == @as(NeonChunk, @splat(@intFromEnum(cls)));
    return @select(u8, eq, @as(NeonChunk, @splat(1)), @as(NeonChunk, @splat(0)));
}

fn printMaskName(name: []const u8) void {
    const truncated_name = if (name.len > 15) name[0..15] else name;
    var extra_spaces_buf: [15]u8 = undefined;
    var extra_spaces: []const u8 = "";
    if (truncated_name.len < 15) {
        const n = 15 - truncated_name.len;
        @memset(extra_spaces_buf[0..n], ' ');
        extra_spaces = extra_spaces_buf[0..n];
    }
    std.debug.print("{s}{s}: ", .{ truncated_name, extra_spaces });
}

fn printMask(name: []const u8, mask: u64) void {
    printMaskName(name);
    for (0..64) |i| {
        const bit = (mask >> @intCast(i)) & 1;
        std.debug.print("{}", .{bit});
    }
    std.debug.print("\n", .{});
}

fn printAllMasks(masks: anytype) void {
    const T = @TypeOf(masks);
    const fields = std.meta.fields(T);
    inline for (fields) |field| {
        printMask(field.name, @field(masks, field.name));
    }
}

fn printInputLine(name: []const u8, input_bytes: *const [64]u8) void {
    printMaskName(name);
    for (input_bytes) |byte| {
        if (byte == '\n') {
            std.debug.print("\\", .{});
        } else if (byte == 0) {
            break; // Stop at null terminator
        } else {
            std.debug.print("{c}", .{byte});
        }
    }
    std.debug.print("\n", .{});
}

fn printBlockResults(input_block: *const Block, classification_block: *const Block, token_masks: Tokenizer.TokenMasks, debug_masks: Tokenizer.DebugMasks, block_idx: usize) void {
    std.debug.print("Block {} (64 bytes):\n", .{block_idx});

    // Convert blocks to byte arrays for printing
    const input_bytes: *const [64]u8 = @ptrCast(input_block);
    const class_bytes: *const [64]u8 = @ptrCast(classification_block);

    // Print input with newlines converted to backslashes
    printInputLine("Input", input_bytes);
    printInputLine("Class", class_bytes);

    // Print token masks
    printAllMasks(token_masks);

    // Print debug masks
    printAllMasks(debug_masks);

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
        ' ', '\t', '\r' => TokenClass.whitespace,
        '#' => TokenClass.hash,
        '\n' => TokenClass.newline,
        '"' => TokenClass.quote,
        '(', ')', '[', ']', '{', '}', ',', '.' => TokenClass.lonely_symbol,
        '!', '@', '$', '%', '^', '&', '*', '-', '=', '+', '\\', '|', ';', ':', '?', '<', '>', '/' => TokenClass.combining_symbol,
        else => TokenClass.other,
    };
}

fn anyTrue(vec: @Vector(16, bool)) bool {
    var result: u32 = undefined;
    asm volatile (
        \\ addv b0, %[vec].16b
        \\ umov w0, v0.b[0]
        \\ mov %[result], x0
        : [result] "=r" (result),
        : [vec] "w" (vec),
        : "v0", "w0", "x0"
    );
    return result > 0;
}

fn tableLookup(table: LookupTable, indices: NeonChunk) NeonChunk {
    var result: NeonChunk = undefined;
    asm volatile (
        \\ mov v0.16b, %[v0].16b
        \\ mov v1.16b, %[v1].16b
        \\ mov v2.16b, %[v2].16b
        \\ mov v3.16b, %[v3].16b
        \\ tbl  %[result].16b, {v0.16b, v1.16b, v2.16b, v3.16b}, %[indices].16b
        : [result] "=w" (result),
        : [v0] "w" (table.v0),
          [v1] "w" (table.v1),
          [v2] "w" (table.v2),
          [v3] "w" (table.v3),
          [indices] "w" (indices),
        : "v0", "v1", "v2", "v3"
    );
    return result;
}

fn vectorToU16Mask(mask: NeonChunk) u16 {
    // Shift left by iota positions within each byte
    const iota: NeonChunk = .{ 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 };
    const shifted = mask << @as(@Vector(16, u3), @truncate(iota));

    // Shuffle to interleave: lanes 0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15
    const shuffle_indices: @Vector(16, i32) = .{ 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };
    const shuffled = @shuffle(u8, shifted, undefined, shuffle_indices);

    // Reinterpret as u16 vector and use addv for horizontal sum
    const v: @Vector(8, u16) = @bitCast(shuffled);
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
    const token_masks = Tokenizer.generateTokenMasks(&classification);
    const debug_masks = Tokenizer.generateIdentifierMask(token_masks);

    // Clear screen and move cursor to top
    std.debug.print("\x1b[2J\x1b[H", .{});
    std.debug.print("Interactive Mode - Type characters (Ctrl+C to exit)\n\n", .{});
    printBlockResults(block, &classification, token_masks, debug_masks, 0);
}

fn runBenchmarkMode(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const blocks = loadFileAsBlocks(allocator, file_path) catch |err| {
        std.debug.print("Error loading file '{s}': {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(blocks);

    const tokenizer = Tokenizer.init();
    
    // Warmup run
    _ = tokenizer.processBlocksBenchmark(blocks);
    
    // Benchmark runs
    const num_runs = 100;
    var total_time: u64 = 0;
    
    for (0..num_runs) |_| {
        const start_time = std.time.nanoTimestamp();
        const result = tokenizer.processBlocksBenchmark(blocks);
        const end_time = std.time.nanoTimestamp();
        total_time += @intCast(end_time - start_time);
        
        // Use result to prevent optimization
        if (result == 0xDEADBEEF) {
            std.debug.print("Unlikely result\n", .{});
        }
    }
    
    const avg_time = total_time / num_runs;
    const throughput = (@as(f64, @floatFromInt(blocks.len * 64)) / @as(f64, @floatFromInt(avg_time))) * 1_000_000_000;
    
    std.debug.print("Benchmark Results:\n", .{});
    std.debug.print("  File: {s}\n", .{file_path});
    std.debug.print("  Blocks processed: {}\n", .{blocks.len});
    std.debug.print("  Total bytes: {}\n", .{blocks.len * 64});
    std.debug.print("  Average time per run: {d:.2} ns\n", .{@as(f64, @floatFromInt(avg_time))});
    std.debug.print("  Throughput: {d:.2} bytes/second\n", .{throughput});
    std.debug.print("  Throughput: {d:.2} MB/s\n", .{throughput / (1024 * 1024)});
    
    // Final result to prevent optimization
    const final_result = tokenizer.processBlocksBenchmark(blocks);
    std.debug.print("  Final token starts mask: 0x{x}\n", .{final_result});
}

fn runDetailedBenchmarkMode(allocator: std.mem.Allocator, file_path: []const u8) !void {
    const blocks = loadFileAsBlocks(allocator, file_path) catch |err| {
        std.debug.print("Error loading file '{s}': {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(blocks);

    const tokenizer = Tokenizer.init();
    
    // Warmup run
    _ = tokenizer.processBlocksBenchmarkTimed(blocks);
    
    // Benchmark runs
    const num_runs = 100;
    var total_classification_time: u64 = 0;
    var total_token_mask_time: u64 = 0;
    var total_debug_mask_time: u64 = 0;
    var total_overall_time: u64 = 0;
    
    for (0..num_runs) |_| {
        const timing = tokenizer.processBlocksBenchmarkTimed(blocks);
        total_classification_time += timing.classification_time;
        total_token_mask_time += timing.token_mask_time;
        total_debug_mask_time += timing.debug_mask_time;
        total_overall_time += timing.total_time;
        
        // Use result to prevent optimization
        if (timing.accumulated_result == 0xDEADBEEF) {
            std.debug.print("Unlikely result\n", .{});
        }
    }
    
    const avg_classification_time = total_classification_time / num_runs;
    const avg_token_mask_time = total_token_mask_time / num_runs;
    const avg_debug_mask_time = total_debug_mask_time / num_runs;
    const avg_overall_time = total_overall_time / num_runs;
    
    const throughput = (@as(f64, @floatFromInt(blocks.len * 64)) / @as(f64, @floatFromInt(avg_overall_time))) * 1_000_000_000;
    
    std.debug.print("Detailed Benchmark Results:\n", .{});
    std.debug.print("  File: {s}\n", .{file_path});
    std.debug.print("  Blocks processed: {}\n", .{blocks.len});
    std.debug.print("  Total bytes: {}\n", .{blocks.len * 64});
    std.debug.print("  Average times per run:\n", .{});
    std.debug.print("    Classification: {d:.2} ns ({d:.1}%)\n", .{
        @as(f64, @floatFromInt(avg_classification_time)),
        (@as(f64, @floatFromInt(avg_classification_time)) / @as(f64, @floatFromInt(avg_overall_time))) * 100
    });
    std.debug.print("    Token masks:    {d:.2} ns ({d:.1}%)\n", .{
        @as(f64, @floatFromInt(avg_token_mask_time)),
        (@as(f64, @floatFromInt(avg_token_mask_time)) / @as(f64, @floatFromInt(avg_overall_time))) * 100
    });
    std.debug.print("    Debug masks:    {d:.2} ns ({d:.1}%)\n", .{
        @as(f64, @floatFromInt(avg_debug_mask_time)),
        (@as(f64, @floatFromInt(avg_debug_mask_time)) / @as(f64, @floatFromInt(avg_overall_time))) * 100
    });
    std.debug.print("    Total:          {d:.2} ns\n", .{@as(f64, @floatFromInt(avg_overall_time))});
    std.debug.print("  Throughput: {d:.2} bytes/second\n", .{throughput});
    std.debug.print("  Throughput: {d:.2} MB/s\n", .{throughput / (1024 * 1024)});
    
    // Final result to prevent optimization
    const final_result = tokenizer.processBlocksBenchmarkTimed(blocks);
    std.debug.print("  Final token starts mask: 0x{x}\n", .{final_result.accumulated_result});
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

        // Handle enter - insert actual newline character
        if (c == '\n' or c == '\r') {
            if (cursor_pos < 64) {
                input_buffer[cursor_pos] = '\n';
                cursor_pos += 1;
                processAndDisplay(&tokenizer, &input_buffer);
            }
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
        std.debug.print("Usage: {s} <file_path> [--debug] OR {s} --interactive OR {s} --benchmark <file_path> OR {s} --detailed-benchmark <file_path>\n", .{ args[0], args[0], args[0], args[0] });
        return;
    }

    if (std.mem.eql(u8, args[1], "--interactive")) {
        try runInteractiveMode(allocator);
        return;
    }

    if (std.mem.eql(u8, args[1], "--benchmark")) {
        if (args.len < 3) {
            std.debug.print("Error: --benchmark requires a file path\n", .{});
            return;
        }
        try runBenchmarkMode(allocator, args[2]);
        return;
    }

    if (std.mem.eql(u8, args[1], "--detailed-benchmark")) {
        if (args.len < 3) {
            std.debug.print("Error: --detailed-benchmark requires a file path\n", .{});
            return;
        }
        try runDetailedBenchmarkMode(allocator, args[2]);
        return;
    }

    const file_path = args[1];
    const debug_mode = args.len > 2 and std.mem.eql(u8, args[2], "--debug");
    
    const blocks = loadFileAsBlocks(allocator, file_path) catch |err| {
        std.debug.print("Error loading file '{s}': {}\n", .{ file_path, err });
        return;
    };
    defer allocator.free(blocks);

    const tokenizer = Tokenizer.init();
    if (debug_mode) {
        tokenizer.processBlocks(blocks);
    } else {
        const result = tokenizer.processBlocksBenchmark(blocks);
        std.debug.print("Processed {} blocks, accumulated token starts: 0x{x}\n", .{ blocks.len, result });
    }
}
