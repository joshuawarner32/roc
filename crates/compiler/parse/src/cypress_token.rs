
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum T {
    Newline,

    Indent, // Not produced by the tokenizer. Used in the formatter.
    Dedent, // Not produced by the tokenizer. Used in the formatter.
    CommentNewline, // Not produced by the tokenizer. Used in the formatter.
    
    Float,
    String,
    SingleQuote,

    UpperIdent,
    LowerIdent,
    Underscore,

    OpenRound,
    CloseRound,
    OpenSquare,
    CloseSquare,
    OpenCurly,
    CloseCurly,

    OpPlus,
    OpStar,
    OpPizza,
    OpAssign,
    OpBinaryMinus, // trailing whitespace
    OpUnaryMinus, // no trailing whitespace
    OpNotEquals,
    OpBang,
    OpAnd,
    OpAmpersand,
    OpComma,
    OpQuestion,
    OpOr,
    OpBar,
    OpDoubleSlash,
    OpSlash,
    OpPercent,
    OpCaret,
    OpGreaterThanOrEq,
    OpGreaterThan,
    OpLessThanOrEq,
    OpBackArrow,
    OpLessThan,
    OpEquals,
    OpColonEqual,

    Comma,
    Dot,
    DoubleDot,
    TripleDot,
    Colon,
    OpArrow,
    OpBackarrow,
    OpBackslash,


    // Keywords
    KwIf,
    KwThen,
    KwElse,
    KwWhen,
    KwIs,
    KwAs,
    KwDbg,
    KwExpect,
    KwCrash,
    KwHas,
    KwExposes,
    KwImports,
    KwWith,
    KwGenerates,
    KwPackage,
    KwPackages,
    KwRequires,
    KwProvides,
    KwTo,
    KwInterface,
    KwApp,
    KwPlatform,
    KwHosted,
    NoSpace,
    NamedUnderscore,
    OpaqueName,
    IntBase10,
    IntNonBase10,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct Indent {
    pub(crate) num_spaces: u16,
    pub(crate) num_tabs: u16,
}

impl Indent {
    pub fn is_indented_more_than(&self, indent: Indent) -> Option<bool> {
        if self.num_spaces == indent.num_spaces {
            Some(self.num_tabs > indent.num_tabs)
        } else if self.num_tabs == indent.num_tabs {
            Some(self.num_spaces > indent.num_spaces)
        } else {
            None
        }
    }
}

pub struct TokenenizedBuffer {
    pub(crate) kinds: Vec<T>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) indents: Vec<Indent>,
}

impl TokenenizedBuffer {
    fn new() -> TokenenizedBuffer {
        TokenenizedBuffer {
            kinds: Vec::new(),
            offsets: Vec::new(),
            indents: Vec::new(),
        }
    }

    fn push_token(&mut self, kind: T, offset: usize) {
        self.kinds.push(kind);
        self.offsets.push(offset as u32);
    }
}

pub struct Tokenizer<'a> {
    cursor: Cursor<'a>,

    output: TokenenizedBuffer,
}

pub struct Message {
    pub kind: MessageKind,
    pub offset: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MessageKind {
    MisplacedCarriageReturn,
    AsciiControl,
    LeadingZero,
    UnknownToken,
    OpaqueNameWithoutName,
    UppercaseBase,
    InvalidUnicodeEscapeSequence,
    InvalidEscapeSequence,
    UnclosedString,
    UnclosedSingleQuote,
}

impl<'a> Tokenizer<'a> {
    pub fn new(text: &'a str) -> Tokenizer<'a> {
        Tokenizer {
            cursor: Cursor {
                buf: text.as_bytes(),
                offset: 0,
                messages: Vec::new(),
            },
            output: TokenenizedBuffer::new(),
        }
    }

    pub fn tokenize(&mut self) {
        macro_rules! push_token {
            ($kind:expr) => {{
                let offset = self.cursor.offset;
                let tok = $kind;
                self.output.push_token(tok, offset);
            }};
        }
        macro_rules! simple_token {
            ($len:expr, $name:ident) => {
                {
                    push_token!(T::$name);
                    self.cursor.offset += $len;
                }
            };
        }

        while let Some(b) = self.cursor.peek() {
            let offset = self.cursor.offset;
            match b {
                b' ' | b'\t' | b'\n' | b'\r' | b'#' | b'\x00'..=b'\x1f' => {
                    if let Some(indent) = self.cursor.chomp_trivia() {
                        self.output.push_token(T::Newline, offset);
                        self.output.indents.push(indent);
                    }
                }
                b'.' => {
                    if self.cursor.peek_at(1) == Some(b'.') {
                        if self.cursor.peek_at(2) == Some(b'.') {
                            simple_token!(3, TripleDot)
                        } else {
                            simple_token!(2, DoubleDot)
                        }
                    } else {
                        todo!("handle .<ident> and .<num> syntax");
                    }
                }
                b'-' => {
                    match self.cursor.peek_at(1) {
                        Some(b'>') => simple_token!(2, OpArrow),
                        Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => simple_token!(1, OpBinaryMinus),
                        Some(b @ b'0'..=b'9') => {
                            self.cursor.offset += 2;
                            let tok = self.cursor.chomp_number(b, true);
                            self.output.push_token(tok, offset); // we start at the original offset, not the offset after the '-'
                        },
                        _ => simple_token!(1, OpUnaryMinus),
                    }
                }
                b'!' => {
                    if self.cursor.peek_at(1) == Some(b'=') {
                        simple_token!(2, OpNotEquals)
                    } else {
                        simple_token!(1, OpBang)
                    }
                }
                b'&' => {
                    if self.cursor.peek_at(1) == Some(b'&') {
                        simple_token!(2, OpAnd)
                    } else {
                        simple_token!(1, OpAmpersand)
                    }
                }
                b',' => simple_token!(1, OpComma),
                b'?' => simple_token!(1, OpQuestion),
                b'|' => {
                    match self.cursor.peek_at(1) {
                        Some(b'|') => simple_token!(2, OpOr),
                        Some(b'>') => simple_token!(2, OpPizza),
                        _ => simple_token!(1, OpBar),
                    }
                }
                b'+' => simple_token!(1, OpPlus),
                b'*' => simple_token!(1, OpStar),
                b'/' => {
                    if self.cursor.peek_at(1) == Some(b'/') {
                        simple_token!(2, OpDoubleSlash)
                    } else {
                        simple_token!(1, OpSlash)
                    }
                }
                b'\\' => simple_token!(1, OpBackslash),
                b'%' => simple_token!(1, OpPercent),
                b'^' => simple_token!(1, OpCaret),
                b'>' => {
                    if self.cursor.peek_at(1) == Some(b'=') {
                        simple_token!(2, OpGreaterThanOrEq)
                    } else {
                        simple_token!(1, OpGreaterThan)
                    }
                }
                b'<' => {
                    match self.cursor.peek_at(1) {
                        Some(b'=') => simple_token!(2, OpLessThanOrEq),
                        Some(b'-') => simple_token!(2, OpBackArrow),
                        _ => simple_token!(1, OpLessThan),
                    }
                }
                b'=' => {
                    if self.cursor.peek_at(1) == Some(b'=') {
                        simple_token!(2, OpEquals)
                    } else {
                        simple_token!(1, OpAssign)
                    }
                }
                b':' => {
                    if self.cursor.peek_at(1) == Some(b'=') {
                        simple_token!(2, OpColonEqual)
                    } else {
                        simple_token!(1, Colon)
                    }
                }

                b'(' => simple_token!(1, OpenRound),
                b')' => {
                    match self.cursor.peek_at(1) {
                        None | Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => simple_token!(1, CloseRound),
                        _ => {
                            simple_token!(1, CloseRound);
                            push_token!(T::NoSpace);
                        }
                    }
                }
                b'[' => simple_token!(1, OpenSquare),
                b']' => {
                    match self.cursor.peek_at(1) {
                        None | Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => simple_token!(1, CloseSquare),
                        _ => {
                            simple_token!(1, CloseSquare);
                            push_token!(T::NoSpace);
                        }
                    }
                }
                b'{' => simple_token!(1, OpenCurly),
                b'}' => {
                    match self.cursor.peek_at(1) {
                        None | Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => simple_token!(1, CloseCurly),
                        _ => {
                            simple_token!(1, CloseCurly);
                            push_token!(T::NoSpace);
                        }
                    }
                }

                b'_' => {
                    match self.cursor.peek_at(1) {
                        Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9') => {
                            self.cursor.offset += 2;
                            self.cursor.chomp_ident_general();
                            self.output.push_token(T::NamedUnderscore, offset);
                        }
                        // TODO: handle unicode named underscores
                        _ => {
                            simple_token!(1, Underscore)
                        }
                    }
                }

                b'@' => {
                    match self.cursor.peek_at(1) {
                        Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9') => {
                            self.cursor.offset += 2;
                            self.cursor.chomp_ident_general();
                            self.output.push_token(T::OpaqueName, offset);
                        }
                        // TODO: handle unicode opaque names
                        _ => {
                            self.cursor.messages.push(Message {
                                kind: MessageKind::OpaqueNameWithoutName,
                                offset: self.cursor.offset as u32,
                            });
                            simple_token!(1, OpaqueName)
                        }
                    }
                }

                b'0'..=b'9' => push_token!(self.cursor.chomp_number(b, false)),

                b'a'..=b'z' => push_token!(self.cursor.chomp_ident_lower()),

                b'A'..=b'Z' => push_token!(self.cursor.chomp_ident_upper()),

                // TODO: handle unicode idents

                b'"' => push_token!(self.cursor.chomp_string_like_literal(b'"')),
                b'\'' => push_token!(self.cursor.chomp_string_like_literal(b'\'')),

                _ => {
                    // Fall back to skipping the token
                    self.cursor.messages.push(Message {
                        kind: MessageKind::UnknownToken,
                        offset: self.cursor.offset as u32,
                    });
                    self.cursor.offset += 1;
                    continue;
                }
            }
        }
    }

    pub(crate) fn finish(self) -> TokenenizedBuffer {
        self.output
    }
}

struct Cursor<'a> {
    buf: &'a [u8],
    offset: usize,
    messages: Vec<Message>,
}

impl<'a> Cursor<'a> {
    fn chomp_trivia(&mut self) -> Option<Indent> {
        let mut saw_newline = false;
        let mut indent = Indent::default();

        while self.offset < self.buf.len() {
            match self.buf[self.offset] {
                b' ' => {
                    self.offset += 1;
                    if saw_newline {
                        indent.num_spaces += 1;
                    }
                },
                b'\t' => {
                    self.offset += 1;
                    if saw_newline {
                        indent.num_tabs += 1;
                    }
                }
                b'\n' => {
                    self.offset += 1;
                    saw_newline = true;
                    indent = Indent::default();
                }
                b'\r' => {
                    self.offset += 1;
                    saw_newline = true;
                    indent = Indent::default();

                    if self.offset < self.buf.len() && self.buf[self.offset] == b'\n' {
                        self.offset += 1;
                    } else {
                        self.messages.push(Message {
                            kind: MessageKind::MisplacedCarriageReturn,
                            offset: self.offset as u32 - 1,
                        });
                        // we'll treat it as a newline anyway for better error recovery
                    }
                }
                b'#' => {
                    self.offset += 1;
                    while self.offset < self.buf.len() && self.buf[self.offset] != b'\n' {
                        self.offset += 1;
                    }
                }
                b'\x00'..=b'\x1f' => {
                    self.messages.push(Message {
                        kind: MessageKind::AsciiControl,
                        offset: self.offset as u32,
                    });
                    self.offset += 1
                }
                _ => break,
            }
        }
        
        if saw_newline {
            Some(indent)
        } else {
            None
        }
    }

    fn chomp_number(&mut self, b: u8, neg: bool) -> T {
        let start = self.offset;
        self.offset += 1;

        macro_rules! maybe_message_for_uppercase_base {
            ($b:expr) => {
                if b == b'X' || b == b'O' || b == b'B' {
                    self.messages.push(Message {
                        kind: MessageKind::UppercaseBase,
                        offset: self.offset as u32,
                    });
                }
            };
        }

        let tok = if b == b'0' {
            match self.peek() {
                Some(b @ (b'x' | b'X')) => {
                    maybe_message_for_uppercase_base!(b);
                    self.offset += 1;
                    self.chomp_number_base16(start)
                }
                Some(b @ (b'o' | b'O')) => {
                    maybe_message_for_uppercase_base!(b);
                    self.offset += 1;
                    self.chomp_number_base8(start)
                }
                Some(b @ (b'b' | b'B')) => {
                    maybe_message_for_uppercase_base!(b);
                    self.offset += 1;
                    self.chomp_number_base2(start)
                }
                Some(b'0'..=b'9') => {
                    self.messages.push(Message {
                        kind: MessageKind::LeadingZero,
                        offset: self.offset as u32,
                    });
                    self.chomp_number_base10(start)
                }
                _ => T::IntBase10,
            }
        } else {
            self.chomp_number_base10(start)
        };

        // TODO: check for trailing ident chars

        tok
    }

    fn chomp_number_base10(&mut self, start: usize) -> T {
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' => self.offset += 1,
                _ => break,
            }
        }

        T::IntBase10
    }

    fn chomp_number_base16(&mut self, start: usize) -> T {
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => self.offset += 1,
                _ => break,
            }
        }

        T::IntNonBase10
    }

    fn chomp_number_base8(&mut self, start: usize) -> T {
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'7' => self.offset += 1,
                _ => break,
            }
        }

        T::IntNonBase10
    }

    fn chomp_number_base2(&mut self, start: usize) -> T {
        while let Some(b) = self.peek() {
            match b {
                b'0' | b'1' => self.offset += 1,
                _ => break,
            }
        }

        T::IntNonBase10
    }

    fn chomp_ident_lower(&mut self) -> T {
        let start = self.offset;

        let mut kw_check = true;
    
        while let Some(b) = self.peek() {
            match b {
                b'a'..=b'z' => self.offset += 1,
                b'A'..=b'Z' | b'0'..=b'9' => {
                    self.offset += 1;
                    kw_check = false;
                }
                _ => break,
            }
        }

        // check for keywords
        // (if then else when as is dbg expect crash has exposes imports with generates package packages requires provides to)
        if kw_check && self.offset - start <= 9 {
            match &self.buf[start..self.offset] {
                b"if" => T::KwIf,
                b"then" => T::KwThen,
                b"else" => T::KwElse,
                b"when" => T::KwWhen,
                b"as" => T::KwAs,
                b"is" => T::KwIs,
                b"dbg" => T::KwDbg,
                b"expect" => T::KwExpect,
                b"crash" => T::KwCrash,
                b"has" => T::KwHas,
                b"exposes" => T::KwExposes,
                b"imports" => T::KwImports,
                b"with" => T::KwWith,
                b"generates" => T::KwGenerates,
                b"package" => T::KwPackage,
                b"packages" => T::KwPackages,
                b"requires" => T::KwRequires,
                b"provides" => T::KwProvides,
                b"to" => T::KwTo,
                b"interface" => T::KwInterface,
                b"app" => T::KwApp,
                b"platform" => T::KwPlatform,
                b"hosted" => T::KwHosted,
                _ => T::LowerIdent,
            }
        } else {
            T::LowerIdent
        }
    }

    fn chomp_ident_upper(&mut self) -> T {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => self.offset += 1,
                _ => break,
            }
        }

        T::UpperIdent
    }

    fn chomp_ident_general(&mut self) {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => self.offset += 1,
                _ => break,
            }
        }
    }

    fn chomp_string_like_literal(&mut self, term: u8) -> T {
        self.offset += 1; // skip the initial "/'

        let mut multiline = false;

        if self.peek() == Some(term) && self.peek_at(1) == Some(term) {
            self.offset += 2;
            multiline = true;
        }

        let mut escape = false;
        while let Some(b) = self.peek() {
            if escape {
                match b {
                    b'\\' | b'"' | b'\'' | b'n' | b'r' | b't' => {
                        escape = false;
                        self.offset += 1;
                    }
                    b'u' => {
                        escape = false;
                        self.offset += 1;
                        self.require(b'(', MessageKind::InvalidUnicodeEscapeSequence);
                        loop {
                            match self.peek() {
                                Some(b')') => {
                                    self.offset += 1;
                                    break;
                                }
                                Some(b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F') => {
                                    self.offset += 1;
                                }
                                None | Some(b'"' | b'\'' | b'\\' | b'\r' | b'\n') => {
                                    // We've swerved way off the rails; just quit trying to process the unicode escape sequence.
                                    self.messages.push(Message {
                                        kind: MessageKind::InvalidUnicodeEscapeSequence,
                                        offset: self.offset as u32,
                                    });
                                    break;
                                }
                                Some(_) => {
                                    // Eat it, try to continue
                                    self.messages.push(Message {
                                        kind: MessageKind::InvalidUnicodeEscapeSequence,
                                        offset: self.offset as u32,
                                    });
                                }
                            }
                        }
                    }
                    b'(' => {
                        self.offset += 1;

                        // TODO...
                    }
                    _ => {
                        self.messages.push(Message {
                            kind: MessageKind::InvalidEscapeSequence,
                            offset: self.offset as u32,
                        });
                        escape = false;
                        self.offset += 1;
                    }
                }
            } else {
                match b {
                    b'\\' => {
                        escape = true;
                        self.offset += 1;
                    }
                    b'\n' => {
                        if !multiline {
                            self.messages.push(Message {
                                kind: MessageKind::UnclosedString,
                                offset: self.offset as u32,
                            });
                            return T::String;
                        } else {
                            self.offset += 1;
                        }
                    }
                    _ => {
                        if !multiline && b == term {
                            self.offset += 1;
                            return T::String;
                        } else if multiline && b == term && self.peek_at(1) == Some(term) && self.peek_at(2) == Some(term) {
                            self.offset += 3;
                            return T::String;
                        }
                        self.offset += 1;
                    }
                }
            }
        }

        // We got to the end of the file without finding a closing quote.
        self.messages.push(Message {
            kind: if term == b'"' { MessageKind::UnclosedString } else { MessageKind::UnclosedSingleQuote },
            offset: self.offset as u32,
        });

        if term == b'"' {
            T::String
        } else {
            T::SingleQuote
        }
    }

    fn peek(&self) -> Option<u8> {
        self.buf.get(self.offset).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.buf.get(self.offset + offset).copied()
    }

    fn require(&mut self, ch: u8, kind: MessageKind) {
        if self.peek() == Some(ch) {
            self.offset += 1;
        } else {
            self.messages.push(Message {
                kind,
                offset: self.offset as u32,
            });
        }
    }

    fn at_eof(&self) -> bool {
        self.offset >= self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tokenize_trivial() {
        let mut tokenizer = Tokenizer::new("");
        tokenizer.tokenize();
        assert_eq!(tokenizer.output.kinds, vec![]);
    }

    #[test]
    fn test_tokenize_plus() {
        let mut tokenizer = Tokenizer::new("1 + 2");
        tokenizer.tokenize();
        assert_eq!(tokenizer.output.kinds, vec![T::IntBase10, T::OpPlus, T::IntBase10]);
    }

    #[test]
    fn test_tokenize_unary_minus_x() {
        let mut tokenizer = Tokenizer::new("1 -x");
        tokenizer.tokenize();
        assert_eq!(tokenizer.output.kinds, vec![T::IntBase10, T::OpUnaryMinus, T::LowerIdent]);
    }

    #[test]
    fn test_tokenize_unary_minus_num() {
        let mut tokenizer = Tokenizer::new("1 -2");
        tokenizer.tokenize();
        assert_eq!(tokenizer.output.kinds, vec![T::IntBase10, T::IntBase10]);
    }

    #[test]
    fn test_tokenize_binary_minus() {
        let mut tokenizer = Tokenizer::new("1 - 2");
        tokenizer.tokenize();
        assert_eq!(tokenizer.output.kinds, vec![T::IntBase10, T::OpBinaryMinus, T::IntBase10]);
    }

    #[test]
    fn test_tokenize_newline() {
        let mut tokenizer = Tokenizer::new("1\n  \t2");
        tokenizer.tokenize();
        assert_eq!(tokenizer.output.kinds, vec![T::IntBase10, T::Newline, T::IntBase10]);
        assert_eq!(tokenizer.output.indents, vec![Indent { num_spaces: 2, num_tabs: 1 }]);
    }
}
