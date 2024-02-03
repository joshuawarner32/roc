#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum T {
    Float,
    String,
    SingleQuote,

    UpperIdent,
    LowerIdent,
    Underscore,
    DotLowerIdent,
    DotNumber,

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
    OpUnaryMinus,  // no trailing whitespace
    OpNotEquals,
    OpBang,
    OpAnd,
    OpAmpersand,
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
    OpColon,
    OpArrow,
    OpBackslash,

    // Keywords
    KwIf,
    KwThen,
    KwElse,
    KwWhen,
    KwIs,
    KwAs,
    KwDbg,
    KwCrash,
    KwHas,
    KwWhere,
    KwImplements,
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
    KwDebug,
    KwExpect,
    KwExpectFx,

    
    NoSpace,
    NamedUnderscore,
    OpaqueName,
    IntBase10,
    IntNonBase10,
    NoSpaceDotNumber,
    NoSpaceDotLowerIdent,
    DotUpperIdent,
    NoSpaceDotUpperIdent,
}

impl T {
    pub fn is_keyword(self) -> bool {
        match self {
            T::KwIf
            | T::KwThen
            | T::KwElse
            | T::KwWhen
            | T::KwIs
            | T::KwAs
            | T::KwDbg
            | T::KwExpect
            | T::KwCrash
            | T::KwHas
            | T::KwWhere
            | T::KwImplements
            | T::KwExposes
            | T::KwImports
            | T::KwWith
            | T::KwGenerates
            | T::KwPackage
            | T::KwPackages
            | T::KwRequires
            | T::KwProvides
            | T::KwTo
            | T::KwInterface
            | T::KwApp
            | T::KwPlatform
            | T::KwHosted
            | T::KwExpectFx => true,
            _ => false,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub struct Indent {
    pub(crate) num_spaces: u16,
    pub(crate) num_tabs: u16,
}

impl Indent {
    pub fn is_indented_more_than(&self, indent: Option<Indent>) -> Option<bool> {
        if let Some(indent) = indent {
            if self.num_spaces == indent.num_spaces {
                Some(self.num_tabs > indent.num_tabs)
            } else if self.num_tabs == indent.num_tabs {
                Some(self.num_spaces > indent.num_spaces)
            } else {
                None
            }
        } else {
            // There's no min indent, so we're always indented more than that
            Some(true)
        }
    }
}

pub struct TokenenizedBuffer {
    pub(crate) kinds: Vec<T>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) lengths: Vec<u32>, // TODO: assess if it's better to just (re)compute this on the fly when accessing later
    pub(crate) lines: Vec<(u32, Indent)>,
}

impl TokenenizedBuffer {
    fn new() -> TokenenizedBuffer {
        TokenenizedBuffer {
            kinds: Vec::new(),
            offsets: Vec::new(),
            lengths: Vec::new(),
            lines: Vec::new(),
        }
    }

    fn push_token(&mut self, kind: T, offset: usize, length: usize) {
        self.kinds.push(kind);
        self.offsets.push(offset as u32);
        self.lengths.push(length as u32);
    }

    pub fn offset(&self, idx: u32) -> usize {
        self.offsets[idx as usize] as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Comment {
    pub begin: usize,
    pub end: usize,
}

pub struct Tokenizer<'a> {
    cursor: Cursor<'a, Vec<Message>>,

    output: TokenenizedBuffer,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
        let mut t = Tokenizer {
            cursor: Cursor {
                buf: text.as_bytes(),
                offset: 0,
                messages: Vec::new(),
            },
            output: TokenenizedBuffer::new(),
        };

        // TODO: consume initial indent, push it onto the stack instead of just assuming 0
        t.output.lines.push((0, Indent::default()));

        t
    }

    pub fn tokenize(&mut self) {
        macro_rules! push_token {
            ($kind:expr) => {{
                let offset = self.cursor.offset;
                let tok = $kind;
                let len = self.cursor.offset - offset;
                self.output.push_token(tok, offset, len);
            }};
        }
        macro_rules! simple_token {
            ($len:expr, $name:ident) => {{
                let offset = self.cursor.offset;
                self.output.push_token(T::$name, offset, $len);
                self.cursor.offset += $len;
            }};
        }

        let mut saw_whitespace = true;

        while let Some(b) = self.cursor.peek() {
            let offset = self.cursor.offset;
            let sp = saw_whitespace;
            saw_whitespace = false;
            match b {
                b' ' | b'\t' | b'\n' | b'\r' | b'#' | b'\x00'..=b'\x1f' => {
                    if let Some(indent) = self.cursor.chomp_trivia(&mut ()) {
                        let line_start = self.output.kinds.len() as u32;
                        self.output.lines.push((line_start, indent));
                    }
                    saw_whitespace = true;
                }
                b'.' => {
                    match self.cursor.peek_at(1) {
                        Some(b'.') => {
                            if self.cursor.peek_at(2) == Some(b'.') {
                                simple_token!(3, TripleDot)
                            } else {
                                simple_token!(2, DoubleDot)
                            }
                        }
                        Some(b'0'..=b'9') => {
                            self.cursor.offset += 1;
                            self.cursor.chomp_integer();
                            self.output
                                .push_token(if sp { T::DotNumber } else { T::NoSpaceDotNumber }, offset, self.cursor.offset - offset);
                        }
                        Some(b'a'..=b'z') => {
                            self.cursor.offset += 1;
                            self.cursor.chomp_ident_general();
                            self.output
                                .push_token(if sp { T::DotLowerIdent } else { T::NoSpaceDotLowerIdent }, offset, self.cursor.offset - offset);
                        }
                        Some(b'A'..=b'Z') => {
                            self.cursor.offset += 1;
                            self.cursor.chomp_ident_general();
                            self.output
                                .push_token(if sp { T::DotUpperIdent } else { T::NoSpaceDotUpperIdent }, offset, self.cursor.offset - offset);
                        }
                        Some(b'{') => {
                            self.cursor.offset += 1;
                            self.output
                                .push_token(T::Dot, offset, self.cursor.offset - offset);
                        }
                        Some(b) => todo!("handle: {:?}", b as char),
                        None => {
                            simple_token!(1, Dot);
                        }
                    }
                }
                b'-' => {
                    match self.cursor.peek_at(1) {
                        Some(b'>') => simple_token!(2, OpArrow),
                        Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => {
                            simple_token!(1, OpBinaryMinus)
                        }
                        Some(b @ b'0'..=b'9') if sp => {
                            self.cursor.offset += 1;
                            let tok = self.cursor.chomp_number(b);
                            // we start at the original offset, not the offset after the '-'
                            self.output
                                .push_token(tok, offset, self.cursor.offset - offset);
                        }
                        _ => {
                            if sp {
                                simple_token!(1, OpUnaryMinus)
                            } else {
                                simple_token!(1, OpBinaryMinus)
                            }
                        },
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
                b',' => simple_token!(1, Comma),
                b'?' => simple_token!(1, OpQuestion),
                b'|' => match self.cursor.peek_at(1) {
                    Some(b'|') => simple_token!(2, OpOr),
                    Some(b'>') => simple_token!(2, OpPizza),
                    _ => simple_token!(1, OpBar),
                },
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
                b'<' => match self.cursor.peek_at(1) {
                    Some(b'=') => simple_token!(2, OpLessThanOrEq),
                    Some(b'-') => simple_token!(2, OpBackArrow),
                    _ => simple_token!(1, OpLessThan),
                },
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
                        simple_token!(1, OpColon)
                    }
                }

                b'(' => simple_token!(1, OpenRound),
                b')' => {
                    simple_token!(1, CloseRound);
                    self.maybe_add_nospace();
                }
                b'[' => simple_token!(1, OpenSquare),
                b']' => {
                    simple_token!(1, CloseSquare);
                    self.maybe_add_nospace();
                }
                b'{' => simple_token!(1, OpenCurly),
                b'}' => {
                    simple_token!(1, CloseCurly);
                    self.maybe_add_nospace();
                }

                b'_' => {
                    match self.cursor.peek_at(1) {
                        Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9') => {
                            self.cursor.offset += 2;
                            self.cursor.chomp_ident_general();
                            self.output.push_token(
                                T::NamedUnderscore,
                                offset,
                                self.cursor.offset - offset,
                            );
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
                            self.output.push_token(
                                T::OpaqueName,
                                offset,
                                self.cursor.offset - offset,
                            );
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

                b'0'..=b'9' => push_token!(self.cursor.chomp_number(b)),

                b'a'..=b'z' => push_token!(self.cursor.chomp_ident_lower()),

                b'A'..=b'Z' => push_token!(self.cursor.chomp_ident_upper()),

                // TODO: handle unicode idents
                b'"' => push_token!(self.cursor.chomp_string_like_literal(b'"')),
                b'\'' => push_token!(self.cursor.chomp_string_like_literal(b'\'')),

                _ => {
                    // Fall back to skipping a single byte
                    self.cursor.messages.push(Message {
                        kind: MessageKind::UnknownToken,
                        offset: self.cursor.offset as u32,
                    });
                    self.cursor.offset += 1;
                    continue;
                }
            }
        }

        if self.output.lines.last().unwrap().0 == self.output.kinds.len() as u32 {
            self.output.lines.pop();
        }

        self.output.offsets.push(self.cursor.offset as u32);
    }

    fn maybe_add_nospace(&mut self) {
        match self.cursor.peek() {
            None | Some(b' ' | b'\t' | b'\r' | b'\n' | b'#' | b',' | b']' | b'}' | b')' | b'-' | b'.') => {}
            _ => {
                self.output.push_token(T::NoSpace, self.cursor.offset, 0);
            }
        }
    }

    pub fn finish(self) -> TokenenizedBuffer {
        self.output
    }
}

trait MessageSink {
    fn push(&mut self, msg: Message);
}

impl MessageSink for Vec<Message> {
    fn push(&mut self, msg: Message) {
        self.push(msg);
    }
}

impl MessageSink for () {
    fn push(&mut self, _msg: Message) {
        // do nothing
    }
}

struct Cursor<'a, M: MessageSink> {
    buf: &'a [u8],
    offset: usize,
    messages: M,
}

trait TriviaSink {
    fn push(&mut self, comment: Comment);
}

impl TriviaSink for Vec<Comment> {
    fn push(&mut self, comment: Comment) {
        self.push(comment);
    }
}

impl TriviaSink for () {
    fn push(&mut self, _trivia: Comment) {
        // do nothing
    }
}

impl<'a, M: MessageSink> Cursor<'a, M> {
    fn chomp_trivia(&mut self, sink: &mut impl TriviaSink) -> Option<Indent> {
        let mut saw_newline = false;
        let mut indent = Indent::default();

        while self.offset < self.buf.len() {
            match self.buf[self.offset] {
                b' ' => {
                    self.offset += 1;
                    if saw_newline {
                        indent.num_spaces += 1;
                    }
                }
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
                    let start = self.offset;
                    self.offset += 1;
                    while self.offset < self.buf.len() && self.buf[self.offset] != b'\n' {
                        self.offset += 1;
                    }
                    sink.push(Comment {
                        begin: start,
                        end: self.offset,
                    })
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

    fn chomp_number(&mut self, b: u8) -> T {
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
            loop {
                let res = match self.peek() {
                    Some(b'x' | b'X') => {
                        maybe_message_for_uppercase_base!(b);
                        self.offset += 1;
                        self.chomp_integer_base16()
                    }
                    Some(b'o' | b'O') => {
                        maybe_message_for_uppercase_base!(b);
                        self.offset += 1;
                        self.chomp_integer_base8()
                    }
                    Some(b'b' | b'B') => {
                        maybe_message_for_uppercase_base!(b);
                        self.offset += 1;
                        self.chomp_integer_base2()
                    }
                    Some(b'0'..=b'9') => {
                        self.messages.push(Message {
                            kind: MessageKind::LeadingZero,
                            offset: self.offset as u32,
                        });
                        self.chomp_number_base10()
                    }
                    Some(b'_') => {
                        self.offset += 1;
                        continue;
                    }
                    Some(b'.') => {
                        self.offset += 1;
                        self.chomp_integer_base10();
                        T::Float
                    }
                    _ => T::IntBase10,
                };
                break res
            }
        } else {
            self.chomp_number_base10()
        };

        // TODO: check for trailing ident chars

        tok
    }

    fn chomp_number_base10(&mut self) -> T {
        self.chomp_integer_base10();
        if self.peek() == Some(b'.') {
            self.offset += 1;
            self.chomp_integer_base10();
            T::Float
        } else {
            T::IntBase10
        }
    }

    fn chomp_integer_base10(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' => self.offset += 1,
                b'_' => {
                    self.offset += 1;
                }
                _ => break,
            }
        }
    }

    fn chomp_integer_base16(&mut self) -> T {
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => self.offset += 1,
                b'_' => {
                    self.offset += 1;
                }
                _ => break,
            }
        }

        T::IntNonBase10
    }

    fn chomp_integer_base8(&mut self) -> T {
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'7' => self.offset += 1,
                b'_' => {
                    self.offset += 1;
                }
                _ => break,
            }
        }

        T::IntNonBase10
    }

    fn chomp_integer_base2(&mut self) -> T {
        while let Some(b) = self.peek() {
            match b {
                b'0' | b'1' => self.offset += 1,
                b'_' => {
                    self.offset += 1;
                }
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

        if kw_check && self.offset - start <= 10 {
            match &self.buf[start..self.offset] {
                b"if" => T::KwIf,
                b"then" => T::KwThen,
                b"else" => T::KwElse,
                b"when" => T::KwWhen,
                b"as" => T::KwAs,
                b"is" => T::KwIs,
                b"dbg" => T::KwDbg,
                b"crash" => T::KwCrash,
                b"has" => T::KwHas,
                b"where" => T::KwWhere,
                b"implements" => T::KwImplements,
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
                b"expect" => {
                    if self.peek() == Some(b'-') && self.peek_at(1) == Some(b'f') && self.peek_at(2) == Some(b'x') {
                        self.offset += 3;
                        T::KwExpectFx
                    } else {
                        T::KwExpect
                    }
                },
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

    fn chomp_integer(&mut self) {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'0'..=b'9' => self.offset += 1,
                _ => break,
            }
        }
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
                        } else if multiline
                            && b == term
                            && self.peek_at(1) == Some(term)
                            && self.peek_at(2) == Some(term)
                        {
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
            kind: if term == b'"' {
                MessageKind::UnclosedString
            } else {
                MessageKind::UnclosedSingleQuote
            },
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
        assert_eq!(
            tokenizer.output.kinds,
            vec![T::IntBase10, T::OpPlus, T::IntBase10]
        );
    }

    #[test]
    fn test_tokenize_unary_minus_x() {
        let mut tokenizer = Tokenizer::new("1 -x");
        tokenizer.tokenize();
        assert_eq!(
            tokenizer.output.kinds,
            vec![T::IntBase10, T::OpUnaryMinus, T::LowerIdent]
        );
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
        assert_eq!(
            tokenizer.output.kinds,
            vec![T::IntBase10, T::OpBinaryMinus, T::IntBase10]
        );
    }

    #[test]
    fn test_tokenize_newline() {
        let mut tokenizer = Tokenizer::new("1\n  \t2");
        tokenizer.tokenize();
        assert_eq!(
            tokenizer.output.kinds,
            vec![T::IntBase10, T::IntBase10]
        );
        assert_eq!(
            tokenizer.output.lines,
            vec![
                (0, Indent { num_spaces: 0, num_tabs: 0 }),
                (1, Indent { num_spaces: 2, num_tabs: 1 }),
            ]
        );
    }

    #[test]
    fn test_tokenize_all_files() {
        // list all .roc files under ../test_syntax/tests/snapshots/pass
        let files = std::fs::read_dir("../test_syntax/tests/snapshots/pass")
            .unwrap()
            .map(|res| res.map(|e| e.path()))
            .collect::<Result<Vec<_>, std::io::Error>>()
            .unwrap();

        assert!(files.len() > 0, "no files found in ../test_syntax/tests/snapshots/pass");

        for file in files {
            if file.extension().map(|e| e != "roc").unwrap_or(true) {
                continue;
            }

            eprintln!("tokenizing {:?}", file);
            let text = std::fs::read_to_string(&file).unwrap();
            let mut tokenizer = Tokenizer::new(&text);
            tokenizer.tokenize(); // make sure we don't panic!

            // check that we don't have any messages
            assert_eq!(tokenizer.cursor.messages, vec![]);
        }
    }
}
