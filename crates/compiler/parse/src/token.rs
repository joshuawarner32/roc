

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Token {
    // LowerIdent-like, all of which match the pattern 0b000x_xxxx
    LowerIdent = 0b0000_0000,
    // Body keywords, which all match the pattern 0b0000_xxxx
    KwIf = 0b0000_0001,
    KwThen = 0b0000_0010,
    KwElse = 0b0000_0011,
    KwWhen = 0b0000_0100,
    KwAs = 0b0000_0101,
    KwIs = 0b0000_0110,
    KwDbg = 0b0000_0111,
    KwExpect = 0b0000_1000,
    KwCrash = 0b0000_1001,
    KwHas = 0b0000_1010,
    // Header keywords, which all match the pattern 0b0001_xxxx
    KwExposes = 0b0001_0000,
    KwImports = 0b0001_0001,
    KwWith = 0b0001_0010,
    KwGenerates = 0b0001_0011,
    KwPackage = 0b0001_0100,
    KwPackages = 0b0001_0101,
    KwRequires = 0b0001_0110,
    KwProvides = 0b0001_0111,
    KwTo = 0b0001_1000,
    KwInterface = 0b0001_1001,
    KwApp = 0b0001_1010,
    KwPlatform = 0b0001_1011,
    KwHosted = 0b0001_1100,

    // "Other", which all match the pattern 0b0010_xxxx
    UpperIdent = 0b0010_0000,
    NamedUnderscore = 0b0010_0001,
    OpaqueName = 0b0010_0010,

    // Hack! This is inserted by the tokenizer after a )/]/} to be able to distinguish between
    // e.g. `Foo {String, Int} a` (i.e. Foo with two type arguments)
    // and `Foo {String, Int}a` (i.e. Foo with one type argument that has an ext var called a)
    NoSpace = 0b0010_0011,
    
    // String pieces, which all match the pattern 0b0100_0xxx
    String = 0b0100_0000,
    StringStart = 0b0100_0001,
    StringEscape = 0b0100_0010,
    StringUnicodeEscape = 0b0100_0011,
    StringInterpolation = 0b0100_0100,
    StringEnd = 0b0100_0101,
    MalformedStringPiece = 0b0100_0110,

    // Char pieces, which all match the pattern 0b0100_1xxx
    SingleQuote = 0b0100_1000,
    SingleQuoteStart = 0b0100_1001,
    SingleQuoteEscape = 0b0100_1010,
    SingleQuoteUnicodeEscape = 0b0100_1011,
    SingleQuoteInterpolation = 0b0100_1100, // malformed
    SingleQuoteEnd = 0b0100_1101,
    MalformedSingleQuotePiece = 0b0100_1110,

    // Number literals, which all match the pattern 0b0101_0xxx
    IntBase10 = 0b0101_0000,
    IntNonBase10 = 0b0101_0001,
    Float = 0b0101_0010,
    IntBase10Neg = 0b0101_0100,

    // Symbols, which all match the pattern 0b10xx_xxxx
    Bang = 0b1000_0000, // !
    Underscore = 0b1000_0001, // _
    Ampersand = 0b1000_0010, // &
    Comma = 0b1000_0011, // ,
    DotLeadingWhitespace = 0b1000_0100, // ., in 'a .b' or 'a .1'
    DoubleDot = 0b1000_0101, // ..
    TripleDot = 0b1000_0110, // ...
    Colon = 0b1000_0111, // :
    Question = 0b1000_1000, // ?
    Bar = 0b1000_1001, // |
    ForwardArrow = 0b1000_1010, // ->
    BackArrow = 0b1000_1011, // <-
    Plus = 0b1000_1100, // +
    MinusTrailingWhitespace = 0b1000_1101, // -, in 'a - b' or '3- 1'
    MinusNoTrailingWhitespace = 0b1000_1110, // -, in 'a-b' or '3 -1'
    Star = 0b1000_1111, // *
    Slash = 0b1001_0000, // /
    Percent = 0b1001_0001, // %
    Caret = 0b1001_0010, // ^
    GreaterThan = 0b1001_0011, // >
    LessThan = 0b1001_0100, // <
    Assignment = 0b1001_0101, // =
    ColonEqual = 0b1001_0110, // :=
    Pizza = 0b1001_1000, // |>
    Equals = 0b1001_1001, // ==
    NotEquals = 0b1001_1010, // !=
    GreaterThanOrEq = 0b1001_1011, // >=
    LessThanOrEq = 0b1001_1100, // <=
    And = 0b1001_1101, // &&
    Or = 0b1001_1110, // ||
    DoubleSlash = 0b1001_1111, // //

    // TODO: insert this above
    DotNoLeadingWhitespace = 0b1010_0000, // ., in 'a.b' or 'a.1'
    Backslash = 0b1010_0001, // \

    // Groups - ( ) / [ ] / { } / INDENT DEDENT, which all match the pattern 0b1100_xxxx
    // The open always has the low bit cleared and the close always has the low bit set 
    OpenParen = 0b1100_0000, // (
    CloseParen = 0b1100_0001, // )
    OpenSquare = 0b1100_0010, // [
    CloseSquare = 0b1100_0011, // ]
    OpenCurly = 0b1100_0100, // {
    CloseCurly = 0b1100_0101, // }
    OpenIndent = 0b1100_0110, // INDENT
    CloseIndent = 0b1100_0111, // DEDENT

    CloseParenNoTrailingWhitespace = 0b1101_0001, // )
    CloseSquareNoTrailingWhitespace = 0b1101_0011, // ]
    CloseCurlyNoTrailingWhitespace = 0b1101_0101, // }

    Newline = 0b1110_0000, // \n
}

impl Token {
    pub fn mask_close_group_whitespace(self) -> Token {
        // Turn CloseParenNoTrailingWhitespace into CloseParen, etc
        match self {
            Token::CloseParenNoTrailingWhitespace => Token::CloseParen,
            Token::CloseSquareNoTrailingWhitespace => Token::CloseSquare,
            Token::CloseCurlyNoTrailingWhitespace => Token::CloseCurly,
            _ => self,
        }
    }

    pub fn is_terminator(self) -> bool {
        match self {
            Token::CloseCurly | Token::CloseSquare | Token::CloseParen | Token::CloseIndent |
            Token::CloseCurlyNoTrailingWhitespace | Token::CloseSquareNoTrailingWhitespace |
            Token::CloseParenNoTrailingWhitespace => true,
            _ => false,
        }
    }

    pub fn to_binop(self) -> Option<BinOp> {
        match self {
            Token::And => Some(BinOp::And),
            Token::Or => Some(BinOp::Or),
            Token::Equals => Some(BinOp::Equals),
            Token::NotEquals => Some(BinOp::NotEquals),
            Token::LessThan => Some(BinOp::LessThan),
            Token::GreaterThan => Some(BinOp::GreaterThan),
            Token::LessThanOrEq => Some(BinOp::LessThanOrEq),
            Token::GreaterThanOrEq => Some(BinOp::GreaterThanOrEq),
            Token::Plus => Some(BinOp::Plus),
            Token::MinusTrailingWhitespace => Some(BinOp::Minus),
            Token::Star => Some(BinOp::Star),
            Token::Slash => Some(BinOp::Slash),
            Token::DoubleSlash => Some(BinOp::DoubleSlash),
            Token::Percent => Some(BinOp::Percent),
            Token::Caret => Some(BinOp::Caret),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinOp {
    // highest precedence
    Caret,
    Star,
    Slash,
    DoubleSlash,
    Percent,
    Plus,
    Minus,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanOrEq,
    GreaterThanOrEq,
    And,
    Or,
    Pizza,
    // lowest precedence
}

pub struct Trivia {
    kind: TriviaKind,
    offset: u32,
}

pub enum TriviaKind {
    Newline,
    LineComment,
    DocComment,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Message {
    kind: MessageKind,
    offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageKind {
    Tab,
    MisplacedCarriageReturn,
    AsciiControl,
    UnknownToken,
    UppercaseBase,
    LeadingZero,
    UnclosedString,
    UnclosedSingleQuote,
    InvalidEscapeSequence,
    InvalidUnicodeEscapeSequence,
    MismatchedGroup,
    OpaqueNameWithoutName,
}

pub struct Tokens {
    pub tokens: Vec<Token>,
    pub token_offsets: Vec<u32>,
    pub messages: Vec<Message>,
}

pub fn tokenize(input: &str) -> Tokens {
    let mut ts = TokenState::new(input, ());

    ts.tokenize();

    Tokens {
        tokens: ts.tokens,
        token_offsets: ts.token_offsets,
        messages: ts.cursor.messages,
    }
}

// TODO: pack this into a single u32
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Group(Token),
    Indent(usize),
}

struct Cursor<'a> {
    buf: &'a [u8],
    offset: usize,
    saw_whitespace: bool,
    last_line_start: usize,
    messages: Vec<Message>,
}

struct TokenState<'a, T: TriviaAccum> {
    cursor: Cursor<'a>,

    scopes: Vec<Scope>,

    tokens: Vec<Token>,
    token_offsets: Vec<u32>,

    token_trivia: T,
}

trait TriviaAccum {
    fn add_trivia(&mut self, whitespace: &[u8], tokens: &[Token]);
}

impl TriviaAccum for () {
    fn add_trivia(&mut self, _whitespace: &[u8], _tokens: &[Token]) {}
}

#[derive(Copy, Clone, Debug)]
struct Bundle<'a> {
    trivia: &'a [u8],
    indent: Option<usize>, // None if there was no newline in trivia
    offset: usize,
    token: Option<Token>,
}

impl<'a, T: TriviaAccum> TokenState<'a, T> {
    fn new(text: &'a str, token_trivia: T) -> TokenState<'a, T> {
        TokenState {
            cursor: Cursor {
                buf: text.as_bytes(),
                offset: 0,
                saw_whitespace: true,
                last_line_start: 0,
                messages: Vec::new(),
            },

            scopes: Vec::new(),

            tokens: Vec::new(),
            token_offsets: Vec::new(),

            token_trivia: token_trivia,
        }
    }

    fn tokenize(&mut self) {
        let mut last = self.cursor.tokenize_bundle();

        while !self.cursor.at_eof() {
            let cur = self.cursor.tokenize_bundle();

            let token_index = self.tokens.len();
            self.push_token(last, cur);
            last = cur;
            self.token_trivia.add_trivia(last.trivia, &self.tokens[token_index..]);
        }

        // handle last token
        let token_index = self.tokens.len();
        self.push_last_token(last);
        self.token_trivia.add_trivia(last.trivia, &self.tokens[token_index..]);
    }

    fn push_last_token(&mut self, last: Bundle) {
        self.handle_token_push(last);

        while let Some(scope) = self.scopes.pop() {
            if let Scope::Indent(_indent) = scope {
                self.append(Token::CloseIndent, last.offset);
            }
            // TODO: do we want to also auto-close groups?
        }
    }

    fn append(&mut self, token: Token, offset: usize) {
        // eprintln!("==> {:?}", token);
        self.tokens.push(token);
        self.token_offsets.push(offset as u32);
    }

    fn push_token(&mut self, last: Bundle<'a>, cur: Bundle<'a>) {
        if last.token == Some(Token::Comma) && cur.indent.is_some() {
            // Push any CloseIndent tokens that were waiting for a comma
            let outer_group = self.scopes.iter().enumerate()
                .rfind(|(index, scope)| matches!(scope, Scope::Group(_)))
                .map(|(index, _)| index);


            if let Some(index) = outer_group {
                // eprintln!("Saw comma newline; popping {} scopes", self.scopes.len() - outer_group.unwrap_or(0) - 1);
                // Pop all the scopes up to the outer group
                while self.scopes.len() > index + 1 {
                    let outer = self.scopes.pop();
                    // eprintln!("  Popping scope {:?}", outer.unwrap());
                    self.append(Token::CloseIndent, last.offset);
                    assert!(matches!(outer, Some(Scope::Indent(_))));
                }
            } else {
                // eprintln!("Saw comma newline; no outer group");
            }
        }

        self.handle_token_push(last);

        // Now push any new indents / newlines
        
        if let Some(indent) = cur.indent {
            if allow_indent(self.scopes.last().copied(), last.token, indent, cur.token) {
                self.append(Token::OpenIndent, cur.offset);
                self.scopes.push(Scope::Indent(indent));
            } else {
                while let Some(is_newline) = allow_dedent_or_newline(self.scopes.last().copied(), last.token, indent, cur.token) {
                    if is_newline {
                        self.append(Token::Newline, cur.offset);
                        break;
                    } else {
                        self.append(Token::CloseIndent, cur.offset);
                        let old = self.scopes.pop();
                        assert!(matches!(old, Some(Scope::Indent(_))));
                    }
                }
            }
        }

    }

    fn handle_token_push(&mut self, last: Bundle) {
        // Now if we saw an open/close group, adjust the scope stack
        match last.token.map(|t| t.mask_close_group_whitespace()) {
            Some(Token::OpenParen) => self.scopes.push(Scope::Group(Token::CloseParen)),
            Some(Token::OpenSquare) => self.scopes.push(Scope::Group(Token::CloseSquare)),
            Some(Token::OpenCurly) => self.scopes.push(Scope::Group(Token::CloseCurly)),
            Some(t @ (Token::CloseParen | Token::CloseSquare | Token::CloseCurly)) => {
                // First pop any indents
                while let Some(scope) = self.scopes.pop() {
                    match scope {
                        Scope::Indent(_indent) => {
                            self.append(Token::CloseIndent, last.offset);
                        }
                        Scope::Group(group) => {
                            if group != t {
                                self.cursor.messages.push(Message {
                                    kind: MessageKind::MismatchedGroup,
                                    offset: last.offset as u32,
                                });
                                self.scopes.push(scope)
                            }
                            break;
                        }
                    }
                }
            }
            _ => {}
        }

        if let Some(token) = last.token {
            self.append(token, last.offset);
        }
    }

}

fn allow_indent(scope: Option<Scope>, last: Option<Token>, indent: usize, cur: Option<Token>) -> bool {
    let in_group = matches!((scope, last), (Some(Scope::Group(_)), Some(Token::Comma | Token::OpenParen | Token::OpenSquare | Token::OpenCurly)));

    let next_is_close_group = matches!(cur, Some(Token::CloseParen | Token::CloseSquare | Token::CloseCurly | Token::CloseParenNoTrailingWhitespace | Token::CloseSquareNoTrailingWhitespace | Token::CloseCurlyNoTrailingWhitespace));

    let block_introducing_token = matches!(last, Some(
        Token::OpenCurly| Token::OpenSquare| Token::OpenParen |
        Token::ForwardArrow | Token::Assignment |
        Token::KwIf | Token::KwThen | Token::KwElse | Token::KwWhen | Token::KwIs
    ));

    let is_indented = match scope {
        Some(Scope::Indent(outer_indent)) if indent > outer_indent => true,
        None if indent > 0 => true,
        _ => false,
    };

    (in_group && !next_is_close_group) || (block_introducing_token && is_indented)
}

fn allow_dedent_or_newline(scope: Option<Scope>, last: Option<Token>, indent: usize, cur: Option<Token>) -> Option<bool> {
    if is_bin_op_maybe(last) {
        return None;
    }

    match scope {
        Some(Scope::Indent(outer_indent)) => {
            if indent < outer_indent {
                Some(false)
            } else if indent == outer_indent && !is_bin_op_maybe(cur) {
                Some(true)
            } else {
                None
            }
        }
        None => {
            if indent == 0 && !is_bin_op_maybe(cur) {
                Some(true)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn is_bin_op_maybe(last: Option<Token>) -> bool {
    if let Some(token) = last {
        is_bin_op(token)
    } else {
        false
    }
}

fn is_bin_op(token: Token) -> bool {
    matches!(token,
        Token::Plus |
        Token::MinusTrailingWhitespace |
        Token::Star |
        Token::Slash |
        Token::Percent |
        Token::Caret |
        Token::GreaterThan |
        Token::LessThan |
        Token::Pizza |
        Token::Equals |
        Token::NotEquals |
        Token::GreaterThanOrEq |
        Token::LessThanOrEq |
        Token::And |
        Token::Or |
        Token::DoubleSlash
    )
}

impl<'a> Cursor<'a> {
    fn chomp_trivia(&mut self) -> bool {
        let start = self.offset;
        let mut saw_newline = false;
        while self.offset < self.buf.len() {
            match self.buf[self.offset] {
                b' ' => self.offset += 1,
                b'\t' => {
                    self.messages.push(Message {
                        kind: MessageKind::Tab,
                        offset: self.offset as u32,
                    });
                    self.offset += 1
                }
                b'\n' => {
                    self.offset += 1;
                    saw_newline = true;
                    self.last_line_start = self.offset;
                }
                b'\r' => {
                    self.offset += 1;
                    if self.offset < self.buf.len() && self.buf[self.offset] == b'\n' {
                        self.offset += 1;
                        saw_newline = true;
                        self.last_line_start = self.offset;
                    } else {
                        self.messages.push(Message {
                            kind: MessageKind::MisplacedCarriageReturn,
                            offset: self.offset as u32 - 1,
                        });
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

        self.saw_whitespace = self.offset == 0 || self.offset > start;

        saw_newline
    }

    fn tokenize_bundle(&mut self) -> Bundle<'a> {
        let start = self.offset;

        let saw_newline = self.chomp_trivia();

        let indent = if saw_newline {
            Some(self.offset - self.last_line_start)
        } else {
            None
        };

        let trivia = &self.buf[start..self.offset];

        let offset = self.offset;

        let token = self.tokenize_token();

        Bundle {
            trivia: trivia,
            indent: indent,
            offset: offset,
            token: token,
        }
    }

    fn tokenize_token(&mut self) -> Option<Token> {

        macro_rules! simple_token {
            ($len:expr, $name:ident) => {
                {
                    self.offset += $len;
                    Some(Token::$name)
                }
            };
        }

        while let Some(b) = self.buf.get(self.offset) {
            return match b {

                // TODO:
                // MinusTrailingWhitespace = 0b1000_1101, // -, in 'a - b' or '3- 1'
                // MinusNoTrailingWhitespace = 0b1000_1110, // -, in 'a-b' or '3 -1'
                // ForwardArrow = 0b1000_1010, // ->

                b'.' => {
                    if self.buf.get(self.offset + 1) == Some(&b'.') {
                        if self.buf.get(self.offset + 2) == Some(&b'.') {
                            simple_token!(3, TripleDot)
                        } else {
                            simple_token!(2, DoubleDot)
                        }
                    } else {
                        if self.saw_whitespace {
                            simple_token!(1, DotLeadingWhitespace)
                        } else {
                            simple_token!(1, DotNoLeadingWhitespace)
                        }
                    }
                }
                b'-' => {
                    match self.buf.get(self.offset + 1) {
                        Some(b'>') => simple_token!(2, ForwardArrow),
                        Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => simple_token!(1, MinusTrailingWhitespace),
                        Some(b @ b'0'..=b'9') => {
                            Some(self.chomp_number(*b, true))
                        },
                        _ => simple_token!(1, MinusNoTrailingWhitespace),
                    }
                }
                b'!' => {
                    if self.buf.get(self.offset + 1) == Some(&b'=') {
                        simple_token!(2, NotEquals)
                    } else {
                        simple_token!(1, Bang)
                    }
                }
                b'&' => {
                    if self.buf.get(self.offset + 1) == Some(&b'&') {
                        simple_token!(2, And)
                    } else {
                        simple_token!(1, Ampersand)
                    }
                }
                b',' => simple_token!(1, Comma),
                b'?' => simple_token!(1, Question),
                b'|' => {
                    match self.buf.get(self.offset + 1) {
                        Some(b'|') => simple_token!(2, Or),
                        Some(b'>') => simple_token!(2, Pizza),
                        _ => simple_token!(1, Bar),
                    }
                }
                b'+' => simple_token!(1, Plus),
                b'*' => simple_token!(1, Star),
                b'/' => {
                    if self.buf.get(self.offset + 1) == Some(&b'/') {
                        simple_token!(2, DoubleSlash)
                    } else {
                        simple_token!(1, Slash)
                    }
                }
                b'\\' => simple_token!(1, Backslash),
                b'%' => simple_token!(1, Percent),
                b'^' => simple_token!(1, Caret),
                b'>' => {
                    if self.buf.get(self.offset + 1) == Some(&b'=') {
                        simple_token!(2, GreaterThanOrEq)
                    } else {
                        simple_token!(1, GreaterThan)
                    }
                }
                b'<' => {
                    match self.buf.get(self.offset + 1) {
                        Some(b'=') => simple_token!(2, LessThanOrEq),
                        Some(b'-') => simple_token!(2, BackArrow),
                        _ => simple_token!(1, LessThan),
                    }
                }
                b'=' => {
                    if self.buf.get(self.offset + 1) == Some(&b'=') {
                        simple_token!(2, Equals)
                    } else {
                        simple_token!(1, Assignment)
                    }
                }
                b':' => {
                    if self.buf.get(self.offset + 1) == Some(&b'=') {
                        simple_token!(2, ColonEqual)
                    } else {
                        simple_token!(1, Colon)
                    }
                }

                b'(' => simple_token!(1, OpenParen),
                b')' => {
                    match self.buf.get(self.offset + 1) {
                        None | Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => simple_token!(1, CloseParen),
                        _ => simple_token!(1, CloseParenNoTrailingWhitespace),
                    }
                }
                b'[' => simple_token!(1, OpenSquare),
                b']' => {
                    match self.buf.get(self.offset + 1) {
                        None | Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => simple_token!(1, CloseSquare),
                        _ => simple_token!(1, CloseSquareNoTrailingWhitespace),
                    }
                }
                b'{' => simple_token!(1, OpenCurly),
                b'}' => {
                    match self.buf.get(self.offset + 1) {
                        None | Some(b' ' | b'\t' | b'\r' | b'\n' | b'#') => simple_token!(1, CloseCurly),
                        _ => simple_token!(1, CloseCurlyNoTrailingWhitespace),
                    }
                }

                b'_' => {
                    match self.buf.get(self.offset + 1) {
                        Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9') => {
                            self.offset += 2;
                            self.chomp_ident_general();
                            Some(Token::NamedUnderscore)
                        }
                        // TODO: handle unicode named underscores
                        _ => {
                            simple_token!(1, Underscore)
                        }
                    }
                }

                b'@' => {
                    match self.buf.get(self.offset + 1) {
                        Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9') => {
                            self.offset += 2;
                            self.chomp_ident_general();
                            Some(Token::OpaqueName)
                        }
                        // TODO: handle unicode opaque names
                        _ => {
                            self.messages.push(Message {
                                kind: MessageKind::OpaqueNameWithoutName,
                                offset: self.offset as u32,
                            });
                            simple_token!(1, OpaqueName)
                        }
                    }
                }

                b'0'..=b'9' => Some(self.chomp_number(*b, false)),

                b'a'..=b'z' => Some(self.chomp_ident_lower()),

                b'A'..=b'Z' => Some(self.chomp_ident_upper()),

                // TODO: handle unicode idents

                b'"' => Some(self.chomp_string_like_literal(b'"')),
                b'\'' => Some(self.chomp_string_like_literal(b'\'')),

                _ => {
                    // Fall back to skipping the token
                    self.messages.push(Message {
                        kind: MessageKind::UnknownToken,
                        offset: self.offset as u32,
                    });
                    self.offset += 1;
                    continue;
                }
            }
        }

        None
    }

    fn chomp_number(&mut self, b: u8, neg: bool) -> Token {
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
            match self.buf.get(self.offset) {
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
                _ => Token::IntBase10,
            }
        } else {
            self.chomp_number_base10(start)
        };

        // TODO: check for trailing ident chars

        tok
    }

    fn chomp_number_base10(&mut self, start: usize) -> Token {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'0'..=b'9' => self.offset += 1,
                _ => break,
            }
        }

        Token::IntBase10
    }

    fn chomp_number_base16(&mut self, start: usize) -> Token {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => self.offset += 1,
                _ => break,
            }
        }

        Token::IntNonBase10
    }

    fn chomp_number_base8(&mut self, start: usize) -> Token {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'0'..=b'7' => self.offset += 1,
                _ => break,
            }
        }

        Token::IntNonBase10
    }

    fn chomp_number_base2(&mut self, start: usize) -> Token {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'0' | b'1' => self.offset += 1,
                _ => break,
            }
        }

        Token::IntNonBase10
    }

    fn chomp_ident_lower(&mut self) -> Token {
        let start = self.offset;

        let mut kw_check = true;
    
        while let Some(b) = self.buf.get(self.offset) {
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
                b"if" => Token::KwIf,
                b"then" => Token::KwThen,
                b"else" => Token::KwElse,
                b"when" => Token::KwWhen,
                b"as" => Token::KwAs,
                b"is" => Token::KwIs,
                b"dbg" => Token::KwDbg,
                b"expect" => Token::KwExpect,
                b"crash" => Token::KwCrash,
                b"has" => Token::KwHas,
                b"exposes" => Token::KwExposes,
                b"imports" => Token::KwImports,
                b"with" => Token::KwWith,
                b"generates" => Token::KwGenerates,
                b"package" => Token::KwPackage,
                b"packages" => Token::KwPackages,
                b"requires" => Token::KwRequires,
                b"provides" => Token::KwProvides,
                b"to" => Token::KwTo,
                b"interface" => Token::KwInterface,
                b"app" => Token::KwApp,
                b"platform" => Token::KwPlatform,
                b"hosted" => Token::KwHosted,
                _ => Token::LowerIdent,
            }
        } else {
            Token::LowerIdent
        }
    }

    fn chomp_ident_upper(&mut self) -> Token {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => self.offset += 1,
                _ => break,
            }
        }

        Token::UpperIdent
    }

    fn chomp_ident_general(&mut self) {
        while let Some(b) = self.buf.get(self.offset) {
            match b {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' => self.offset += 1,
                _ => break,
            }
        }
    }

    fn chomp_string_like_literal(&mut self, term: u8) -> Token {
        self.offset += 1; // skip the initial "/'

        let mut multiline = false;

        if self.buf.get(self.offset) == Some(&term) && self.buf.get(self.offset + 1) == Some(&term) {
            self.offset += 2;
            multiline = true;
        }

        let mut escape = false;
        while let Some(b) = self.buf.get(self.offset) {
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
                            return Token::String;
                        } else {
                            self.offset += 1;
                        }
                    }
                    _ => {
                        if !multiline && *b == term {
                            self.offset += 1;
                            return Token::String;
                        } else if multiline && *b == term && self.buf.get(self.offset + 1) == Some(&term) && self.buf.get(self.offset + 2) == Some(&term) {
                            self.offset += 3;
                            return Token::String;
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
            Token::String
        } else {
            Token::SingleQuote
        }
    }

    fn peek(&self) -> Option<u8> {
        self.buf.get(self.offset).copied()
    }

    fn require(&mut self, ch: u8, kind: MessageKind) {
        if self.buf.get(self.offset) == Some(&ch) {
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
    use super::Token::*;

    #[track_caller]
    fn simple_test(input: &str, expected: Vec<Token>) {
        let tokens = tokenize(input);
        assert_eq!(tokens.tokens, expected);
        assert_eq!(tokens.messages, vec![]);
    }

    #[test]
    fn test_empty() {
        simple_test("", vec![]);
    }

    #[test]
    fn test_empty_groups() {
        simple_test("()", vec![OpenParen, CloseParen]);
        simple_test("[]", vec![OpenSquare, CloseSquare]);
        simple_test("{}", vec![OpenCurly, CloseCurly]);
    }

    #[test]
    fn test_plain_operators() {
        simple_test("!", vec![Bang]);
        simple_test("&", vec![Ampersand]);
        simple_test(",", vec![Comma]);
        simple_test("..", vec![DoubleDot]);
        simple_test("...", vec![TripleDot]);
        simple_test(":", vec![Colon]);
        simple_test("?", vec![Question]);
        simple_test("|", vec![Bar]);
        simple_test("->", vec![ForwardArrow]);
        simple_test("<-", vec![BackArrow]);
        simple_test("+", vec![Plus]);
        simple_test("*", vec![Star]);
        simple_test("/", vec![Slash]);
        simple_test("%", vec![Percent]);
        simple_test("^", vec![Caret]);
        simple_test(">", vec![GreaterThan]);
        simple_test("<", vec![LessThan]);
        simple_test("=", vec![Assignment]);
        simple_test(":=", vec![ColonEqual]);
        simple_test("|>", vec![Pizza]);
        simple_test("==", vec![Equals]);
        simple_test("!=", vec![NotEquals]);
        simple_test(">=", vec![GreaterThanOrEq]);
        simple_test("<=", vec![LessThanOrEq]);
        simple_test("&&", vec![And]);
        simple_test("||", vec![Or]);
        simple_test("//", vec![DoubleSlash]);
        simple_test("_", vec![Underscore]);
        simple_test(".", vec![DotNoLeadingWhitespace]);
        simple_test("-", vec![MinusNoTrailingWhitespace]);
    }

    const BINOPS: &[(&str, Token)] = &[
        ("+", Plus),
        ("*", Star),
        ("/", Slash),
        ("%", Percent),
        ("^", Caret),
        (">", GreaterThan),
        ("<", LessThan),
        ("|>", Pizza),
        ("==", Equals),
        ("!=", NotEquals),
        (">=", GreaterThanOrEq),
        ("<=", LessThanOrEq),
        ("&&", And),
        ("||", Or),
        ("//", DoubleSlash),
    ];

    #[test]
    fn test_binops() {
        for (op, token) in BINOPS {
            simple_test(&format!("1 {} 2", op), vec![IntBase10, *token, IntBase10]);
        }
    }

    #[test]
    fn test_indent_after_binops() {
        for (op, token) in BINOPS {
            simple_test(&format!("1 {}\n    2", op), vec![IntBase10, *token, IntBase10]);
        }
    }

    #[test]
    fn test_indent_before_binops() {
        for (op, token) in BINOPS {
            simple_test(&format!("1\n    {} 2", op), vec![IntBase10, *token, IntBase10]);
        }
    }

    #[test]
    fn test_newline_before_binops() {
        for (op, token) in BINOPS {
            simple_test(&format!("1\n{} 2", op), vec![IntBase10, *token, IntBase10]);
        }
    }

    #[test]
    fn test_newline_simple() {
        simple_test("a\nb", vec![LowerIdent, Newline, LowerIdent]);
    }

    #[test]
    fn test_block_simple() {
        simple_test("a =\n    b = 2\n    c", vec![
            LowerIdent,
            Assignment,
            OpenIndent,
            LowerIdent,
            Assignment,
            IntBase10,
            Newline,
            LowerIdent,
            CloseIndent,
        ]);
        simple_test("a =\n    b = 2\nc", vec![
            LowerIdent,
            Assignment,
            OpenIndent,
            LowerIdent,
            Assignment,
            IntBase10,
            CloseIndent,
            Newline,
            LowerIdent,
        ]);
        simple_test("a =\n    b =\n        c", vec![
            LowerIdent,
            Assignment,
            OpenIndent,
            LowerIdent,
            Assignment,
            OpenIndent,
            LowerIdent,
            CloseIndent,
            CloseIndent,
        ]);
        simple_test("a =\n    b =\n        c\n    b+4", vec![
            LowerIdent,
            Assignment,
            OpenIndent,
            LowerIdent,
            Assignment,
            OpenIndent,
            LowerIdent,
            CloseIndent,
            Newline,
            LowerIdent,
            Plus,
            IntBase10,
            CloseIndent,
        ]);
    }

    #[test]
    fn test_string() {
        simple_test("\"hello\"", vec![String]);
        simple_test("\"\\\"hello\\\"\"", vec![String]);
        simple_test("\"hello\\n\"", vec![String]);
    }

    #[test]
    fn test_paren_commas() {
        simple_test("(\n  a)", vec![OpenParen, OpenIndent, LowerIdent, CloseIndent, CloseParen]);
        simple_test("(\n  a,\n  b\n)", vec![OpenParen, OpenIndent, LowerIdent, CloseIndent, Comma, OpenIndent, LowerIdent, CloseIndent, CloseParen]);
        simple_test("(a,\n  b)", vec![OpenParen, LowerIdent, Comma, OpenIndent, LowerIdent, CloseIndent, CloseParen]);
        simple_test("(\n  a,\n  #comment\n)", vec![OpenParen, OpenIndent, LowerIdent, CloseIndent, Comma, CloseParen]);
    }

    #[test]
    fn test_multibackpassing() {
        simple_test("a, b <- myCall", vec![LowerIdent, Comma, LowerIdent, BackArrow, LowerIdent]);
        simple_test("(\n  a, b <- myCall\n)", vec![OpenParen, OpenIndent, LowerIdent, Comma, LowerIdent, BackArrow, LowerIdent, CloseIndent, CloseParen]);
        simple_test("(\n  a, b <- myCall,\n  a, b <- myCall\n)", vec![
            OpenParen,
            OpenIndent, LowerIdent, Comma, LowerIdent, BackArrow, LowerIdent, CloseIndent, Comma,
            OpenIndent, LowerIdent, Comma, LowerIdent, BackArrow, LowerIdent, CloseIndent, 
            CloseParen,
        ]);
    }

    #[test]
    fn test_when() {
        simple_test("when a is\n  1 -> 2\n  2 -> 3", vec![KwWhen, LowerIdent, KwIs, OpenIndent, IntBase10, ForwardArrow, IntBase10, Newline, IntBase10, ForwardArrow, IntBase10, CloseIndent]);

        // This doesn't work because the tokenizer doesn't have a baseline for the indent level, so it can't insert OpenIndent / Newline tokens
        // simple_test("(when a is\n  1 -> 2\n  2 -> 3)", vec![OpenParen, KwWhen, LowerIdent, KwIs, OpenIndent, IntBase10, ForwardArrow, IntBase10, Newline, IntBase10, ForwardArrow, IntBase10, CloseIndent, CloseParen]);
    }
}
