#![allow(dead_code)]
#![allow(unused)]

use std::collections::VecDeque;

pub struct TokenenizedBuffer {
    kinds: Vec<T>,
    offsets: Vec<u32>,
    indents: Vec<Indent>,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum T {
    Newline,

    Indent, // Not produced by the tokenizer. Used in the formatter.
    Dedent, // Not produced by the tokenizer. Used in the formatter.
    CommentNewline, // Not produced by the tokenizer. Used in the formatter.
    
    Float,
    Num,
    String,
    SingleQuote,
    Number,

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
    OpArrow,
    OpBackarrow,
    OpBackslash,

    Comma,
    Dot,
    Colon,

    // Keywords
    KwIf,
    KwThen,
    KwElse,
    KwWhen,
    KwIs,
}

pub struct Tree {
    kinds: Vec<N>,
    tokens: Vec<Token>,
    subtree_start_positions: Vec<u32>,
}

#[derive(Debug, Copy, Clone)]
pub struct Token(u32);

#[derive(Debug, Copy, Clone)]
pub struct Node(u32);

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum N {
    // Number Literals
    Float,

    /// Integer Literals, e.g. `42`
    Num,

    /// String Literals, e.g. `"foo"`
    Str,

    /// eg 'b'
    SingleQuote,

    /// Look up exactly one field on a record or tuple, e.g. `x.foo` or `x.0`.
    Access,

    /// e.g. `.foo` or `.0`
    AccessorFunction,

    /// List literals, e.g. `[1, 2, 3]`
    List,

    /// Record literals, e.g. `{ x: 1, y: 2 }`
    Record,

    /// Record updates (e.g. `{ x & y: 3 }`)
    RecordUpdate,

    Parens,

    /// Tuple literals, e.g. `(1, 2)`
    Tuple,

    Ident,

    /// An underscore, e.g. `_` or `_x`
    Underscore,

    /// The "crash" keyword
    Crash,

    /// Tag
    Tag,

    /// Reference to an opaque type, e.g. @Opaq
    OpaqueRef,

    /// Closure, e.g. `\x -> x`
    Closure,

    /// Indented block of statements and expressions
    BeginBlock, EndBlock,

    /// The special dbg function, e.g. `dbg x`
    Dbg,

    /// Function application, e.g. `f x`
    Apply,

    /// Pizza operator, e.g. `x |> f`
    Pizza,

    /// Assignment declaration, e.g. `x = 1`
    Assign,

    /// Binary operators, e.g. `x + y`
    BinOpPlus,
    BinOpStar,

    /// Unary operator, e.g. `-x`
    UnaryOp,

    /// If expression, e.g. `if x then y else z`
    If,

    /// A when expression, e.g. `when x is y -> z` (you need a newline after the 'is')
    When,
    ArgList,
    EndTopLevelDecls,
    BeginTopLevelDecls,
}

impl N {
    fn is_decl(self) -> bool {
        match self {
            N::Assign => true,
            _ => false,
        }
    }
}



impl TokenenizedBuffer {
    fn new(text: String) -> TokenenizedBuffer {
        todo!()
    }

    fn kind(&self, pos: usize) -> Option<T> {
        self.kinds.get(pos).copied()
    }

    fn from_tokens(kinds: &[T]) -> TokenenizedBuffer {
        TokenenizedBuffer { kinds:  kinds.to_owned(), offsets: Vec::new(), indents: Vec::new() }
    }
}


impl Tree {
    fn new() -> Tree {
        Tree {
            kinds: Vec::new(),
            tokens: Vec::new(),
            subtree_start_positions: Vec::new(),
        }
    }

    fn len(&self) -> u32 {
        self.kinds.len() as u32
    }

    fn debug_vis_grouping(&self) -> String {
        let mut stack = Vec::<(usize, String)>::new();

        for (i, (&kind, &subtree_start_pos)) in self.kinds.iter().zip(self.subtree_start_positions.iter()).enumerate() {

            if (subtree_start_pos as usize) < i {
                let mut items = VecDeque::new();
                while let Some((j, item)) = stack.pop() {
                    if j >= subtree_start_pos as usize {
                        items.push_front(item);
                    } else {
                        stack.push((j, item));
                        break;
                    }
                }

                // (a b c)
                let mut s = String::new();
                s.push('(');
                for (i, item) in items.into_iter().enumerate() {
                    if i > 0 {
                        s.push(' ');
                    }
                    s.push_str(&item);
                }
                s.push(')');

                s.push(' ');
                s.push_str(&format!("{:?}", kind));

                stack.push((i, s));
            } else {
                stack.push((i, format!("{:?}", kind)));
            }
        }
        
        // (a b c)
        let mut s = String::new();
        for (i, (_, item)) in stack.into_iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&item);
        }

        s
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Prec {
    Outer,
    DeclSeq, // BinOp::DeclSeq,
    Decl, // BinOp::Assign, BinOp::Backpassing,
    Pizza, // BinOp::Pizza,
    AndOr, // BinOp::And, BinOp::Or,
    Compare, // BinOp::Equals, BinOp::NotEquals, BinOp::LessThan, BinOp::GreaterThan, BinOp::LessThanOrEq, BinOp::GreaterThanOrEq,
    Add, // BinOp::Plus, BinOp::Minus,
    Multiply, // BinOp::Star, BinOp::Slash, BinOp::DoubleSlash, BinOp::Percent,
    Exponent, // BinOp::Caret
    Apply,
    Atom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    AssignBlock,
    DeclSeq,
    Assign,
    Backpassing,
    Pizza,
    And,
    Or,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanOrEq,
    GreaterThanOrEq,
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    Percent,
    Caret,
    Apply,
}

impl Prec {
    fn next(self) -> Prec {
        match self {
            Prec::Outer => Prec::DeclSeq,
            Prec::DeclSeq => Prec::Decl,
            Prec::Decl => Prec::Pizza,
            Prec::Pizza => Prec::AndOr,
            Prec::AndOr => Prec::Compare,
            Prec::Compare => Prec::Add,
            Prec::Add => Prec::Multiply,
            Prec::Multiply => Prec::Exponent,
            Prec::Exponent => Prec::Apply,
            Prec::Apply => Prec::Atom,
            Prec::Atom => Prec::Atom,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Assoc {
    Left,
    Right,
    NonAssociative,
}

impl BinOp {
    fn prec(self) -> Prec {
        match self {
            // BinOp::AssignBlock => Prec::Outer,
            BinOp::DeclSeq => Prec::DeclSeq,
            BinOp::AssignBlock | BinOp::Assign | BinOp::Backpassing => Prec::Decl,
            BinOp::Apply => Prec::Apply,
            BinOp::Caret => Prec::Exponent,
            BinOp::Star | BinOp::Slash | BinOp::DoubleSlash | BinOp::Percent => Prec::Multiply,
            BinOp::Plus | BinOp::Minus => Prec::Add,
            BinOp::Equals | BinOp::NotEquals | BinOp::LessThan | BinOp::GreaterThan | BinOp::LessThanOrEq | BinOp::GreaterThanOrEq => Prec::Compare,
            BinOp::And | BinOp::Or => Prec::AndOr,
            BinOp::Pizza => Prec::Pizza,
        }
    }

    fn grouping_assoc(self) -> Assoc {
        match self {
            BinOp::AssignBlock => Assoc::Right,
            BinOp::DeclSeq => Assoc::Right,
            BinOp::Assign | BinOp::Backpassing => Assoc::Right,
            BinOp::Apply => Assoc::Left,
            BinOp::Caret => Assoc::Right,
            BinOp::Star | BinOp::Slash | BinOp::DoubleSlash | BinOp::Percent => Assoc::Left,
            BinOp::Plus | BinOp::Minus => Assoc::Left,
            BinOp::Equals | BinOp::NotEquals | BinOp::LessThan | BinOp::GreaterThan | BinOp::LessThanOrEq | BinOp::GreaterThanOrEq => Assoc::NonAssociative,
            BinOp::And | BinOp::Or => Assoc::Left,
            BinOp::Pizza => Assoc::Left,
        }
    }

    fn matching_assoc(self) -> Assoc {
        if self == BinOp::AssignBlock {
            return Assoc::Left;
        } else {
            self.grouping_assoc()
        }
    }

    fn n_arity(self) -> bool {
        match self {
            BinOp::Apply | BinOp::Pizza | BinOp::DeclSeq => true,
            _ => false,
        }
    }
}

impl From<BinOp> for N {
    fn from(op: BinOp) -> Self {
        match op {
            BinOp::Pizza => N::Pizza,
            BinOp::Apply => N::Apply,
            BinOp::Plus => N::BinOpPlus,
            BinOp::Minus => N::BinOpStar,
            BinOp::Assign => N::Assign,
            BinOp::AssignBlock => N::Assign,
            _ => todo!("binop to node {:?}", op),
        }
    }
}

#[derive(Debug)]
enum Frame {
    StartExpr { min_prec: Prec },
    ContinueExpr { min_prec: Prec, cur_op: Option<BinOp>, num_found: usize },
    ContinueBlock,
    FinishParen,
    FinishAssignBlock,
    ContinueTopLevel { num_found: i32 },
    ContinueClosureArgs,
    ContinueIf { next: IfState },
    ContinueWhen { next: WhenState },
    FinishClosure,
}

impl Frame {
    fn start_expr() -> Frame {
        Frame::StartExpr { min_prec: Prec::DeclSeq }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExprCfg {
    when_branch_indent_floor: Option<Indent>, // when branches must be indented more than this. None means no restriction.
    allow_multi_backpassing: bool,
}

impl Default for ExprCfg {
    fn default() -> Self {
        ExprCfg {
            when_branch_indent_floor: None,
            allow_multi_backpassing: true,
        }
    }
}

#[derive(Debug)]
enum IfState {
    Then,
    Else,
    End,
}

#[derive(Debug)]
enum WhenState {
    Is,
    BranchPattern(Indent),
    BranchArrow(Indent),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
struct Indent {
    num_spaces: u16,
    num_tabs: u16,
}

impl Indent {
    fn is_indented_more_than(&self, indent: Indent) -> Result<bool, Error> {
        if self.num_spaces == indent.num_spaces {
            Ok(self.num_tabs > indent.num_tabs)
        } else if self.num_tabs == indent.num_tabs {
            Ok(self.num_spaces > indent.num_spaces)
        } else {
            Err(Error::InconsistentIndent)
        }
    }
}

enum Error {
    InconsistentIndent,
}

struct State {
    frames: Vec<(u32, ExprCfg, Frame)>,
    buf: TokenenizedBuffer,
    pos: usize,
    line: usize,

    tree: Tree,
}

impl State {
    fn new(text: String) -> Self {
        State {
            frames: vec![],
            buf: TokenenizedBuffer::new(text),
            pos: 0,
            line: 0,
            tree: Tree::new(),
        }
    }

    fn from_tokens(tokens: &[T]) -> Self {
        State {
            frames: vec![],
            buf: TokenenizedBuffer::from_tokens(tokens),
            pos: 0,
            line: 0,
            tree: Tree::new(),
        }
    }

    fn cur(&self) -> Option<T> {
        self.buf.kind(self.pos)
    }

    fn cur_indent(&self) -> Indent {
        self.buf.indents.get(self.line).copied().unwrap_or(Indent { num_spaces: 0, num_tabs: 0 })
    }

    fn at_terminator(&self) -> bool {
        matches!(self.cur(), None | Some(T::CloseRound) | Some(T::CloseSquare) | Some(T::CloseCurly))
    }

    fn bump(&mut self) {
        debug_assert!(self.cur() != Some(T::Newline));
        self.pos += 1;
    }

    fn expect(&mut self, tok: T) {
        debug_assert!(tok != T::Newline); // Use expect_newline instead
        if self.cur() != Some(tok) {
            todo!()
        }
        self.bump();
    }

    fn expect_newline(&mut self) {
        if self.cur() != Some(T::Newline) {
            todo!()
        }
        self.pos += 1;
        self.line += 1;
    }

    fn consume(&mut self, tok: T) -> bool {
        debug_assert!(tok != T::Newline); // Use consume_newline instead
        if self.cur() != Some(tok) {
            return false;
        }
        self.bump();
        true
    }

    fn consume_newline(&mut self) -> bool {
        if self.cur() != Some(T::Newline) {
            return false;
        }
        self.pos += 1;
        self.line += 1;
        true
    }

    fn push_node(&mut self, kind: N, subtree_start: Option<u32>) {
        eprintln!("{:indent$}pushing kind {:?} starting at {:?}", "", kind, subtree_start, indent = 2 * self.frames.len() + 2);
        self.tree.kinds.push(kind);
        let pos = subtree_start.unwrap_or(self.tree.subtree_start_positions.len() as u32);
        self.tree.subtree_start_positions.push(pos);
        eprintln!("{:indent$}tree: {:?}", "", self.tree.debug_vis_grouping(), indent = 2 * self.frames.len() + 4);
    }

    fn push_next_frame(&mut self, subtree_start: u32, cfg: ExprCfg, frame: Frame) {
        eprintln!("{:indent$}pushing frame {:?}", "", frame, indent = 2 * self.frames.len() + 2);
        self.frames.push((subtree_start, cfg, frame));
    }

    fn push_next_frame_starting_here(&mut self, cfg: ExprCfg, frame: Frame) {
        let subtree_start = self.tree.len();
        self.push_next_frame(subtree_start, cfg, frame)
    }

    fn pump(&mut self) {
        while let Some((subtree_start, cfg, frame)) = self.frames.pop() {
            eprintln!("{:indent$}@{} pumping frame {:?}", "", self.pos, frame, indent = 2 * self.frames.len());
            match frame {
                Frame::StartExpr { min_prec } => self.pump_start_expr(cfg, min_prec),
                Frame::FinishParen => self.pump_finish_paren(),
                Frame::FinishClosure => self.pump_finish_closure(subtree_start),
                Frame::ContinueExpr { min_prec, cur_op, num_found } => self.pump_continue_expr(subtree_start, cfg, min_prec, cur_op, num_found),
                Frame::ContinueBlock => self.pump_continue_block(subtree_start, cfg),
                Frame::FinishAssignBlock => self.pump_finish_assign_block(subtree_start, cfg),
                Frame::ContinueTopLevel { num_found } => self.pump_continue_top_level(subtree_start, cfg, num_found),
                Frame::ContinueClosureArgs => self.pump_continue_closure_args(subtree_start, cfg),
                Frame::ContinueIf { next } => self.pump_continue_if(subtree_start, cfg, next),
                Frame::ContinueWhen { next } => self.pump_continue_when(subtree_start, cfg, next),
            }
        }
    }

    fn pump_start_expr(&mut self, cfg: ExprCfg, mut min_prec: Prec) {
        loop {
            let subtree_start = self.tree.len();
            match self.cur() {
                Some(T::OpenRound) => {
                    self.bump();
                    self.push_next_frame(subtree_start, cfg, Frame::ContinueExpr { min_prec, cur_op: None, num_found: 1 });
                    self.push_next_frame(subtree_start, cfg, Frame::FinishParen);
                    min_prec = Prec::Outer;
                    continue;
                }
                Some(T::LowerIdent) => {
                    self.bump();
                    self.push_node(N::Ident, None);

                    match self.cur() {
                        None | Some(T::OpArrow) => return,
                        _ => {}
                    }

                    self.push_next_frame(subtree_start, cfg, Frame::ContinueExpr { min_prec, cur_op: None, num_found: 1 });
                    return;
                }
                Some(T::OpBackslash) => {
                    self.bump();
                    self.push_next_frame(subtree_start, cfg, Frame::ContinueClosureArgs);
                    return;
                }
                Some(T::KwIf) => {
                    self.bump();
                    self.push_next_frame(subtree_start, cfg, Frame::ContinueIf { next: IfState::Then });
                    self.start_expr(cfg);
                    return;
                }
                Some(T::KwWhen) => {
                    self.bump();
                    self.push_next_frame(subtree_start, cfg, Frame::ContinueWhen { next: WhenState::Is });
                    self.start_expr(cfg);
                    return;
                }
                k => todo!("{:?}", k),
            }
        }
    }

    fn pump_continue_expr(&mut self, subtree_start: u32, cfg: ExprCfg, min_prec: Prec, cur_op: Option<BinOp>, mut num_found: usize) {
        if let Some(op) = self.next_op(min_prec, cur_op) {
            if let Some(cur_op) = cur_op {
                if op != cur_op || !op.n_arity() {
                    self.push_node(cur_op.into(), Some(subtree_start));
                }
            }

            if op == BinOp::Assign && self.consume_newline() {
                self.push_next_frame(subtree_start, cfg, Frame::FinishAssignBlock);
                self.push_next_frame_starting_here(cfg, Frame::ContinueBlock);
                self.push_node(N::BeginBlock, None);
                self.start_expr(cfg);
                return;
            }

            eprintln!("{:indent$}next op {:?}", "", op, indent = 2 * self.frames.len() + 2);

            let op_prec = op.prec();
            let assoc = op.matching_assoc();

            let next_min_prec = if assoc == Assoc::Left {
                op_prec
            } else {
                op_prec.next()
            };

            self.push_next_frame(subtree_start, cfg, Frame::ContinueExpr { min_prec, cur_op: Some(op), num_found });
            self.push_next_frame_starting_here(cfg, Frame::StartExpr { min_prec: next_min_prec });
            return;
        } else if let Some(cur_op) = cur_op {
            self.push_node(cur_op.into(), Some(subtree_start));
        }

    }

    fn next_op(&mut self, min_prec: Prec, cur_op: Option<BinOp>) -> Option<BinOp> {
        let k = self.buf.kind(self.pos);

        let (op, width) = match k {
            Some(T::LowerIdent) => (BinOp::Apply, 0),
            Some(T::OpPlus) => (BinOp::Plus, 1),
            Some(T::OpStar) => (BinOp::Star, 1),
            Some(T::OpPizza) => (BinOp::Pizza, 1),
            Some(T::OpAssign) => (BinOp::Assign, 1),
            _ => return None,
        };

        if op.prec() < min_prec || (op.prec() == min_prec && op.grouping_assoc() == Assoc::Left) {
            return None;
        }

        self.pos += width;

        Some(op)
    }

    fn pump_finish_paren(&mut self) {
        self.expect(T::CloseRound);
    }

    fn pump_finish_closure(&mut self, subtree_start: u32) {
        self.push_node(N::Closure, Some(subtree_start));
    }

    fn pump_continue_block(&mut self, subtree_start: u32, cfg: ExprCfg) {
        while self.consume_newline() {}

        // need to inspect the expr we just parsed.
        // if it's a decl we keep going; if it's not, we're done.
        if self.tree.kinds.last().copied().unwrap().is_decl() {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueBlock);
            self.start_expr(cfg);
        } else {
            self.push_node(N::EndBlock, Some(subtree_start));
        }
    }

    fn pump_finish_assign_block(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::Assign, Some(subtree_start));
    }

    fn pump_continue_top_level(&mut self, subtree_start: u32, cfg: ExprCfg, num_found: i32) {
        // keep parsing decls until the end
        while self.consume_newline() {}

        if self.pos < self.buf.kinds.len() {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueTopLevel { num_found: num_found + 1 });
            self.push_next_frame_starting_here(cfg, Frame::start_expr());
        } else {
            self.push_node(N::EndTopLevelDecls, Some(subtree_start as u32));
        }
    }

    fn pump_continue_closure_args(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::OpArrow) {
            self.push_next_frame(subtree_start, cfg, Frame::FinishClosure);
            self.start_block_or_expr(cfg);
        } else {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueClosureArgs);
            self.push_next_frame_starting_here(cfg, Frame::start_expr());
        }
    }

    fn pump_continue_if(&mut self, subtree_start: u32, cfg: ExprCfg, next: IfState) {
        match next {
            IfState::Then => {
                self.expect(T::KwThen);
                self.push_next_frame(subtree_start, cfg, Frame::ContinueIf { next: IfState::Else });
                self.start_block_or_expr(cfg);
            }
            IfState::Else => {
                self.expect(T::KwElse);

                let next = if self.consume(T::KwIf) {
                    IfState::Then
                } else {
                    IfState::End
                };

                self.push_next_frame(subtree_start, cfg, Frame::ContinueIf { next });
                self.start_block_or_expr(cfg);
            }
            IfState::End => {
                self.push_node(N::If, Some(subtree_start));
            }
        }
    }

    fn pump_continue_when(&mut self, subtree_start: u32, cfg: ExprCfg, next: WhenState) {
        match next {
            WhenState::Is => {
                self.expect(T::KwIs);
                self.consume_newline();
                let indent = self.cur_indent();
                self.push_next_frame(subtree_start, cfg, Frame::ContinueWhen { next: WhenState::BranchArrow(indent) });
                self.start_expr(cfg);
            },
            WhenState::BranchPattern(indent) => {
                if let Some(min_indent) = cfg.when_branch_indent_floor {
                    if self.check(self.cur_indent().is_indented_more_than(min_indent)) && !self.at_terminator() {
                        self.push_next_frame(subtree_start, cfg, Frame::ContinueWhen { next: WhenState::BranchArrow(indent) });
                        self.start_expr(cfg);
                        return;
                    }
                }
                self.push_node(N::When, Some(subtree_start));
            }
            WhenState::BranchArrow(indent) => {
                self.expect(T::OpArrow);
                self.push_next_frame(subtree_start, cfg, Frame::ContinueWhen { next: WhenState::BranchPattern(indent) });
                self.start_block_or_expr(cfg);
            }
        }
    }

    fn start_top_level_decls(&mut self) {
        self.push_next_frame_starting_here(ExprCfg::default(), Frame::ContinueTopLevel { num_found: 0 });
        self.push_node(N::BeginTopLevelDecls, None);
    }

    fn start_block_or_expr(&mut self, cfg: ExprCfg) {
        if self.consume_newline() {
            self.push_next_frame_starting_here(cfg, Frame::ContinueBlock);
            self.push_node(N::BeginBlock, None);
            self.start_expr(cfg);
        } else {
            self.start_expr(cfg);
        }
    }

    fn start_expr(&mut self, cfg: ExprCfg) {
        self.push_next_frame_starting_here(cfg, Frame::start_expr());
    }

    fn assert_end(&self) {
        assert_eq!(self.pos, self.buf.kinds.len(),
            "Expected to be at the end, but these tokens remain: {:?}",
            &self.buf.kinds[self.pos..]);
    }

    fn check<T>(&self, v: Result<T, Error>) -> T {
        match v {
            Ok(v) => v,
            Err(e) => todo!(),
        }
    }
}

struct FormattedBuffer {
    kinds: VecDeque<T>,
    offsets: VecDeque<u32>,
}

impl FormattedBuffer {
    fn new() -> FormattedBuffer {
        FormattedBuffer {
            kinds: VecDeque::new(),
            offsets: VecDeque::new(),
        }
    }
}

struct Formatter<'a> {
    tree: &'a Tree,
    buf: &'a TokenenizedBuffer,
    out: FormattedBuffer,
}

// fn pretty(tree: &Tree, buf: &TokenenizedBuffer) -> FormattedBuffer {
//     let mut buf = FormattedBuffer::new();
//     for (i, &node) in tree.kinds.iter().enumerate() {
//         match node {
//             N::Ident => T::LowerIdent,
//             N::Assign => T::OpAssign,
//             N::DeclSeq => T::Newline,
//             N::BinOpPlus => T::OpPlus,
//             N::BinOpStar => T::OpStar,
//             N::Apply => T::Newline,
//             N::Pizza => T::OpPizza,
//             N::Closure => T::OpBackslash,
//             N::If => T::KwIf,
//             N::When => T::KwWhen,
//             _ => todo!(),
//         };

//         buf.kinds.push_back(kind);
//         buf.offsets.push_back(i as u32);
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    enum ExpectAtom {
        Seq(Vec<ExpectAtom>),
        Unit(N),
    }

    struct ExpectBuilder {
        kinds: Vec<N>,
        subtree_start_positions: Vec<u32>,
    }

    impl ExpectBuilder {
        fn new() -> ExpectBuilder {
            ExpectBuilder {
                kinds: Vec::new(),
                subtree_start_positions: Vec::new(),
            }
        }

        fn consume_items(&mut self, items: Vec<ExpectAtom>) {
            let mut last_start_pos = None;

            for item in items {
                match item {
                    ExpectAtom::Seq(items) => {
                        assert!(last_start_pos.is_none());
                        last_start_pos = Some(self.kinds.len() as u32);
                        self.consume_items(items);
                    }
                    ExpectAtom::Unit(kind) => {
                        self.kinds.push(kind);
                        let pos = last_start_pos.take().unwrap_or(self.subtree_start_positions.len() as u32);
                        self.subtree_start_positions.push(pos);
                    }
                }
            }
        }

        fn finish(mut self) -> Tree {
            Tree {
                kinds: self.kinds,
                subtree_start_positions: self.subtree_start_positions,
                tokens: Vec::new(), // TODO
            }
        }
    }

    fn build_expect(items: Vec<ExpectAtom>) -> Tree {
        let mut b = ExpectBuilder::new();
        
        b.consume_items(items);

        b.finish()
    }

    macro_rules! cvt_item {
        ($item:ident) => {
            ExpectAtom::Unit(N::$item)
        };

        (($($items:tt)*)) => {
            ExpectAtom::Seq(vec![$(cvt_item!($items)),*])
        };
    }

    macro_rules! expect {
        ($($items:tt)*) => {{
            build_expect(vec![$(cvt_item!($items)),*])
        }};
    }

    fn unindentify(input: &[T]) -> (Vec<T>, Vec<Indent>) {
        let mut output = Vec::new();
        let mut indents = Vec::new();

        let mut cur_indent = Indent::default();

        // loop over tokens, incrementing indent level when we see an indent, and decrementing when we see a dedent
        // when we see a newline, we add the current indent level to the indents list
        // also remove indent/dedent tokens since the parser doesn't need them.

        for &tok in input {
            match tok {
                T::Newline => {
                    indents.push(cur_indent);
                    output.push(tok);
                }
                T::Indent => {
                    cur_indent.num_spaces += 1;
                }
                T::Dedent => {
                    cur_indent.num_spaces -= 1;
                }
                _ => {
                    output.push(tok);
                }
            }
        }

        (output, indents)
    }

    #[track_caller]
    fn expr_test(kinds: &[T], expected: Tree) {
        let (kinds, indents) = unindentify(kinds);

        let mut state = State::from_tokens(&kinds);
        state.buf.indents = indents;
        state.start_expr(ExprCfg::default());
        state.pump();
        state.assert_end();

        assert_eq!(&state.tree.debug_vis_grouping(), &expected.debug_vis_grouping());
    }

    #[test]
    fn test_simple_ident() {
        expr_test(
            &[T::LowerIdent],
            expect!(Ident),
        );
    }

    #[test]
    fn test_simple_apply() {
        expr_test(
            &[T::LowerIdent, T::LowerIdent],
            expect!((Ident Ident) Apply),
        );
    }

    #[test]
    fn test_simple_binop_plus() {
        expr_test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!((Ident Ident) BinOpPlus),
        );
    }

    #[test]
    fn test_complex_apply() {
        expr_test(
            &[T::LowerIdent, T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!(((Ident Ident) Apply Ident) BinOpPlus),
        );
    }

    #[test]
    fn test_complex_binop_plus() {
        expr_test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent, T::LowerIdent],
            expect!((Ident (Ident Ident) Apply) BinOpPlus),
        );
    }

    #[test]
    fn test_nested_binop_plus() {
        expr_test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!(((Ident Ident) BinOpPlus Ident) BinOpPlus),
        );
    }

    #[test]
    fn test_multiple_ident() {
        expr_test(
            &[T::LowerIdent, T::LowerIdent, T::LowerIdent],
            expect!((Ident Ident Ident) Apply),
        );
    }

    #[test]
    fn test_pizza_operator() {
        expr_test(
            &[T::LowerIdent, T::OpPizza, T::LowerIdent, T::OpPizza, T::LowerIdent],
            expect!((Ident Ident Ident) Pizza),
        );
    }

    #[test]
    fn test_closure_expr() {
        expr_test(
            &[T::OpBackslash, T::LowerIdent, T::OpArrow, T::LowerIdent],
            expect!((Ident Ident) Closure),
        );
    }

    #[test]
    fn test_if() {
        expr_test(
            &[T::KwIf, T::LowerIdent, T::KwThen, T::LowerIdent, T::KwElse, T::LowerIdent],
            expect!((Ident Ident Ident) If),
        );
    }

    #[test]
    fn test_when() {
        expr_test(
            &[T::KwWhen, T::LowerIdent, T::KwIs, T::LowerIdent, T::OpArrow, T::LowerIdent],
            expect!((Ident Ident Ident) When),
        );
    }

    #[test]
    fn test_nested_when() {
        expr_test(
            &[
                T::KwWhen, T::LowerIdent, T::KwIs, T::Newline,
                    T::LowerIdent, T::OpArrow, T::Newline,
                        T::KwWhen, T::LowerIdent, T::KwIs, T::Newline,
                            T::Indent,
                            T::LowerIdent, T::OpArrow, T::LowerIdent, T::Newline,
                            T::Dedent,
            ],
            expect!((Ident Ident (BeginBlock (Ident Ident Ident) When) EndBlock) When),
        );
    }
    
    #[track_caller]
    fn decl_test(kinds: &[T], expected: Tree) {
        let (kinds, indents) = unindentify(kinds);
        let mut state = State::from_tokens(&kinds);
        state.buf.indents = indents;
        state.start_top_level_decls();
        state.pump();
        state.assert_end();

        // assert_eq!(&state.tree.kinds, &expected.kinds);
        // eprintln!("{:?}", state.tree.subtree_start_positions);
        // eprintln!("{:?}", expected.subtree_start_positions);
        assert_eq!(&state.tree.debug_vis_grouping(), &expected.debug_vis_grouping());
    }

    #[test]
    fn test_simple_assign_decl() {
        decl_test(
            &[T::LowerIdent, T::OpAssign, T::LowerIdent],
            expect!((BeginTopLevelDecls (Ident Ident) Assign) EndTopLevelDecls),
        );
    }

    #[test]
    fn test_double_assign_decl() {
        decl_test(
            &[T::LowerIdent, T::OpAssign, T::LowerIdent, T::Newline, T::LowerIdent, T::OpAssign, T::LowerIdent],
            expect!((BeginTopLevelDecls (Ident Ident) Assign (Ident Ident) Assign) EndTopLevelDecls),
        );
    }

    #[test]
    fn test_simple_nested_assign_decl() {
        decl_test(
            &[
                T::LowerIdent, T::OpAssign, T::Newline,
                    T::LowerIdent, T::OpAssign, T::LowerIdent, T::Newline,
                    T::LowerIdent],
            expect!(
                (
                    BeginTopLevelDecls
                    (Ident
                        (BeginBlock (Ident Ident) Assign Ident) EndBlock)
                        Assign)
                EndTopLevelDecls),
        );
    }

    #[test]
    fn test_decl_then_top_level_expr() {
        decl_test(
            &[
                T::LowerIdent, T::OpAssign, T::Newline,
                    T::LowerIdent, T::Newline,
                T::LowerIdent], // Note we really should error on the top-level expr
            expect!((BeginTopLevelDecls (Ident (BeginBlock Ident) EndBlock) Assign Ident) EndTopLevelDecls),
        );
    }

    #[test]
    fn test_double_nested_decl() {
        /*
        a =
            b =
                c
            d
        */
        decl_test(
            &[
                T::LowerIdent, T::OpAssign, T::Newline,
                    T::LowerIdent, T::OpAssign, T::Newline,
                        T::LowerIdent, T::Newline,
                    T::LowerIdent],
            expect!((BeginTopLevelDecls
                    (Ident (BeginBlock (Ident (BeginBlock Ident) EndBlock) Assign Ident) EndBlock) Assign
                ) EndTopLevelDecls),
        );
    }

    #[test]
    fn test_double_assign_block_decl() {
        decl_test(
            &[
                T::LowerIdent, T::OpAssign, T::Newline, T::LowerIdent,
                T::Newline,
                T::LowerIdent, T::OpAssign, T::Newline, T::LowerIdent],
            expect!(
                (
                    BeginTopLevelDecls
                    (Ident (BeginBlock Ident) EndBlock) Assign
                    (Ident (BeginBlock Ident) EndBlock) Assign
                ) EndTopLevelDecls
            ),
        );
    }

    #[test]
    fn test_closure_decl() {
        decl_test(
            &[T::LowerIdent, T::OpAssign, T::OpBackslash, T::LowerIdent, T::OpArrow, T::LowerIdent],
            expect!((BeginTopLevelDecls (Ident (Ident Ident) Closure) Assign) EndTopLevelDecls),
        );
    }
}