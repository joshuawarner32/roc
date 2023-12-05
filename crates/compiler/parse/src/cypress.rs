#![allow(dead_code)]
#![allow(unused)]

use std::collections::{VecDeque, btree_map::Keys};

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
    paird_group_ends: Vec<u32>,
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

    /// Lowercase identifier, e.g. `foo`
    Ident,

    /// An underscore, e.g. `_` or `_x`
    Underscore,

    /// The "crash" keyword
    Crash,

    /// The special dbg keyword, as in `dbg x`
    Dbg,

    /// Tag, e.g. `Foo`
    Tag,

    /// Reference to an opaque type, e.g. @Opaq
    OpaqueRef,

    /// Look up exactly one field on a record or tuple, e.g. `x.foo` or `x.0`.
    Access,

    /// e.g. `.foo` or `.0`
    AccessorFunction,

    /// List literals, e.g. `[1, 2, 3]`
    BeginList, EndList,

    /// Record literals, e.g. `{ x: 1, y: 2 }`
    BeginRecord, EndRecord,

    /// Record updates (e.g. `{ x & y: 3 }`)
    BeginRecordUpdate, EndRecordUpdate,

    /// Parentheses, e.g. `(1 + 2)`
    BeginParens, EndParens,

    /// Tuple literals, e.g. `(1, 2)`
    BeginTuple, EndTuple,

    /// Indented block of statements and expressions
    BeginBlock, EndBlock,

    /// Function application, e.g. `f x`
    InlineApply, EndApply,

    /// Pizza operator, e.g. `x |> f`
    InlinePizza, EndPizza,

    /// Assignment declaration, e.g. `x = 1`
    BeginAssign, InlineAssign, EndAssign,

    /// Binary operators, e.g. `x + y`
    InlineBinOpPlus, EndBinOpPlus,
    InlineBinOpStar, EndBinOpStar,

    /// Unary operator, e.g. `-x`
    BeginUnaryOp, EndUnaryOp,

    /// If expression, e.g. `if x then y else z`
    BeginIf, EndIf,

    /// A when expression, e.g. `when x is y -> z` (you need a newline after the 'is')
    BeginWhen, EndWhen,

    /// A lambda expression, e.g. `\x -> x`
    BeginLambda, EndLambda,

    ArgList,
    EndTopLevelDecls,
    BeginTopLevelDecls,
    Dummy,
    
    HintExpr, // EndExpr,
}

impl N {
    fn is_decl(self) -> bool {
        match self {
            N::EndAssign => true,
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
            paird_group_ends: Vec::new(),
        }
    }

    fn len(&self) -> u32 {
        self.kinds.len() as u32
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
            BinOp::Pizza => N::EndPizza,
            BinOp::Apply => N::EndApply,
            BinOp::Plus => N::EndBinOpPlus,
            BinOp::Minus => N::EndBinOpStar,
            BinOp::Assign => N::EndAssign,
            BinOp::AssignBlock => N::EndAssign,
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
    FinishAssign,
    ContinueTopLevel { num_found: i32 },
    ContinueLambdaArgs,
    ContinueIf { next: IfState },
    ContinueWhen { next: WhenState },
    FinishLambda,
    FinishBlockItem,
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

    fn update_end(&mut self, kind: N, subtree_start: u32) {
        eprintln!("{:indent$}@{} updating end {} -> {}", "", self.pos, subtree_start, self.tree.len(), indent = 2 * self.frames.len() + 2);
        assert_eq!(self.tree.kinds[subtree_start as usize], kind);
        self.tree.paird_group_ends[subtree_start as usize] = self.tree.len() as u32;
    }

    fn push_node(&mut self, kind: N, subtree_start: Option<u32>) {
        eprintln!("{:indent$}@{} pushing kind {}:{:?} starting at {:?}", "", self.pos, self.tree.kinds.len(), kind, subtree_start, indent = 2 * self.frames.len() + 2);
        self.tree.kinds.push(kind);
        let pos = subtree_start.unwrap_or(self.tree.paird_group_ends.len() as u32);
        self.tree.paird_group_ends.push(pos);
        // eprintln!("{:indent$}tree: {:?}", "", self.tree.debug_vis_grouping(), indent = 2 * self.frames.len() + 4);
    }

    fn push_next_frame(&mut self, subtree_start: u32, cfg: ExprCfg, frame: Frame) {
        eprintln!("{:indent$}@{} pushing frame {:?}", "", self.pos, frame, indent = 2 * self.frames.len() + 2);
        self.frames.push((subtree_start, cfg, frame));
    }

    fn push_next_frame_starting_here(&mut self, cfg: ExprCfg, frame: Frame) {
        let subtree_start = self.tree.len();
        self.push_next_frame(subtree_start, cfg, frame)
    }

    fn pump(&mut self) {
        while let Some((subtree_start, cfg, frame)) = self.frames.pop() {
            eprintln!("{:indent$}@{} pumping frame {:?} starting at {}", "", self.pos, frame, subtree_start, indent = 2 * self.frames.len());
            match frame {
                Frame::StartExpr { min_prec } => self.pump_start_expr(subtree_start, cfg, min_prec),
                Frame::FinishParen => self.pump_finish_paren(),
                Frame::FinishLambda => self.pump_finish_lambda(subtree_start),
                Frame::FinishBlockItem => self.pump_finish_block_item(subtree_start),
                Frame::ContinueExpr { min_prec, cur_op, num_found } => self.pump_continue_expr(subtree_start, cfg, min_prec, cur_op, num_found),
                Frame::ContinueBlock => self.pump_continue_block(subtree_start, cfg),
                Frame::FinishAssign => self.pump_finish_assign(subtree_start, cfg),
                Frame::ContinueTopLevel { num_found } => self.pump_continue_top_level(subtree_start, cfg, num_found),
                Frame::ContinueLambdaArgs => self.pump_continue_lambda_args(subtree_start, cfg),
                Frame::ContinueIf { next } => self.pump_continue_if(subtree_start, cfg, next),
                Frame::ContinueWhen { next } => self.pump_continue_when(subtree_start, cfg, next),
            }
        }
    }

    fn pump_start_expr(&mut self, subtree_start: u32, cfg: ExprCfg, mut min_prec: Prec) {
        loop {
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
                    self.push_next_frame_starting_here(cfg, Frame::ContinueLambdaArgs);
                    self.push_node(N::BeginLambda, None);
                    return;
                }
                Some(T::KwIf) => {
                    self.bump();
                    self.push_next_frame_starting_here(cfg, Frame::ContinueIf { next: IfState::Then });
                    self.push_node(N::BeginIf, None);
                    self.start_expr(cfg);
                    return;
                }
                Some(T::KwWhen) => {
                    self.bump();
                    self.push_next_frame_starting_here(cfg, Frame::ContinueWhen { next: WhenState::Is });
                    self.push_node(N::BeginWhen, None);
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
                self.push_next_frame(subtree_start, cfg, Frame::FinishAssign);
                self.start_block(cfg);
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

    fn pump_finish_lambda(&mut self, subtree_start: u32) {
        self.push_node(N::EndLambda, Some(subtree_start));
        self.update_end(N::BeginLambda, subtree_start);
    }

    fn pump_finish_block_item(&mut self, subtree_start: u32) {
        let k = self.tree.kinds.last().copied().unwrap(); // Find out what we just parsed
        match k {
            N::EndAssign =>{
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginAssign;
            }
            N::Ident | N::EndWhen | N::EndIf => {
                self.tree.kinds[subtree_start as usize] = N::HintExpr;
            }
            k => todo!("{:?}", k),
        };
    }

    fn pump_continue_block(&mut self, subtree_start: u32, cfg: ExprCfg) {
        while self.consume_newline() {}

        // need to inspect the expr we just parsed.
        // if it's a decl we keep going; if it's not, we're done.
        if self.tree.kinds.last().copied().unwrap().is_decl() {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueBlock);
            self.start_block_item(cfg);
        } else {
            self.push_node(N::EndBlock, Some(subtree_start));
            self.update_end(N::BeginBlock, subtree_start);
        }
    }

    fn pump_finish_assign(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::EndAssign, Some(subtree_start));
    }

    fn pump_continue_top_level(&mut self, subtree_start: u32, cfg: ExprCfg, num_found: i32) {
        // keep parsing decls until the end
        while self.consume_newline() {}

        if self.pos < self.buf.kinds.len() {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueTopLevel { num_found: num_found + 1 });
            self.start_top_level_item();
        } else {
            self.push_node(N::EndTopLevelDecls, Some(subtree_start as u32));
            self.update_end(N::BeginTopLevelDecls, subtree_start);
        }
    }

    fn pump_continue_lambda_args(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::OpArrow) {
            self.push_next_frame(subtree_start, cfg, Frame::FinishLambda);
            self.start_block_or_expr(cfg);
        } else {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueLambdaArgs);
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
                self.push_node(N::EndIf, Some(subtree_start));
                self.update_end(N::BeginIf, subtree_start);
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
                self.push_node(N::EndWhen, Some(subtree_start));
                self.update_end(N::BeginWhen, subtree_start);
            }
            WhenState::BranchArrow(indent) => {
                self.expect(T::OpArrow);
                self.push_next_frame(subtree_start, cfg, Frame::ContinueWhen { next: WhenState::BranchPattern(indent) });
                self.start_block_or_expr(cfg);
            }
        }
    }

    fn start_top_level_item(&mut self) {
        self.push_next_frame_starting_here(ExprCfg::default(), Frame::FinishBlockItem);
        self.start_expr(ExprCfg::default());
        self.push_node(N::Dummy, None); // will be replaced by the actual node in pump_finish_block_item
    }

    fn start_top_level_decls(&mut self) {
        self.push_next_frame_starting_here(ExprCfg::default(), Frame::ContinueTopLevel { num_found: 1 });
        self.push_node(N::BeginTopLevelDecls, None);
        self.start_top_level_item();
    }

    fn start_block_or_expr(&mut self, cfg: ExprCfg) {
        if self.consume_newline() {
            self.start_block(cfg);
        } else {
            self.start_expr(cfg);
        }
    }

    fn start_expr(&mut self, cfg: ExprCfg) {
        self.push_next_frame_starting_here(cfg, Frame::start_expr());
    }

    fn start_block_item(&mut self, cfg: ExprCfg) {
        self.push_next_frame_starting_here(cfg, Frame::FinishBlockItem);
        self.start_expr(cfg);
        self.push_node(N::Dummy, None); // will be replaced by the actual node in pump_finish_block_item
    }

    fn start_block(&mut self, cfg: ExprCfg) {
        self.push_next_frame_starting_here(cfg, Frame::ContinueBlock);
        self.push_node(N::BeginBlock, None);
        self.start_block_item(cfg);
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

fn pretty(tree: &Tree, buf: &TokenenizedBuffer) -> FormattedBuffer {
    let mut buf = FormattedBuffer::new();
    
    let mut has_newline = Vec::with_capacity(tree.len() as usize);

    for (i, &node) in tree.kinds.iter().enumerate() {
        let hn = match node {
            N::Underscore | N::Float | N::Num | N::Str | N::SingleQuote => false,
            _ => todo!(),
        };

        has_newline.push(hn);
    }
    


    for (i, &node) in tree.kinds.iter().enumerate() {
        match node {
            N::Ident => {
                buf.kinds.push_back(T::LowerIdent);
            }
            N::BeginTopLevelDecls => {}
            N::EndTopLevelDecls => {}
            N::HintExpr => {}
            N::BeginIf => {
                buf.kinds.push_back(T::KwIf);
            }
            _ => todo!("{:?}", node),
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format_test(kinds: &[T]) {
        let (unindented_kinds, indents) = unindentify(kinds);

        let mut state = State::from_tokens(&unindented_kinds);
        state.buf.indents = indents;
        state.start_top_level_decls();
        state.pump();
        state.assert_end();
        
        let res = pretty(&state.tree, &state.buf);

        assert_eq!(res.kinds, kinds);
    }

    // #[test]
    // fn test_format_simple() {
    //     format_test(&[T::LowerIdent]);
    // }

    // #[test]
    // fn test_format_if() {
    //     format_test(&[
    //         T::KwIf, T::LowerIdent, T::KwThen, T::Newline,
    //             T::Indent, T::LowerIdent, T::Newline, T::Dedent,
    //         T::KwElse, T::Newline,
    //             T::Indent, T::LowerIdent, T::Newline, T::Dedent,
    //     ]);
    // }

    impl Tree {
        fn to_expect_atom(&self) -> ExpectAtom {
            let (i, atom) = self.expect_atom(self.kinds.len());
            assert_eq!(i, 0);
            atom
        }
        
        fn expect_atom(&self, end: usize) -> (u32, ExpectAtom) {
            let index = self.paird_group_ends[end - 1];
            if index as usize >= end - 1 {
                (index, ExpectAtom::Unit(self.kinds[end - 1]))
            } else {
                let mut res = VecDeque::new();
                res.push_front(ExpectAtom::Unit(self.kinds[end - 1]));
                let begin = index as usize;

                let mut i = end - 1;

                while i > begin {
                    let (i2, atom) = self.expect_atom(i);
                    println!("{}: {:?}", i2, atom);
                    if i2 as usize >= i {
                        res.push_front(atom);
                        i = i - 1;
                    } else {
                        assert!(i2 as usize >= begin);
                        i = i2 as usize;
                        res.push_front(atom);
                        if i == begin {
                            res.push_front(ExpectAtom::Empty);
                        }
                    }
                }

                (begin as u32, ExpectAtom::Seq(res.into()))
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ExpectAtom {
        Seq(Vec<ExpectAtom>),
        Unit(N),
        Empty,
    }

    impl ExpectAtom {
        fn debug_vis(&self) -> String {
            match self {
                ExpectAtom::Seq(items) => format!("({})", items.iter().map(|i| i.debug_vis()).collect::<Vec<_>>().join(" ")),
                ExpectAtom::Unit(kind) => format!("{:?}", kind),
                ExpectAtom::Empty => format!("*"),
            }
        }
    }

    struct ExpectBuilder {
        kinds: Vec<N>,
        paird_group_ends: Vec<u32>,
    }

    impl ExpectBuilder {
        fn new() -> ExpectBuilder {
            ExpectBuilder {
                kinds: Vec::new(),
                paird_group_ends: Vec::new(),
            }
        }

        fn consume_items(&mut self, items: &[ExpectAtom]) {
            assert!(items.len() > 0);
            assert!(!matches!(items.first().unwrap(), ExpectAtom::Seq(_)));
            assert!(!matches!(items.last().unwrap(), ExpectAtom::Seq(_)));

            let start = self.kinds.len();

            for item in items {
                match item {
                    ExpectAtom::Seq(items) => self.consume_items(&items),
                    ExpectAtom::Unit(kind) => {
                        self.kinds.push(*kind);
                        self.paird_group_ends.push(self.paird_group_ends.len() as u32);
                    }
                    ExpectAtom::Empty => {}
                }
            }

            let end = self.kinds.len();

            if matches!(items.first().unwrap(), ExpectAtom::Unit(_) ) {
                self.paird_group_ends[start] = end as u32;
            }

            if matches!(items.last().unwrap(), ExpectAtom::Unit(_) ) {
                self.paird_group_ends[end - 1] = start as u32;
            }
        }

        fn finish(mut self) -> Tree {
            Tree {
                kinds: self.kinds,
                paird_group_ends: self.paird_group_ends,
                tokens: Vec::new(), // TODO
            }
        }
    }

    fn build_expect(items: Vec<ExpectAtom>) -> Tree {
        let mut b = ExpectBuilder::new();
        
        b.consume_items(&items);

        let t = b.finish();

        let reconstituted = t.to_expect_atom();
        match reconstituted {
            ExpectAtom::Seq(new_items) => assert_eq!(items, new_items),
            ExpectAtom::Unit(item) => assert_eq!(items, vec![ExpectAtom::Unit(item)]),
            ExpectAtom::Empty => panic!(),
        };

        t
    }

    macro_rules! cvt_item {
        (*)                 => { ExpectAtom::Empty };
        ($item:ident)       => { ExpectAtom::Unit(N::$item) };
        (($($items:tt)*))   => { ExpectAtom::Seq(vec![$(cvt_item!($items)),*]) };
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

        assert_eq!(state.tree.to_expect_atom().debug_vis(), expected.to_expect_atom().debug_vis());
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
            expect!(* Ident Ident EndApply),
        );
    }

    #[test]
    fn test_simple_binop_plus() {
        expr_test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!(* Ident Ident EndBinOpPlus),
        );
    }

    #[test]
    fn test_complex_apply() {
        expr_test(
            &[T::LowerIdent, T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!(* (* Ident Ident EndApply) Ident EndBinOpPlus),
        );
    }

    #[test]
    fn test_complex_binop_plus() {
        expr_test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent, T::LowerIdent],
            expect!(* Ident (* Ident Ident EndApply) EndBinOpPlus),
        );
    }

    #[test]
    fn test_nested_binop_plus() {
        expr_test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!(* (* Ident Ident EndBinOpPlus) Ident EndBinOpPlus),
        );
    }

    #[test]
    fn test_multiple_ident() {
        expr_test(
            &[T::LowerIdent, T::LowerIdent, T::LowerIdent],
            expect!(* Ident Ident Ident EndApply),
        );
    }

    #[test]
    fn test_pizza_operator() {
        expr_test(
            &[T::LowerIdent, T::OpPizza, T::LowerIdent, T::OpPizza, T::LowerIdent],
            expect!(* Ident Ident Ident EndPizza),
        );
    }

    #[test]
    fn test_lambda_expr() {
        expr_test(
            &[T::OpBackslash, T::LowerIdent, T::OpArrow, T::LowerIdent],
            expect!(BeginLambda Ident Ident EndLambda),
        );
    }

    #[test]
    fn test_if() {
        expr_test(
            &[T::KwIf, T::LowerIdent, T::KwThen, T::LowerIdent, T::KwElse, T::LowerIdent],
            expect!(BeginIf Ident Ident Ident EndIf),
        );
    }

    #[test]
    fn test_when() {
        expr_test(
            &[T::KwWhen, T::LowerIdent, T::KwIs, T::LowerIdent, T::OpArrow, T::LowerIdent],
            expect!(BeginWhen Ident Ident Ident EndWhen),
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
            expect!(BeginWhen Ident Ident (BeginBlock HintExpr (BeginWhen Ident Ident Ident EndWhen) EndBlock) EndWhen),
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
        // eprintln!("{:?}", state.tree.paird_group_ends);
        // eprintln!("{:?}", expected.paird_group_ends);
        assert_eq!(state.tree.to_expect_atom().debug_vis(), expected.to_expect_atom().debug_vis());
    }

    #[test]
    fn test_simple_assign_decl() {
        decl_test(
            &[T::LowerIdent, T::OpAssign, T::LowerIdent],
            expect!(BeginTopLevelDecls (BeginAssign Ident Ident EndAssign) EndTopLevelDecls),
        );
    }

    #[test]
    fn test_double_assign_decl() {
        decl_test(
            &[T::LowerIdent, T::OpAssign, T::LowerIdent, T::Newline, T::LowerIdent, T::OpAssign, T::LowerIdent],
            expect!(BeginTopLevelDecls (BeginAssign Ident Ident EndAssign) (BeginAssign Ident Ident EndAssign) EndTopLevelDecls),
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
                BeginTopLevelDecls
                    (BeginAssign
                        Ident
                        (BeginBlock (BeginAssign Ident Ident EndAssign) HintExpr Ident EndBlock)
                    EndAssign)
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
            expect!(BeginTopLevelDecls (BeginAssign Ident (BeginBlock HintExpr Ident EndBlock) EndAssign) HintExpr Ident EndTopLevelDecls),
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
            expect!(
                BeginTopLevelDecls
                    (BeginAssign Ident (BeginBlock (BeginAssign Ident (BeginBlock HintExpr Ident EndBlock) EndAssign) HintExpr Ident EndBlock) EndAssign)
                EndTopLevelDecls),
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
                BeginTopLevelDecls
                    (BeginAssign Ident (BeginBlock HintExpr Ident EndBlock) EndAssign)
                    (BeginAssign Ident (BeginBlock HintExpr Ident EndBlock) EndAssign)
                EndTopLevelDecls
            ),
        );
    }

    #[test]
    fn test_lambda_decl() {
        decl_test(
            &[T::LowerIdent, T::OpAssign, T::OpBackslash, T::LowerIdent, T::OpArrow, T::LowerIdent],
            expect!(BeginTopLevelDecls (BeginAssign Ident (BeginLambda Ident Ident EndLambda) EndAssign) EndTopLevelDecls),
        );
    }
}