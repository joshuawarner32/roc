//! Cypress: an experimental (and very incomplete) parser for the Roc language.
//! Inspired by the Carbon language parser, described here: https://docs.google.com/document/d/1RRYMm42osyqhI2LyjrjockYCutQ5dOf8Abu50kTrkX0/edit?resourcekey=0-kHyqOESbOHmzZphUbtLrTw
//! Cypress is more-or-less a recursive descent parser, but written in a very unusual style, and outputting an unusual AST.
//!
//! Notably, the tree we produce is made of two vecs: one vec of N (node) enums - just u8's basically - and one vec of some arbitrary index data, the interpretation of which varies by node type.
//! Here's an example of what the tree looks like:
//!
//! Original text: `f = \x -> x + 1`
//! Tokens:            LowerIdent,   OpAssign,   OpBackslash, LowerIdent,  OpArrow,   LowerIdent,    OpPlus,   Num
//!  Tree:                  ^            ^                         ^           ^            ^           ^       ^
//!             ____________|____________|_________________________|___________|____________|___________|_______|_________________
//!            /            |            |            _____________|___________|____________|___________|_______|_______          \
//!           v             |            |           V             |           |            |           |       |       V          V
//! Nodes: BeginAssign, LowerIdent, InlineAssign, BeginLambda, LowerIdent, InlineArrow, LowerIdent, InlinePlus, Num, EndLambda, EndAssign
//!
//! The nodes are a flat list of nodes, and the arrows represent what each index is referring to.
//! The indices associated with BeginAssign/EndAssign and BeginLambda/EndLambda point to each other, because they're paired groups.
//! The indicies for most of the remaining nodes refer back to the token buffer - and specifically to the token they come from.
//!
//! This is a very unusual way to represent a tree, but it has some nice properties:
//! * It's very compact. The tree is just two vecs - five bytes per node.
//!     (note however that in order to make the tree easy to traverse, we add some redundant nodes - so it's harder to directly compare to other tree encodings)
//! * It's very fast to traverse. We can traverse the tree without any indirection - usually by walking in a straight line through the vecs.
//! * We still have enough information to skip around the tree if we want - for example, in order to collect a hash map of the identifiers defined at the top level.
//! * It's very fast to build. The current parser doesn't take great advantage of this - but you could imagine a future where we fast-path the common case of a sequence of lower idents and use SIMD ops to splat the nodes and indices into the tree.
//!
//! Note that this tree isn't really designed to work well with the later stages of compiling.
//! We'll be able to do some simple things like:
//! * name resolution
//! * syntax desugaring
//! * canonicalization
//!
//! ... but we won't be able to do more complex things like type inference/checking, code generation, etc.
//!
//! To facilitate this, we'll do name resolution / desugaring / canonicalization in a single pass over the tree,
//! and then convert it to a more traditional IR representation for the later parts of the compiler.
//!
//! Notably however, interactive operations like formatting can be done without converting to a different representation.
//!
//! ------------------------------------------------------------------
//!
//! The other unusual aspect of the parser is that it's written in a style where we reify the parser state as a stack of frames, instead of using recursion.
//!
//! Which is to say, we take a function like the following:
//!
//! ```ignore
//! fn parse_if() -> Result<If, Error> {
//!    expect(T::KwIf)?; // start state
//!    let cond = parse_expr()?;
//!    expect(T::KwThen)?; // where we'll return to after parsing the condition
//!    let then = parse_expr()?;
//!    expect(T::KwElse)?; // where we'll return to after parsing the then branch
//!    let else_ = parse_expr()?;
//!    Ok(If { cond, then, else }) // where we'll return to after parsing the else branch
//! }
//! ```
//!
//! And we rewrite it to a series of enum variants that look something like:
//!
//! ```ignore
//! enum Frame {
//!     StartIf, // the initial state, at the start of the function
//!     ContinueAtThenKw, // the return point after parsing the condition
//!     ContinueAtElseKw, // the return point after parsing the then branch
//!     FinishIf, // the return point after parsing the else branch
//!
//!     // ... other frames for other constructs
//!     StartExpr,
//! }
//! ```
//! (this is a lie in several ways; but it's close enough for now)
//!
//! The parser is then just a loop:
//! * pop the top frame off the stack
//! * call the corresponding `pump` function
//!
//! In the `pump_*` functions, each time we want to call a parsing "function" recursively (e.g. `parse_expr()`), we:
//! * push the state for the return pos onto the stack, e.g. `ContinueAtThenKw`
//! * push the state for the start of the function onto the stack, e.g. `StartExpr`
//!
//! In practice however, we can short-circuit the enum variant + loop iteration in a few cases. e.g. We don't technically need a `StartIf` frame,
//! since we can immediately (1) check there's a `KwIf` token, (2) parse the condition, (3) push the `ContinueAtThenKw` frame, and (4) push the `StartExpr` frame.
//!
//!
//! Overall this design has some advantages:
//! * When an error occurs, we can simply copy the enums representing stack frames and directly use those for generating error messages. We don't need any other mechanism for tracking the context around an error.
//! * (minor) This allows us to handle much deeper trees, since we don't have to worry about blowing the stack.
//! * (minor) We also consume less memory for deep trees (the explicit stack can be more compact)
//! * (in the future...) We can use the fact that the stack is reified to do some interesting things, like:
//!     * Store the stack at the location of the user's cursor in the editor, and use that to incrementally re-parse the tree as the user types.
//!     * Build debugging visualizations of the parser state, without having to resort to a debugger.
//!
//! That said, I haven't benchmarked this, and I don't really have a good intuition for how it compares to a normal recursive descent parser.
//! On the one hand, we can in principle reduce overall memory traffic associated with stack frame setup, and on the other hand, we're not able to take advantage of the return address branch prediction in the CPU.

#![allow(dead_code)] // temporarily during development
#![allow(unused)] // temporarily during development

use crate::token::{Comment, Indent, TokenenizedBuffer, T};
use crate::tree::{NodeIndexKind, Tree, N};
use std::fmt;
use std::{
    collections::{btree_map::Keys, VecDeque},
    f32::consts::E,
};

impl TokenenizedBuffer {
    pub fn kind(&self, pos: usize) -> Option<T> {
        self.kinds.get(pos).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TypePrec {
    WhereClauses,
    As,
    Lambda,
    Apply,
    Atom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Prec {
    Outer,
    Decl,                  // BinOp::Assign, BinOp::Backpassing,
    MultiBackpassingComma, // BinOp::MultiBackpassingComma. Used for parsing the comma in `x, y, z <- foo`
    Pizza,                 // BinOp::Pizza,
    AndOr,                 // BinOp::And, BinOp::Or,
    Compare, // BinOp::Equals, BinOp::NotEquals, BinOp::LessThan, BinOp::GreaterThan, BinOp::LessThanOrEq, BinOp::GreaterThanOrEq,
    Add,     // BinOp::Plus, BinOp::Minus,
    Multiply, // BinOp::Star, BinOp::Slash, BinOp::DoubleSlash, BinOp::Percent,
    Exponent, // BinOp::Caret
    Apply,
    Atom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    // shouldn't be pub. only pub because of canfmt. should just use a different enum.
    AssignBlock,
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
    DefineTypeOrTypeAlias,
    DefineOtherTypeThing,
    MultiBackpassingComma,
    Implements,
    As,
}

impl Prec {
    fn next(self) -> Prec {
        match self {
            Prec::Outer => Prec::MultiBackpassingComma,
            Prec::MultiBackpassingComma => Prec::Pizza,
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
            BinOp::AssignBlock
            | BinOp::Assign
            | BinOp::As // is this right?
            | BinOp::DefineTypeOrTypeAlias
            | BinOp::DefineOtherTypeThing
            | BinOp::Backpassing
            | BinOp::Implements => Prec::Decl,
            BinOp::MultiBackpassingComma => Prec::MultiBackpassingComma,
            BinOp::Apply => Prec::Apply,
            BinOp::Caret => Prec::Exponent,
            BinOp::Star | BinOp::Slash | BinOp::DoubleSlash | BinOp::Percent => Prec::Multiply,
            BinOp::Plus | BinOp::Minus => Prec::Add,
            BinOp::Equals
            | BinOp::NotEquals
            | BinOp::LessThan
            | BinOp::GreaterThan
            | BinOp::LessThanOrEq
            | BinOp::GreaterThanOrEq => Prec::Compare,
            BinOp::And | BinOp::Or => Prec::AndOr,
            BinOp::Pizza => Prec::Pizza,
        }
    }

    fn grouping_assoc(self) -> Assoc {
        match self {
            BinOp::AssignBlock => Assoc::Right,
            BinOp::Assign
            | BinOp::As // is this right?
            | BinOp::Backpassing
            | BinOp::DefineTypeOrTypeAlias
            | BinOp::DefineOtherTypeThing
            | BinOp::MultiBackpassingComma
            | BinOp::Implements => Assoc::Right,
            BinOp::Apply => Assoc::Left,
            BinOp::Caret => Assoc::Right,
            BinOp::Star | BinOp::Slash | BinOp::DoubleSlash | BinOp::Percent => Assoc::Left,
            BinOp::Plus | BinOp::Minus => Assoc::Left,
            BinOp::Equals
            | BinOp::NotEquals
            | BinOp::LessThan
            | BinOp::GreaterThan
            | BinOp::LessThanOrEq
            | BinOp::GreaterThanOrEq => Assoc::NonAssociative,
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
            BinOp::Apply | BinOp::Pizza => true,
            _ => false,
        }
    }

    fn to_inline(&self) -> N {
        match self {
            BinOp::AssignBlock => todo!(),
            BinOp::As => N::InlineKwAs,
            BinOp::Assign => N::InlineAssign,
            BinOp::Implements => N::InlineAbilityImplements,
            BinOp::DefineTypeOrTypeAlias => N::InlineTypeColon,
            BinOp::DefineOtherTypeThing => N::InlineTypeColonEqual,
            BinOp::Backpassing => N::InlineBackArrow,
            BinOp::MultiBackpassingComma => N::InlineMultiBackpassingComma,
            BinOp::Pizza => N::InlinePizza,
            BinOp::And => N::InlineBinOpAnd,
            BinOp::Or => N::InlineBinOpOr,
            BinOp::Equals => N::InlineBinOpEquals,
            BinOp::NotEquals => N::InlineBinOpNotEquals,
            BinOp::LessThan => N::InlineBinOpLessThan,
            BinOp::GreaterThan => N::InlineBinOpGreaterThan,
            BinOp::LessThanOrEq => N::InlineBinOpLessThanOrEq,
            BinOp::GreaterThanOrEq => N::InlineBinOpGreaterThanOrEq,
            BinOp::Plus => N::InlineBinOpPlus,
            BinOp::Minus => N::InlineBinOpMinus,
            BinOp::Star => N::InlineBinOpStar,
            BinOp::Slash => N::InlineBinOpSlash,
            BinOp::DoubleSlash => N::InlineBinOpDoubleSlash,
            BinOp::Percent => N::InlineBinOpPercent,
            BinOp::Caret => N::InlineBinOpCaret,
            BinOp::Apply => N::InlineApply,
        }
    }
}

impl From<BinOp> for N {
    fn from(op: BinOp) -> Self {
        match op {
            BinOp::Pizza => N::EndPizza,
            BinOp::Apply => N::EndApply,
            BinOp::Plus => N::EndBinOpPlus,
            BinOp::Minus => N::EndBinOpMinus,
            BinOp::Star => N::EndBinOpStar,

            BinOp::LessThan => N::EndBinOpLessThan,
            BinOp::GreaterThan => N::EndBinOpGreaterThan,
            BinOp::LessThanOrEq => N::EndBinOpLessThanOrEq,
            BinOp::GreaterThanOrEq => N::EndBinOpGreaterThanOrEq,
            BinOp::Assign => N::EndAssign,
            BinOp::AssignBlock => N::EndAssign,
            BinOp::Backpassing => N::EndBackpassing,
            BinOp::MultiBackpassingComma => N::EndMultiBackpassingArgs,
            BinOp::Slash => N::EndBinOpSlash,
            BinOp::DoubleSlash => N::EndBinOpDoubleSlash,
            BinOp::Percent => N::EndBinOpPercent,
            BinOp::Caret => N::EndBinOpCaret,
            BinOp::And => N::EndBinOpAnd,
            BinOp::Or => N::EndBinOpOr,
            BinOp::Equals => N::EndBinOpEquals,
            BinOp::NotEquals => N::EndBinOpNotEquals,
            BinOp::As => N::EndPatternAs,
            _ => todo!("binop to node {:?}", op),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum IfState {
    Then,
    Else,
    End,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum WhenState {
    Is,
    BranchPattern(Indent),
    BranchBarOrArrow(Indent),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Frame {
    StartExpr {
        min_prec: Prec,
    },
    ContinueExpr {
        min_prec: Prec,
        cur_op: Option<BinOp>,
        num_found: usize,
    },
    StartPattern,
    ContinuePattern,
    ContinueBlock,
    ContinueExprTupleOrParen,
    ContinuePatternTupleOrParen,
    FinishAssign,
    ContinueTopLevel {
        num_found: i32,
    },
    ContinueLambdaArgs,
    ContinueIf {
        next: IfState,
    },
    ContinueWhen {
        next: WhenState,
    },
    FinishLambda,
    FinishBlockItem,
    StartType {
        allow_clauses: bool,
        allow_commas: bool,
    },
    ContinueType {
        in_apply: Option<bool>,
        allow_clauses: bool,
        allow_commas: bool,
    },
    ContinueTypeCommaSep {
        allow_clauses: bool,
    },
    FinishTypeFunction,
    ContinueWhereClause,
    FinishTypeOrTypeAlias,
    ContinueRecord {
        start: bool,
    },
    ContinuePatternRecord {
        start: bool,
    },
    PushEndOnly(N),
    ContinueTypeTupleOrParen,
    ContinueExprList,
    ContinuePatternList,
    ContinueTypeTagUnion,
    ContinueTypeTagUnionArgs,
    ContinueImplementsMethodDecl,
    ContinueTypeRecord,
    PushEnd(N, N),
}

#[derive(Clone, Copy)]
struct ExprCfg {
    expr_indent_floor: Option<Indent>, // expression continuations must be indented more than this.
    block_indent_floor: Option<Indent>, // when branches must be indented more than this. None means no restriction.
    allow_multi_backpassing: bool,
}

impl fmt::Debug for ExprCfg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // shorthand for debugging

        write!(
            f,
            "{}",
            if self.allow_multi_backpassing {
                "mb"
            } else {
                "/"
            }
        )?;

        if let Some(i) = self.expr_indent_floor {
            write!(f, "e")?;
            write!(f, "{}.{}", i.num_spaces, i.num_tabs)?;
        }

        if let Some(i) = self.block_indent_floor {
            write!(f, "w")?;
            write!(f, "{}.{}", i.num_spaces, i.num_tabs)?;
        }

        Ok(())
    }
}

impl ExprCfg {
    fn disable_multi_backpassing(mut self) -> ExprCfg {
        ExprCfg {
            allow_multi_backpassing: false,
            ..self
        }
    }

    fn set_block_indent_floor(&self, cur_indent: Option<Indent>) -> ExprCfg {
        ExprCfg {
            block_indent_floor: cur_indent,
            ..*self
        }
    }

    fn set_expr_indent_floor(&self, cur_indent: Option<Indent>) -> ExprCfg {
        ExprCfg {
            expr_indent_floor: cur_indent,
            ..*self
        }
    }
}

impl Default for ExprCfg {
    fn default() -> Self {
        ExprCfg {
            expr_indent_floor: None,
            block_indent_floor: None, // for when branches and implements methods
            allow_multi_backpassing: true,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Error {
    InconsistentIndent,
    ExpectViolation(T, Option<T>),
    ExpectedExpr(Option<T>),
    ExpectedPattern(Option<T>),
    Todo(Option<T>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Message {
    pub kind: Error,
    pub frames: Vec<Frame>,
    pub pos: u32,
}

pub struct State {
    frames: Vec<(u32, ExprCfg, Frame)>,
    pub buf: TokenenizedBuffer,
    pos: usize,
    line: usize,

    pub tree: Tree,

    pumping: Option<Frame>,
    pub messages: Vec<Message>,
}

// wrapper around an eprintln that's only enabled in debug mode
macro_rules! debug_print {
    ($($arg:tt)*) => {
        if cfg!(debug_assertions) {
            eprintln!($($arg)*);
        }
    }

}

impl State {
    pub fn from_buf(buf: TokenenizedBuffer) -> Self {
        State {
            frames: vec![],
            buf: buf,
            pos: 0,
            line: 0,
            tree: Tree::new(),
            pumping: None,
            messages: vec![],
        }
    }

    fn cur(&self) -> Option<T> {
        self.buf.kind(self.pos)
    }

    fn peek_at(&self, offset: usize) -> Option<T> {
        self.buf.kind(self.pos + offset)
    }

    fn _update_line(&mut self) {
        while self.line + 1 < self.buf.lines.len()
            && self.pos >= self.buf.lines[self.line + 1].0 as usize
        {
            self.line += 1;
        }
        debug_assert!(self.pos >= self.buf.lines[self.line].0 as usize);
        if self.line + 1 < self.buf.lines.len() {
            debug_assert!(self.pos < self.buf.lines[self.line + 1].0 as usize);
        }
    }

    fn at_newline(&mut self) -> bool {
        self._update_line();
        self.pos == self.buf.lines[self.line].0 as usize && self.pos < self.buf.kinds.len()
    }

    fn cur_indent(&mut self) -> Indent {
        self._update_line();
        self.buf
            .lines
            .get(self.line)
            .copied()
            .map(|(_, i)| i)
            .unwrap_or(Indent::default())
    }

    fn at_terminator(&self) -> bool {
        matches!(
            self.cur(),
            None | Some(T::CloseRound) | Some(T::CloseSquare) | Some(T::CloseCurly)
        )
    }

    fn bump(&mut self) {
        debug_print!(
            "{:indent$}@{} consuming \x1b[32m{:?}\x1b[0m",
            "",
            self.pos,
            self.cur().unwrap(),
            indent = 2 * self.frames.len() + 2
        );

        self.pos += 1;
    }

    fn bump_n(&mut self, n: usize) {
        for _ in 0..n {
            self.bump();
        }
    }

    fn push_error(&mut self, kind: Error) {
        debug_print!(
            "{:indent$}@{} pushing error \x1b[31m{:?}\x1b[0m",
            "",
            self.pos,
            kind,
            indent = 2 * self.frames.len() + 2
        );

        let mut frames: Vec<Frame> = self.frames.iter().map(|(_, _, f)| *f).collect();
        if let Some(f) = self.pumping.clone() {
            frames.push(f);
        }
        self.messages.push(Message {
            kind,
            frames,
            pos: self.pos as u32,
        });
    }

    #[track_caller]
    fn expect(&mut self, tok: T) {
        if self.cur() == Some(tok) {
            self.bump();
        } else {
            self.push_error(Error::ExpectViolation(tok, self.cur()));
            self.fast_forward_past_newline();
        }
    }

    fn fast_forward_past_newline(&mut self) {
        if self.line + 1 < self.buf.lines.len() {
            self.line += 1;
            self.pos = self.buf.lines[self.line].0 as usize;
        } else {
            self.pos = self.buf.kinds.len();
        }
    }

    #[track_caller]
    fn expect_and_push_node(&mut self, tok: T, node: N) {
        self.expect(tok);
        self.push_node(node, Some(self.pos as u32 - 1));
    }

    fn consume(&mut self, tok: T) -> bool {
        if self.cur() != Some(tok) {
            return false;
        }
        self.bump();
        true
    }

    fn consume_and_push_node(&mut self, tok: T, node: N) -> bool {
        if self.consume(tok) {
            self.push_node(node, Some(self.pos as u32 - 1));
            return true;
        }
        false
    }

    // Like consume, except we will also return true and add a message if we're at EOF
    fn consume_end(&mut self, tok: T) -> bool {
        match self.cur() {
            None => {
                self.push_error(Error::ExpectViolation(tok, None));
                true // intentional! We want the outer scope to think we've consumed the end token!
            }
            Some(t) => {
                if t == tok {
                    self.bump();
                    true
                } else {
                    false
                }
            }
        }
    }

    fn consume_comma_terminator(&mut self, close: T) -> bool {
        if self.consume_end(close) {
            return true;
        }
        self.expect(T::Comma);
        if self.consume_end(close) {
            return true;
        }
        false
    }

    #[track_caller]
    fn consume_and_push_node_end(&mut self, tok: T, begin: N, end: N, subtree_start: u32) -> bool {
        if self.consume(tok) {
            self.push_node_end(begin, end, subtree_start);
            true
        } else {
            false
        }
    }

    #[track_caller]
    fn update_end(&mut self, kind: N, subtree_start: u32) {
        debug_print!(
            "{:indent$}@{} updating end {} -> {}",
            "",
            self.pos,
            subtree_start,
            self.tree.len(),
            indent = 2 * self.frames.len() + 2
        );
        assert_eq!(
            self.tree.kinds[subtree_start as usize],
            kind,
            "Expected {:?} but found {:?}; tree: {:?}",
            kind,
            self.tree.kinds[subtree_start as usize],
            ShowTreePosition(&self.tree, subtree_start)
        );
        self.tree.indices[subtree_start as usize] = self.tree.len() as u32;
    }

    fn push_node(&mut self, kind: N, subtree_start: Option<u32>) {
        debug_print!(
            "{:indent$}@{} pushing kind {}:\x1b[34m{:?}\x1b[0m starting at {:?}",
            "",
            self.pos,
            self.tree.kinds.len(),
            kind,
            subtree_start,
            indent = 2 * self.frames.len() + 2
        );
        self.tree.kinds.push(kind);
        let pos = subtree_start.unwrap_or(self.tree.indices.len() as u32);
        self.tree.indices.push(pos);
    }

    fn push_node_begin(&mut self, kind: N) -> u32 {
        let index = self.tree.kinds.len() as u32;
        debug_assert_eq!(kind.index_kind(), NodeIndexKind::Begin);
        self.push_node(kind, None);
        index
    }

    fn push_node_end(&mut self, begin: N, end: N, subtree_start: u32) {
        debug_assert_eq!(end.index_kind(), NodeIndexKind::End);
        self.push_node(end, Some(subtree_start));
        self.update_end(begin, subtree_start);
    }

    fn push_node_begin_end(&mut self, begin: N, end: N) {
        let index = self.push_node_begin(begin);
        self.push_node_end(begin, end, index);
    }

    fn push_next_frame(&mut self, subtree_start: u32, cfg: ExprCfg, frame: Frame) {
        debug_print!(
            "{:indent$}@{} *{:?} pushing frame {:?}",
            "",
            self.pos,
            cfg,
            frame,
            indent = 2 * self.frames.len() + 2
        );
        self.frames.push((subtree_start, cfg, frame));
    }

    fn push_next_frame_starting_here(&mut self, cfg: ExprCfg, frame: Frame) {
        let subtree_start = self.tree.len();
        self.push_next_frame(subtree_start, cfg, frame)
    }

    fn pump_frame_now(&mut self) {
        let mut i = 0;
        while let Some((subtree_start, cfg, frame)) = self.frames.pop() {
            debug_print!(
                "{:indent$}@{} *{:?} pumping frame {:?} starting at {}",
                "",
                self.pos,
                cfg,
                frame,
                subtree_start,
                indent = 2 * self.frames.len()
            );
            let mut start = self.pos;
            self._pump_single_frame(subtree_start, cfg, frame);
            // debug_assert!(self.pos > start, "pump didn't advance the cursor");
            i += 1;
            debug_assert!(i < 1000, "pump looped too many times");
        }
    }

    pub fn pump(&mut self) {
        self._pump(0);
    }

    fn force_pump_single_frame(&mut self, subtree_start: u32, expr_cfg: ExprCfg, frame: Frame) {
        let depth = self.frames.len();
        self.push_next_frame(subtree_start, expr_cfg, frame);
        self._pump(depth);
    }

    fn _pump(&mut self, min_depth: usize) {
        let mut i = 0;
        loop {
            if self.frames.len() < min_depth + 1 {
                return;
            }
            if let Some((subtree_start, cfg, frame)) = self.frames.pop() {
                debug_print!(
                    "{:indent$}@{} *{:?} pumping frame {:?} starting at {}",
                    "",
                    self.pos,
                    cfg,
                    frame,
                    subtree_start,
                    indent = 2 * self.frames.len()
                );
                let mut start = self.pos;
                self._pump_single_frame(subtree_start, cfg, frame);
                i += 1;
                debug_assert!(i < 1000, "pump looped too many times");
            }

            if self.messages.len() > 0 {
                self.pos = self.buf.kinds.len();
                return;
            }
        }
    }

    fn _pump_single_frame(&mut self, subtree_start: u32, cfg: ExprCfg, frame: Frame) {
        self.pumping = Some(frame);
        match frame {
            Frame::StartExpr { min_prec } => self.pump_start_expr(subtree_start, cfg, min_prec),
            Frame::ContinueExprTupleOrParen => {
                self.pump_continue_expr_tuple_or_paren(subtree_start, cfg)
            }
            Frame::ContinuePatternTupleOrParen => {
                self.pump_continue_pattern_tuple_or_paren(subtree_start, cfg)
            }
            Frame::FinishLambda => self.pump_finish_lambda(subtree_start),
            Frame::FinishBlockItem => self.pump_finish_block_item(subtree_start),
            Frame::PushEndOnly(end) => self.pump_push_end_only(subtree_start, end),
            Frame::PushEnd(begin, end) => self.pump_push_end(subtree_start, begin, end),
            Frame::ContinueExpr {
                min_prec,
                cur_op,
                num_found,
            } => self.pump_continue_expr(subtree_start, cfg, min_prec, cur_op, num_found),
            Frame::StartPattern => self.pump_start_pattern(subtree_start, cfg),
            Frame::ContinuePattern => self.pump_continue_pattern(subtree_start, cfg),
            Frame::ContinueBlock => self.pump_continue_block(subtree_start, cfg),
            Frame::FinishAssign => self.pump_finish_assign(subtree_start, cfg),
            Frame::ContinueTopLevel { num_found } => {
                self.pump_continue_top_level(subtree_start, cfg, num_found)
            }
            Frame::ContinueLambdaArgs => self.pump_continue_lambda_args(subtree_start, cfg),
            Frame::ContinueIf { next } => self.pump_continue_if(subtree_start, cfg, next),
            Frame::ContinueWhen { next } => self.pump_continue_when(subtree_start, cfg, next),
            Frame::StartType {
                allow_clauses,
                allow_commas,
            } => self.pump_start_type(subtree_start, cfg, allow_clauses, allow_commas),
            Frame::ContinueType {
                in_apply,
                allow_clauses,
                allow_commas,
            } => self.pump_continue_type(subtree_start, cfg, in_apply, allow_clauses, allow_commas),
            Frame::ContinueTypeCommaSep { allow_clauses } => {
                self.pump_continue_type_comma_sep(subtree_start, cfg, allow_clauses)
            }
            Frame::FinishTypeFunction => self.pump_finish_type_function(subtree_start, cfg),
            Frame::ContinueWhereClause => self.pump_continue_where_clause(subtree_start, cfg),
            Frame::FinishTypeOrTypeAlias => self.pump_finish_type_or_type_alias(subtree_start, cfg),
            Frame::ContinueRecord { start } => self.pump_continue_record(subtree_start, cfg, start),
            Frame::ContinuePatternRecord { start } => {
                self.pump_continue_pattern_record(subtree_start, cfg, start)
            }
            Frame::ContinueExprList => self.pump_continue_expr_list(subtree_start, cfg),
            Frame::ContinuePatternList => self.pump_continue_pattern_list(subtree_start, cfg),
            Frame::ContinueTypeTupleOrParen => {
                self.pump_continue_type_tuple_or_paren(subtree_start, cfg)
            }
            Frame::ContinueTypeTagUnion => self.pump_continue_type_tag_union(subtree_start, cfg),
            Frame::ContinueTypeTagUnionArgs => {
                self.pump_continue_tag_union_args(subtree_start, cfg)
            }
            Frame::ContinueImplementsMethodDecl => {
                self.pump_continue_implements_method_decl(subtree_start, cfg)
            }
            Frame::ContinueTypeRecord => self.pump_continue_type_record(subtree_start, cfg),
        }
        self.pumping = None;
    }

    fn pump_start_expr(&mut self, mut subtree_start: u32, mut cfg: ExprCfg, mut min_prec: Prec) {
        macro_rules! maybe_return {
            () => {
                match self.cur() {
                    None | Some(T::OpArrow) => return,
                    _ => {}
                }

                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueExpr {
                        min_prec,
                        cur_op: None,
                        num_found: 1,
                    },
                );
                return;
            };
        }

        macro_rules! atom {
            ($n:expr) => {{
                self.bump();
                let subtree_start = self.tree.len();
                self.push_node($n, Some(self.pos as u32 - 1));
                self.handle_field_access_suffix(subtree_start);
                maybe_return!();
            }};
        }

        loop {
            match self.cur() {
                Some(T::LowerIdent) => atom!(N::Ident),
                Some(T::KwCrash) => atom!(N::Crash),
                // TODO: do these need to be distinguished in the node?
                Some(T::Underscore | T::NamedUnderscore) => atom!(N::Underscore),
                Some(T::Int) => atom!(N::Num),
                Some(T::String) => atom!(N::String),
                Some(T::Float) => atom!(N::Float),
                Some(T::DotNumber) => atom!(N::TupleAccessFunction),
                Some(T::DotLowerIdent) => atom!(N::FieldAccessFunction),
                Some(T::OpaqueName) => atom!(N::OpaqueName),

                Some(T::UpperIdent) => {
                    self.bump();
                    let subtree_start = self.tree.len();
                    if self.consume(T::NoSpaceDotLowerIdent) {
                        self.push_node(N::ModuleName, Some(self.pos as u32 - 2));
                        self.push_node(N::DotModuleLowerIdent, Some(self.pos as u32 - 1));
                    } else if self.consume(T::NoSpaceDotUpperIdent) {
                        self.push_node(N::ModuleName, Some(self.pos as u32 - 2));
                        self.push_node(N::DotModuleUpperIdent, Some(self.pos as u32 - 1));
                        if self.consume(T::NoSpaceDotUpperIdent) {
                            self.push_node(N::DotModuleUpperIdent, Some(self.pos as u32 - 1));
                        }
                    } else {
                        // TODO: this is probably wrong
                        self.push_node(N::UpperIdent, Some(self.pos as u32 - 1));
                    }
                    self.handle_field_access_suffix(subtree_start);
                    maybe_return!();
                }

                Some(T::OpenRound) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueExpr {
                            min_prec,
                            cur_op: None,
                            num_found: 1,
                        },
                    );
                    self.push_next_frame_starting_here(cfg, Frame::ContinueExprTupleOrParen);
                    self.push_node(N::BeginParens, None);
                    subtree_start = self.tree.len();
                    min_prec = Prec::Outer;
                    cfg = cfg.disable_multi_backpassing();
                    continue;
                }
                Some(T::OpenCurly) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueExpr {
                            min_prec,
                            cur_op: None,
                            num_found: 1,
                        },
                    );
                    self.push_next_frame_starting_here(cfg, Frame::ContinueRecord { start: true });
                    self.push_node(N::BeginRecord, None); // index will be updated later
                    return;
                }
                Some(T::OpenSquare) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueExpr {
                            min_prec,
                            cur_op: None,
                            num_found: 1,
                        },
                    );
                    self.start_list(cfg);
                    return;
                }
                Some(T::OpBackslash) => {
                    self.bump();
                    self.push_next_frame_starting_here(cfg, Frame::ContinueLambdaArgs);
                    self.push_node(N::BeginLambda, None);
                    self.start_pattern(cfg);
                    return;
                }
                Some(T::KwIf) => {
                    if self.plausible_expr_continue_comes_next() {
                        atom!(N::Ident);
                    } else {
                        self.bump();
                        self.push_next_frame_starting_here(
                            cfg,
                            Frame::ContinueIf {
                                next: IfState::Then,
                            },
                        );
                        self.push_node(N::BeginIf, None);
                        self.start_expr(cfg);
                        return;
                    }
                }
                Some(T::KwWhen) => {
                    if self.plausible_expr_continue_comes_next() {
                        atom!(N::Ident);
                    } else {
                        self.bump();
                        self.push_next_frame_starting_here(
                            cfg,
                            Frame::ContinueWhen {
                                next: WhenState::Is,
                            },
                        );
                        self.push_node(N::BeginWhen, None);
                        self.start_expr(cfg);
                        return;
                    }
                }
                Some(T::OpBang) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueExpr {
                            min_prec,
                            cur_op: None,
                            num_found: 1,
                        },
                    );
                    self.push_next_frame_starting_here(cfg, Frame::PushEndOnly(N::EndUnaryNot));
                    self.push_next_frame_starting_here(
                        cfg,
                        Frame::StartExpr {
                            min_prec: Prec::Atom,
                        },
                    );
                    return;
                }
                Some(T::OpUnaryMinus) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueExpr {
                            min_prec,
                            cur_op: None,
                            num_found: 1,
                        },
                    );
                    self.push_next_frame_starting_here(cfg, Frame::PushEndOnly(N::EndUnaryMinus));
                    self.push_next_frame_starting_here(
                        cfg,
                        Frame::StartExpr {
                            min_prec: Prec::Atom,
                        },
                    );
                    return;
                }
                Some(T::OpArrow) => return, // when arrow; handled in outer scope
                Some(T::KwDbg) => {
                    self.bump();
                    self.push_next_frame(subtree_start, cfg, Frame::PushEnd(N::Dummy, N::EndDbg));
                    self.start_block_or_expr(cfg);
                    return;
                }
                Some(T::KwExpect) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::PushEnd(N::Dummy, N::EndExpect),
                    );
                    self.start_block_or_expr(cfg);
                    return;
                }
                Some(T::KwExpectFx) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::PushEnd(N::Dummy, N::EndExpectFx),
                    );
                    self.start_block_or_expr(cfg);
                    return;
                }
                Some(k) if k.is_keyword() => {
                    // treat as an identifier
                    atom!(N::Ident);
                }
                k => {
                    self.push_error(Error::ExpectedExpr(k));
                    self.fast_forward_past_newline();
                    return;
                }
            }
        }
    }

    fn pump_start_pattern(&mut self, mut subtree_start: u32, mut cfg: ExprCfg) {
        macro_rules! maybe_return {
            () => {
                match self.cur() {
                    None | Some(T::OpArrow) => return,
                    _ => {}
                }

                self.push_next_frame(subtree_start, cfg, Frame::ContinuePattern);
                return;
            };
        }

        macro_rules! atom {
            ($n:expr) => {{
                self.bump();
                let subtree_start = self.tree.len();
                self.push_node($n, Some(self.pos as u32 - 1));
                self.handle_field_access_suffix(subtree_start);
                maybe_return!();
            }};
        }

        loop {
            match self.cur() {
                Some(T::DoubleDot) => atom!(N::PatternDoubleDot),
                Some(T::LowerIdent) => atom!(N::Ident),
                Some(T::KwCrash) => atom!(N::Crash),
                // TODO: do these need to be distinguished in the node?
                Some(T::Underscore | T::NamedUnderscore) => atom!(N::Underscore),
                Some(T::Int) => atom!(N::Num),
                Some(T::String) => atom!(N::String),
                Some(T::Float) => atom!(N::Float),
                Some(T::OpaqueName) => atom!(N::OpaqueName),

                Some(T::UpperIdent) => {
                    self.bump();
                    let subtree_start = self.tree.len();
                    if self.consume(T::NoSpaceDotLowerIdent) {
                        self.push_node(N::ModuleName, Some(self.pos as u32 - 2));
                        self.push_node(N::DotModuleLowerIdent, Some(self.pos as u32 - 1));
                    } else {
                        // TODO: this is probably wrong
                        self.push_node(N::UpperIdent, Some(self.pos as u32 - 1));
                    }
                    self.handle_field_access_suffix(subtree_start);
                    maybe_return!();
                }

                Some(T::OpenRound) => {
                    self.bump();
                    self.push_next_frame(subtree_start, cfg, Frame::ContinuePattern);
                    self.push_next_frame_starting_here(cfg, Frame::ContinuePatternTupleOrParen);
                    self.push_node(N::BeginPatternParens, None);
                    subtree_start = self.tree.len();
                    cfg = cfg.disable_multi_backpassing();
                    continue;
                }
                Some(T::OpenCurly) => {
                    self.bump();
                    self.push_next_frame_starting_here(
                        cfg,
                        Frame::ContinuePatternRecord { start: true },
                    );
                    self.push_node(N::BeginPatternRecord, None); // index will be updated later
                    return;
                }
                Some(T::OpenSquare) => {
                    self.bump();
                    self.start_pattern_list(cfg);
                    return;
                }
                Some(T::OpArrow) => return, // when arrow; handled in outer scope
                Some(k) if k.is_keyword() => {
                    // treat as an identifier
                    atom!(N::Ident);
                }
                k => {
                    self.push_error(Error::ExpectedExpr(k));
                    self.fast_forward_past_newline();
                    return;
                }
            }
        }
    }

    fn at_pattern_continue(&mut self, cfg: ExprCfg) -> bool {
        match self.cur() {
            // TODO: check for other things that can start an expr
            Some(
                T::LowerIdent
                | T::UpperIdent
                | T::Underscore
                | T::OpenCurly
                | T::Int
                | T::String
                | T::OpenRound
                | T::OpenCurly
                | T::OpenSquare
                | T::OpUnaryMinus
                | T::OpBang,
            ) => {
                if self.at_newline() {
                    // are we at the start of a line?
                    if !self.buf.lines[self.line]
                        .1
                        .is_indented_more_than(cfg.expr_indent_floor)
                        .expect("TODO: error handling")
                    {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn pump_continue_pattern(&mut self, subtree_start: u32, cfg: ExprCfg) {
        loop {
            if self.consume(T::KwAs) {
                self.expect_and_push_node(T::LowerIdent, N::Ident);
                self.push_node(N::EndPatternAs, Some(subtree_start));
                continue;
            }

            // TODO: only allow calls / application in patterns (remove this generic op stuff)
            if self.at_pattern_continue(cfg) {
                self.push_next_frame(subtree_start, cfg, Frame::ContinuePattern);
                self.push_next_frame_starting_here(cfg, Frame::StartPattern);
            }
            return;
        }
    }

    fn start_list(&mut self, cfg: ExprCfg) {
        if self.consume(T::CloseSquare) {
            let subtree_start = self.tree.len();
            self.push_node(N::BeginList, Some(subtree_start + 2));
            self.push_node(N::EndList, Some(subtree_start));
            return;
        }

        self.push_next_frame_starting_here(cfg, Frame::ContinueExprList);
        self.push_node(N::BeginList, None);
        // index will be updated later
        self.start_expr(cfg.disable_multi_backpassing());
    }

    fn start_pattern_list(&mut self, cfg: ExprCfg) {
        if self.consume(T::CloseSquare) {
            let subtree_start = self.tree.len();
            self.push_node(N::BeginPatternList, Some(subtree_start + 2));
            self.push_node(N::EndPatternList, Some(subtree_start));
            return;
        }

        self.push_next_frame_starting_here(cfg, Frame::ContinuePatternList);
        self.push_node(N::BeginPatternList, None);
        // index will be updated later
        self.start_pattern(cfg.disable_multi_backpassing());
    }

    fn handle_field_access_suffix(&mut self, subtree_start: u32) {
        loop {
            match self.cur() {
                Some(T::NoSpaceDotLowerIdent) => {
                    self.bump();
                    self.push_node(N::DotIdent, Some(self.pos as u32 - 1));
                    self.push_node(N::EndFieldAccess, Some(subtree_start));
                }
                Some(T::NoSpaceDotNumber) => {
                    self.bump();
                    self.push_node(N::DotNumber, Some(self.pos as u32 - 1));
                    self.push_node(N::EndIndexAccess, Some(subtree_start));
                }
                _ => return,
            }
        }
    }

    fn pump_continue_expr(
        &mut self,
        subtree_start: u32,
        cfg: ExprCfg,
        min_prec: Prec,
        cur_op: Option<BinOp>,
        mut num_found: usize,
    ) {
        let cur_indent = self.cur_indent();
        if let Some(op) = self.next_op(min_prec, cfg) {
            if let Some(cur_op) = cur_op {
                if op != cur_op || !op.n_arity() {
                    self.push_node(cur_op.into(), Some(subtree_start));
                }
            }

            self.push_node(op.to_inline(), Some(self.pos as u32 - 1));

            match op {
                BinOp::Assign => {
                    if self.at_newline() {
                        self.push_next_frame(subtree_start, cfg, Frame::FinishAssign);
                        self.start_block(cfg);
                        return;
                    }
                }
                BinOp::DefineTypeOrTypeAlias => {
                    self.push_next_frame(subtree_start, cfg, Frame::FinishTypeOrTypeAlias);
                    self.start_type(cfg, true, true);
                    return;
                }
                BinOp::DefineOtherTypeThing => {
                    // TODO: is this correct????
                    self.push_next_frame(subtree_start, cfg, Frame::FinishTypeOrTypeAlias);
                    self.start_type(cfg, true, true);
                    return;
                }
                BinOp::Implements => {
                    let cfg = cfg.set_block_indent_floor(Some(cur_indent));
                    self.continue_implements_method_decl_body(subtree_start, cfg);
                    return;
                }
                _ => {}
            }

            debug_print!(
                "{:indent$}next op {:?}",
                "",
                op,
                indent = 2 * self.frames.len() + 2
            );

            let op_prec = op.prec();
            let assoc = op.matching_assoc();

            let next_min_prec = if assoc == Assoc::Left {
                op_prec
            } else {
                op_prec.next()
            };

            self.push_next_frame(
                subtree_start,
                cfg,
                Frame::ContinueExpr {
                    min_prec,
                    cur_op: Some(op),
                    num_found,
                },
            );
            self.push_next_frame_starting_here(
                cfg,
                Frame::StartExpr {
                    min_prec: next_min_prec,
                },
            );
            return;
        } else if let Some(cur_op) = cur_op {
            self.push_node(cur_op.into(), Some(subtree_start));
        }
    }

    fn next_op(&mut self, min_prec: Prec, cfg: ExprCfg) -> Option<BinOp> {
        let (op, width) = match self.cur() {
            // TODO: check for other things that can start an expr
            Some(
                T::LowerIdent
                | T::UpperIdent
                | T::Underscore
                | T::OpenCurly
                | T::Int
                | T::String
                | T::OpenRound
                | T::OpenCurly
                | T::OpenSquare
                | T::OpUnaryMinus
                | T::OpBang
                | T::OpBackslash
                | T::DotLowerIdent,
            ) => {
                if self.at_newline() {
                    // are we at the start of a line?
                    if !self.buf.lines[self.line]
                        .1
                        .is_indented_more_than(cfg.expr_indent_floor)
                        .expect("TODO: error handling")
                    {
                        return None;
                    }
                }
                (BinOp::Apply, 0)
            }

            Some(T::OpPlus) => (BinOp::Plus, 1),
            Some(T::OpBinaryMinus) => (BinOp::Minus, 1),
            Some(T::OpStar) => (BinOp::Star, 1),
            Some(T::OpPizza) => (BinOp::Pizza, 1),
            Some(T::OpPercent) => (BinOp::Percent, 1),
            Some(T::OpAssign) => (BinOp::Assign, 1),
            Some(T::OpColon) => (BinOp::DefineTypeOrTypeAlias, 1),
            Some(T::OpColonEqual) => (BinOp::DefineOtherTypeThing, 1),
            Some(T::OpBackArrow) => (BinOp::Backpassing, 1),
            Some(T::KwImplements) => (BinOp::Implements, 1),
            Some(T::OpEquals) => (BinOp::Equals, 1),
            Some(T::OpLessThan) => (BinOp::LessThan, 1),
            Some(T::OpLessThanOrEq) => (BinOp::LessThanOrEq, 1),
            Some(T::OpGreaterThan) => (BinOp::GreaterThan, 1),
            Some(T::OpGreaterThanOrEq) => (BinOp::GreaterThanOrEq, 1),
            Some(T::OpAnd) => (BinOp::And, 1),
            Some(T::OpOr) => (BinOp::Or, 1),
            Some(T::Comma) if cfg.allow_multi_backpassing => (BinOp::MultiBackpassingComma, 1),
            Some(T::KwAs) => (BinOp::As, 1),
            _ => return None,
        };

        if op.prec() < min_prec || (op.prec() == min_prec && op.grouping_assoc() == Assoc::Left) {
            return None;
        }

        self.bump_n(width);

        Some(op)
    }

    fn pump_start_type(
        &mut self,
        subtree_start: u32,
        cfg: ExprCfg,
        allow_clauses: bool,
        allow_commas: bool,
    ) {
        loop {
            match self.cur() {
                Some(T::OpenRound) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType {
                            in_apply: None,
                            allow_clauses,
                            allow_commas,
                        },
                    );
                    self.push_node(N::BeginParens, None);
                    self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeTupleOrParen);
                    continue;
                }
                Some(T::OpenSquare) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType {
                            in_apply: None,
                            allow_clauses,
                            allow_commas,
                        },
                    );
                    self.start_tag_union(subtree_start, cfg);
                    return;
                }
                Some(T::OpenCurly) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType {
                            in_apply: None,
                            allow_clauses,
                            allow_commas,
                        },
                    );
                    self.start_type_record(cfg);
                    return;
                }
                Some(T::LowerIdent) => {
                    self.bump();
                    self.push_node(N::Ident, Some(self.pos as u32 - 1));

                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType {
                            in_apply: None,
                            allow_clauses,
                            allow_commas,
                        },
                    );
                    return;
                }
                Some(T::Underscore) => {
                    self.bump();
                    self.push_node(N::Underscore, Some(self.pos as u32 - 1));

                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType {
                            in_apply: None,
                            allow_clauses,
                            allow_commas,
                        },
                    );
                    return;
                }
                Some(T::OpStar) => {
                    self.bump();
                    self.push_node(N::TypeWildcard, Some(self.pos as u32 - 1));

                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType {
                            in_apply: None,
                            allow_clauses,
                            allow_commas,
                        },
                    );
                    return;
                }
                Some(T::UpperIdent) => {
                    self.bump();
                    let subtree_start = self.tree.len();
                    if self.consume(T::NoSpaceDotUpperIdent) {
                        self.push_node(N::ModuleName, Some(self.pos as u32 - 2));
                        self.push_node(N::DotModuleUpperIdent, Some(self.pos as u32 - 1));
                        if self.consume(T::NoSpaceDotUpperIdent) {
                            self.push_node(N::DotModuleUpperIdent, Some(self.pos as u32 - 1));
                        }
                    } else {
                        self.push_node(N::TypeName, Some(self.pos as u32 - 1));
                    }
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType {
                            in_apply: Some(false),
                            allow_clauses,
                            allow_commas,
                        },
                    );
                    return;
                }
                Some(k) if k.is_keyword() => {
                    // treat as an identifier
                    self.bump();
                    self.push_node(N::Ident, Some(self.pos as u32 - 1));

                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType {
                            in_apply: None,
                            allow_clauses,
                            allow_commas,
                        },
                    );
                    return;
                }
                k => todo!("start type, unexpected: {:?}", k),
            }
        }
    }

    fn start_type_ext(&mut self, subtree_start: u32, cfg: ExprCfg) {
        loop {
            match self.cur() {
                Some(T::OpenRound) => {
                    self.bump();
                    self.push_node(N::BeginParens, None);
                    self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeTupleOrParen);
                    continue;
                }
                Some(T::OpenSquare) => {
                    self.bump();
                    self.start_tag_union(subtree_start, cfg);
                    return;
                }
                Some(T::OpenCurly) => {
                    self.bump();
                    self.start_type_record(cfg);
                    return;
                }
                Some(T::LowerIdent) => {
                    self.bump();
                    self.push_node(N::Ident, Some(self.pos as u32 - 1));
                    return;
                }
                Some(T::OpStar) => {
                    self.bump();
                    self.push_node(N::TypeWildcard, Some(self.pos as u32 - 1));
                    return;
                }
                Some(k) if k.is_keyword() => {
                    // treat as an identifier
                    self.bump();
                    self.push_node(N::Ident, Some(self.pos as u32 - 1));
                    return;
                }
                k => todo!("start type ext, unexpected: {:?}", k),
            }
        }
    }

    fn pump_continue_type(
        &mut self,
        subtree_start: u32,
        cfg: ExprCfg,
        in_apply: Option<bool>,
        allow_clauses: bool,
        allow_commas: bool,
    ) {
        match self.cur() {
            Some(T::Comma) if allow_commas => {
                if (self.peek_at(1) != Some(T::LowerIdent) || self.peek_at(2) != Some(T::OpColon))
                    && self.peek_at(1) != Some(T::CloseCurly)
                {
                    self.bump();
                    if in_apply == Some(true) {
                        self.push_node(N::EndTypeApply, Some(subtree_start));
                    }
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueTypeCommaSep { allow_clauses },
                    );
                    self.start_type(cfg, false, false);
                    return;
                }

                if in_apply == Some(true) {
                    self.push_node(N::EndTypeApply, Some(subtree_start));
                }
                return;
            }
            Some(T::OpArrow) => {
                self.bump();
                if in_apply == Some(true) {
                    self.push_node(N::EndTypeApply, Some(subtree_start));
                }
                self.push_node(N::InlineLambdaArrow, Some(self.pos as u32 - 1));
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueType {
                        in_apply,
                        allow_clauses,
                        allow_commas,
                    },
                );
                self.push_next_frame(subtree_start, cfg, Frame::FinishTypeFunction);
                self.start_type(cfg, false, false);
                return;
            }
            Some(T::KwWhere) if allow_clauses => {
                if !self.plausible_expr_continue_comes_next() {
                    self.bump();
                    // TODO: should write a plausible_type_continue_comes_next
                    if in_apply == Some(true) {
                        self.push_node(N::EndTypeApply, Some(subtree_start));
                    }

                    self.push_node(N::InlineKwWhere, Some(self.pos as u32 - 1));
                    self.push_next_frame(subtree_start, cfg, Frame::ContinueWhereClause);
                    self.start_type(cfg, false, false);
                    return;
                }
            }
            Some(T::KwAs) if allow_clauses => {
                self.bump();

                if in_apply == Some(true) {
                    self.push_node(N::EndTypeApply, Some(subtree_start));
                }

                self.push_node(N::InlineKwAs, Some(self.pos as u32 - 1));
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueType {
                        in_apply: None,
                        allow_clauses,
                        allow_commas,
                    },
                );
                self.push_next_frame(subtree_start, cfg, Frame::PushEndOnly(N::EndTypeAs));
                self.start_type(cfg, false, false);
            }
            // TODO: check for other things that can start an expr
            Some(
                T::LowerIdent
                | T::UpperIdent
                | T::OpenCurly
                | T::OpenSquare
                | T::OpenRound
                | T::Underscore,
            ) if in_apply.is_some() => {
                if self.at_newline() {
                    // are we at the start of a line?
                    if !self.buf.lines[self.line]
                        .1
                        .is_indented_more_than(cfg.expr_indent_floor)
                        .expect("TODO: error handling")
                    {
                        // Not indented enough; we're done.
                        if in_apply == Some(true) {
                            self.push_node(N::EndTypeApply, Some(subtree_start));
                        }
                        return;
                    }
                }

                // We need to keep processing args
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueType {
                        in_apply: Some(true),
                        allow_clauses,
                        allow_commas,
                    },
                );
                self.start_type(cfg, false, false);
            }
            _ => {
                if in_apply == Some(true) {
                    self.push_node(N::EndTypeApply, Some(subtree_start));
                }
            }
        }
    }

    fn pump_continue_type_comma_sep(
        &mut self,
        subtree_start: u32,
        cfg: ExprCfg,
        allow_clauses: bool,
    ) {
        match self.cur() {
            Some(T::Comma) => {
                if (self.peek_at(1) != Some(T::LowerIdent) || self.peek_at(2) != Some(T::OpColon))
                    && self.peek_at(1) != Some(T::CloseCurly)
                {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueTypeCommaSep { allow_clauses },
                    );
                    self.start_type(cfg, false, false);
                }
            }
            Some(T::OpArrow) => {
                self.bump();
                self.push_node(N::InlineLambdaArrow, Some(self.pos as u32 - 1));
                self.push_next_frame(subtree_start, cfg, Frame::FinishTypeFunction);
                self.start_type(cfg, false, false);
            }

            // TODO: if there isn't an outer square/round/curly and we don't eventually get the arrow,
            // we should error
            Some(T::CloseSquare | T::CloseRound | T::CloseCurly) => return, // Outer scope will handle
            Some(T::LowerIdent | T::UpperIdent) | None => return, // if the inner type didn't consume, we know we're at the end of the line
            k => todo!("continue type comma sep, unexpected: {:?}", k),
        }
    }

    fn pump_continue_where_clause(&mut self, subtree_start: u32, cfg: ExprCfg) {
        // We should have just parsed the `where` followed by a type.
        // ... or we might be after the following , and type (e.g. as in the following:)
        //    where a implements Hash, b implements Eq, c implements Ord
        self.expect_and_push_node(T::KwImplements, N::InlineKwImplements);
        loop {
            self.expect_and_push_node(T::UpperIdent, N::AbilityName);
            if !self.consume(T::OpAmpersand) {
                break;
            }
        }
        self.push_node(N::EndWhereClause, Some(subtree_start));
        if self.consume(T::Comma) {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueWhereClause);
            self.start_type(cfg, false, false);
        }
    }

    fn pump_finish_type_function(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::EndTypeLambda, Some(subtree_start));
    }

    fn pump_continue_expr_tuple_or_paren(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume_comma_terminator(T::CloseRound) {
            self.push_node_end(N::BeginParens, N::EndParens, subtree_start);
            self.handle_field_access_suffix(subtree_start);
        } else {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueExprTupleOrParen);
            self.start_expr(cfg.disable_multi_backpassing());
        }
    }

    fn pump_continue_pattern_tuple_or_paren(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume_comma_terminator(T::CloseRound) {
            self.push_node_end(N::BeginPatternParens, N::EndPatternParens, subtree_start);
        } else {
            self.push_next_frame(subtree_start, cfg, Frame::ContinuePatternTupleOrParen);
            self.start_expr(cfg.disable_multi_backpassing());
        }
    }

    fn pump_finish_lambda(&mut self, subtree_start: u32) {
        self.push_node_end(N::BeginLambda, N::EndLambda, subtree_start);
    }

    fn pump_finish_block_item(&mut self, subtree_start: u32) {
        let k = self.tree.kinds.last().copied().unwrap(); // Find out what we just parsed
        match k {
            N::EndAssign => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginAssign;
            }
            N::EndBackpassing => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginBackpassing;
            }
            N::EndTypeOrTypeAlias => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginTypeOrTypeAlias;
            }
            N::EndImplements => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginImplements;
            }
            N::EndDbg => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginDbg;
            }
            N::EndExpect => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginExpect;
            }
            N::EndExpectFx => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginExpectFx;
            }
            N::Ident
            | N::UpperIdent
            | N::Underscore
            | N::OpaqueName
            | N::Num
            | N::Crash
            | N::String
            | N::Float
            | N::EndWhen
            | N::EndIf
            | N::EndApply
            | N::EndBinOpPlus
            | N::EndBinOpMinus
            | N::EndBinOpStar
            | N::EndBinOpLessThan
            | N::EndBinOpGreaterThan
            | N::EndBinOpLessThanOrEq
            | N::EndBinOpGreaterThanOrEq
            | N::EndBinOpSlash
            | N::EndBinOpDoubleSlash
            | N::EndBinOpPercent
            | N::EndBinOpCaret
            | N::EndBinOpAnd
            | N::EndBinOpOr
            | N::EndBinOpEquals
            | N::EndBinOpNotEquals
            | N::EndUnaryNot
            | N::EndUnaryMinus
            | N::EndPizza
            | N::EndParens
            | N::EndList
            | N::EndLambda
            | N::EndRecord
            | N::EndFieldAccess
            | N::EndIndexAccess
            | N::EndBinOpSlash
            | N::EndBinOpDoubleSlash
            | N::EndBinOpPercent
            | N::EndBinOpCaret
            | N::EndBinOpAnd
            | N::EndBinOpOr
            | N::EndBinOpPercent
            | N::EndTypeAs
            | N::EndBinOpEquals
            | N::EndBinOpNotEquals
            | N::DotModuleLowerIdent => {
                self.tree.kinds[subtree_start as usize] = N::HintExpr;
            }
            k => todo!("finish block item, unexpected: {:?}", k),
        };
    }

    fn pump_continue_block(&mut self, subtree_start: u32, cfg: ExprCfg) {
        // need to inspect the expr we just parsed.
        // if it's a decl we keep going; if it's not, we're done.
        if self.tree.kinds.last().copied().unwrap().is_decl() {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueBlock);
            self.start_block_item(cfg);
        } else {
            self.push_node_end(N::BeginBlock, N::EndBlock, subtree_start);
        }
    }

    fn pump_finish_assign(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::EndAssign, Some(subtree_start));
    }

    fn pump_finish_type_or_type_alias(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::EndTypeOrTypeAlias, Some(subtree_start));
    }

    fn pump_continue_top_level(&mut self, subtree_start: u32, cfg: ExprCfg, num_found: i32) {
        // keep parsing decls until the end
        if self.pos < self.buf.kinds.len() {
            self.push_next_frame(
                subtree_start,
                cfg,
                Frame::ContinueTopLevel {
                    num_found: num_found + 1,
                },
            );
            self.start_top_level_item();
        } else {
            self.push_node_end(N::BeginTopLevelDecls, N::EndTopLevelDecls, subtree_start);
        }
    }

    fn pump_continue_lambda_args(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::OpArrow) {
            self.push_node(N::InlineLambdaArrow, Some(self.pos as u32 - 1));
            self.push_next_frame(subtree_start, cfg, Frame::FinishLambda);
            self.start_block_or_expr(cfg);
        } else if self.consume(T::Comma) {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueLambdaArgs);
            self.start_pattern(cfg);
        } else {
            self.push_error(Error::ExpectedPattern(self.cur()));
            self.fast_forward_past_newline();
        }
    }

    fn pump_continue_if(&mut self, subtree_start: u32, cfg: ExprCfg, next: IfState) {
        match next {
            IfState::Then => {
                self.expect_and_push_node(T::KwThen, N::InlineKwThen);
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueIf {
                        next: IfState::Else,
                    },
                );
                self.start_block_or_expr(cfg);
            }
            IfState::Else => {
                self.expect_and_push_node(T::KwElse, N::InlineKwElse);

                let next = if self.consume(T::KwIf) {
                    IfState::Then
                } else {
                    IfState::End
                };

                self.push_next_frame(subtree_start, cfg, Frame::ContinueIf { next });
                self.start_block_or_expr(cfg);
            }
            IfState::End => {
                self.push_node_end(N::BeginIf, N::EndIf, subtree_start);
            }
        }
    }

    fn pump_continue_when(&mut self, subtree_start: u32, cfg: ExprCfg, next: WhenState) {
        match next {
            WhenState::Is => {
                self.expect_and_push_node(T::KwIs, N::InlineKwIs);
                let indent = self.cur_indent();
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueWhen {
                        next: WhenState::BranchBarOrArrow(indent),
                    },
                );
                self.start_pattern(cfg);
            }
            WhenState::BranchPattern(indent) => {
                if
                /*dbg!(self
                .cur_indent()
                .is_indented_more_than(cfg.block_indent_floor)
                .ok_or(Error::InconsistentIndent)
                .expect("TODO: error handling"))
                && */
                self.at_plausible_when_pattern() {
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueWhen {
                            next: WhenState::BranchBarOrArrow(indent),
                        },
                    );
                    self.start_pattern(cfg);
                    return;
                }
                self.push_node_end(N::BeginWhen, N::EndWhen, subtree_start);
            }
            WhenState::BranchBarOrArrow(indent) => {
                if self.consume(T::OpBar) {
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueWhen {
                            next: WhenState::BranchBarOrArrow(indent),
                        },
                    );
                    self.start_pattern(cfg);
                } else if self.consume_and_push_node(T::KwIf, N::InlineKwIf) {
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueWhen {
                            next: WhenState::BranchBarOrArrow(indent),
                        },
                    );
                    self.start_block_or_expr(cfg);
                } else {
                    self.expect_and_push_node(T::OpArrow, N::InlineWhenArrow);
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueWhen {
                            next: WhenState::BranchPattern(indent),
                        },
                    );
                    let indent = self.cur_indent();
                    let cfg = cfg
                        .set_block_indent_floor(Some(indent))
                        .set_expr_indent_floor(Some(indent));
                    self.start_block_or_expr(cfg);
                }
            }
        }
    }

    fn start_top_level_item(&mut self) {
        self.push_next_frame_starting_here(ExprCfg::default(), Frame::FinishBlockItem);
        let indent = self.cur_indent();
        let cfg = ExprCfg::default()
            .set_expr_indent_floor(Some(indent))
            .set_block_indent_floor(Some(indent));
        self.start_expr(cfg);
        self.push_node(N::Dummy, None); // will be replaced by the actual node in pump_finish_block_item
    }

    fn start_top_level_decls(&mut self) {
        if self.pos == self.buf.kinds.len() {
            let subtree_start = self.tree.len();
            self.push_node_begin_end(N::BeginTopLevelDecls, N::EndTopLevelDecls);
        } else {
            self.push_next_frame_starting_here(
                ExprCfg::default(),
                Frame::ContinueTopLevel { num_found: 1 },
            );
            self.push_node(N::BeginTopLevelDecls, None);
            self.start_top_level_item();
        }
    }

    pub fn start_file(&mut self) {
        let subtree_start = self.push_node_begin(N::BeginFile);
        match self.cur() {
            Some(T::KwApp) => {
                self.bump();
                let subtree_start = self.push_node_begin(N::BeginHeaderApp);
                self.expect_and_push_node(T::String, N::String);
                if self.consume(T::KwPackages) {
                    self.expect_collection(
                        T::OpenCurly,
                        N::BeginCollection,
                        T::CloseCurly,
                        N::EndCollection,
                        |s| {
                            s.expect_and_push_node(T::LowerIdent, N::Ident);
                            s.expect(T::OpColon);
                            s.expect_and_push_node(T::String, N::String);
                        },
                    );
                } else {
                    self.push_node_begin_end(N::BeginCollection, N::EndCollection);
                }
                self.expect_header_imports();
                self.expect(T::KwProvides);
                self.expect_collection(
                    T::OpenSquare,
                    N::BeginCollection,
                    T::CloseSquare,
                    N::EndCollection,
                    |s| {
                        s.expect_and_push_node(T::LowerIdent, N::Ident);
                    },
                );
                if self.cur() == Some(T::OpenCurly) {
                    self.expect_collection(
                        T::OpenCurly,
                        N::BeginCollection,
                        T::CloseCurly,
                        N::EndCollection,
                        |s| {
                            s.expect_and_push_node(T::UpperIdent, N::Ident);
                        },
                    );
                }
                self.expect(T::KwTo);
                match self.cur() {
                    Some(T::LowerIdent) => {
                        self.bump();
                        self.push_node(N::Ident, Some(self.pos as u32 - 1));
                    }
                    Some(T::String) => {
                        self.bump();
                        self.push_node(N::String, Some(self.pos as u32 - 1));
                    }
                    _ => {
                        self.push_error(Error::ExpectViolation(T::LowerIdent, self.cur()));
                        self.fast_forward_past_newline();
                    }
                }
                self.push_node_end(N::BeginHeaderApp, N::EndHeaderApp, subtree_start);
            }
            Some(T::KwPlatform) => {
                self.bump();
                let subtree_start = self.push_node_begin(N::BeginHeaderPlatform);
                self.expect(T::String);
                self.expect(T::KwRequires);
                self.expect_collection(
                    T::OpenCurly,
                    N::BeginCollection,
                    T::CloseCurly,
                    N::EndCollection,
                    |s| {
                        s.expect_and_push_node(T::UpperIdent, N::Ident);
                    },
                );
                self.consume(T::NoSpace);
                self.expect_collection(
                    T::OpenCurly,
                    N::BeginCollection,
                    T::CloseCurly,
                    N::EndCollection,
                    |s| {
                        // This needs to be key:value pairs for lower ident keys and type values
                        s.expect_and_push_node(T::LowerIdent, N::Ident); // TODO: correct node type
                        s.expect(T::OpColon);
                        let subtree_start = s.tree.len();
                        s.force_pump_single_frame(
                            subtree_start,
                            ExprCfg::default(),
                            Frame::StartType {
                                allow_clauses: false,
                                allow_commas: true,
                            },
                        );
                    },
                );
                self.expect(T::KwExposes);
                self.expect_collection(
                    T::OpenSquare,
                    N::BeginCollection,
                    T::CloseSquare,
                    N::EndCollection,
                    |s| {
                        s.expect_and_push_node(T::UpperIdent, N::Ident); // TODO: correct node type
                    },
                );
                self.expect(T::KwPackages);
                self.expect_collection(
                    T::OpenCurly,
                    N::BeginCollection,
                    T::CloseCurly,
                    N::EndCollection,
                    |s| {
                        s.expect_and_push_node(T::LowerIdent, N::Ident);
                        s.expect(T::OpColon);
                        s.expect_and_push_node(T::String, N::String);
                    },
                );
                self.expect_header_imports();
                self.expect(T::KwProvides);
                self.expect_collection(
                    T::OpenSquare,
                    N::BeginCollection,
                    T::CloseSquare,
                    N::EndCollection,
                    |s| {
                        s.expect_and_push_node(T::LowerIdent, N::Ident); // TODO: correct node type
                    },
                );
                self.push_node_end(N::BeginHeaderPlatform, N::EndHeaderPlatform, subtree_start);
            }
            Some(T::KwHosted) => {
                self.bump();
                let subtree_start = self.push_node_begin(N::BeginHeaderHosted);
                self.expect(T::UpperIdent);
                self.expect(T::KwExposes);
                self.expect_collection(
                    T::OpenSquare,
                    N::BeginCollection,
                    T::CloseSquare,
                    N::EndCollection,
                    |s| {
                        match s.cur() {
                            Some(T::LowerIdent) => {
                                s.bump();
                                s.push_node(N::Ident, Some(s.pos as u32 - 1));
                            }
                            Some(T::UpperIdent) => {
                                s.bump();
                                s.push_node(N::TypeName, Some(s.pos as u32 - 1));
                            }
                            t => {
                                s.push_error(Error::ExpectViolation(T::LowerIdent, t));
                                s.fast_forward_past_newline(); // TODO: this should fastforward to the close square
                            }
                        }
                    },
                );
                self.expect_header_imports();
                self.expect(T::KwGenerates);
                self.expect(T::UpperIdent);
                self.expect(T::KwWith);
                self.expect_collection(
                    T::OpenSquare,
                    N::BeginCollection,
                    T::CloseSquare,
                    N::EndCollection,
                    |s| {
                        s.expect_and_push_node(T::LowerIdent, N::Ident); // TODO: correct node type
                    },
                );
                self.push_node_end(N::BeginHeaderHosted, N::EndHeaderHosted, subtree_start);
            }
            Some(T::KwInterface) => {
                self.bump();
                let subtree_start = self.push_node_begin(N::BeginHeaderInterface);
                self.expect_and_push_node(T::UpperIdent, N::ModuleName);

                while self.consume(T::NoSpaceDotUpperIdent) {
                    self.push_node(N::DotModuleUpperIdent, Some(self.pos as u32 - 1));
                }

                self.expect(T::KwExposes);
                self.expect_collection(
                    T::OpenSquare,
                    N::BeginCollection,
                    T::CloseSquare,
                    N::EndCollection,
                    |s| {
                        s.expect_and_push_node(T::UpperIdent, N::Ident); // TODO: correct node type
                    },
                );

                self.expect_header_imports();

                self.push_node_end(
                    N::BeginHeaderInterface,
                    N::EndHeaderInterface,
                    subtree_start,
                );
            }
            Some(T::KwPackage) => {
                self.bump();
                let subtree_start = self.push_node_begin(N::BeginHeaderPackage);
                self.expect_and_push_node(T::String, N::String);
                self.expect(T::KwExposes);
                self.expect_collection(
                    T::OpenSquare,
                    N::BeginCollection,
                    T::CloseSquare,
                    N::EndCollection,
                    |s| {
                        match s.cur() {
                            Some(T::LowerIdent) => {
                                s.bump();
                                s.push_node(N::Ident, Some(s.pos as u32 - 1));
                            }
                            Some(T::UpperIdent) => {
                                s.bump();
                                s.push_node(N::TypeName, Some(s.pos as u32 - 1));
                            }
                            t => {
                                s.push_error(Error::ExpectViolation(T::LowerIdent, t));
                                s.fast_forward_past_newline(); // TODO: this should fastforward to the close square
                            }
                        }
                    },
                );
                self.expect(T::KwPackages);
                self.expect_collection(
                    T::OpenCurly,
                    N::BeginCollection,
                    T::CloseCurly,
                    N::EndCollection,
                    |s| {
                        s.expect(T::LowerIdent);
                        s.expect(T::OpColon);
                        s.expect(T::String);
                    },
                );

                self.push_node_end(N::BeginHeaderPackage, N::EndHeaderPackage, subtree_start);
            }
            _ => {}
        }

        self.push_next_frame(
            subtree_start,
            ExprCfg::default(),
            Frame::PushEnd(N::BeginFile, N::EndFile),
        );

        self.start_top_level_decls();
    }

    fn expect_header_imports(&mut self) {
        if self.consume(T::KwImports) {
            self.expect_collection(
                T::OpenSquare,
                N::BeginCollection,
                T::CloseSquare,
                N::EndCollection,
                |s| {
                    match s.cur() {
                        Some(T::LowerIdent) => {
                            s.bump();
                            s.push_node(N::Ident, Some(s.pos as u32 - 1));
                            s.consume_and_push_node(
                                T::NoSpaceDotUpperIdent,
                                N::DotModuleUpperIdent,
                            );
                            s.consume_and_push_node(
                                T::NoSpaceDotUpperIdent,
                                N::DotModuleUpperIdent,
                            );

                            if s.consume(T::Dot) {
                                s.expect_collection(
                                    T::OpenCurly,
                                    N::BeginCollection,
                                    T::CloseCurly,
                                    N::EndCollection,
                                    |s| {
                                        match s.cur() {
                                            Some(T::LowerIdent) => {
                                                s.bump();
                                                s.push_node(N::Ident, Some(s.pos as u32 - 1));
                                            }
                                            Some(T::UpperIdent) => {
                                                s.bump();
                                                s.push_node(N::TypeName, Some(s.pos as u32 - 1));
                                            }
                                            t => {
                                                s.push_error(Error::ExpectViolation(
                                                    T::LowerIdent,
                                                    t,
                                                ));
                                                s.fast_forward_past_newline(); // TODO: this should fastforward to the close square
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        Some(T::String) => {
                            s.bump();
                            s.push_node(N::String, Some(s.pos as u32 - 1));
                            s.expect(T::KwAs);
                            s.expect(T::LowerIdent);
                            s.expect(T::OpColon);
                            s.expect(T::UpperIdent);
                        }
                        Some(T::UpperIdent) => {
                            s.bump();
                            s.push_node(N::TypeName, Some(s.pos as u32 - 1));
                        }
                        t => {
                            s.push_error(Error::ExpectViolation(T::LowerIdent, t));
                            s.fast_forward_past_newline(); // TODO: this should fastforward to the close square
                        }
                    }
                },
            );
        } else {
            self.push_node_begin_end(N::BeginCollection, N::EndCollection);
        }
    }

    fn start_block_or_expr(&mut self, cfg: ExprCfg) {
        if self.at_newline() {
            self.start_block(cfg);
        } else {
            self.start_expr(cfg);
        }
    }

    fn start_expr(&mut self, mut cfg: ExprCfg) {
        cfg.expr_indent_floor = Some(self.cur_indent());
        self.push_next_frame_starting_here(
            cfg,
            Frame::StartExpr {
                min_prec: Prec::Outer,
            },
        );
    }

    fn start_pattern(&mut self, mut cfg: ExprCfg) {
        cfg.expr_indent_floor = Some(self.cur_indent());
        self.push_next_frame_starting_here(cfg, Frame::StartPattern);
    }

    fn start_block_item(&mut self, cfg: ExprCfg) {
        self.push_next_frame_starting_here(cfg, Frame::FinishBlockItem);
        let cfg = cfg.set_expr_indent_floor(Some(self.cur_indent()));
        self.start_expr(cfg);
        self.push_node(N::Dummy, None); // will be replaced by the actual node in pump_finish_block_item
    }

    fn start_block(&mut self, cfg: ExprCfg) {
        self.push_next_frame_starting_here(cfg, Frame::ContinueBlock);
        self.push_node(N::BeginBlock, None);
        self.start_block_item(cfg);
    }

    fn start_type(&mut self, cfg: ExprCfg, allow_clauses: bool, allow_commas: bool) {
        self.push_next_frame_starting_here(
            cfg,
            Frame::StartType {
                allow_clauses,
                allow_commas,
            },
        );
    }

    pub fn assert_end(&self) {
        assert_eq!(
            self.pos,
            self.buf.kinds.len(),
            "Expected to be at the end, but these tokens remain: {:?}",
            &self.buf.kinds[self.pos..]
        );
    }

    #[track_caller]
    fn check<T>(&self, v: Result<T, Error>) -> T {
        match v {
            Ok(v) => v,
            Err(e) => todo!("indentation error"),
        }
    }

    fn pump_continue_record(&mut self, subtree_start: u32, cfg: ExprCfg, mut start: bool) {
        // TODO: allow { existingRecord & foo: bar } syntax
        loop {
            if !start {
                if self.consume_end(T::CloseCurly) {
                    self.push_node(N::EndRecord, Some(subtree_start));
                    self.update_end(N::BeginRecord, subtree_start);
                    self.handle_field_access_suffix(subtree_start);
                    return;
                }

                self.expect(T::Comma);
            }

            start = false;

            if self.consume_end(T::CloseCurly) {
                self.push_node(N::EndRecord, Some(subtree_start));
                self.update_end(N::BeginRecord, subtree_start);
                self.handle_field_access_suffix(subtree_start);
                return;
            }

            let field_subtree_start = self.tree.len();

            self.expect_lower_ident_and_push_node();

            if self.consume_and_push_node(T::OpColon, N::InlineColon) {
                self.push_next_frame(subtree_start, cfg, Frame::ContinueRecord { start });
                self.push_next_frame(
                    field_subtree_start,
                    cfg,
                    Frame::PushEndOnly(N::EndRecordFieldPair),
                );
                self.start_expr(cfg.disable_multi_backpassing());
                return;
            }
        }
    }

    fn pump_continue_pattern_record(&mut self, subtree_start: u32, cfg: ExprCfg, mut start: bool) {
        loop {
            if !start {
                if self.consume_end(T::CloseCurly) {
                    self.push_node(N::EndPatternRecord, Some(subtree_start));
                    self.update_end(N::BeginPatternRecord, subtree_start);
                    return;
                }

                self.expect(T::Comma);
            }

            if self.consume_end(T::CloseCurly) {
                self.push_node(N::EndPatternRecord, Some(subtree_start));
                self.update_end(N::BeginPatternRecord, subtree_start);
                return;
            }

            self.expect_lower_ident_and_push_node();

            if self.consume_and_push_node(T::OpColon, N::InlineColon) {
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinuePatternRecord { start: false },
                );
                self.start_pattern(cfg.disable_multi_backpassing());
                return;
            } else if self.cur() == Some(T::Comma) || self.cur() == Some(T::CloseCurly) {
                self.push_node(N::PatternAny, Some(self.pos as u32));
            }

            start = false;
        }
    }

    fn expect_lower_ident_and_push_node(&mut self) {
        match self.cur() {
            Some(k) if k.is_keyword() => {
                // treat as an identifier
                self.bump();
                self.push_node(N::Ident, Some(self.pos as u32 - 1));
            }
            Some(T::LowerIdent) => {
                self.bump();
                self.push_node(N::Ident, Some(self.pos as u32 - 1));
            }
            _ => {
                self.push_error(Error::ExpectViolation(T::LowerIdent, self.cur()));
                self.fast_forward_past_newline();
            }
        }
    }

    fn pump_push_end_only(&mut self, subtree_start: u32, end: N) {
        self.push_node(end, Some(subtree_start));
    }

    fn pump_push_end(&mut self, subtree_start: u32, begin: N, end: N) {
        self.push_node(end, Some(subtree_start));
        self.update_end(begin, subtree_start);
    }

    fn pump_continue_type_tuple_or_paren(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::Comma) {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeTupleOrParen);
            self.start_type(cfg, false, true);
        } else {
            self.expect(T::CloseRound);
            self.push_node_end(N::BeginParens, N::EndParens, subtree_start);

            // Pseudo-token that the tokenizer produces for inputs like (a, b)c - there will be a NoSpace after the parens.
            if self.consume(T::NoSpace) {
                self.push_next_frame(subtree_start, cfg, Frame::PushEndOnly(N::EndTypeAdendum));
                let subtree_start = self.tree.len();
                self.start_type_ext(subtree_start, cfg);
            }
        }
    }

    fn pump_continue_expr_list(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume_end(T::CloseSquare) {
            self.push_node_end(N::BeginList, N::EndList, subtree_start);
            self.handle_field_access_suffix(subtree_start);
            return;
        }

        self.expect(T::Comma);

        if self.consume_end(T::CloseSquare) {
            self.push_node_end(N::BeginList, N::EndList, subtree_start);
            self.handle_field_access_suffix(subtree_start);
            return;
        }

        self.push_next_frame(subtree_start, cfg, Frame::ContinueExprList);
        self.start_expr(cfg.disable_multi_backpassing());
    }

    fn pump_continue_pattern_list(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume_end(T::CloseSquare) {
            self.push_node_end(N::BeginPatternList, N::EndPatternList, subtree_start);
            return;
        }

        self.expect(T::Comma);

        if self.consume_end(T::CloseSquare) {
            self.push_node_end(N::BeginPatternList, N::EndPatternList, subtree_start);
            return;
        }

        self.push_next_frame(subtree_start, cfg, Frame::ContinuePatternList);
        self.start_pattern(cfg.disable_multi_backpassing());
    }

    fn start_tag_union(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::BeginTypeTagUnion, None);
        if self.consume_end_tag_union(subtree_start, cfg) {
            return;
        }

        self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeTagUnion);

        self.expect_and_push_node(T::UpperIdent, N::Tag); // tag name

        self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeTagUnionArgs);
    }

    fn pump_continue_type_tag_union(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume_end_tag_union(subtree_start, cfg) {
            return;
        }

        self.expect(T::Comma);
        if self.consume_end_tag_union(subtree_start, cfg) {
            return;
        }

        self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeTagUnion);

        self.expect_and_push_node(T::UpperIdent, N::Tag); // tag name

        self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeTagUnionArgs);
    }

    fn consume_end_tag_union(&mut self, subtree_start: u32, cfg: ExprCfg) -> bool {
        if self.consume_and_push_node_end(
            T::CloseSquare,
            N::BeginTypeTagUnion,
            N::EndTypeTagUnion,
            subtree_start,
        ) {
            self.maybe_start_type_adendum(subtree_start, cfg);
            return true;
        }

        false
    }

    fn pump_continue_tag_union_args(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if matches!(self.cur(), Some(T::CloseSquare | T::Comma)) {
            return;
        }
        self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeTagUnionArgs);
        self.start_type(cfg, false, true);
    }

    fn pump_continue_implements_method_decl(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.continue_implements_method_decl_body(subtree_start, cfg);
    }

    fn continue_implements_method_decl_body(&mut self, subtree_start: u32, cfg: ExprCfg) {
        // We continue as long as the indent is the same as the start of the implements block.
        if dbg!(self
            .cur_indent()
            .is_indented_more_than(cfg.block_indent_floor)
            .ok_or(Error::InconsistentIndent)
            .expect("TODO: error handling"))
        {
            self.expect_and_push_node(T::LowerIdent, N::Ident);
            self.expect(T::OpColon); // TODO: need to add a node?
            self.push_next_frame(subtree_start, cfg, Frame::ContinueImplementsMethodDecl);
            self.start_type(cfg, false, true);
        } else {
            self.push_node(N::EndImplements, Some(subtree_start));
        }
    }

    fn start_type_record(&mut self, cfg: ExprCfg) {
        let subtree_start = self.push_node_begin(N::BeginTypeRecord);

        if self.consume_and_push_node_end(
            T::CloseCurly,
            N::BeginTypeRecord,
            N::EndTypeRecord,
            subtree_start,
        ) {
            return;
        }

        self.expect_lower_ident_and_push_node();
        // TODO: ? for optional fields
        self.expect(T::OpColon); // TODO: need to add a node?

        self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeRecord);
        self.start_type(cfg, false, true);
    }

    fn consume_end_type_record(&mut self, subtree_start: u32, cfg: ExprCfg) -> bool {
        if self.consume_and_push_node_end(
            T::CloseCurly,
            N::BeginTypeRecord,
            N::EndTypeRecord,
            subtree_start,
        ) {
            self.maybe_start_type_adendum(subtree_start, cfg);
            return true;
        }

        false
    }

    fn pump_continue_type_record(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume_end_type_record(subtree_start, cfg) {
            return;
        }

        self.expect(T::Comma);
        if self.consume_end_type_record(subtree_start, cfg) {
            return;
        }

        self.expect_and_push_node(T::LowerIdent, N::Ident);
        // TODO: ? for optional fields
        self.expect(T::OpColon); // TODO: need to add a node?

        self.push_next_frame(subtree_start, cfg, Frame::ContinueTypeRecord);
        self.start_type(cfg, false, true);
    }

    fn maybe_start_type_adendum(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::NoSpace) {
            self.push_next_frame(subtree_start, cfg, Frame::PushEndOnly(N::EndTypeAdendum));
            self.start_type(cfg, false, false);
        }
    }

    fn consume_token_with_negative_lookahead(
        &mut self,
        tok: T,
        negative_following: &[T],
    ) -> Option<usize> {
        let mut pos = self.pos;

        if self.buf.kind(pos) != Some(tok) {
            return None;
        }

        let found = pos;

        pos += 1;

        for &neg in negative_following {
            if self.buf.kind(pos) == Some(neg) {
                return None;
            }

            pos += 1;
        }
        self.pos = found + 1;
        Some(found)
    }

    fn expect_collection(
        &mut self,
        open: T,
        open_node: N,
        close: T,
        close_node: N,
        f: impl Fn(&mut Self),
    ) {
        self.expect(open);
        let subtree_start = self.push_node_begin(open_node);
        if self.consume(close) {
            self.push_node_end(open_node, close_node, subtree_start);
            return;
        }
        loop {
            f(self);

            if self.consume_end(close) {
                self.push_node_end(open_node, close_node, subtree_start);
                return;
            }
            self.expect(T::Comma);
            if self.consume_end(close) {
                self.push_node_end(open_node, close_node, subtree_start);
                return;
            }
        }
    }

    fn plausible_expr_continue_comes_next(&self) -> bool {
        match self.buf.kind(self.pos + 1) {
            Some(T::OpAssign | T::OpColon | T::OpColonEqual) => true,
            None | Some(T::CloseRound) | Some(T::CloseSquare) | Some(T::CloseCurly) => true,
            // TODO: maybe also allow binops / etc?
            // note: we can't allow expr start tokens here (i.e. implicitly allowing `if foo` to be concidered a call)
            _ => false,
        }
    }

    fn at_plausible_when_pattern(&self) -> bool {
        if self.at_terminator() {
            return false;
        }

        match self.cur() {
            Some(T::KwWhen | T::KwIf) => return false,
            _ => {}
        }

        let mut depth = 0;
        let mut offset = 0;

        loop {
            match self.peek_at(offset) {
                Some(T::OpenRound | T::OpenSquare | T::OpenCurly) => {
                    depth += 1;
                }
                Some(T::CloseRound | T::CloseSquare | T::CloseCurly) => {
                    depth -= 1;
                }
                Some(T::OpBar | T::OpArrow | T::KwIf) if depth == 0 => {
                    return true;
                }
                Some(T::OpAssign | T::Comma | T::OpColon | T::OpColonEqual) => {
                    if depth == 0 {
                        return false;
                    }
                }
                Some(
                    T::OpPlus
                    | T::OpBinaryMinus
                    | T::OpStar
                    | T::OpSlash
                    | T::OpPercent
                    | T::OpPizza
                    | T::OpBackslash
                    | T::KwWhen
                    | T::KwElse
                    | T::OpBang
                    | T::OpPercent,
                ) =>
                // TODO: more binops
                {
                    return false;
                }
                Some(
                    T::UpperIdent
                    | T::LowerIdent
                    | T::Underscore
                    | T::NamedUnderscore
                    | T::Int
                    | T::Float
                    | T::String
                    | T::SingleQuote
                    | T::NoSpaceDotLowerIdent
                    | T::DotLowerIdent
                    | T::DoubleDot,
                ) => {
                    // ok!
                }
                None => {
                    return false;
                }
                t => todo!("{:?} at depth {}", t, depth),
            }
            offset += 1;
        }
    }
}

struct FormattedBuffer {
    text: String,
    indent: usize,
    pending_spaces: usize,
    pending_newlines: usize,

    pretend_space: bool,
}

impl FormattedBuffer {
    fn new() -> FormattedBuffer {
        FormattedBuffer {
            text: String::new(),
            indent: 0,
            pending_spaces: 0,
            pending_newlines: 0,

            pretend_space: false,
        }
    }

    fn _flush(&mut self) {
        // set indent spaces
        if self.pending_newlines > 0 {
            self.pending_spaces = self.indent * 4;
        }
        for _ in 0..self.pending_newlines {
            self.text.push('\n');
        }
        for _ in 0..self.pending_spaces {
            self.text.push(' ');
        }
        self.pending_newlines = 0;
        self.pending_spaces = 0;
        self.pretend_space = false;
    }

    fn push_str(&mut self, s: &str) {
        self._flush();
        self.text.push_str(s);
    }

    fn push_sp_str_sp(&mut self, s: &str) {
        self.space();
        self.push_str(s);
        self.space();
    }

    fn push_sp_str(&mut self, s: &str) {
        self.space();
        self.push_str(s);
    }

    fn push_newline(&mut self) {
        self.pending_spaces = 0;
        self.pending_newlines += 1;
    }

    fn space(&mut self) {
        if !self.pretend_space
            && self.pending_spaces == 0
            && self.pending_newlines == 0
            && self.text.len() > 0
        {
            self.pending_spaces = 1;
        }
    }

    fn indent(&mut self) {
        self.indent += 1;
    }

    fn dedent(&mut self) {
        self.indent -= 1;
    }
}

struct ShowTreePosition<'a>(&'a Tree, u32);

impl fmt::Debug for ShowTreePosition<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // print a list of nodes in the tree, with the current node highlighted
        for (i, &node) in self.0.kinds.iter().enumerate() {
            if i == self.1 as usize {
                write!(f, "<-({:?})-> ", node)?;
            } else {
                write!(f, "{:?} ", node)?;
            }
        }
        Ok(())
    }
}

pub struct TreeWalker<'b> {
    kinds: &'b [N],
    indices: &'b [u32],
    pos: usize,
}

impl<'b> TreeWalker<'b> {
    pub fn new(tree: &'b Tree) -> Self {
        Self {
            kinds: &tree.kinds,
            indices: &tree.indices,
            pos: 0,
        }
    }

    pub fn next(&mut self) -> Option<N> {
        let res = self.kinds.get(self.pos).copied();
        self.pos += 1;
        res
    }

    pub fn cur(&self) -> Option<N> {
        self.kinds.get(self.pos).copied()
    }

    pub fn cur_index(&self) -> Option<(N, usize, u32)> {
        self.cur().map(|n| (n, self.pos, self.indices[self.pos]))
    }

    pub fn next_index(&mut self) -> Option<(N, usize, u32)> {
        let res = self.cur_index();
        self.pos += 1;
        res
    }
}

#[derive(Copy, Clone)]
pub struct ParsedCtx<'a> {
    pub tree: &'a Tree,
    pub toks: &'a TokenenizedBuffer,
    pub text: &'a str,
}

impl<'a> ParsedCtx<'a> {
    pub fn text(&self, token_index: u32) -> &'a str {
        let offset = self.toks.offsets[token_index as usize] as usize;
        let len = self.toks.lengths[token_index as usize] as usize;
        &self.text[offset..offset + len]
    }
}

struct TokenizedBufferFollower<'a> {
    text: &'a str,
    toks: &'a TokenenizedBuffer,
    pos: usize,
}

impl<'a> TokenizedBufferFollower<'a> {
    fn _bump(&mut self, trivia: &mut Vec<Comment>) {
        panic!();
        // self.toks
        //     .extract_comments_at(self.text, self.pos, trivia);
        self.pos += 1;
    }

    fn check_next_token(&mut self, tok: T, trivia: &mut Vec<Comment>) {
        panic!();
        // if tok != T::Newline {
        //     // fast forward past newlines in the underlying buffer
        //     while self.toks.kind(self.pos) == Some(T::Newline) {
        //         self._bump(trivia);
        //     }
        // }
        if self.toks.kind(self.pos) != Some(tok) {
            panic!(
                "programming error: misaligned token stream when formatting.\n\
                Expected {:?} at position {}, found {:?} instead.",
                tok,
                self.pos,
                self.toks.kind(self.pos)
            );
        }
        self._bump(trivia);
    }
}

trait UpProp {
    type Out: Copy;
    type Accum;

    fn leaf(&mut self, node: N, index: u32) -> Self::Out;
    fn init(&mut self) -> Self::Accum;
    fn pump(&mut self, accum: Self::Accum, next_sibling: Self::Out) -> Self::Accum;
    fn finish(&mut self, accum: Self::Accum, parent: N) -> Self::Out;
}

fn bubble_up<P: UpProp>(prop: &mut P, tree: &Tree) -> Vec<P::Out> {
    let mut res = Vec::with_capacity(tree.len() as usize);
    let mut stack: Vec<(u32, P::Out)> = Vec::new();
    let mut accum = prop.init();

    for (i, &node) in tree.kinds.iter().enumerate() {
        let index = tree.indices[i];

        let item = match node.index_kind() {
            NodeIndexKind::Begin | NodeIndexKind::Token | NodeIndexKind::Unused => {
                prop.leaf(node, index)
            }
            NodeIndexKind::EndOnly | NodeIndexKind::End => {
                let mut begin = stack.len();
                while begin > 0 && stack[begin - 1].0 > index {
                    begin -= 1;
                }

                let mut accum = prop.init();

                for (_, item) in stack.drain(begin..) {
                    accum = prop.pump(accum, item);
                }

                prop.finish(accum, node)
            }
            NodeIndexKind::EndSingleToken => {
                prop.leaf(node, index);
                let (_, item) = stack.pop().unwrap();
                let mut accum = prop.init();
                accum = prop.pump(accum, item);
                prop.finish(accum, node)
            }
        };
        stack.push((index, item));
        res.push(item);
    }

    res
}

trait DownProp: Copy {
    fn update_from_parent(&mut self, node: N, index: u32, parent: Self, parent_node: N);
}

fn bubble_down_mut<P: DownProp>(tree: &Tree, state: &mut Vec<P>) {
    assert_eq!(tree.len() as usize, state.len());
    let mut stack = Vec::<(usize, P, N)>::new();
    for (i, &node) in tree.kinds.iter().enumerate().rev() {
        let index = tree.indices[i];

        while let Some(&(begin_index, parent_state, parent_node)) = stack.last() {
            if begin_index > i {
                stack.pop();
                continue;
            }

            state[i].update_from_parent(node, index, parent_state, parent_node);
            break;
        }

        match node.index_kind() {
            NodeIndexKind::Begin | NodeIndexKind::Token | NodeIndexKind::Unused => {}
            NodeIndexKind::EndOnly | NodeIndexKind::End | NodeIndexKind::EndSingleToken => {
                stack.push((index as usize, state[i], node));
            }
        }
    }
}

#[derive(Default, Copy, Clone)]
struct FmtInfo {
    has_newline: bool,
    has_comment: bool,
}

struct FmtInfoProp<'a> {
    text: &'a str,
    toks: &'a TokenenizedBuffer,
    pos: usize,
    trivia: Vec<Comment>,
    comment_indices: Vec<(usize, usize)>,
}

impl<'a> FmtInfoProp<'a> {
    fn _bump(&mut self) {
        panic!();
        // self.toks
        //     .extract_comments_at(self.text, self.pos, &mut self.trivia);
        self.pos += 1;
    }

    fn check_next_token(&mut self, tok: T) -> FmtInfo {
        panic!();
        // debug_assert!(tok != T::Newline);

        let comment_start = self.pos;
        let mut newline = false;

        // fast forward past newlines in the underlying buffer
        // also capture trivia at this stage
        // while self.toks.kind(self.pos) == Some(T::Newline) {
        //     newline = true;
        //     self._bump();
        // }

        if self.toks.kind(self.pos) != Some(tok) {
            panic!(
                "programming error: misaligned token stream when formatting.\n\
                Expected {:?} at position {}, found {:?} instead.",
                tok,
                self.pos,
                self.toks.kind(self.pos)
            );
        }

        self.pos += 1;

        let comment_end = self.pos;
        FmtInfo {
            has_newline: newline,
            has_comment: comment_start != comment_end,
        }
    }
}

impl<'a> UpProp for FmtInfoProp<'a> {
    type Out = FmtInfo;
    type Accum = FmtInfo;

    fn init(&mut self) -> Self::Accum {
        FmtInfo::default()
    }

    fn leaf(&mut self, node: N, index: u32) -> Self::Out {
        let kind = node.index_kind();
        match node {
            N::BeginAssign
            | N::BeginTopLevelDecls
            | N::HintExpr
            | N::EndIf
            | N::EndWhen
            | N::InlineApply
            | N::EndLambda
            | N::EndBlock
            | N::EndApply
            | N::EndBinOpPlus
            | N::EndBinOpStar
            | N::EndPizza
            | N::BeginBlock => FmtInfo::default(),

            N::EndTopLevelDecls | N::EndAssign => {
                panic!("not expected in ::leaf");
            }

            N::Ident => self.check_next_token(T::LowerIdent),
            N::InlineAssign => self.check_next_token(T::OpAssign),

            N::BeginIf => self.check_next_token(T::KwIf),
            N::InlineKwThen => self.check_next_token(T::KwThen),
            N::InlineKwElse => self.check_next_token(T::KwElse),

            N::BeginWhen => self.check_next_token(T::KwWhen),
            N::InlineKwIs => self.check_next_token(T::KwIs),
            N::InlineWhenArrow => self.check_next_token(T::OpArrow),

            N::InlineBinOpPlus => self.check_next_token(T::OpPlus),
            N::InlineBinOpStar => self.check_next_token(T::OpStar),
            N::InlinePizza => self.check_next_token(T::OpPizza),

            N::BeginLambda => self.check_next_token(T::OpBackslash),
            N::InlineLambdaArrow => self.check_next_token(T::OpArrow),

            _ => todo!("leaf {:?}", node),
        }
    }

    fn pump(&mut self, accum: Self::Accum, next_sibling: Self::Out) -> Self::Accum {
        // or together the fields
        FmtInfo {
            has_newline: accum.has_newline || next_sibling.has_newline,
            has_comment: accum.has_comment || next_sibling.has_comment,
        }
    }

    fn finish(&mut self, accum: Self::Accum, _parent: N) -> Self::Out {
        accum
    }
}

impl DownProp for FmtInfo {
    fn update_from_parent(&mut self, node: N, index: u32, parent: Self, parent_node: N) {
        match parent_node {
            N::EndTopLevelDecls | N::EndBlock => return,
            // TODO: exempt some other nodes from this
            _ => {}
        }
        self.has_newline |= parent.has_newline;
        self.has_comment |= parent.has_comment;
    }
}

fn pretty(tree: &Tree, toks: &TokenenizedBuffer, text: &str) -> FormattedBuffer {
    let ctx = ParsedCtx { tree, toks, text };

    let mut prop = FmtInfoProp {
        text,
        toks,
        pos: 0,
        trivia: Vec::new(),
        comment_indices: Vec::new(),
    };

    let mut info = bubble_up(&mut prop, tree);
    bubble_down_mut(tree, &mut info);

    let mut buf = FormattedBuffer::new();

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    enum St {
        Assign0,
        Assign1,
        Expr,
    }

    let mut stack = Vec::new();

    // dbg!(&ctx.toks.kinds);

    for (i, &node) in tree.kinds.iter().enumerate() {
        let index = tree.indices[i];
        match node {
            N::Ident => buf.push_sp_str_sp(ctx.text(index)),
            N::BeginTopLevelDecls => {}
            N::EndTopLevelDecls => {}

            N::BeginAssign => {
                stack.push(St::Assign0);
            }
            N::InlineAssign => {
                assert_eq!(stack.pop(), Some(St::Assign0));
                buf.push_sp_str_sp("=");
                stack.push(St::Assign1);
            }
            N::EndAssign => {
                assert_eq!(stack.pop(), Some(St::Assign1));
                buf.push_newline();
            }

            N::HintExpr => {}

            N::BeginIf => {
                buf.push_sp_str_sp("if");
            }
            N::InlineKwThen => {
                buf.push_sp_str_sp("then");
                buf.push_newline();
                buf.indent();
            }
            N::InlineKwElse => {
                buf.dedent();
                buf.push_newline();
                buf.push_sp_str_sp("else");
                buf.push_newline();
                buf.indent();
            }
            N::EndIf => {
                buf.dedent();
                buf.push_newline();
            }

            N::BeginWhen => {
                buf.push_sp_str_sp("when");
            }
            N::InlineKwIs => {
                buf.push_sp_str_sp("is");
            }
            N::InlineWhenArrow => {
                buf.push_sp_str_sp("->");
            }
            N::EndWhen => {}

            N::InlineApply => {}
            N::InlineBinOpPlus => buf.push_sp_str_sp("+"),
            N::InlineBinOpStar => buf.push_sp_str_sp("*"),
            N::InlinePizza => buf.push_sp_str_sp("|>"),

            N::BeginLambda => {
                buf.push_sp_str("\\");
                // hack!!!!
                buf.pretend_space = true;
            }
            N::InlineLambdaArrow => buf.push_sp_str_sp("->"),
            N::EndLambda => {}

            N::BeginBlock => {
                buf.push_newline();
                buf.indent();
            }
            N::EndBlock => {
                buf.dedent();
                buf.push_newline();
            }
            N::EndApply | N::EndBinOpPlus | N::EndBinOpStar | N::EndPizza => {}
            _ => todo!("pretty {:?}", node),
        }
    }
    buf
}

struct CommentExtractor<'a> {
    text: &'a str,
    toks: &'a TokenenizedBuffer,
    pos: usize,
    comments: Vec<Comment>,
    comment_indices: Vec<(usize, usize)>,
}

impl<'a> CommentExtractor<'a> {
    fn new(text: &'a str, toks: &'a TokenenizedBuffer) -> Self {
        Self {
            text,
            toks,
            pos: 0,
            comments: Vec::new(),
            comment_indices: Vec::new(),
        }
    }

    fn check_next_token(&mut self, tok: T) {
        panic!();
        // debug_assert!(tok != T::Newline);

        // while self.toks.kind(self.pos) == Some(T::Newline) {
        //     self.toks.extract_comments_at(self.text, self.pos, &mut self.comments);
        //     self.pos += 1;
        // }

        if self.toks.kind(self.pos) != Some(tok) {
            panic!(
                "programming error: misaligned token stream when formatting.\n\
                Expected {:?} at position {}, found {:?} instead.",
                tok,
                self.pos,
                self.toks.kind(self.pos)
            );
        }

        self.pos += 1;
    }
}

impl<'a> CommentExtractor<'a> {
    fn consume(&mut self, node: N) -> &[Comment] {
        let begin = self.comments.len();
        let kind = node.index_kind();
        match node {
            N::BeginAssign
            | N::BeginTopLevelDecls
            | N::HintExpr
            | N::EndIf
            | N::EndWhen
            | N::InlineApply
            | N::EndLambda
            | N::EndBlock
            | N::EndApply
            | N::EndBinOpPlus
            | N::EndBinOpStar
            | N::EndBinOpMinus
            | N::EndPizza
            | N::BeginBlock
            | N::EndTopLevelDecls
            | N::EndAssign => {}

            N::Ident => self.check_next_token(T::LowerIdent),
            N::UpperIdent => self.check_next_token(T::UpperIdent),
            N::Num => self.check_next_token(T::Int),
            N::Float => self.check_next_token(T::Float),
            N::InlineAssign => self.check_next_token(T::OpAssign),

            N::BeginIf => self.check_next_token(T::KwIf),
            N::InlineKwThen => self.check_next_token(T::KwThen),
            N::InlineKwElse => self.check_next_token(T::KwElse),

            N::BeginWhen => self.check_next_token(T::KwWhen),
            N::InlineKwIs => self.check_next_token(T::KwIs),
            N::InlineWhenArrow => self.check_next_token(T::OpArrow),

            N::InlineBinOpPlus => self.check_next_token(T::OpPlus),
            N::InlineBinOpStar => self.check_next_token(T::OpStar),
            N::InlineBinOpMinus => self.check_next_token(T::OpBinaryMinus),
            N::InlinePizza => self.check_next_token(T::OpPizza),

            N::BeginLambda => self.check_next_token(T::OpBackslash),
            N::InlineLambdaArrow => self.check_next_token(T::OpArrow),

            _ => todo!("comment extract {:?}", node),
        }

        &self.comments[begin..]
    }
}
