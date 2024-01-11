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

use crate::cypress_token::{Comment, Indent, TokenenizedBuffer, T};
use std::{collections::{btree_map::Keys, VecDeque}, f32::consts::E};

pub struct Tree {
    kinds: Vec<N>,
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
    BeginList,
    EndList,

    /// Record literals, e.g. `{ x: 1, y: 2 }`
    BeginRecord,
    EndRecord,

    /// Record updates (e.g. `{ x & y: 3 }`)
    BeginRecordUpdate,
    EndRecordUpdate,

    /// Parentheses, e.g. `(1 + 2)`
    BeginParens,
    EndParens,

    /// Tuple literals, e.g. `(1, 2)`
    BeginTuple,
    EndTuple,

    /// Indented block of statements and expressions
    BeginBlock,
    EndBlock,

    /// Function application, e.g. `f x`
    InlineApply,
    EndApply,

    /// Pizza operator, e.g. `x |> f`
    InlinePizza,
    EndPizza,

    /// Assignment declaration, e.g. `x = 1`
    BeginAssign,
    InlineAssign,
    EndAssign,

    /// Binary operators, e.g. `x + y`
    InlineBinOpPlus,
    EndBinOpPlus,
    InlineBinOpStar,
    EndBinOpStar,

    /// Unary operator, e.g. `-x`
    BeginUnaryOp,
    EndUnaryOp,

    /// If expression, e.g. `if x then y else z`
    BeginIf,
    InlineKwThen,
    InlineKwElse,
    EndIf,

    /// A when expression, e.g. `when x is y -> z` (you need a newline after the 'is')
    BeginWhen,
    InlineKwIs,
    InlineWhenArrow,
    EndWhen,

    /// A lambda expression, e.g. `\x -> x`
    BeginLambda,
    InlineLambdaArrow,
    EndLambda,

    EndTopLevelDecls,
    BeginTopLevelDecls,
    Dummy,

    HintExpr,
    InlineColon,

    EndTypeLambda,
    TypeName,
    AbilityName,
    InlineKwImplements,
    EndTypeOrTypeAlias,
    BeginTypeOrTypeAlias,
}

enum NodeIndexKind {
    Begin,   // this is a begin node; the index points one past the corresponding end node
    End,     // this is an end node; the index points to the corresponding begin node
    EndOnly, // this is an end node; the index points to the first child and there is no corresponding begin node
    Token,   // the index points to a token

    Unused, // we don't use the index for this node
}

impl N {
    fn is_decl(self) -> bool {
        match self {
            N::EndAssign => true,
            _ => false,
        }
    }

    fn index_kind(self) -> NodeIndexKind {
        match self {
            N::BeginList
            | N::BeginRecord
            | N::BeginRecordUpdate
            | N::BeginParens
            | N::BeginTuple
            | N::BeginBlock
            | N::BeginAssign
            | N::BeginTypeOrTypeAlias
            | N::BeginIf
            | N::BeginWhen
            | N::BeginLambda
            | N::BeginUnaryOp
            | N::BeginTopLevelDecls => NodeIndexKind::Begin,
            N::EndList
            | N::EndRecord
            | N::EndRecordUpdate
            | N::EndParens
            | N::EndTuple
            | N::EndBlock
            | N::EndAssign
            | N::EndTypeOrTypeAlias
            | N::EndIf
            | N::EndWhen
            | N::EndLambda
            | N::EndTopLevelDecls => NodeIndexKind::End,
            N::InlineApply
            | N::InlinePizza
            | N::InlineAssign
            | N::InlineBinOpPlus
            | N::InlineBinOpStar
            | N::InlineKwThen
            | N::InlineKwElse
            | N::InlineKwImplements
            | N::InlineKwIs
            | N::InlineLambdaArrow
            | N::InlineColon
            | N::InlineWhenArrow => NodeIndexKind::Token,
            N::Num
            | N::Str
            | N::Ident
            | N::TypeName
            | N::AbilityName
            | N::Tag
            | N::OpaqueRef
            | N::Access
            | N::AccessorFunction => NodeIndexKind::Token,
            N::Dummy | N::HintExpr => NodeIndexKind::Unused,
            N::EndApply | N::EndPizza | N::EndBinOpPlus | N::EndBinOpStar | N::EndUnaryOp
            | N::EndTypeLambda => {
                NodeIndexKind::EndOnly
            }
            N::Float | N::SingleQuote | N::Underscore | N::Crash | N::Dbg => NodeIndexKind::Token,
        }
    }
}

impl TokenenizedBuffer {
    fn kind(&self, pos: usize) -> Option<T> {
        self.kinds.get(pos).copied()
    }
}

impl Tree {
    fn new() -> Tree {
        Tree {
            kinds: Vec::new(),
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
    DeclSeq,  // BinOp::DeclSeq,
    Decl,     // BinOp::Assign, BinOp::Backpassing,
    Pizza,    // BinOp::Pizza,
    AndOr,    // BinOp::And, BinOp::Or,
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
    DefineTypeOrTypeAlias,
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
            BinOp::AssignBlock | BinOp::Assign | BinOp::DefineTypeOrTypeAlias | BinOp::Backpassing => Prec::Decl,
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
            BinOp::DeclSeq => Assoc::Right,
            BinOp::Assign | BinOp::Backpassing | BinOp::DefineTypeOrTypeAlias => Assoc::Right,
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
            BinOp::Apply | BinOp::Pizza | BinOp::DeclSeq => true,
            _ => false,
        }
    }

    fn to_inline(&self) -> N {
        match self {
            BinOp::AssignBlock => todo!(),
            BinOp::DeclSeq => todo!(),
            BinOp::Assign => N::InlineAssign,
            BinOp::DefineTypeOrTypeAlias => N::InlineColon,
            BinOp::Backpassing => todo!(),
            BinOp::Pizza => N::InlinePizza,
            BinOp::And => todo!(),
            BinOp::Or => todo!(),
            BinOp::Equals => todo!(),
            BinOp::NotEquals => todo!(),
            BinOp::LessThan => todo!(),
            BinOp::GreaterThan => todo!(),
            BinOp::LessThanOrEq => todo!(),
            BinOp::GreaterThanOrEq => todo!(),
            BinOp::Plus => N::InlineBinOpPlus,
            BinOp::Minus => todo!(),
            BinOp::Star => N::InlineBinOpStar,
            BinOp::Slash => todo!(),
            BinOp::DoubleSlash => todo!(),
            BinOp::Percent => todo!(),
            BinOp::Caret => todo!(),
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
            BinOp::Minus => N::EndBinOpStar,
            BinOp::Assign => N::EndAssign,
            BinOp::AssignBlock => N::EndAssign,
            _ => todo!("binop to node {:?}", op),
        }
    }
}

#[derive(Debug)]
enum Frame {
    StartExpr {
        min_prec: Prec,
    },
    ContinueExpr {
        min_prec: Prec,
        cur_op: Option<BinOp>,
        num_found: usize,
    },
    ContinueBlock,
    FinishParen,
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
    StartType,
    ContinueType,
    ContinueTypeFunction,
    FinishTypeFunction,
    ContinueWhereClause,
    FinishTypeOrTypeAlias,
}

impl Frame {
    fn start_expr() -> Frame {
        Frame::StartExpr {
            min_prec: Prec::DeclSeq,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExprCfg {
    expr_indent_floor: Option<Indent>, // expression continuations must be indented more than this.
    when_branch_indent_floor: Option<Indent>, // when branches must be indented more than this. None means no restriction.
    allow_multi_backpassing: bool,
}

impl Default for ExprCfg {
    fn default() -> Self {
        ExprCfg {
            expr_indent_floor: None,
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
    fn from_buf(buf: TokenenizedBuffer) -> Self {
        State {
            frames: vec![],
            buf: buf,
            pos: 0,
            line: 0,
            tree: Tree::new(),
        }
    }

    fn cur(&self) -> Option<T> {
        self.buf.kind(self.pos)
    }

    fn cur_indent(&self) -> Indent {
        self.buf
            .indents
            .get(self.line)
            .copied()
            .unwrap_or(Indent::default())
    }

    fn at_terminator(&self) -> bool {
        matches!(
            self.cur(),
            None | Some(T::CloseRound) | Some(T::CloseSquare) | Some(T::CloseCurly)
        )
    }

    fn bump(&mut self) {
        debug_assert!(self.cur() != Some(T::Newline));
        self.pos += 1;
    }

    fn expect(&mut self, tok: T) {
        debug_assert!(tok != T::Newline); // Use expect_newline instead
        if self.cur() != Some(tok) {
            todo!("expecting {:?} but found {:?}", tok, self.cur());
        }
        self.bump();
    }

    fn expect_newline(&mut self) {
        if self.cur() != Some(T::Newline) {
            todo!("expecting newline")
        }
        self.pos += 1;
        self.line += 1;
    }

    // Advances the cursor if the next tokens are <tok>, <tok newline> or <newline tok>
    // Returns Some(pos) for the token index of the given tok, or None if there wasn't a match
    fn consume_newline_agnostic(&mut self, tok: T) -> Option<u32> {
        match self.cur() {
            Some(T::Newline) => {
                if self.buf.kind(self.pos + 1) == Some(tok) {
                    let res = Some(self.pos as u32 + 1);
                    self.pos += 2;
                    self.line += 1;
                    res
                } else {
                    None
                }
            }
            Some(t) => {
                if t == tok {
                    let res = Some(self.pos as u32);
                    self.pos += 1;
                    self.consume_newline();
                    res
                } else {
                    None
                }
            }
            None => None,
        }
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
        eprintln!(
            "{:indent$}@{} updating end {} -> {}",
            "",
            self.pos,
            subtree_start,
            self.tree.len(),
            indent = 2 * self.frames.len() + 2
        );
        assert_eq!(self.tree.kinds[subtree_start as usize], kind);
        self.tree.paird_group_ends[subtree_start as usize] = self.tree.len() as u32;
    }

    fn push_node(&mut self, kind: N, subtree_start: Option<u32>) {
        eprintln!(
            "{:indent$}@{} pushing kind {}:{:?} starting at {:?}",
            "",
            self.pos,
            self.tree.kinds.len(),
            kind,
            subtree_start,
            indent = 2 * self.frames.len() + 2
        );
        self.tree.kinds.push(kind);
        let pos = subtree_start.unwrap_or(self.tree.paird_group_ends.len() as u32);
        self.tree.paird_group_ends.push(pos);
        // eprintln!("{:indent$}tree: {:?}", "", self.tree.debug_vis_grouping(), indent = 2 * self.frames.len() + 4);
    }

    fn push_next_frame(&mut self, subtree_start: u32, cfg: ExprCfg, frame: Frame) {
        eprintln!(
            "{:indent$}@{} pushing frame {:?}",
            "",
            self.pos,
            frame,
            indent = 2 * self.frames.len() + 2
        );
        self.frames.push((subtree_start, cfg, frame));
    }

    fn push_next_frame_starting_here(&mut self, cfg: ExprCfg, frame: Frame) {
        let subtree_start = self.tree.len();
        self.push_next_frame(subtree_start, cfg, frame)
    }

    fn pump(&mut self) {
        while let Some((subtree_start, cfg, frame)) = self.frames.pop() {
            eprintln!(
                "{:indent$}@{} pumping frame {:?} starting at {}",
                "",
                self.pos,
                frame,
                subtree_start,
                indent = 2 * self.frames.len()
            );
            match frame {
                Frame::StartExpr { min_prec } => self.pump_start_expr(subtree_start, cfg, min_prec),
                Frame::FinishParen => self.pump_finish_paren(),
                Frame::FinishLambda => self.pump_finish_lambda(subtree_start),
                Frame::FinishBlockItem => self.pump_finish_block_item(subtree_start),
                Frame::ContinueExpr {
                    min_prec,
                    cur_op,
                    num_found,
                } => self.pump_continue_expr(subtree_start, cfg, min_prec, cur_op, num_found),
                Frame::ContinueBlock => self.pump_continue_block(subtree_start, cfg),
                Frame::FinishAssign => self.pump_finish_assign(subtree_start, cfg),
                Frame::ContinueTopLevel { num_found } => {
                    self.pump_continue_top_level(subtree_start, cfg, num_found)
                }
                Frame::ContinueLambdaArgs => self.pump_continue_lambda_args(subtree_start, cfg),
                Frame::ContinueIf { next } => self.pump_continue_if(subtree_start, cfg, next),
                Frame::ContinueWhen { next } => self.pump_continue_when(subtree_start, cfg, next),
                Frame::StartType => self.pump_start_type(subtree_start, cfg),
                Frame::ContinueType => self.pump_continue_type(subtree_start, cfg),
                Frame::ContinueTypeFunction => self.pump_continue_type_function(subtree_start, cfg),
                Frame::FinishTypeFunction => self.pump_finish_type_function(subtree_start, cfg),
                Frame::ContinueWhereClause => self.pump_continue_where_clause(subtree_start, cfg),
                Frame::FinishTypeOrTypeAlias => self.pump_finish_type_or_type_alias(subtree_start, cfg),
            }
        }
    }

    fn pump_start_expr(&mut self, subtree_start: u32, cfg: ExprCfg, mut min_prec: Prec) {
        loop {
            match self.cur() {
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
                    self.push_next_frame(subtree_start, cfg, Frame::FinishParen);
                    min_prec = Prec::Outer;
                    continue;
                }
                Some(T::LowerIdent) => {
                    self.bump();
                    self.push_node(N::Ident, Some(self.pos as u32 - 1));

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
                }
                Some(T::IntBase10) => {
                    self.bump();
                    self.push_node(N::Num, Some(self.pos as u32 - 1));

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
                }
                Some(T::OpBackslash) => {
                    self.bump();
                    self.push_next_frame_starting_here(cfg, Frame::ContinueLambdaArgs);
                    self.push_node(N::BeginLambda, None);
                    return;
                }
                Some(T::KwIf) => {
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
                Some(T::KwWhen) => {
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
                k => todo!("{:?}", k),
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
        if let Some(op) = self.next_op(min_prec, cur_op) {
            if let Some(cur_op) = cur_op {
                if op != cur_op || !op.n_arity() {
                    self.push_node(cur_op.into(), Some(subtree_start));
                }
            }

            self.push_node(op.to_inline(), Some(self.pos as u32 - 1));

            match op {
                BinOp::Assign => {
                    if self.consume_newline() {
                        self.push_next_frame(subtree_start, cfg, Frame::FinishAssign);
                        self.start_block(cfg);
                        return;
                    }
                }
                BinOp::DefineTypeOrTypeAlias => {
                    self.push_next_frame(subtree_start, cfg, Frame::FinishTypeOrTypeAlias);
                    self.start_type(cfg);
                    return;
                }
                _ => {}
            }

            eprintln!(
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

    fn next_op(&mut self, min_prec: Prec, cur_op: Option<BinOp>) -> Option<BinOp> {
        let k = self.buf.kind(self.pos);

        let (op, width) = match k {
            Some(T::LowerIdent) => (BinOp::Apply, 0),
            Some(T::OpPlus) => (BinOp::Plus, 1),
            Some(T::OpStar) => (BinOp::Star, 1),
            Some(T::OpPizza) => (BinOp::Pizza, 1),
            Some(T::OpAssign) => (BinOp::Assign, 1),
            Some(T::OpColon) => (BinOp::DefineTypeOrTypeAlias, 1),
            _ => return None,
        };

        if op.prec() < min_prec || (op.prec() == min_prec && op.grouping_assoc() == Assoc::Left) {
            return None;
        }

        self.pos += width;

        Some(op)
    }

    fn pump_start_type(&mut self, subtree_start: u32, cfg: ExprCfg) {
        loop {
            match self.cur() {
                Some(T::OpenRound) => {
                    self.bump();
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType,
                    );
                    self.push_next_frame(subtree_start, cfg, Frame::FinishParen);
                    continue;
                }
                Some(T::LowerIdent) => {
                    self.bump();
                    self.push_node(N::Ident, Some(self.pos as u32 - 1));

                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType,
                    );
                    return;
                }
                Some(T::UpperIdent) => {
                    self.bump();
                    self.push_node(N::TypeName, Some(self.pos as u32 - 1));

                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueType,
                    );
                    return;
                }
                k => todo!("{:?}", k),
            }
        }
    }

    fn pump_continue_type(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if let Some(_pos) = self.consume_newline_agnostic(T::Comma) {
            // this must be the first comma of a function type!
            self.push_next_frame(
                subtree_start,
                cfg,
                Frame::ContinueTypeFunction,
            );
            self.start_type(cfg);
            return;
        }

        if let Some(pos) = self.consume_newline_agnostic(T::OpArrow) {
            self.push_node(N::InlineLambdaArrow, Some(pos));
            self.push_next_frame(subtree_start, cfg, Frame::ContinueType);
            self.push_next_frame(subtree_start, cfg, Frame::FinishTypeFunction);
            self.start_type(cfg);
            return;
        }

        if let Some(pos) = self.consume_newline_agnostic(T::KwWhere) {
            // Is it valid to have some other clause follow a where clause????
            // If so, uncomment:
            // self.push_next_frame(subtree_start, cfg, Frame::ContinueType);
            self.push_next_frame(subtree_start, cfg, Frame::ContinueWhereClause);
            self.start_type(cfg);
            return;
        }
    }

    fn pump_continue_type_function(&mut self, subtree_start: u32, cfg: ExprCfg) {
        match self.cur() {
            Some(T::Comma) => {
                self.bump();
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueTypeFunction,
                );
                self.start_type(cfg);
            }
            Some(T::OpArrow) => {
                self.bump();
                self.push_node(N::InlineLambdaArrow, Some(self.pos as u32 - 1));
                self.push_next_frame(subtree_start, cfg, Frame::FinishTypeFunction);
                self.start_type(cfg);
            }
            k => todo!("{:?}", k),
        }
    }

    fn pump_continue_where_clause(&mut self, subtree_start: u32, cfg: ExprCfg) {
        // We should have just parsed the `where` followed by a type.
        self.expect(T::KwImplements);
        self.push_node(N::InlineKwImplements, Some(self.pos as u32 - 1));
        self.expect(T::UpperIdent);
        self.push_node(N::AbilityName, Some(self.pos as u32 - 1));
    }

    fn pump_finish_type_function(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::EndTypeLambda, Some(subtree_start));
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
            N::EndAssign => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginAssign;
            }
            N::EndTypeOrTypeAlias => {
                self.update_end(N::Dummy, subtree_start);
                self.tree.kinds[subtree_start as usize] = N::BeginTypeOrTypeAlias;
            }
            N::Ident
            | N::EndWhen
            | N::EndIf
            | N::EndApply
            | N::EndBinOpPlus
            | N::EndBinOpStar
            | N::EndPizza
            | N::EndLambda => {
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

    fn pump_finish_type_or_type_alias(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::EndTypeOrTypeAlias, Some(subtree_start));
    }

    fn pump_continue_top_level(&mut self, subtree_start: u32, cfg: ExprCfg, num_found: i32) {
        // keep parsing decls until the end
        while self.consume_newline() {}

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
            self.push_node(N::EndTopLevelDecls, Some(subtree_start as u32));
            self.update_end(N::BeginTopLevelDecls, subtree_start);
        }
    }

    fn pump_continue_lambda_args(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::OpArrow) {
            self.push_node(N::InlineLambdaArrow, Some(self.pos as u32 - 1));
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
                self.push_node(N::InlineKwThen, Some(self.pos as u32 - 1));
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
                self.expect(T::KwElse);
                self.push_node(N::InlineKwElse, Some(self.pos as u32 - 1));

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
                self.push_node(N::InlineKwIs, Some(self.pos as u32 - 1));
                self.consume_newline();
                let indent = self.cur_indent();
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueWhen {
                        next: WhenState::BranchArrow(indent),
                    },
                );
                self.start_expr(cfg);
            }
            WhenState::BranchPattern(indent) => {
                if let Some(min_indent) = cfg.when_branch_indent_floor {
                    if self.check(
                        self.cur_indent()
                            .is_indented_more_than(min_indent)
                            .ok_or(Error::InconsistentIndent),
                    ) && !self.at_terminator()
                    {
                        self.push_next_frame(
                            subtree_start,
                            cfg,
                            Frame::ContinueWhen {
                                next: WhenState::BranchArrow(indent),
                            },
                        );
                        self.start_expr(cfg);
                        return;
                    }
                }
                self.push_node(N::EndWhen, Some(subtree_start));
                self.update_end(N::BeginWhen, subtree_start);
            }
            WhenState::BranchArrow(indent) => {
                self.expect(T::OpArrow);
                self.push_node(N::InlineWhenArrow, Some(self.pos as u32 - 1));
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueWhen {
                        next: WhenState::BranchPattern(indent),
                    },
                );
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
        while self.consume_newline() {}
        self.push_next_frame_starting_here(
            ExprCfg::default(),
            Frame::ContinueTopLevel { num_found: 1 },
        );
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

    fn start_type(&mut self, cfg: ExprCfg) {
        self.push_next_frame_starting_here(cfg, Frame::StartType);
    }

    fn assert_end(&self) {
        assert_eq!(
            self.pos,
            self.buf.kinds.len(),
            "Expected to be at the end, but these tokens remain: {:?}",
            &self.buf.kinds[self.pos..]
        );
    }

    fn check<T>(&self, v: Result<T, Error>) -> T {
        match v {
            Ok(v) => v,
            Err(e) => todo!(),
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

#[derive(Copy, Clone)]
pub struct FormatCtx<'a> {
    tree: &'a Tree,
    toks: &'a TokenenizedBuffer,
    text: &'a str,
}
impl<'a> FormatCtx<'a> {
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
        self.toks
            .extract_comments_at(self.text, self.pos, trivia);
        self.pos += 1;
    }

    fn check_next_token(&mut self, tok: T, trivia: &mut Vec<Comment>) {
        if tok != T::Newline {
            // fast forward past newlines in the underlying buffer
            while self.toks.kind(self.pos) == Some(T::Newline) {
                self._bump(trivia);
            }
        }
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
        let index = tree.paird_group_ends[i];

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
        let index = tree.paird_group_ends[i];

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
            NodeIndexKind::EndOnly | NodeIndexKind::End => {
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
        self.toks
            .extract_comments_at(self.text, self.pos, &mut self.trivia);
        self.pos += 1;
    }

    fn check_next_token(&mut self, tok: T) -> FmtInfo {
        debug_assert!(tok != T::Newline);

        let comment_start = self.pos;
        let mut newline = false;

        // fast forward past newlines in the underlying buffer
        // also capture trivia at this stage
        while self.toks.kind(self.pos) == Some(T::Newline) {
            newline = true;
            self._bump();
        }

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

            _ => todo!("{:?}", node),
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
    let ctx = FormatCtx { tree, toks, text };

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

    dbg!(&ctx.toks.kinds);

    for (i, &node) in tree.kinds.iter().enumerate() {
        let index = tree.paird_group_ends[i];
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
            _ => todo!("{:?}", node),
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
        dbg!(tok);
        debug_assert!(tok != T::Newline);

        while self.toks.kind(self.pos) == Some(T::Newline) {
            self.toks.extract_comments_at(self.text, self.pos, &mut self.comments);
            self.pos += 1;
        }

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
            | N::EndPizza
            | N::BeginBlock
            | N::EndTopLevelDecls
            | N::EndAssign => {}

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

            _ => todo!("{:?}", node),
        }

        dbg!(&self.comments[begin..])
    }
}

mod canfmt {
    use super::*;
    use bumpalo::collections::vec::Vec;
    use bumpalo::Bump;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Expr<'a> {
        Ident(&'a str),
        Apply(&'a Expr<'a>, &'a [Expr<'a>]),
        BinOp(&'a Expr<'a>, BinOp, &'a Expr<'a>),
        Pizza(&'a [Expr<'a>]),
        Lambda(&'a [Expr<'a>], &'a Expr<'a>),
        If(&'a Expr<'a>, &'a Expr<'a>, &'a Expr<'a>),
        When(&'a Expr<'a>, &'a [Expr<'a>]),
        Block(&'a [Expr<'a>]),

        // Not really expressions, but considering them as such to make the formatter as error tolerant as possible
        Assign(&'a str, &'a Expr<'a>),
        Expr(&'a Expr<'a>),
        Comment(&'a str),
    }

    struct ExprStack<'a> {
        stack: std::vec::Vec<(usize, Expr<'a>)>,
    }

    impl<'a> std::fmt::Debug for ExprStack<'a> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.stack.fmt(f)
        }
    }

    impl<'a> ExprStack<'a> {
        fn new() -> Self {
            Self {
                stack: std::vec::Vec::new(),
            }
        }

        fn push(&mut self, index: usize, expr: Expr<'a>) {
            self.stack.push((index, expr));
        }

        fn drain_to_index(&mut self, index: u32) -> impl ExactSizeIterator<Item = Expr<'a>> + '_ {
            let mut begin = self.stack.len();
            while begin > 0 && self.stack[begin - 1].0 >= index as usize {
                begin -= 1;
            }
            self.stack.drain(begin..).map(|(_, e)| e)
        }
    }

    pub fn build<'a, 'b: 'a>(bump: &'a Bump, ctx: FormatCtx<'b>) -> &'a [Expr<'a>] {
        let mut ce = CommentExtractor::new(ctx.text, ctx.toks);

        let mut stack = ExprStack::new();

        dbg!(&ctx.tree.kinds);

        for (i, &node) in ctx.tree.kinds.iter().enumerate() {
            let comments = dbg!(ce.consume(node));
            if comments.len() > 0 {
                for comment in comments {
                    stack.push(i, Expr::Comment(ctx.text[comment.begin..comment.end].trim()));
                }
            }
            let index = ctx.tree.paird_group_ends[i];
            eprintln!("{}: {:?}@{}: {:?}", i, node, index, stack);
            match node {
                N::BeginTopLevelDecls | N::EndTopLevelDecls => {}
                N::HintExpr => {}
                N::Ident => stack.push(i, Expr::Ident(ctx.text(index))),
                N::EndApply => {
                    let mut values = stack.drain_to_index(index);
                    let first = bump.alloc(values.next().unwrap());
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Expr::Apply(first, args));
                }
                N::EndPizza => {
                    let values = stack.drain_to_index(index);
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Expr::Pizza(args));
                }
                N::EndBinOpPlus => {
                    let mut values = stack.drain_to_index(index);
                    assert_eq!(values.len(), 2);
                    let a = values.next().unwrap();
                    let b = values.next().unwrap();
                    drop(values);
                    stack.push(i, Expr::BinOp(bump.alloc(a), BinOp::Plus, bump.alloc(b)));
                }
                N::BeginLambda => {}
                N::EndLambda => {
                    let mut values = stack.drain_to_index(index);
                    let count = values.len() - 1;
                    let args = bump.alloc_slice_fill_iter(values.by_ref().take(count));
                    let body = values.next().unwrap();
                    drop(values);
                    stack.push(i, Expr::Lambda(args, bump.alloc(body)));
                }
                N::BeginAssign => {}
                N::EndAssign => {
                    let mut values = stack.drain_to_index(index);
                    assert_eq!(values.len(), 2);
                    let name = values.next().unwrap();
                    let name_text = match name {
                        Expr::Ident(name) => name,
                        _ => panic!("Expected ident"),
                    };
                    let value = values.next().unwrap();
                    drop(values);
                    stack.push(i, Expr::Assign(name_text, bump.alloc(value)));
                }
                N::BeginIf => {}
                N::InlineKwThen => {}
                N::InlineKwElse => {}
                N::EndIf => {
                    // pop three elements (cond, then, else)
                    let mut values = stack.drain_to_index(index);
                    assert_eq!(values.len(), 3);
                    let cond = values.next().unwrap();
                    let then = values.next().unwrap();
                    let els = values.next().unwrap();
                    drop(values);
                    stack.push(
                        i,
                        Expr::If(bump.alloc(cond), bump.alloc(then), bump.alloc(els)),
                    );
                }
                N::BeginWhen => {}
                N::InlineKwIs => {}
                N::InlineWhenArrow => {}
                N::EndWhen => {
                    let mut values = stack.drain_to_index(index);
                    let cond = values.next().unwrap();
                    let arms = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Expr::When(bump.alloc(cond), arms));
                }
                N::InlineApply
                | N::InlineAssign
                | N::InlinePizza
                | N::InlineBinOpPlus
                | N::InlineBinOpStar
                | N::InlineLambdaArrow => {}
                N::BeginBlock => {}
                N::EndBlock => {
                    let values = bump.alloc_slice_fill_iter(stack.drain_to_index(index));
                    stack.push(i, Expr::Block(values));
                }
                _ => todo!("{:?}", node),
            }
        }
        bump.alloc_slice_fill_iter(stack.drain_to_index(0))
    }

    enum Doc<'a> {
        Empty,
        Text(&'a str, &'a Doc<'a>),
        Line(usize, &'a Doc<'a>),
        Union(&'a Doc<'a>, &'a Doc<'a>),
    }
}

#[cfg(test)]
mod tests {
    use crate::cypress_token::Tokenizer;

    use super::*;

    // #[track_caller]
    // fn format_test(kinds: &[T]) {
    //     let (unindented_kinds, indents) = unindentify(kinds);

    //     let mut state = State::from_tokens(&unindented_kinds);
    //     state.buf.indents = indents;
    //     state.start_top_level_decls();
    //     state.pump();
    //     state.assert_end();

    //     let res = pretty(&state.tree, &state.buf);

    //     assert_eq!(res.kinds, kinds);
    // }

    // #[track_caller]
    // fn format_test_2(kinds: &[T], expected_kinds: &[T]) {
    //     let (unindented_kinds, indents) = unindentify(kinds);

    //     let mut state = State::from_tokens(&unindented_kinds);
    //     state.buf.indents = indents;
    //     state.start_top_level_decls();
    //     state.pump();
    //     state.assert_end();

    //     let res = pretty(&state.tree, &state.buf);

    //     assert_eq!(res.kinds, expected_kinds);
    // }

    // #[test]
    // fn test_format_ident() {
    //     format_test(&[T::LowerIdent]);
    // }

    // #[test]
    // fn test_format_assign() {
    //     format_test(&[T::LowerIdent, T::OpAssign, T::LowerIdent, T::Newline]);
    // }

    // #[test]
    // fn test_format_double_assign() {
    //     format_test(&[T::LowerIdent, T::OpAssign, T::LowerIdent, T::Newline, T::LowerIdent, T::OpAssign, T::LowerIdent, T::Newline]);
    // }

    // #[test]
    // fn test_format_if() {
    //     format_test_2(
    //         &[
    //             T::KwIf, T::LowerIdent, T::KwThen, T::Newline,
    //                 T::Indent, T::LowerIdent, T::Dedent, T::Newline,
    //             T::KwElse, // missing newline - should be added by the formatter!
    //                 T::Indent, T::LowerIdent, T::Dedent, T::Newline,
    //         ],
    //         &[
    //             T::KwIf, T::LowerIdent, T::KwThen, T::Newline,
    //                 T::Indent, T::LowerIdent, T::Dedent, T::Newline,
    //             T::KwElse, T::Newline,
    //                 T::Indent, T::LowerIdent, T::Dedent, T::Newline,
    //         ],
    //     );
    // }

    impl State {
        fn to_expect_atom(&self) -> ExpectAtom {
            self.tree.to_expect_atom(&self.buf)
        }
    }

    impl Tree {
        fn to_expect_atom(&self, buf: &TokenenizedBuffer) -> ExpectAtom {
            let (i, atom) = self.expect_atom(self.kinds.len(), buf);
            assert_eq!(i, 0);
            atom
        }

        fn expect_atom(&self, end: usize, buf: &TokenenizedBuffer) -> (usize, ExpectAtom) {
            let node = self.kinds[end - 1];
            let index = self.paird_group_ends[end - 1];

            let has_begin = match node.index_kind() {
                NodeIndexKind::Begin => {
                    return (end - 1, ExpectAtom::Unit(node));
                }

                NodeIndexKind::Token => {
                    if let Some(token) = buf.kind(index as usize) {
                        return (end - 1, ExpectAtom::Token(node, token, index as usize));
                    } else {
                        return (end - 1, ExpectAtom::BrokenToken(node, index as usize));
                    }
                }
                NodeIndexKind::Unused => {
                    return (end - 1, ExpectAtom::Unit(node));
                }
                NodeIndexKind::End => true,
                NodeIndexKind::EndOnly => false,
            };

            let mut res = VecDeque::new();
            res.push_front(ExpectAtom::Unit(node));
            let begin = index as usize;

            let mut i = end - 1;

            while i > begin {
                let (new_i, atom) = self.expect_atom(i, buf);
                assert!(new_i < i);
                assert!(new_i >= begin);
                res.push_front(atom);
                i = new_i;
            }

            if has_begin {
                assert_eq!(self.paird_group_ends[begin], end as u32);
            } else {
                res.push_front(ExpectAtom::Empty);
            }

            (begin, ExpectAtom::Seq(res.into()))
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum ExpectAtom {
        Seq(Vec<ExpectAtom>),
        Unit(N),
        Token(N, T, usize),
        Empty,
        BrokenToken(N, usize),
    }

    impl ExpectAtom {
        fn debug_vis(&self) -> String {
            self.debug_vis_indent(0)
        }

        fn debug_vis_indent(&self, indent: usize) -> String {
            match self {
                ExpectAtom::Seq(items) => {
                    // format!("({})", items.iter().map(|i| i.debug_vis()).collect::<Vec<_>>().join(" "))

                    // first let's build up a list of items that have been formatted:
                    let formatted_items = items
                        .iter()
                        .map(|i| i.debug_vis_indent(indent + 4))
                        .collect::<Vec<_>>();

                    // now, if the total length of the formatted items is less than 80, we can just return them as a list
                    let total_len = formatted_items.iter().map(|s| s.len()).sum::<usize>()
                        + formatted_items.len()
                        - 1
                        + 2;
                    if total_len < 80 {
                        return format!("({})", formatted_items.join(" "));
                    }

                    // otherwise, we need to format them as an indented block
                    // somewhat strangely, we format like this:
                    // (first
                    //     second
                    //     ...
                    // last)

                    let mut res = String::new();
                    res.push_str(&format!("({}", formatted_items[0]));
                    for item in &formatted_items[1..formatted_items.len() - 1] {
                        res.push_str(&format!("\n{:indent$}{}", "", item, indent = indent + 4));
                    }
                    res.push_str(&format!(
                        "\n{:indent$}{})",
                        "",
                        formatted_items[formatted_items.len() - 1],
                        indent = indent
                    ));

                    res
                }
                ExpectAtom::Unit(kind) => format!("{:?}", kind),
                ExpectAtom::Token(kind, token, token_index) => {
                    format!("{:?}=>{:?}@{}", kind, token, token_index)
                }
                ExpectAtom::BrokenToken(kind, token_index) => {
                    format!("{:?}=>?broken?@{}", kind, token_index)
                }
                ExpectAtom::Empty => format!("*"),
            }
        }
    }

    // struct ExpectBuilder {
    //     kinds: Vec<N>,
    //     paird_group_ends: Vec<u32>,
    // }

    // impl ExpectBuilder {
    //     fn new() -> ExpectBuilder {
    //         ExpectBuilder {
    //             kinds: Vec::new(),
    //             paird_group_ends: Vec::new(),
    //         }
    //     }

    //     fn consume_items(&mut self, items: &[ExpectAtom]) {
    //         assert!(items.len() > 0);
    //         assert!(!matches!(items.first().unwrap(), ExpectAtom::Seq(_)));
    //         assert!(!matches!(items.last().unwrap(), ExpectAtom::Seq(_)));

    //         let start = self.kinds.len();

    //         for item in items {
    //             match item {
    //                 ExpectAtom::Seq(items) => self.consume_items(&items),
    //                 ExpectAtom::Unit(kind) => {
    //                     self.kinds.push(*kind);
    //                     self.paird_group_ends.push(self.paird_group_ends.len() as u32);
    //                 }
    //                 ExpectAtom::Empty => {}
    //                 ExpectAtom::Token(..) => todo!(),
    //             }
    //         }

    //         let end = self.kinds.len();

    //         if matches!(items.first().unwrap(), ExpectAtom::Unit(_) ) {
    //             self.paird_group_ends[start] = end as u32;
    //         }

    //         if matches!(items.last().unwrap(), ExpectAtom::Unit(_) ) {
    //             self.paird_group_ends[end - 1] = start as u32;
    //         }
    //     }

    //     fn finish(mut self) -> Tree {
    //         Tree {
    //             kinds: self.kinds,
    //             paird_group_ends: self.paird_group_ends,
    //         }
    //     }
    // }

    // fn build_expect(items: Vec<ExpectAtom>) -> Tree {
    //     let mut b = ExpectBuilder::new();

    //     b.consume_items(&items);

    //     let t = b.finish();

    //     let reconstituted = t.to_expect_atom();
    //     match reconstituted {
    //         ExpectAtom::Seq(new_items) => assert_eq!(items, new_items),
    //         ExpectAtom::Unit(item) => assert_eq!(items, vec![ExpectAtom::Unit(item)]),
    //         ExpectAtom::Empty => panic!(),
    //         ExpectAtom::Token(..) => panic!(),
    //     };

    //     t
    // }

    // macro_rules! cvt_item {
    //     (*)                 => { ExpectAtom::Empty };
    //     ($item:ident)       => { ExpectAtom::Unit(N::$item) };
    //     (($($items:tt)*))   => { ExpectAtom::Seq(vec![$(cvt_item!($items)),*]) };
    // }

    // macro_rules! expect {
    //     ($($items:tt)*) => {{
    //         build_expect(vec![$(cvt_item!($items)),*])
    //     }};
    // }

    // fn unindentify(input: &[T]) -> (Vec<T>, Vec<Indent>) {
    //     let mut output = Vec::new();
    //     let mut indents = Vec::new();

    //     let mut cur_indent = Indent::default();

    //     // loop over tokens, incrementing indent level when we see an indent, and decrementing when we see a dedent
    //     // when we see a newline, we add the current indent level to the indents list
    //     // also remove indent/dedent tokens since the parser doesn't need them.

    //     for &tok in input {
    //         match tok {
    //             T::Newline => {
    //                 indents.push(cur_indent);
    //                 output.push(tok);
    //             }
    //             T::Indent => {
    //                 cur_indent.num_spaces += 1;
    //             }
    //             T::Dedent => {
    //                 cur_indent.num_spaces -= 1;
    //             }
    //             _ => {
    //                 output.push(tok);
    //             }
    //         }
    //     }

    //     (output, indents)
    // }

    // #[track_caller]
    // fn decl_test(kinds: &[T], expected: Tree) {
    //     let (kinds, indents) = unindentify(kinds);
    //     let mut state = State::from_tokens(&kinds);
    //     state.buf.indents = indents;
    //     state.start_top_level_decls();
    //     state.pump();
    //     state.assert_end();

    //     // assert_eq!(&state.tree.kinds, &expected.kinds);
    //     // eprintln!("{:?}", state.tree.paird_group_ends);
    //     // eprintln!("{:?}", expected.paird_group_ends);
    //     assert_eq!(state.tree.to_expect_atom().debug_vis(), expected.to_expect_atom().debug_vis());
    // }

    macro_rules! snapshot_test {
        ($text:expr) => {{
            let text = $text;
            let text: &str = text.as_ref();
            let mut tokenizer = Tokenizer::new(text);
            tokenizer.tokenize();
            let tb = tokenizer.finish();
            eprintln!("tokens: {:?}", tb.kinds);
            let mut state = State::from_buf(tb);
            state.start_top_level_decls();
            state.pump();
            state.assert_end();

            let tree_output = state.to_expect_atom().debug_vis();

            let canfmt_output = {
                let bump = bumpalo::Bump::new();
                let canfmt = canfmt::build(&bump, FormatCtx {
                    tree: &state.tree,
                    toks: &state.buf,
                    text,
                });
                canfmt.iter().map(|i| format!("{:?}", i)).collect::<Vec<_>>().join("\n")
            };

            let format_output = pretty(&state.tree, &state.buf, text).text;

            let output = format!("{}\n\n[=== canfmt below ===]\n{}\n\n[=== formatted below ===]\n{}",
                tree_output,
                canfmt_output,
                format_output);

            insta::with_settings!({
                // info => &ctx, // not sure what to put here?
                description => text, // the template source code
                omit_expression => true // do not include the default expression
            }, {
                insta::assert_display_snapshot!(output);
            });

            // Now let's verify that we can replace all the newlines in the original text with comments
            // and everything still works.

            // Iterate thru text and replace newlines with "# <line number>\n"
            // let mut new_text = String::with_capacity(text.len());
            // let mut line_num = 0;
            // for (i, c) in text.chars().enumerate() {
            //     if c == '\n' {
            //         line_num += 1;
            //         new_text.push_str(&format!("# {}\n", line_num));
            //     } else {
            //         new_text.push(c);
            //     }
            // }
            // let text = new_text;
            // eprintln!("commentified text {:?}", text);

            // let mut tokenizer = Tokenizer::new(&text);
            // tokenizer.tokenize();
            // let tb = tokenizer.finish();

            // let mut state = State::from_buf(tb);
            // state.start_top_level_decls();
            // state.pump();
            // state.assert_end();

            // let tree_output = state.to_expect_atom().debug_vis();

            // let bump = bumpalo::Bump::new();
            // let canfmt = canfmt::build(&bump, FormatCtx {
            //     tree: &state.tree,
            //     toks: &state.buf,
            //     text: &text,
            // });

        }};
    }

    #[test]
    fn test_ident() {
        snapshot_test!("abc");
    }

    #[test]
    fn test_apply() {
        snapshot_test!("abc def");
    }

    #[test]
    fn test_simple_binop_plus() {
        snapshot_test!("abc + def");
    }

    #[test]
    fn test_complex_apply() {
        snapshot_test!("abc def + ghi");
    }

    #[test]
    fn test_complex_binop_plus() {
        snapshot_test!("abc + def ghi");
    }

    #[test]
    fn test_nested_binop_plus() {
        snapshot_test!("abc + def + ghi");
    }

    #[test]
    fn test_multiple_ident() {
        snapshot_test!("abc def ghi");
    }

    #[test]
    fn test_pizza_operator() {
        snapshot_test!("abc |> def |> ghi |> jkl");
    }

    #[test]
    fn test_lambda_expr() {
        snapshot_test!("\\abc -> def");
    }

    #[test]
    fn test_if() {
        snapshot_test!("if abc then def else ghi");
    }

    #[test]
    fn test_when() {
        snapshot_test!("when abc is def -> ghi");
    }

    fn block_indentify(text: &str) -> String {
        // remove the leading | from each line, along with any whitespace before that.
        // if the line is completely whitespace, remove it entirely.

        assert_eq!(text.chars().next(), Some('\n'));
        let mut res = String::new();
        let mut saw_newline = true;
        for ch in text.chars().skip(1) {
            if ch == '\n' {
                res.push(ch);
                saw_newline = true;
            } else if saw_newline {
                if ch.is_ascii_whitespace() {
                    continue;
                } else if ch == '|' {
                    saw_newline = false;
                }
            } else {
                res.push(ch);
            }
        }

        res
    }

    #[test]
    fn test_block_indentify() {
        assert_eq!(
            block_indentify(
                r#"
        |abc
        |def
        |ghi
        "#
            ),
            "abc\ndef\nghi\n"
        );
    }

    #[test]
    fn test_nested_when() {
        snapshot_test!(block_indentify(
            r#"
        |when abc is def ->
        |    when ghi is jkl ->
        |        mno
        "#
        ));
    }

    #[test]
    fn test_simple_assign_decl() {
        snapshot_test!(block_indentify(
            r#"
        |abc = def
        "#
        ))
    }

    #[test]
    fn test_double_assign_decl() {
        snapshot_test!(block_indentify(
            r#"
        |abc = def
        |ghi = jkl
        "#
        ))
    }

    #[test]
    fn test_simple_nested_assign_decl() {
        snapshot_test!(block_indentify(
            r#"
        |abc =
        |    def = ghi
        |    jkl
        "#
        ))
    }

    #[test]
    fn test_decl_then_top_level_expr() {
        snapshot_test!(block_indentify(
            r#"
        |abc =
        |    def
        |ghi
        "#
        ))
    }

    #[test]
    fn test_double_nested_decl() {
        snapshot_test!(block_indentify(
            r#"
        |a =
        |    b =
        |        c
        |    d
        "#
        ))
    }

    #[test]
    fn test_double_assign_block_decl() {
        snapshot_test!(block_indentify(
            r#"
        |abc =
        |    def
        |ghi =
        |    jkl
        "#
        ))
    }

    #[test]
    fn test_lambda_decl() {
        snapshot_test!(block_indentify(
            r#"
        |abc = \def ->
        |    ghi
        "#
        ))
    }

    #[test]
    fn test_leading_comment() {
        snapshot_test!(block_indentify(
            r#"
        |# hello
        |abc
        "#
        ))
    }

    #[test]
    fn test_parse_all_files() {
        // list all .roc files under ../test_syntax/tests/snapshots/pass
        let files = std::fs::read_dir("../test_syntax/tests/snapshots/pass")
            .unwrap()
            .map(|res| res.map(|e| e.path()))
            .collect::<Result<Vec<_>, std::io::Error>>()
            .unwrap();

        assert!(files.len() > 0, "no files found in ../test_syntax/tests/snapshots/pass");

        for file in files {
            // if the extension is not .roc, continue
            if file.extension().map(|e| e != "roc").unwrap_or(true) {
                continue;
            }

            eprintln!("parsing {:?}", file);
            let text = std::fs::read_to_string(&file).unwrap();
            eprintln!("---------------------\n{}\n---------------------", text);
            let mut tokenizer = Tokenizer::new(&text);
            tokenizer.tokenize(); // make sure we don't panic!
            let tb = tokenizer.finish();
            eprintln!("tokens: {:?}", tb.kinds);
            let mut state = State::from_buf(tb);
            state.start_top_level_decls();
            state.pump();
            state.assert_end();
        }
    }
}
