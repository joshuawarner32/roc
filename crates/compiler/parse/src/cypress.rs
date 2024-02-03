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
use std::fmt;
use std::{
    collections::{btree_map::Keys, VecDeque},
    f32::consts::E,
};

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
    String,

    /// eg 'b'
    SingleQuote,

    /// Lowercase identifier, e.g. `foo`
    Ident,

    /// Uppercase identifier, when parsed as a type, e.g. `Foo`
    TypeName,

    /// Uppercase identifier, when parsed as a module name
    ModuleName,

    /// Uppercase identifier, when parsed as an ability name
    AbilityName,

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
    InlineBinOpLessThan,
    InlineBinOpGreaterThan,
    InlineBinOpLessThanOrEq,
    InlineBinOpGreaterThanOrEq,
    InlineBinOpSlash,
    InlineBinOpDoubleSlash,
    InlineBinOpPercent,
    InlineBinOpCaret,
    InlineBinOpAnd,
    InlineBinOpOr,
    InlineBinOpEquals,
    InlineBinOpNotEquals,
    EndBinOpMinus,
    EndBinOpLessThan,
    EndBinOpGreaterThan,
    EndBinOpLessThanOrEq,
    EndBinOpGreaterThanOrEq,
    EndBinOpSlash,
    EndBinOpDoubleSlash,
    EndBinOpPercent,
    EndBinOpCaret,
    EndBinOpAnd,
    EndBinOpOr,
    EndBinOpEquals,
    EndBinOpNotEquals,
    InlineBinOpMinus,

    /// Unary not, e.g. `!x`
    EndUnaryNot,

    /// Unary minus, e.g. `-x`
    EndUnaryMinus,

    /// If expression, e.g. `if x then y else z`
    BeginIf,
    InlineKwThen,
    InlineKwElse,
    EndIf,

    /// As expression, e.g. `x as y`
    InlineKwAs,
    EndTypeAs,

    /// A when expression, e.g. `when x is y -> z` (you need a newline after the 'is')
    BeginWhen,
    InlineKwIs,
    InlineWhenArrow,
    EndWhen,

    /// A lambda expression, e.g. `\x -> x`
    BeginLambda,
    InlineLambdaArrow,
    EndLambda,

    BeginTopLevelDecls,
    EndTopLevelDecls,

    /// A type application of some kind, including types or tags
    EndTypeApply,

    Dummy,

    InlineKwWhere,

    BeginFile,
    EndFile,

    BeginHeaderApp,
    BeginHeaderHosted,
    BeginHeaderInterface,
    BeginHeaderPackage,
    BeginHeaderPlatform,
    EndHeaderApp,
    EndHeaderHosted,
    EndHeaderInterface,
    EndHeaderPackage,
    EndHeaderPlatform,

    HintExpr,
    InlineColon,
    InlineTypeColon,

    EndTypeLambda,
    InlineKwImplements,
    EndTypeOrTypeAlias,
    BeginTypeOrTypeAlias,
    InlineBackArrow,
    EndBackpassing,
    BeginBackpassing,
    UpperIdent,
    EndTypeAdendum,
    InlineMultiBackpassingComma,
    EndMultiBackpassingArgs,
    EndFieldAccess,
    EndIndexAccess,
    InlineColonEqual,
    BeginTypeTagUnion,
    EndTypeTagUnion,
    TypeWildcard,

    BeginImplements,
    EndImplements,
    BeginTypeRecord,
    EndTypeRecord,
    BeginCollection,
    EndCollection,
    DotIdent,
    DotModuleLowerIdent,
    DotModuleUpperIdent,
    DotNumber,
    EndWhereClause,

    BeginDbg,
    BeginExpect,
    BeginExpectFx,
    EndDbg,
    EndExpect,
    EndExpectFx,
    BeginPatternList,
    EndPatternList,
    EndPatternParens,
    BeginPatternParens,
    EndPatternRecord,
    BeginPatternRecord,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum NodeIndexKind {
    Begin,          // this is a begin node; the index points one past the corresponding end node
    End,            // this is an end node; the index points to the corresponding begin node
    EndOnly, // this is an end node; the index points to the first child and there is no corresponding begin node
    EndSingleToken, // this is an end node that only contains one item; the index points to the token
    Token,          // the index points to a token

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
            N::BeginFile
            | N::BeginList
            | N::BeginRecord
            | N::BeginRecordUpdate
            | N::BeginParens
            | N::BeginTuple
            | N::BeginBlock
            | N::BeginAssign
            | N::BeginTypeOrTypeAlias
            | N::BeginTypeTagUnion
            | N::BeginIf
            | N::BeginWhen
            | N::BeginLambda
            | N::BeginTopLevelDecls
            | N::BeginImplements
            | N::BeginTypeRecord
            | N::BeginCollection
            | N::BeginDbg
            | N::BeginExpect
            | N::BeginExpectFx
            | N::BeginPatternList
            | N::BeginHeaderApp
            | N::BeginHeaderHosted
            | N::BeginHeaderInterface
            | N::BeginHeaderPackage
            | N::BeginHeaderPlatform
            | N::BeginBackpassing => NodeIndexKind::Begin,
            N::EndFile
            | N::EndList
            | N::EndRecord
            | N::EndRecordUpdate
            | N::EndParens
            | N::EndTuple
            | N::EndBlock
            | N::EndAssign
            | N::EndTypeOrTypeAlias
            | N::EndTypeTagUnion
            | N::EndIf
            | N::EndWhen
            | N::EndLambda
            | N::EndTopLevelDecls
            | N::EndImplements
            | N::EndTypeRecord
            | N::EndCollection
            | N::EndDbg
            | N::EndExpect
            | N::EndExpectFx
            | N::EndPatternList
            | N::EndPatternParens
            | N::EndPatternRecord
            | N::EndHeaderApp
            | N::EndHeaderHosted
            | N::EndHeaderInterface
            | N::EndHeaderPackage
            | N::EndHeaderPlatform
            | N::EndBackpassing => NodeIndexKind::End,
            N::InlineApply
            | N::InlinePizza
            | N::InlineAssign
            | N::InlineBinOpPlus
            | N::InlineBinOpStar
            | N::InlineKwThen
            | N::InlineKwElse
            | N::InlineKwWhere
            | N::InlineKwAs
            | N::InlineKwImplements
            | N::InlineKwIs
            | N::InlineLambdaArrow
            | N::InlineColon
            | N::InlineTypeColon
            | N::InlineColonEqual
            | N::InlineBackArrow
            | N::InlineWhenArrow
            | N::InlineBinOpLessThan
            | N::InlineBinOpGreaterThan
            | N::InlineBinOpLessThanOrEq
            | N::InlineBinOpGreaterThanOrEq
            | N::InlineBinOpSlash
            | N::InlineBinOpDoubleSlash
            | N::InlineBinOpPercent
            | N::InlineBinOpCaret
            | N::InlineBinOpAnd
            | N::InlineBinOpOr
            | N::InlineBinOpEquals
            | N::InlineBinOpNotEquals
            | N::InlineBinOpMinus
            | N::BeginPatternParens
            | N::BeginPatternRecord
            | N::InlineMultiBackpassingComma => NodeIndexKind::Token,
            N::Num
            | N::String
            | N::Ident
            | N::DotIdent
            | N::DotNumber
            | N::UpperIdent
            | N::TypeName
            | N::ModuleName
            | N::AbilityName
            | N::Tag
            | N::OpaqueRef
            | N::Access
            | N::AccessorFunction => NodeIndexKind::Token,
            N::Dummy | N::HintExpr => NodeIndexKind::Unused,
            N::EndApply
            | N::EndPizza
            | N::EndBinOpPlus
            | N::EndBinOpMinus
            | N::EndBinOpStar
            | N::EndUnaryNot
            | N::EndUnaryMinus
            | N::EndTypeLambda
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
            | N::EndTypeAdendum
            | N::EndMultiBackpassingArgs
            | N::EndFieldAccess
            | N::EndIndexAccess
            | N::EndTypeAs
            | N::EndWhereClause
            | N::EndTypeApply => NodeIndexKind::EndOnly,
            N::DotModuleLowerIdent | N::DotModuleUpperIdent => NodeIndexKind::EndSingleToken,
            N::Float | N::SingleQuote | N::Underscore | N::TypeWildcard | N::Crash | N::Dbg => {
                NodeIndexKind::Token
            }
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
    DeclSeq,               // BinOp::DeclSeq,
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
    DefineOtherTypeThing,
    MultiBackpassingComma,
    Implements,
}

impl Prec {
    fn next(self) -> Prec {
        match self {
            Prec::Outer => Prec::DeclSeq,
            Prec::DeclSeq => Prec::MultiBackpassingComma,
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
            BinOp::DeclSeq => Prec::DeclSeq,
            BinOp::AssignBlock
            | BinOp::Assign
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
            BinOp::DeclSeq => Assoc::Right,
            BinOp::Assign
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
            BinOp::Apply | BinOp::Pizza | BinOp::DeclSeq => true,
            _ => false,
        }
    }

    fn to_inline(&self) -> N {
        match self {
            BinOp::AssignBlock => todo!(),
            BinOp::DeclSeq => todo!(),
            BinOp::Assign => N::InlineAssign,
            BinOp::Implements => N::InlineKwImplements,
            BinOp::DefineTypeOrTypeAlias => N::InlineTypeColon,
            BinOp::DefineOtherTypeThing => N::InlineColonEqual,
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
            _ => todo!("binop to node {:?}", op),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Frame {
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
    },
    ContinueType {
        in_apply: Option<bool>,
        allow_clauses: bool,
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
enum IfState {
    Then,
    Else,
    End,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum WhenState {
    Is,
    BranchPattern(Indent),
    BranchArrow(Indent),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Error {
    InconsistentIndent,
    ExpectViolation(T, Option<T>),
    ExpectedExpr(Option<T>),
    ExpectedPattern(Option<T>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Message {
    kind: Error,
    frames: Vec<Frame>,
    pos: u32,
}

struct State {
    frames: Vec<(u32, ExprCfg, Frame)>,
    buf: TokenenizedBuffer,
    pos: usize,
    line: usize,

    tree: Tree,

    pumping: Option<Frame>,
    messages: Vec<Message>,
}

impl State {
    fn from_buf(buf: TokenenizedBuffer) -> Self {
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

    fn peek_at(&self, pos: usize) -> Option<T> {
        self.buf.kind(pos)
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
        eprintln!(
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
        eprintln!(
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

    #[track_caller]
    fn expect_newline(&mut self) {
        todo!();
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
        eprintln!(
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
        self.tree.paird_group_ends[subtree_start as usize] = self.tree.len() as u32;
    }

    fn push_node(&mut self, kind: N, subtree_start: Option<u32>) {
        eprintln!(
            "{:indent$}@{} pushing kind {}:\x1b[34m{:?}\x1b[0m starting at {:?}",
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
        eprintln!(
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
            eprintln!(
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
            debug_assert!(i < 100, "pump looped too many times");
        }
    }

    fn pump(&mut self) {
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
                eprintln!(
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
                debug_assert!(i < 100, "pump looped too many times");
            }
        }
    }

    fn _pump_single_frame(&mut self, subtree_start: u32, cfg: ExprCfg, frame: Frame) {
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
            Frame::StartType { allow_clauses } => {
                self.pump_start_type(subtree_start, cfg, allow_clauses)
            }
            Frame::ContinueType {
                in_apply,
                allow_clauses,
            } => self.pump_continue_type(subtree_start, cfg, in_apply, allow_clauses),
            Frame::ContinueTypeCommaSep { allow_clauses } => {
                self.pump_continue_type_comma_sep(subtree_start, cfg, allow_clauses)
            }
            Frame::FinishTypeFunction => self.pump_finish_type_function(subtree_start, cfg),
            Frame::ContinueWhereClause => self.pump_continue_where_clause(subtree_start, cfg),
            Frame::FinishTypeOrTypeAlias => {
                self.pump_finish_type_or_type_alias(subtree_start, cfg)
            }
            Frame::ContinueRecord { start } => {
                self.pump_continue_record(subtree_start, cfg, start)
            }
            Frame::ContinuePatternRecord { start } => {
                self.pump_continue_pattern_record(subtree_start, cfg, start)
            }
            Frame::ContinueExprList => self.pump_continue_expr_list(subtree_start, cfg),
            Frame::ContinuePatternList => self.pump_continue_pattern_list(subtree_start, cfg),
            Frame::ContinueTypeTupleOrParen => {
                self.pump_continue_type_tuple_or_paren(subtree_start, cfg)
            }
            Frame::ContinueTypeTagUnion => {
                self.pump_continue_type_tag_union(subtree_start, cfg)
            }
            Frame::ContinueTypeTagUnionArgs => {
                self.pump_continue_tag_union_args(subtree_start, cfg)
            }
            Frame::ContinueImplementsMethodDecl => {
                self.pump_continue_implements_method_decl(subtree_start, cfg)
            }
            Frame::ContinueTypeRecord => self.pump_continue_type_record(subtree_start, cfg),
        }
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
                Some(T::IntBase10) => atom!(N::Num),
                Some(T::String) => atom!(N::String),
                Some(T::Float) => atom!(N::Float),

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
                Some(T::LowerIdent) => atom!(N::Ident),
                Some(T::KwCrash) => atom!(N::Crash),
                // TODO: do these need to be distinguished in the node?
                Some(T::Underscore | T::NamedUnderscore) => atom!(N::Underscore),
                Some(T::IntBase10) => atom!(N::Num),
                Some(T::String) => atom!(N::String),
                Some(T::Float) => atom!(N::Float),

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
                | T::OpenCurly
                | T::IntBase10
                | T::IntNonBase10
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
        // TODO: only allow calls / application in patterns (remove this generic op stuff)
        if self.at_pattern_continue(cfg) {
            self.push_next_frame(subtree_start, cfg, Frame::ContinuePattern);
            self.push_next_frame_starting_here(cfg, Frame::StartPattern);
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
                    self.start_type(cfg, true);
                    return;
                }
                BinOp::DefineOtherTypeThing => {
                    // TODO: is this correct????
                    self.push_next_frame(subtree_start, cfg, Frame::FinishTypeOrTypeAlias);
                    self.start_type(cfg, true);
                    return;
                }
                BinOp::Implements => {
                    let cfg = cfg.set_block_indent_floor(Some(self.cur_indent()));
                    self.continue_implements_method_decl_body(subtree_start, cfg);
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

    fn next_op(&mut self, min_prec: Prec, cfg: ExprCfg) -> Option<BinOp> {
        let (op, width) = match self.cur() {
            // TODO: check for other things that can start an expr
            Some(
                T::LowerIdent
                | T::UpperIdent
                | T::OpenCurly
                | T::IntBase10
                | T::IntNonBase10
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
                        return None;
                    }
                }
                (BinOp::Apply, 0)
            }

            Some(T::OpPlus) => (BinOp::Plus, 1),
            Some(T::OpBinaryMinus) => (BinOp::Minus, 1),
            Some(T::OpStar) => (BinOp::Star, 1),
            Some(T::OpPizza) => (BinOp::Pizza, 1),
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
            _ => return None,
        };

        if op.prec() < min_prec || (op.prec() == min_prec && op.grouping_assoc() == Assoc::Left) {
            return None;
        }

        self.pos += width;

        Some(op)
    }

    fn pump_start_type(&mut self, subtree_start: u32, cfg: ExprCfg, allow_clauses: bool) {
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
                        },
                    );
                    return;
                }
                k => todo!("{:?}", k),
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
                k => todo!("{:?}", k),
            }
        }
    }

    fn pump_continue_type(
        &mut self,
        subtree_start: u32,
        cfg: ExprCfg,
        in_apply: Option<bool>,
        allow_clauses: bool,
    ) {
        match self.cur() {
            Some(T::Comma) => {
                if self.peek_at(1) != Some(T::LowerIdent) || self.peek_at(2) != Some(T::OpColon) {
                    self.bump();
                    if in_apply == Some(true) {
                        self.push_node(N::EndTypeApply, Some(subtree_start));
                    }
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueTypeCommaSep { allow_clauses },
                    );
                    self.start_type(cfg, false);
                    return;
                }
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
                    },
                );
                self.push_next_frame(subtree_start, cfg, Frame::FinishTypeFunction);
                self.start_type(cfg, false);
                return;
            }
            Some(T::KwWhere) if allow_clauses => {
                if !self.plausible_expr_continue_comes_next() {
                    self.bump();
                    // TODO: should write a plausible_type_continue_comes_next
                    if in_apply == Some(true) {
                        self.push_node(N::EndTypeApply, Some(subtree_start));
                    }

                    let clause_subtree_start = self.tree.len();
                    self.push_node(N::InlineKwWhere, Some(self.pos as u32 - 1));
                    self.push_next_frame(subtree_start, cfg, Frame::PushEndOnly(N::EndWhereClause));
                    self.push_next_frame(clause_subtree_start, cfg, Frame::ContinueWhereClause);
                    self.start_type(cfg, false);
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
                    },
                );
                self.push_next_frame(subtree_start, cfg, Frame::PushEndOnly(N::EndTypeAs));
                self.start_type(cfg, false);
            }
            // TODO: check for other things that can start an expr
            Some(T::LowerIdent | T::UpperIdent | T::OpenCurly | T::OpenSquare | T::OpenRound)
                if in_apply.is_some() =>
            {
                if dbg!(self.at_newline()) {
                    // are we at the start of a line?
                    if !dbg!(self.buf.lines[self.line]
                        .1
                        .is_indented_more_than(cfg.expr_indent_floor)
                        .expect("TODO: error handling"))
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
                    },
                );
                self.start_type(cfg, false);
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
                self.bump();
                self.push_next_frame(
                    subtree_start,
                    cfg,
                    Frame::ContinueTypeCommaSep { allow_clauses },
                );
                self.start_type(cfg, false);
            }
            Some(T::OpArrow) => {
                self.bump();
                self.push_node(N::InlineLambdaArrow, Some(self.pos as u32 - 1));
                self.push_next_frame(subtree_start, cfg, Frame::FinishTypeFunction);
                self.start_type(cfg, false);
            }

            // TODO: if there isn't an outer square/round/curly and we don't eventually get the arrow,
            // we should error
            Some(T::CloseSquare | T::CloseRound | T::CloseCurly) => return, // Outer scope will handle
            k => todo!("{:?}", k),
        }
    }

    fn pump_continue_where_clause(&mut self, subtree_start: u32, cfg: ExprCfg) {
        // We should have just parsed the `where` followed by a type.
        self.expect_and_push_node(T::KwImplements, N::InlineKwImplements);
        self.expect_and_push_node(T::UpperIdent, N::AbilityName);
    }

    fn pump_finish_type_function(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.push_node(N::EndTypeLambda, Some(subtree_start));
    }

    fn pump_continue_expr_tuple_or_paren(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::Comma) {
            self.push_next_frame(subtree_start, cfg, Frame::ContinueExprTupleOrParen);
            self.start_expr(cfg.disable_multi_backpassing());
        } else {
            self.expect(T::CloseRound);
            self.push_node_end(N::BeginParens, N::EndParens, subtree_start);
            self.handle_field_access_suffix(subtree_start);
        }
    }

    fn pump_continue_pattern_tuple_or_paren(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::Comma) {
            self.push_next_frame(subtree_start, cfg, Frame::ContinuePatternTupleOrParen);
            self.start_expr(cfg.disable_multi_backpassing());
        } else {
            self.expect(T::CloseRound);
            self.push_node_end(N::BeginPatternParens, N::EndPatternParens, subtree_start);
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
            | N::EndTypeAs
            | N::EndBinOpEquals
            | N::EndBinOpNotEquals => {
                self.tree.kinds[subtree_start as usize] = N::HintExpr;
            }
            k => todo!("{:?}: {:?}", k, self.tree.kinds),
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
                        next: WhenState::BranchArrow(indent),
                    },
                );
                self.start_pattern(cfg);
            }
            WhenState::BranchPattern(indent) => {
                if self
                    .cur_indent()
                    .is_indented_more_than(cfg.block_indent_floor)
                    .ok_or(Error::InconsistentIndent)
                    .expect("TODO: error handling")
                    && !self.at_terminator()
                {
                    self.push_next_frame(
                        subtree_start,
                        cfg,
                        Frame::ContinueWhen {
                            next: WhenState::BranchArrow(indent),
                        },
                    );
                    self.start_pattern(cfg);
                    return;
                }
                self.push_node_end(N::BeginWhen, N::EndWhen, subtree_start);
            }
            WhenState::BranchArrow(indent) => {
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

    fn start_file(&mut self) {
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
                        s.expect_and_push_node(T::UpperIdent, N::Ident); // TODO: correct node type
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
                        loop {
                            match s.cur() {
                                Some(T::UpperIdent) => {
                                    s.bump();
                                    s.push_node(N::TypeName, Some(s.pos as u32 - 1));
                                }
                                Some(T::OpenCurly) => {
                                    s.bump();
                                    // TODO: stuff in the middle here...
                                    s.expect(T::CloseCurly);
                                }
                                _ => break,
                            }
                        }
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
                self.expect(T::KwImports);
                self.expect_collection(
                    T::OpenSquare,
                    N::BeginCollection,
                    T::CloseSquare,
                    N::EndCollection,
                    |s| {
                        s.expect_and_push_node(T::UpperIdent, N::Ident); // TODO: correct node type
                    },
                );
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

                self.push_node_end(N::BeginHeaderInterface, N::EndHeaderInterface, subtree_start);
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

        self.push_next_frame(subtree_start, ExprCfg::default(), Frame::PushEnd(N::BeginFile, N::EndFile));

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
                            s.consume_and_push_node(T::NoSpaceDotUpperIdent, N::DotModuleUpperIdent);
                            s.consume_and_push_node(T::NoSpaceDotUpperIdent, N::DotModuleUpperIdent);
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
                                        s.push_error(Error::ExpectViolation(T::LowerIdent, t));
                                        s.fast_forward_past_newline(); // TODO: this should fastforward to the close square
                                    }
                                }
                            },
                        );
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
                min_prec: Prec::DeclSeq,
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

    fn start_type(&mut self, cfg: ExprCfg, allow_clauses: bool) {
        self.push_next_frame_starting_here(cfg, Frame::StartType { allow_clauses });
    }

    fn assert_end(&self) {
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
            Err(e) => todo!(),
        }
    }

    fn pump_continue_record(&mut self, subtree_start: u32, cfg: ExprCfg, start: bool) {
        // TODO: allow { existingRecord & foo: bar } syntax

        if !start {
            if self.consume_end(T::CloseCurly) {
                self.push_node(N::EndRecord, Some(subtree_start));
                self.update_end(N::BeginRecord, subtree_start);
                self.handle_field_access_suffix(subtree_start);
                return;
            }

            self.expect(T::Comma);
        }

        if self.consume_end(T::CloseCurly) {
            self.push_node(N::EndRecord, Some(subtree_start));
            self.update_end(N::BeginRecord, subtree_start);
            self.handle_field_access_suffix(subtree_start);
            return;
        }

        self.expect_lower_ident_and_push_node();

        self.expect_and_push_node(T::OpColon, N::InlineColon);

        self.push_next_frame(subtree_start, cfg, Frame::ContinueRecord { start: false });
        self.start_expr(cfg.disable_multi_backpassing());
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
            self.start_type(cfg, false);
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
        self.start_expr(cfg.disable_multi_backpassing());
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
        self.start_type(cfg, false);
    }

    fn pump_continue_implements_method_decl(&mut self, subtree_start: u32, cfg: ExprCfg) {
        self.expect_newline();
        self.continue_implements_method_decl_body(subtree_start, cfg);
    }

    fn continue_implements_method_decl_body(&mut self, subtree_start: u32, cfg: ExprCfg) {
        // We continue as long as the indent is the same as the start of the implements block.
        if self
            .cur_indent()
            .is_indented_more_than(cfg.block_indent_floor)
            .ok_or(Error::InconsistentIndent)
            .expect("TODO: error handling")
        {
            self.expect_and_push_node(T::LowerIdent, N::Ident);
            self.expect(T::OpColon); // TODO: need to add a node?
            self.push_next_frame(subtree_start, cfg, Frame::ContinueImplementsMethodDecl);
            self.start_type(cfg, false);
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
        self.start_type(cfg, false);
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
        self.start_type(cfg, false);
    }

    fn maybe_start_type_adendum(&mut self, subtree_start: u32, cfg: ExprCfg) {
        if self.consume(T::NoSpace) {
            self.push_next_frame(subtree_start, cfg, Frame::PushEndOnly(N::EndTypeAdendum));
            self.start_type(cfg, false);
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

fn format_message(text: &str, buf: &TokenenizedBuffer, msg: &Message) -> String {
    // binary search to find the line of msg.pos
    let (line_start, line_end) = match buf
        .lines
        .binary_search_by_key(&msg.pos, |(offset, _indent)| *offset)
    {
        Ok(i) => (buf.lines[i].0, buf.lines[i].0),
        Err(i) => {
            if i > 0 {
                if i < buf.lines.len() {
                    (buf.lines[i - 1].0, buf.lines[i].0)
                } else {
                    (buf.lines[i - 1].0, buf.kinds.len() as u32)
                }
            } else {
                (0, buf.lines[i].0)
            }
        }
    };

    debug_assert!(line_start <= msg.pos && msg.pos <= line_end);

    let mut res = String::new();

    res.push_str(&format!(
        "Error at token {} (offset {}):\n",
        msg.pos,
        buf.offset(msg.pos)
    ));

    // print the first line (tokens)
    let mut pointer_offset = 0;
    let mut pointer_len = 0;
    for (i, kind) in buf
        .kinds
        .iter()
        .enumerate()
        .skip(line_start as usize)
        .take((line_end - line_start) as usize)
    {
        let text = format!("{:?} ", kind);
        if i < msg.pos as usize {
            pointer_offset += text.len();
        } else if i == msg.pos as usize {
            pointer_len = text.len();
        }
        res.push_str(&text);
    }
    res.push('\n');

    // print the pointer
    for _ in 0..pointer_offset {
        res.push(' ');
    }
    for _ in 0..pointer_len {
        res.push('^');
    }
    res.push('\n');

    // print the text
    res.push_str(&text[buf.offset(line_start)..buf.offset(line_end)]);
    res.push('\n');

    let pointer_offset =
        buf.offsets[msg.pos as usize] as usize - buf.offsets[line_start as usize] as usize;
    let pointer_len = buf
        .lengths
        .get(msg.pos as usize)
        .map(|o| *o as usize)
        .unwrap_or(0);

    // print the pointer
    for _ in 0..pointer_offset {
        res.push(' ');
    }
    for _ in 0..pointer_len {
        res.push('^');
    }
    res.push('\n');

    res.push_str(&format!("{:?}", msg.kind));
    for frame in &msg.frames {
        res.push_str(&format!("\n  in {:?}", frame));
    }

    res
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

    // dbg!(&ctx.toks.kinds);

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
            N::Num => self.check_next_token(T::IntBase10),
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

            _ => todo!("{:?}", node),
        }

        &self.comments[begin..]
    }
}

mod canfmt {
    use std::fmt::Debug;

    use super::*;
    use bumpalo::collections::vec::Vec;
    use bumpalo::Bump;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Expr<'a> {
        Crash,
        Ident(&'a str),
        Underscore(&'a str),
        UpperIdent(&'a str),
        IntBase10(&'a str),
        Float(&'a str),
        String(&'a str),
        DotNumber(&'a str),
        Apply(&'a Expr<'a>, &'a [Expr<'a>]),
        BinOp(&'a Expr<'a>, BinOp, &'a Expr<'a>),
        UnaryOp(UnaryOp, &'a Expr<'a>),
        Pizza(&'a [Expr<'a>]),
        Lambda(&'a [Expr<'a>], &'a Expr<'a>),
        If(&'a [(&'a Expr<'a>, &'a Expr<'a>)], &'a Expr<'a>),
        When(&'a Expr<'a>, &'a [Expr<'a>]),
        Block(&'a [Expr<'a>]),
        Record(&'a [(&'a str, &'a Expr<'a>)]),
        RecordAccess(&'a Expr<'a>, &'a str),
        TupleAccess(&'a Expr<'a>, &'a str),
        ModuleLowerName(&'a str, &'a str),
        Tuple(&'a [Expr<'a>]),
        List(&'a [Expr<'a>]),

        // Not really expressions, but considering them as such to make the formatter as error tolerant as possible
        Assign(&'a str, &'a Expr<'a>),
        Comment(&'a str),
        TypeAlias(&'a Expr<'a>, &'a Type<'a>),
        AbilityName(&'a str),
        Dbg(&'a Expr<'a>),
        Expect(&'a Expr<'a>),
        ExpectFx(&'a Expr<'a>),

        PatternRecord(&'a [&'a str]),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum UnaryOp {
        Minus,
        Not,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Type<'a> {
        Wildcard,
        Name(&'a str),
        Record(&'a [(&'a str, &'a Type<'a>)]),
        Apply(&'a Type<'a>, &'a [Type<'a>]),
        Lambda(&'a [Type<'a>], &'a Type<'a>),
        WhereClause(&'a Type<'a>, &'a Type<'a>, &'a Type<'a>),
        Tuple(&'a [Type<'a>]),
        TagUnion(&'a [Type<'a>]),
        Adendum(&'a Type<'a>, &'a Type<'a>),
        ModuleType(&'a str, &'a str),
        As(&'a Type<'a>, &'a Type<'a>),
    }

    struct TreeStack<V> {
        stack: std::vec::Vec<(usize, V)>,
    }

    impl<'a, V: Debug> std::fmt::Debug for TreeStack<V> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.stack.fmt(f)
        }
    }

    impl<V> TreeStack<V> {
        fn new() -> Self {
            Self {
                stack: std::vec::Vec::new(),
            }
        }

        fn push(&mut self, index: usize, item: V) {
            self.stack.push((index, item));
        }

        fn drain_to_index(&mut self, index: u32) -> impl ExactSizeIterator<Item = V> + '_ {
            let mut begin = self.stack.len();
            while begin > 0 && self.stack[begin - 1].0 >= index as usize {
                begin -= 1;
            }
            self.stack.drain(begin..).map(|(_, e)| e)
        }

        fn pop(&mut self) -> Option<V> {
            self.stack.pop().map(|(_, e)| e)
        }
    }

    struct TreeWalker<'b> {
        kinds: &'b [N],
        indices: &'b [u32],
        pos: usize,
    }

    impl<'b> TreeWalker<'b> {
        fn new(tree: &'b Tree) -> Self {
            Self {
                kinds: &tree.kinds,
                indices: &tree.paird_group_ends,
                pos: 0,
            }
        }

        fn next(&mut self) -> Option<N> {
            let res = self.kinds.get(self.pos).copied();
            self.pos += 1;
            res
        }

        fn cur(&self) -> Option<N> {
            self.kinds.get(self.pos).copied()
        }

        fn cur_index(&self) -> Option<(N, usize, u32)> {
            self.cur().map(|n| (n, self.pos, self.indices[self.pos]))
        }

        fn next_index(&mut self) -> Option<(N, usize, u32)> {
            let res = self.cur_index();
            self.pos += 1;
            res
        }
    }

    fn build_type<'a, 'b: 'a>(
        bump: &'a Bump,
        ctx: FormatCtx<'b>,
        w: &mut TreeWalker<'b>,
    ) -> (Type<'a>, usize, u32) {
        let mut stack = TreeStack::new();
        while let Some((node, i, index)) = w.next_index() {
            match node {
                N::Ident => stack.push(i, Type::Name(ctx.text(index))),
                N::UpperIdent => stack.push(i, Type::Name(ctx.text(index))),
                N::TypeName => stack.push(i, Type::Name(ctx.text(index))),
                N::ModuleName => stack.push(i, Type::Name(ctx.text(index))), // TODO!
                N::TypeWildcard => stack.push(i, Type::Wildcard),
                N::Tag => stack.push(i, Type::Name(ctx.text(index))), // TODO
                N::AbilityName => stack.push(i, Type::Name(ctx.text(index))), // TODO
                N::DotModuleUpperIdent => {
                    let last = stack.pop().unwrap();
                    let name = match last {
                        Type::Name(name) => ctx.text(index),
                        Type::ModuleType(name, _) => name, // TODO! THIS IS WRONG
                        _ => panic!("Expected name"),
                    };
                    stack.push(i, Type::ModuleType(name, ctx.text(index)));
                }
                N::EndTypeApply => {
                    let mut values = stack.drain_to_index(index);
                    let first = bump.alloc(values.next().unwrap());
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Type::Apply(first, args));
                }
                N::EndTypeLambda => {
                    let mut values = stack.drain_to_index(index);
                    let count = values.len() - 1;
                    let args = bump.alloc_slice_fill_iter(values.by_ref().take(count));
                    let body = values.next().unwrap();
                    drop(values);
                    stack.push(i, Type::Lambda(args, bump.alloc(body)));
                }
                N::EndTypeAdendum => {
                    let mut values = stack.drain_to_index(index);
                    assert_eq!(values.len(), 2);
                    let first = values.next().unwrap();
                    let second = values.next().unwrap();
                    drop(values);
                    stack.push(i, Type::Adendum(bump.alloc(first), bump.alloc(second)));
                }
                N::EndTypeRecord => {
                    let mut values = stack.drain_to_index(index);
                    let mut pairs: Vec<(&'a str, &'a Type<'a>)> = Vec::new_in(bump);
                    loop {
                        let name = match values.next() {
                            Some(Type::Name(name)) => name,
                            None => break,
                            _ => panic!("Expected name"),
                        };
                        let value = values.next().unwrap();
                        pairs.push((name, bump.alloc(value)));
                    }
                    drop(values);
                    stack.push(i, Type::Record(pairs.into_bump_slice()));
                }
                N::EndWhereClause => {
                    let mut values = stack.drain_to_index(index);
                    let first = bump.alloc(values.next().unwrap());
                    let second = bump.alloc(values.next().unwrap());
                    let third = bump.alloc(values.next().unwrap());
                    assert_eq!(values.next(), None);
                    drop(values);
                    stack.push(i, Type::WhereClause(first, second, third));
                }
                N::EndTypeAs => {
                    let mut values = stack.drain_to_index(index);
                    let first = bump.alloc(values.next().unwrap());
                    let second = bump.alloc(values.next().unwrap());
                    assert_eq!(values.next(), None);
                    drop(values);
                    stack.push(i, Type::As(first, second));
                }
                N::EndParens => {
                    let mut values = stack.drain_to_index(index);
                    if values.len() == 1 {
                        // we don't create a tuple for a single element
                        let value = values.next().unwrap();
                        drop(values);
                        stack.push(i, value);
                    } else {
                        let mut values = values.into_iter();
                        let args = bump.alloc_slice_fill_iter(values);
                        stack.push(i, Type::Tuple(args));
                    }
                }
                N::EndTypeTagUnion => {
                    let mut values = stack.drain_to_index(index);
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Type::TagUnion(args));
                }
                N::InlineLambdaArrow
                | N::InlineKwWhere
                | N::InlineKwImplements
                | N::BeginTypeRecord
                | N::BeginParens
                | N::InlineKwAs
                | N::BeginTypeTagUnion => {}
                N::EndTypeOrTypeAlias => {
                    assert_eq!(stack.stack.len(), 1, "{:?}", stack.stack);

                    return (stack.pop().unwrap(), i, index);
                }
                _ => todo!("{:?}", node),
            }
        }

        panic!("didn't find EndTypeOrTypeAlias");
    }

    pub fn build<'a, 'b: 'a>(bump: &'a Bump, ctx: FormatCtx<'b>) -> &'a [Expr<'a>] {
        // let mut ce = CommentExtractor::new(ctx.text, ctx.toks);

        let mut w = TreeWalker::new(&ctx.tree);

        if w.cur() == Some(N::BeginFile) {
            w.next();
        }

        if matches!(w.cur(), 
            Some(N::BeginHeaderApp
        | N::BeginHeaderHosted
        | N::BeginHeaderInterface
        | N::BeginHeaderPackage
        | N::BeginHeaderPlatform)
    ) {
            w.next();
            while let Some((node, i, index)) = w.cur_index() {
                match node {
                    N::EndHeaderApp
                    | N::EndHeaderHosted
                    | N::EndHeaderInterface
                    | N::EndHeaderPackage
                    | N::EndHeaderPlatform => {
                        w.next();
                        break;
                    }
                    _ => {
                        // TODO: consume and construct app headers!
                        w.next();
                    },
                }
            }
        }


        let mut stack = TreeStack::new();

        while let Some((node, i, index)) = w.next_index() {
            // let comments = ce.consume(node);
            // if comments.len() > 0 {
            //     for comment in comments {
            //         stack.push(i, Expr::Comment(ctx.text[comment.begin..comment.end].trim()));
            //     }
            // }

            macro_rules! binop {
                ($op:expr) => {{
                    let mut values = stack.drain_to_index(index);
                    assert_eq!(values.len(), 2);
                    let a = values.next().unwrap();
                    let b = values.next().unwrap();
                    drop(values);
                    stack.push(i, Expr::BinOp(bump.alloc(a), $op, bump.alloc(b)));
                }};
            }

            match node {
                N::Ident => stack.push(i, Expr::Ident(ctx.text(index))),
                N::DotNumber => stack.push(i, Expr::DotNumber(ctx.text(index))),
                N::Crash => stack.push(i, Expr::Crash),
                N::Underscore => stack.push(i, Expr::Underscore(ctx.text(index))),
                N::UpperIdent => stack.push(i, Expr::UpperIdent(ctx.text(index))),
                N::Num => stack.push(i, Expr::IntBase10(ctx.text(index))),
                N::Float => stack.push(i, Expr::Float(ctx.text(index))),
                N::String => stack.push(i, Expr::String(ctx.text(index))),
                N::DotIdent => stack.push(i, Expr::Ident(ctx.text(index))),
                N::ModuleName => {
                    // stack.push(i, Expr::ModuleName(ctx.text(index)))
                    if let Some((N::DotModuleLowerIdent, _, index2)) = w.cur_index() {
                        w.next();
                        // if the next node is also a module name, then this is a module name
                        stack.push(i, Expr::ModuleLowerName(ctx.text(index), ctx.text(index2)));
                    } else {
                        // otherwise it's a type name
                        // stack.push(i, Expr::TypeName(ctx.text(index)));
                        todo!();
                    }
                }
                N::EndFieldAccess => {
                    let mut values = stack.drain_to_index(index);
                    let value = values.next().unwrap();
                    let name = match values.next().unwrap() {
                        Expr::Ident(name) => name,
                        _ => panic!("Expected ident"),
                    };
                    drop(values);
                    stack.push(i, Expr::RecordAccess(bump.alloc(value), name));
                }
                N::EndIndexAccess => {
                    let mut values = stack.drain_to_index(index);
                    let value = values.next().unwrap();
                    let name = match values.next().unwrap() {
                        Expr::DotNumber(name) => name,
                        _ => panic!("Expected ident"),
                    };
                    drop(values);
                    stack.push(i, Expr::TupleAccess(bump.alloc(value), name));
                }
                N::EndApply => {
                    let mut values = stack.drain_to_index(index);
                    let first = bump.alloc(values.next().unwrap());
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Expr::Apply(first, args));
                }
                N::EndList => {
                    let mut values = stack.drain_to_index(index);
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Expr::List(args));
                }
                // N::EndTypeApply => {
                //     let mut values = stack.drain_to_index(index);
                //     let first = bump.alloc(values.next().unwrap());
                //     let args = bump.alloc_slice_fill_iter(values);
                //     stack.push(i, Expr::TypeApply(first, args));
                // }
                N::EndPizza => {
                    let values = stack.drain_to_index(index);
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Expr::Pizza(args));
                }
                N::EndBinOpPlus => binop!(BinOp::Plus),
                N::EndBinOpMinus => binop!(BinOp::Minus),
                N::EndBinOpStar => binop!(BinOp::Star),
                N::EndBinOpSlash => binop!(BinOp::Slash),
                N::EndBinOpDoubleSlash => binop!(BinOp::DoubleSlash),
                N::EndBinOpPercent => binop!(BinOp::Percent),
                N::EndBinOpCaret => binop!(BinOp::Caret),
                N::EndBinOpAnd => binop!(BinOp::And),
                N::EndBinOpOr => binop!(BinOp::Or),
                N::EndBinOpEquals => binop!(BinOp::Equals),
                N::EndBinOpNotEquals => binop!(BinOp::NotEquals),
                N::EndLambda => {
                    let mut values = stack.drain_to_index(index);
                    let count = values.len() - 1;
                    let args = bump.alloc_slice_fill_iter(values.by_ref().take(count));
                    let body = values.next().unwrap();
                    drop(values);
                    stack.push(i, Expr::Lambda(args, bump.alloc(body)));
                }
                N::EndAssign => {
                    let mut values = stack.drain_to_index(index);
                    assert_eq!(
                        values.len(),
                        2,
                        "{:?}",
                        values.collect::<std::vec::Vec<_>>()
                    );
                    let name = values.next().unwrap();
                    let name_text = match name {
                        Expr::Ident(name) => name,
                        Expr::Underscore(name) => name, // TODO: allow patterns
                        _ => panic!("Expected ident, found {:?}", name),
                    };
                    let value = values.next().unwrap();
                    drop(values);
                    stack.push(i, Expr::Assign(name_text, bump.alloc(value)));
                }
                N::InlineTypeColon => {
                    let (ty, i, index) = build_type(bump, ctx, &mut w);
                    let mut values = stack.drain_to_index(index);
                    assert_eq!(
                        values.len(),
                        1,
                        "{:?}",
                        values.collect::<std::vec::Vec<_>>()
                    );
                    let name = values.next().unwrap();
                    drop(values);
                    stack.push(i, Expr::TypeAlias(bump.alloc(name), bump.alloc(ty)));
                }
                N::EndIf => {
                    // pop three elements (cond, then, else)
                    let mut values = stack.drain_to_index(index);
                    assert!(
                        values.len() >= 3 && (values.len() - 1) % 2 == 0,
                        "{:?}",
                        values.collect::<std::vec::Vec<_>>()
                    );
                    let mut condthenseq = Vec::<(&'a Expr<'a>, &'a Expr<'a>)>::with_capacity_in(
                        (values.len() - 1) / 2,
                        bump,
                    );
                    for _ in 0..(values.len() - 1) / 2 {
                        let cond = values.next().unwrap();
                        let then = values.next().unwrap();
                        condthenseq.push((bump.alloc(cond), bump.alloc(then)));
                    }
                    let els = values.next().unwrap();
                    assert!(values.next().is_none());
                    drop(values);
                    stack.push(i, Expr::If(condthenseq.into_bump_slice(), bump.alloc(els)));
                }
                N::EndWhen => {
                    let mut values = stack.drain_to_index(index);
                    let cond = values.next().unwrap();
                    let arms = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Expr::When(bump.alloc(cond), arms));
                }
                N::EndTypeOrTypeAlias => {
                    panic!("should already be handled via earlier build_type call");
                }
                N::EndRecord => {
                    let mut values = stack.drain_to_index(index);
                    let mut pairs: Vec<(&'a str, &'a Expr<'a>)> = Vec::new_in(bump);
                    loop {
                        let name = match values.next() {
                            Some(Expr::Ident(name)) => name,
                            None => break,
                            _ => panic!("Expected ident"),
                        };
                        let value = values.next().unwrap();
                        pairs.push((name, bump.alloc(value)));
                    }
                    drop(values);
                    stack.push(i, Expr::Record(pairs.into_bump_slice()));
                }
                N::EndPatternRecord => {
                    let mut values = stack.drain_to_index(index);
                    let mut items: Vec<&'a str> = Vec::new_in(bump);
                    loop {
                        let name = match values.next() {
                            Some(Expr::Ident(name)) => name,
                            None => break,
                            _ => panic!("Expected ident"),
                        };
                        // TODO: allow optional :/? for field patterns
                        items.push(name);
                    }
                    drop(values);
                    stack.push(i, Expr::PatternRecord(items.into_bump_slice()));
                }
                N::EndParens => {
                    let mut values = stack.drain_to_index(index);
                    if values.len() == 1 {
                        // we don't create a tuple for a single element
                        let value = values.next().unwrap();
                        drop(values);
                        stack.push(i, value);
                    } else {
                        let mut values = values.into_iter();
                        let args = bump.alloc_slice_fill_iter(values);
                        stack.push(i, Expr::Tuple(args));
                    }
                }
                N::EndPatternParens => {
                    let mut values = stack.drain_to_index(index);
                    if values.len() == 1 {
                        // we don't create a tuple for a single element
                        let value = values.next().unwrap();
                        drop(values);
                        stack.push(i, value);
                    } else {
                        let mut values = values.into_iter();
                        let args = bump.alloc_slice_fill_iter(values);
                        stack.push(i, Expr::Tuple(args));
                    }
                }
                N::EndBlock => {
                    let values = bump.alloc_slice_fill_iter(stack.drain_to_index(index));
                    stack.push(i, Expr::Block(values));
                }
                N::EndDbg => {
                    let mut values = stack.drain_to_index(index);
                    let body = bump.alloc(values.next().unwrap());
                    assert_eq!(values.next(), None);
                    drop(values);
                    stack.push(i, Expr::Dbg(body));
                }
                N::EndExpect => {
                    let mut values = stack.drain_to_index(index);
                    let body = bump.alloc(values.next().unwrap());
                    assert_eq!(values.next(), None);
                    drop(values);
                    stack.push(i, Expr::Expect(body));
                }
                N::EndExpectFx => {
                    let mut values = stack.drain_to_index(index);
                    let body = bump.alloc(values.next().unwrap());
                    assert_eq!(values.next(), None);
                    drop(values);
                    stack.push(i, Expr::ExpectFx(body));
                }
                N::EndUnaryMinus => {
                    let mut values = stack.drain_to_index(index);
                    let body = bump.alloc(values.next().unwrap());
                    assert_eq!(values.next(), None);
                    drop(values);
                    stack.push(i, Expr::UnaryOp(UnaryOp::Minus, body));
                }
                N::EndUnaryNot => {
                    let mut values = stack.drain_to_index(index);
                    let body = bump.alloc(values.next().unwrap());
                    assert_eq!(values.next(), None);
                    drop(values);
                    stack.push(i, Expr::UnaryOp(UnaryOp::Not, body));
                }
                N::BeginFile
                | N::EndFile
                | N::InlineApply
                | N::InlineAssign
                | N::InlinePizza
                | N::InlineColon
                | N::InlineBinOpPlus
                | N::InlineBinOpStar
                | N::InlineBinOpMinus
                | N::InlineBinOpSlash
                | N::InlineBinOpDoubleSlash
                | N::InlineBinOpPercent
                | N::InlineBinOpCaret
                | N::InlineBinOpAnd
                | N::InlineBinOpOr
                | N::InlineBinOpEquals
                | N::InlineBinOpNotEquals
                | N::InlineLambdaArrow
                | N::InlineKwWhere
                | N::InlineKwImplements
                | N::BeginBlock
                | N::BeginParens
                | N::BeginRecord
                | N::BeginTypeOrTypeAlias
                | N::BeginWhen
                | N::BeginList
                | N::InlineKwIs
                | N::InlineWhenArrow
                | N::BeginIf
                | N::InlineKwThen
                | N::InlineKwElse
                | N::BeginLambda
                | N::BeginAssign
                | N::BeginTopLevelDecls
                | N::EndTopLevelDecls
                | N::BeginDbg
                | N::BeginExpect
                | N::BeginExpectFx
                | N::BeginPatternRecord
                | N::BeginPatternParens
                | N::HintExpr
                | N::InlineMultiBackpassingComma => {}
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

                NodeIndexKind::EndSingleToken => {
                    let mut res = VecDeque::new();
                    res.push_front(ExpectAtom::Unit(node));
                    let (new_i, atom) = self.expect_atom(end - 1, buf);
                    assert!(new_i < end - 1);
                    res.push_front(atom);
                    res.push_front(ExpectAtom::Empty);
                    return (new_i, ExpectAtom::Seq(res.into()));
                }
            };

            let mut res = VecDeque::new();
            res.push_front(ExpectAtom::Unit(node));
            let begin = index as usize;

            let mut i = end - 1;

            while i > begin {
                let (new_i, atom) = self.expect_atom(i, buf);
                assert!(new_i < i);
                assert!(
                    new_i >= begin,
                    "new_i={}, begin={}, node={:?}",
                    new_i,
                    begin,
                    node
                );
                res.push_front(atom);
                i = new_i;
            }

            if has_begin {
                assert_eq!(
                    self.paird_group_ends[begin], end as u32,
                    "begin/end mismatch at {}->{}, with node {:?}",
                    begin, end, node
                );
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

    macro_rules! snapshot_test {
        ($text:expr) => {{
            let text = $text;
            eprintln!("text:\n{}", text);
            let text: &str = text.as_ref();
            let mut tokenizer = Tokenizer::new(text);
            tokenizer.tokenize();
            let tb = tokenizer.finish();
            eprint!("tokens:");
            let mut last = 0;
            for (i, (begin, indent)) in tb.lines.iter().enumerate() {
                for tok in &tb.kinds[last as usize .. *begin as usize] {
                    eprint!(" {:?}", tok);
                }
                eprint!("\n{}: {:?} {}.{}:", i, begin, indent.num_spaces, indent.num_tabs);
                last = *begin;
            }
            for tok in &tb.kinds[last as usize..] {
                eprint!(" {:?}", tok);
            }
            eprintln!();

            let mut state = State::from_buf(tb);
            state.start_file();
            state.pump();
            state.assert_end();
            if state.messages.len() > 0 {
                for msg in state.messages.iter().take(3) {
                    eprintln!("{}", format_message(text, &state.buf, msg));
                }

                panic!("unexpected messages: {:?}", state.messages);
            }

            eprintln!("raw tree: {:?}", state.tree.kinds);

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

            // let format_output = pretty(&state.tree, &state.buf, text).text;
            let format_output = String::new(); // TODO!

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

    // #[test]
    // fn test_parse_all_files() {
    //     // list all .roc files under ../test_syntax/tests/snapshots/pass
    //     let files = std::fs::read_dir("../test_syntax/tests/snapshots/pass")
    //         .unwrap()
    //         .map(|res| res.map(|e| e.path()))
    //         .collect::<Result<Vec<_>, std::io::Error>>()
    //         .unwrap();

    //     assert!(files.len() > 0, "no files found in ../test_syntax/tests/snapshots/pass");

    //     for file in files {
    //         // if the extension is not .roc, continue
    //         if file.extension().map(|e| e != "roc").unwrap_or(true) {
    //             continue;
    //         }

    //         eprintln!("parsing {:?}", file);
    //         let text = std::fs::read_to_string(&file).unwrap();
    //         eprintln!("---------------------\n{}\n---------------------", text);
    //         let mut tokenizer = Tokenizer::new(&text);
    //         tokenizer.tokenize(); // make sure we don't panic!
    //         let tb = tokenizer.finish();
    //         eprintln!("tokens: {:?}", tb.kinds);
    //         let mut state = State::from_buf(tb);
    //         state.start_file();
    //         state.pump();
    //         state.assert_end();
    //     }
    // }

    #[test]
    fn test_where_clause_on_newline_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f : a -> U64
            |    where a implements Hash
            |
            |f
            |
            "#
        ));
    }

    #[test]
    fn test_one_plus_two_expr() {
        snapshot_test!(block_indentify(
            r#"
            |1+2
            "#
        ));
    }

    #[test]
    fn test_nested_backpassing_no_newline_before_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |main =
            |    task =
            |        file <-
            |            foo
            |        bar
            |    task
            |42
            "#
        ));
    }

    #[test]
    fn test_tag_pattern_expr() {
        snapshot_test!(block_indentify(
            r#"
            |\Thing -> 42
            "#
        ));
    }

    #[test]
    fn test_newline_and_spaces_before_less_than_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x =
            |    1
            |    < 2
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_outdented_record_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x = foo {
            |    bar: blah,
            |}
            |x
            "#
        ));
    }

    #[test]
    fn test_minimal_app_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |app "test-app" provides [] to "./blah"
            |
            "#
        ));
    }

    #[test]
    fn test_newline_after_paren_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(
            |A)
            "#
        ));
    }

    #[test]
    fn test_unary_not_with_parens_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |!(whee 12 foo)
            "#
        ));
    }

    #[test]
    fn test_when_with_tuple_in_record_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when { foo: (1, 2) } is
            |    { foo: (1, x) } -> x
            |    { foo: (_, b) } -> 3 + b
            "#
        ));
    }

    #[test]
    fn test_unary_negation_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-foo
            "#
        ));
    }

    #[test]
    fn test_type_signature_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |foo : Int
            |foo = 4
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_function_with_tuple_ext_type_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f : (Str)a -> (Str)a
            |f = \x -> x
            |
            |f ("Str", 42)
            |
            "#
        ));
    }

    #[test]
    fn test_multi_backpassing_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x, y <- List.map2 [] []
            |
            |x + y
            |
            "#
        ));
    }

    #[test]
    fn test_opaque_simple_moduledefs() {
        snapshot_test!(block_indentify(
            r#"
            |Age := U8
            |
            "#
        ));
    }

    #[test]
    fn test_list_closing_same_indent_no_trailing_comma_expr() {
        snapshot_test!(block_indentify(
            r#"
            |myList = [
            |    0,
            |    1
            |]
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_call_with_newlines_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |f
            |    -5
            |    2
            "#
        ));
    }

    #[test]
    fn test_list_closing_indent_not_enough_expr() {
        snapshot_test!(block_indentify(
            r#"
            |myList = [
            |    0,
            |    [
            |        a,
            |        b,
            |],
            |    1,
            |]
            |42
            |
            "#
        ));
    }

    // This test no longer works on the new parser; h needs to be indented
    #[ignore]
    #[test]
    fn test_comment_after_tag_in_def_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |Z #
            |h
            | : a
            |j
            "#
        ));
    }

    #[test]
    fn test_lambda_in_chain_expr() {
        snapshot_test!(block_indentify(
            r#"
            |"a string"
            ||> Str.toUtf8
            ||> List.map \byte -> byte + 1
            ||> List.reverse
            "#
        ));
    }

    #[test]
    fn test_minus_twelve_minus_five_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |-12 - 5
            "#
        ));
    }

    #[test]
    fn test_newline_in_type_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |R
            |:D
            |a
            "#
        ));
    }

    #[test]
    fn test_tuple_type_ext_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f: (Str, Str)a -> (Str, Str)a
            |f = \x -> x
            |
            |f (1, 2)
            "#
        ));
    }

    #[test]
    fn test_closure_with_underscores_expr() {
        snapshot_test!(block_indentify(
            r#"
            |\_, _name -> 42
            "#
        ));
    }

    #[test]
    fn test_negative_in_apply_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a=A
            | -g a
            |a
            "#
        ));
    }

    #[test]
    fn test_ann_open_union_expr() {
        snapshot_test!(block_indentify(
            r#"
            |foo : [True, Perhaps Thing]*
            |foo = True
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_newline_before_sub_expr() {
        snapshot_test!(block_indentify(
            r#"
            |3
            |- 4
            "#
        ));
    }

    #[test]
    fn test_annotated_tag_destructure_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |UserId x : [UserId I64]
            |(UserId x) = UserId 42
            |
            |x
            "#
        ));
    }

    #[test]
    fn test_unary_negation_arg_expr() {
        snapshot_test!(block_indentify(
            r#"
            |whee  12 -foo
            "#
        ));
    }

    #[test]
    fn test_parenthetical_var_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |whee
            "#
        ));
    }

    #[test]
    fn test_one_backpassing_expr() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |x <- (\y -> y)
            |
            |x
            |
            "#
        ));
    }

    #[test]
    fn test_list_pattern_weird_indent_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when [] is
            |    [1, 2, 3] -> ""
            "#
        ));
    }

    #[test]
    fn test_negative_float_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-42.9
            "#
        ));
    }

    #[test]
    fn test_value_def_confusion_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |a : F
            |F : h
            |abc
            "#
        ));
    }

    #[test]
    fn test_ability_demand_signature_is_multiline_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |Hash implements
            |    hash : a
            |        -> U64
            |
            |1
            "#
        ));
    }

    #[test]
    fn test_opaque_with_type_arguments_moduledefs() {
        snapshot_test!(block_indentify(
            r#"
            |Bookmark a := { chapter: Str, stanza: Str, notes: a }
            |
            "#
        ));
    }

    #[test]
    fn test_var_when_expr() {
        snapshot_test!(block_indentify(
            r#"
            |whenever
            "#
        ));
    }

    #[test]
    fn test_function_effect_types_header() {
        snapshot_test!(block_indentify(
            r#"
            |platform "cli"
            |    requires {}{ main : Task {} [] } # TODO FIXME
            |    exposes []
            |    packages {}
            |    imports [ Task.{ Task } ]
            |    provides [ mainForHost ]
            |
            "#
        ));
    }

    #[test]
    fn test_apply_unary_not_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |!whee 12 foo
            "#
        ));
    }

    #[test]
    fn test_basic_apply_expr() {
        snapshot_test!(block_indentify(
            r#"
            |whee 1
            "#
        ));
    }

    #[test]
    fn test_list_minus_newlines_expr() {
        snapshot_test!(block_indentify(
            r#"
            |[K,
            |]-i
            "#
        ));
    }

    #[test]
    fn test_pattern_with_space_in_parens_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when Delmin (Del rx) 0 is
            |    Delmin (Del ry) _ -> Node Black 0 Bool.false ry
            "#
        ));
    }

    #[test]
    fn test_list_closing_indent_not_enough_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |myList = [
            |    0,
            |    [
            |        a,
            |        b,
            |    ],
            |    1,
            |]
            |42
            "#
        ));
    }

    #[test]
    fn test_underscore_backpassing_expr() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |_ <- (\y -> y)
            |
            |4
            |
            "#
        ));
    }

    #[test]
    fn test_ann_closed_union_expr() {
        snapshot_test!(block_indentify(
            r#"
            |foo : [True, Perhaps Thing]
            |foo = True
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_outdented_record_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x = foo {
            |  bar: blah
            |}
            |x
            |
            "#
        ));
    }

    #[test]
    fn test_record_destructure_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |{ x, y } = 5
            |y = 6
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_space_only_after_minus_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x- y
            "#
        ));
    }

    #[test]
    fn test_minus_twelve_minus_five_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-12-5
            "#
        ));
    }

    #[test]
    fn test_basic_field_expr() {
        snapshot_test!(block_indentify(
            r#"
            |rec.field
            "#
        ));
    }

    #[test]
    fn test_add_var_with_spaces_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x + 2
            "#
        ));
    }

    #[test]
    fn test_annotated_record_destructure_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |{ x, y } : Foo
            |{ x, y } = { x: "foo", y: 3.14 }
            |
            |x
            "#
        ));
    }

    #[test]
    fn test_empty_string_expr() {
        snapshot_test!(block_indentify(
            r#"
            |""
            |
            "#
        ));
    }

    #[test]
    fn test_where_ident_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |where : { where : I32 }
            |where = { where: 1 }
            |
            |where.where
            "#
        ));
    }

    #[test]
    fn test_apply_parenthetical_tag_args_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Whee (12) (34)
            "#
        ));
    }

    #[test]
    fn test_when_with_tuples_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when (1, 2) is
            |    (1, x) -> x
            |    (_, b) -> 3 + b
            "#
        ));
    }

    #[test]
    fn test_comment_after_annotation_expr() {
        snapshot_test!(block_indentify(
            r#"
            |F:e#
            |
            |
            |q
            "#
        ));
    }

    #[test]
    fn test_when_with_negative_numbers_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            | 1 -> 2
            | -3 -> 4
            |
            "#
        ));
    }

    #[test]
    fn test_newline_in_type_def_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |R : D
            |a
            "#
        ));
    }

    #[test]
    fn test_var_minus_two_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x - 2
            "#
        ));
    }

    #[test]
    fn test_opaque_reference_expr_expr() {
        snapshot_test!(block_indentify(
            r#"
            |@Age
            |
            "#
        ));
    }

    #[test]
    fn test_equals_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x==y
            "#
        ));
    }

    #[test]
    fn test_var_is_expr() {
        snapshot_test!(block_indentify(
            r#"
            |isnt
            "#
        ));
    }

    #[test]
    fn test_function_effect_types_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |platform "cli"
            |    requires {} { main : Task {} [] } # TODO FIXME
            |    exposes []
            |    packages {}
            |    imports [Task.{ Task }]
            |    provides [mainForHost]
            |
            "#
        ));
    }

    #[test]
    fn test_list_pattern_weird_indent_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when [] is
            |    [1, 2,
            |3] -> ""
            |
            "#
        ));
    }

    #[test]
    fn test_multiple_fields_expr() {
        snapshot_test!(block_indentify(
            r#"
            |rec.abc.def.ghi
            "#
        ));
    }

    #[test]
    fn test_nonempty_hosted_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |hosted Foo
            |    exposes
            |        [
            |            Stuff,
            |            Things,
            |            somethingElse,
            |        ]
            |    imports
            |        [
            |            Blah,
            |            Baz.{ stuff, things },
            |        ]
            |    generates Bar with
            |        [
            |            map,
            |            after,
            |            loop,
            |        ]
            |
            "#
        ));
    }

    #[test]
    fn test_basic_tag_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Whee
            "#
        ));
    }

    #[test]
    fn test_newline_before_operator_with_defs_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |7
            |== (
            |    Q : c
            |    42
            |)
            "#
        ));
    }

    #[test]
    fn test_requires_type_header() {
        snapshot_test!(block_indentify(
            r#"
            |platform "test/types"
            |    requires { Flags, Model, } { main : App Flags Model }
            |    exposes []
            |    packages {}
            |    imports []
            |    provides [ mainForHost ]
            |
            |mainForHost : App Flags Model
            |mainForHost = main
            |
            "#
        ));
    }

    #[test]
    fn test_empty_hosted_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |hosted Foo exposes [] imports [] generates Bar with []
            |
            "#
        ));
    }

    #[test]
    fn test_multiline_string_in_apply_expr() {
        snapshot_test!(block_indentify(
            r#"
            |e""""\""""
            "#
        ));
    }

    #[test]
    fn test_empty_interface_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |interface Foo exposes [] imports []
            |
            "#
        ));
    }

    #[test]
    fn test_pattern_as_spaces_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when 0 is
            |    0 # foobar
            |        as # barfoo
            |        n -> {}
            |
            "#
        ));
    }

    #[test]
    fn test_where_clause_on_newline_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |f : a -> U64 where a implements Hash
            |
            |f
            "#
        ));
    }

    #[test]
    fn test_tuple_type_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f: (Str, Str) -> (Str, Str)
            |f = \x -> x
            |
            |f (1, 2)
            "#
        ));
    }

    #[test]
    fn test_underscore_in_assignment_pattern_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Pair x _ = Pair 0 1
            |Pair _ y = Pair 0 1
            |Pair _ _ = Pair 0 1
            |_ = Pair 0 1
            |Pair (Pair x _) (Pair _ y) = Pair (Pair 0 1) (Pair 2 3)
            |
            |0
            |
            "#
        ));
    }

    #[test]
    fn test_multiline_type_signature_with_comment_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f :# comment
            |    {}
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_basic_docs_expr() {
        snapshot_test!(block_indentify(
            r#"
            |## first line of docs
            |##     second line
            |##  third line
            |## fourth line
            |##
            |## sixth line after doc new line
            |x = 5
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_lambda_indent_expr() {
        snapshot_test!(block_indentify(
            r#"
            |\x ->
            |  1
            "#
        ));
    }

    #[test]
    fn test_space_only_after_minus_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x - y
            "#
        ));
    }

    #[test]
    fn test_parse_alias_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Blah a b : Foo.Bar.Baz x y
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_two_arg_closure_expr() {
        snapshot_test!(block_indentify(
            r#"
            |\a, b -> 42
            "#
        ));
    }

    #[test]
    fn test_opaque_type_def_with_newline_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |a : e
            |Na :=
            |    e
            |e0
            "#
        ));
    }

    #[test]
    fn test_newline_in_type_alias_application_expr() {
        snapshot_test!(block_indentify(
            r#"
            |A:A
            | A
            |p
            "#
        ));
    }

    #[test]
    fn test_two_spaced_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |x = 5
            |y = 6
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_record_func_type_decl_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |f : {
            |    getLine : Effect Str,
            |    putLine : Str -> Effect Int,
            |    text : Str,
            |    value : Int *,
            |}
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_comment_before_equals_def_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |t #
            | = 3
            |e
            "#
        ));
    }

    #[test]
    fn test_apply_three_args_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a b c d
            "#
        ));
    }

    #[test]
    fn test_module_def_newline_moduledefs_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |main =
            |    i = 64
            |
            |    i
            |
            "#
        ));
    }

    #[test]
    fn test_var_if_expr() {
        snapshot_test!(block_indentify(
            r#"
            |iffy
            "#
        ));
    }

    #[test]
    fn test_empty_interface_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |interface Foo exposes [] imports []
            |
            "#
        ));
    }

    #[test]
    fn test_closure_in_binop_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |a
            |&& (\x -> x)
            |    8
            "#
        ));
    }

    #[test]
    fn test_float_with_underscores_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-1_23_456.0_1_23_456
            "#
        ));
    }

    #[test]
    fn test_nested_def_annotation_moduledefs() {
        snapshot_test!(block_indentify(
            r#"
            |main =
            |    wrappedNotEq : a, a -> Bool
            |    wrappedNotEq = \num1, num2 ->
            |        num1 != num2
            |
            |    wrappedNotEq 2 3
            |
            "#
        ));
    }

    #[test]
    fn test_crash_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |_ = crash ""
            |_ = crash "" ""
            |_ = crash 15 123
            |_ = try foo (\_ -> crash "")
            |_ =
            |    _ = crash ""
            |    crash
            |
            |{ f: crash "" }
            "#
        ));
    }

    #[test]
    fn test_parenthesized_type_def_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |D : b
            |a
            "#
        ));
    }

    #[test]
    fn test_apply_tag_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Whee 12 34
            "#
        ));
    }

    #[test]
    fn test_tuple_access_after_record_expr() {
        snapshot_test!(block_indentify(
            r#"
            |{ a: (1, 2) }.a.0
            "#
        ));
    }

    #[test]
    fn test_positive_int_expr() {
        snapshot_test!(block_indentify(
            r#"
            |42
            "#
        ));
    }

    #[test]
    fn test_parenthesized_type_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(D):b
            |a
            "#
        ));
    }

    #[test]
    fn test_space_before_colon_full_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |app "example"
            |    packages { pf: "path" }
            |    imports [pf.Stdout]
            |    provides [main] to pf
            |
            |main = Stdout.line "Hello"
            |
            "#
        ));
    }

    #[test]
    fn test_comment_after_def_moduledefs_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |foo = 1 # comment after
            |
            "#
        ));
    }

    #[test]
    fn test_qualified_field_expr() {
        snapshot_test!(block_indentify(
            r#"
            |One.Two.rec.abc.def.ghi
            "#
        ));
    }

    #[test]
    fn test_comment_before_equals_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |t#
            |=3
            |e
            "#
        ));
    }

    #[test]
    fn test_empty_record_update_expr() {
        snapshot_test!(block_indentify(
            r#"
            |{e&}
            "#
        ));
    }

    #[test]
    fn test_when_in_parens_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(when x is
            |    Ok ->
            |        3)
            |
            "#
        ));
    }

    #[test]
    fn test_multi_backpassing_in_def_moduledefs_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |main =
            |    arg1, arg2 <- f {}
            |    "Roc <3 Zig!\n"
            |
            "#
        ));
    }

    #[test]
    fn test_parens_in_value_def_annotation_expr() {
        snapshot_test!(block_indentify(
            r#"
            |i
            |(#
            |N):b
            |a
            "#
        ));
    }

    #[test]
    fn test_record_update_expr() {
        snapshot_test!(block_indentify(
            r#"
            |{ Foo.Bar.baz & x: 5, y: 0 }
            "#
        ));
    }

    #[test]
    fn test_newline_and_spaces_before_less_than_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x = 1
            |    < 2
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_underscore_in_assignment_pattern_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |(Pair x _) = Pair 0 1
            |(Pair _ y) = Pair 0 1
            |(Pair _ _) = Pair 0 1
            |_ = Pair 0 1
            |(Pair (Pair x _) (Pair _ y)) = Pair (Pair 0 1) (Pair 2 3)
            |
            |0
            "#
        ));
    }

    #[test]
    fn test_parens_in_type_def_apply_expr() {
        snapshot_test!(block_indentify(
            r#"
            |U(b a):b
            |a
            "#
        ));
    }

    #[test]
    fn test_newline_inside_empty_list_expr() {
        snapshot_test!(block_indentify(
            r#"
            |[
            |]
            "#
        ));
    }

    #[test]
    fn test_function_with_tuple_type_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f : I64 -> (I64, I64)
            |f = \x -> (x, x + 1)
            |
            |f 42
            |
            "#
        ));
    }

    #[test]
    fn test_control_characters_in_scalar_expr() {
        snapshot_test!(block_indentify(
            r#"
            |''
            "#
        ));
    }

    #[test]
    fn test_multiple_operators_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |31 * 42 + 534
            "#
        ));
    }

    #[test]
    fn test_nested_module_header() {
        snapshot_test!(block_indentify(
            r#"
            |interface Foo.Bar.Baz exposes [] imports []
            |
            "#
        ));
    }

    #[test]
    fn test_multiple_operators_expr() {
        snapshot_test!(block_indentify(
            r#"
            |31*42+534
            "#
        ));
    }

    #[test]
    fn test_call_with_newlines_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f
            |-5
            |2
            "#
        ));
    }

    #[test]
    fn test_newline_singleton_list_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |[
            |    1,
            |]
            "#
        ));
    }

    #[test]
    fn test_basic_var_expr() {
        snapshot_test!(block_indentify(
            r#"
            |whee
            "#
        ));
    }

    #[test]
    fn test_tuple_type_ext_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |f : (Str, Str)a -> (Str, Str)a
            |f = \x -> x
            |
            |f (1, 2)
            "#
        ));
    }

    #[test]
    fn test_full_app_header_trailing_commas_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |app "quicksort"
            |    packages { pf: "./platform" }
            |    imports [foo.Bar.{
            |        Baz,
            |        FortyTwo,
            |        # I'm a happy comment
            |    }]
            |    provides [quicksort] to pf
            |
            "#
        ));
    }

    #[test]
    fn test_negate_multiline_string_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |-(
            |    """
            |    """)
            "#
        ));
    }

    #[test]
    #[ignore]
    fn test_comment_before_colon_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |w#
            |:n
            |Q
            "#
        ));
    }

    #[test]
    fn test_provides_type_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |app "test"
            |    packages { pf: "./platform" }
            |    imports [foo.Bar.Baz]
            |    provides [quicksort] { Flags, Model } to pf
            |
            "#
        ));
    }

    #[test]
    fn test_outdented_colon_in_record_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x = foo {
            |    bar
            |    : blah,
            |}
            |x
            "#
        ));
    }

    #[test]
    fn test_opaque_destructure_first_item_in_body_expr() {
        snapshot_test!(block_indentify(
            r#"
            |@Thunk it = id (@A {})
            |it {}
            |
            "#
        ));
    }

    #[test]
    fn test_annotated_tuple_destructure_expr() {
        snapshot_test!(block_indentify(
            r#"
            |( x, y ) : Foo
            |( x, y ) = ( "foo", 3.14 )
            |
            |x
            |
            "#
        ));
    }

    #[test]
    fn test_record_func_type_decl_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f :
            |    {
            |        getLine : Effect Str,
            |        putLine : Str -> Effect Int,
            |        text: Str,
            |        value: Int *
            |    }
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_requires_type_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |platform "test/types"
            |    requires { Flags, Model } { main : App Flags Model }
            |    exposes []
            |    packages {}
            |    imports []
            |    provides [mainForHost]
            |
            "#
        ));
    }

    #[test]
    fn test_comment_before_colon_def_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |w #
            | : n
            |Q
            "#
        ));
    }

    #[test]
    fn test_pattern_as_list_rest_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when myList is
            |    [first, .. as rest] -> 0
            |
            "#
        ));
    }

    #[test]
    fn test_when_with_numbers_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    1 -> 2
            |    3 -> 4
            "#
        ));
    }

    #[test]
    fn test_equals_with_spaces_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x == y
            "#
        ));
    }

    #[test]
    fn test_nested_def_annotation_moduledefs_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |main =
            |    wrappedNotEq : a, a -> Bool
            |    wrappedNotEq = \num1, num2 ->
            |        num1 != num2
            |
            |    wrappedNotEq 2 3
            |
            "#
        ));
    }

    #[test]
    fn test_nonempty_hosted_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |hosted Foo
            |    exposes
            |    [
            |        Stuff,
            |        Things,
            |        somethingElse,
            |    ]
            |    imports
            |    [
            |        Blah,
            |        Baz.{ stuff, things },
            |    ]
            |    generates Bar with
            |    [
            |        map,
            |        after,
            |        loop,
            |    ]
            |
            "#
        ));
    }

    #[test]
    fn test_sub_with_spaces_expr() {
        snapshot_test!(block_indentify(
            r#"
            |1  -   2
            "#
        ));
    }

    #[test]
    fn test_highest_float_expr() {
        snapshot_test!(block_indentify(
            r#"
            |179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0
            "#
        ));
    }

    #[test]
    fn test_comment_with_non_ascii_expr() {
        snapshot_test!(block_indentify(
            r#"
            |3  # 2 × 2
            |+ 4
            "#
        ));
    }

    #[test]
    fn test_newline_after_mul_expr() {
        snapshot_test!(block_indentify(
            r#"
            |3  *
            |  4
            "#
        ));
    }

    #[test]
    fn test_nested_def_without_newline_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x=a:n 4
            |_
            "#
        ));
    }

    #[test]
    fn test_one_backpassing_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |x <- \y -> y
            |
            |x
            "#
        ));
    }

    #[test]
    fn test_outdented_app_with_record_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x = foo (baz {
            |  bar: blah
            |})
            |x
            |
            "#
        ));
    }

    #[test]
    fn test_when_with_alternative_patterns_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    "blah" | "blop" -> 1
            |    "foo"
            |    | "bar"
            |    | "baz" -> 2
            |
            |    "stuff" -> 4
            "#
        ));
    }

    #[test]
    fn test_comment_after_def_moduledefs() {
        snapshot_test!(block_indentify(
            r#"
            |foo = 1 # comment after
            |
            "#
        ));
    }

    #[test]
    fn test_interface_with_newline_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |interface T exposes [] imports []
            |
            "#
        ));
    }

    #[test]
    fn test_when_with_function_application_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    1 -> Num.neg
            |     2
            |    _ -> 4
            |
            "#
        ));
    }

    #[test]
    fn test_zero_float_expr() {
        snapshot_test!(block_indentify(
            r#"
            |0.0
            "#
        ));
    }

    #[test]
    fn test_opaque_simple_moduledefs_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |Age := U8
            |
            "#
        ));
    }

    #[test]
    fn test_opaque_destructure_first_item_in_body_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |(@Thunk it) = id (@A {})
            |it {}
            "#
        ));
    }

    #[test]
    fn test_when_in_parens_indented_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    Ok -> 3
            "#
        ));
    }

    #[test]
    fn test_empty_app_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |app "test-app" packages {} imports [] provides [] to blah
            |
            "#
        ));
    }

    #[test]
    fn test_extra_newline_in_parens_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |B : {}
            |
            |a
            "#
        ));
    }

    #[test]
    fn test_highest_int_expr() {
        snapshot_test!(block_indentify(
            r#"
            |9223372036854775807
            "#
        ));
    }

    #[test]
    fn test_str_block_multiple_newlines_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |"""
            |
            |
            |#
            |""" #
            "#
        ));
    }

    #[test]
    fn test_apply_unary_negation_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-whee  12 foo
            "#
        ));
    }

    #[test]
    fn test_nonempty_platform_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |platform "foo/barbaz"
            |    requires {Model} { main : {} }
            |    exposes []
            |    packages { foo: "./foo" }
            |    imports []
            |    provides [ mainForHost ]
            |
            "#
        ));
    }

    #[test]
    fn test_multiline_type_signature_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f :
            |    {}
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_if_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |iffy=5
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_one_plus_two_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |1 + 2
            "#
        ));
    }

    #[test]
    fn test_dbg_expr() {
        snapshot_test!(block_indentify(
            r#"
            |dbg 1 == 1
            |
            |4
            |
            "#
        ));
    }

    #[test]
    fn test_newline_before_operator_with_defs_expr() {
        snapshot_test!(block_indentify(
            r#"
            |7
            |==(Q:c 42)
            "#
        ));
    }

    #[test]
    fn test_int_with_underscore_expr() {
        snapshot_test!(block_indentify(
            r#"
            |1__23
            "#
        ));
    }

    #[test]
    fn test_bound_variable_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a:
            |c 0
            "#
        ));
    }

    #[test]
    fn test_when_in_function_python_style_indent_expr() {
        snapshot_test!(block_indentify(
            r#"
            |func = \x -> when n is
            |    0 -> 0
            |42
            "#
        ));
    }

    #[test]
    fn test_ability_single_line_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Hash implements hash : a -> U64 where a implements Hash
            |
            |1
            |
            "#
        ));
    }

    #[test]
    fn test_add_with_spaces_expr() {
        snapshot_test!(block_indentify(
            r#"
            |1  +   2
            "#
        ));
    }

    #[test]
    fn test_comment_after_op_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |12
            |* # test!
            |92
            "#
        ));
    }

    #[test]
    fn test_spaces_inside_empty_list_expr() {
        snapshot_test!(block_indentify(
            r#"
            |[  ]
            "#
        ));
    }

    #[test]
    fn test_spaces_inside_empty_list_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |[]
            "#
        ));
    }

    #[test]
    fn test_comment_after_expr_in_parens_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(i#abc
            |)
            "#
        ));
    }

    #[test]
    fn test_unary_negation_with_parens_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |-(whee 12 foo)
            "#
        ));
    }

    #[test]
    fn test_where_clause_multiple_has_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f : a -> (b -> c) where a implements A, b implements Eq, c implements Ord
            |
            |f
            |
            "#
        ));
    }

    #[test]
    fn test_empty_record_expr() {
        snapshot_test!(block_indentify(
            r#"
            |{}
            "#
        ));
    }

    #[test]
    fn test_where_clause_multiple_has_across_newlines_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |f : a -> (b -> c) where a implements Hash, b implements Eq, c implements Ord
            |
            |f
            "#
        ));
    }

    #[test]
    fn test_var_then_expr() {
        snapshot_test!(block_indentify(
            r#"
            |thenever
            "#
        ));
    }

    #[test]
    fn test_parenthetical_basic_field_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(rec).field
            "#
        ));
    }

    #[test]
    fn test_crash_expr() {
        snapshot_test!(block_indentify(
            r#"
            |_ = crash ""
            |_ = crash "" ""
            |_ = crash 15 123
            |_ = try foo (\_ -> crash "")
            |_ =
            |  _ = crash ""
            |  crash
            |
            |{ f: crash "" }
            |
            "#
        ));
    }

    #[test]
    fn test_opaque_reference_pattern_with_arguments_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when n is
            |    @Add n m -> n + m
            "#
        ));
    }

    #[test]
    fn test_full_app_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |app "quicksort"
            |    packages { pf: "./platform" }
            |    imports [ foo.Bar.Baz ]
            |    provides [ quicksort ] to pf
            |
            "#
        ));
    }

    #[test]
    fn test_empty_platform_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |platform "rtfeldman/blah" requires {} { main : {} } exposes [] packages {} imports [] provides []
            |
            "#
        ));
    }

    #[test]
    fn test_expect_fx_moduledefs() {
        snapshot_test!(block_indentify(
            r#"
            |# expecting some effects
            |expect-fx 5 == 2
            |
            "#
        ));
    }

    #[test]
    fn test_unary_negation_access_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-rec1.field
            "#
        ));
    }

    #[test]
    fn test_pattern_as_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when 0 is
            |    _ as n -> n
            |
            "#
        ));
    }

    #[test]
    fn test_pattern_with_space_in_parens_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when Delmin (Del rx) 0 is
            |    Delmin (Del ry ) _ -> Node Black 0 Bool.false ry
            |
            "#
        ));
    }

    #[test]
    fn test_def_without_newline_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a:b i
            "#
        ));
    }

    #[test]
    fn test_multiline_string_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |a = "Hello,\n\nWorld!"
            |b =
            |    """
            |    Hello,\n\nWorld!
            |    """
            |c =
            |    """
            |    Hello,
            |
            |    World!
            |    """
            |42
            "#
        ));
    }

    #[test]
    fn test_empty_app_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |app "test-app" packages {} imports [] provides [] to blah
            |
            "#
        ));
    }

    #[test]
    fn test_closure_in_binop_with_spaces_expr() {
        snapshot_test!(block_indentify(
            r#"
            |i>\s->s
            |-a
            "#
        ));
    }

    #[test]
    fn test_when_with_negative_numbers_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    1 -> 2
            |    -3 -> 4
            "#
        ));
    }

    #[test]
    fn test_newline_before_add_expr() {
        snapshot_test!(block_indentify(
            r#"
            |3
            |+ 4
            "#
        ));
    }

    #[test]
    fn test_one_minus_two_expr() {
        snapshot_test!(block_indentify(
            r#"
            |1-2
            "#
        ));
    }

    #[test]
    fn test_lambda_indent_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |\x ->
            |    1
            "#
        ));
    }

    #[test]
    fn test_tuple_accessor_function_expr() {
        snapshot_test!(block_indentify(
            r#"
            |.1 (1, 2, 3)
            "#
        ));
    }

    #[test]
    fn test_sub_var_with_spaces_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x - 2
            "#
        ));
    }

    #[test]
    fn test_destructure_tag_assignment_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |(Email str) = Email "blah@example.com"
            |str
            "#
        ));
    }

    #[test]
    fn test_ops_with_newlines_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |3
            |+
            |
            |4
            "#
        ));
    }

    #[test]
    fn test_comment_inside_empty_list_expr() {
        snapshot_test!(block_indentify(
            r#"
            |[#comment
            |]
            "#
        ));
    }

    #[test]
    fn test_outdented_colon_in_record_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x = foo {
            |bar
            |:
            |blah
            |}
            |x
            |
            "#
        ));
    }

    #[test]
    fn test_underscore_backpassing_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |_ <- \y -> y
            |
            |4
            "#
        ));
    }

    #[test]
    fn test_where_clause_multiple_has_across_newlines_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f : a -> (b -> c)
            |    where a implements Hash,
            |      b implements Eq,
            |      c implements Ord
            |
            |f
            |
            "#
        ));
    }

    #[test]
    fn test_when_in_function_expr() {
        snapshot_test!(block_indentify(
            r#"
            |func = \x -> when n is
            |              0 -> 0
            |42
            "#
        ));
    }

    #[test]
    fn test_spaced_singleton_list_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |[1]
            "#
        ));
    }

    #[test]
    fn test_mixed_docs_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |# ## not docs!
            |## docs, but with a problem
            |## (namely that this is a mix of docs and regular comments)
            |# not docs
            |x = 5
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_parenthetical_field_qualified_var_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(One.Two.rec).field
            "#
        ));
    }

    #[test]
    fn test_one_minus_two_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |1 - 2
            "#
        ));
    }

    #[test]
    fn test_newline_singleton_list_expr() {
        snapshot_test!(block_indentify(
            r#"
            |[
            |1
            |]
            "#
        ));
    }

    #[test]
    fn test_annotated_record_destructure_expr() {
        snapshot_test!(block_indentify(
            r#"
            |{ x, y } : Foo
            |{ x, y } = { x : "foo", y : 3.14 }
            |
            |x
            |
            "#
        ));
    }

    #[test]
    fn test_mixed_docs_expr() {
        snapshot_test!(block_indentify(
            r#"
            |### not docs!
            |## docs, but with a problem
            |## (namely that this is a mix of docs and regular comments)
            |# not docs
            |x = 5
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_empty_package_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |package "rtfeldman/blah" exposes [] packages {}
            "#
        ));
    }

    #[test]
    fn test_bound_variable_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |a :
            |    c
            |0
            "#
        ));
    }

    #[test]
    fn test_nonempty_platform_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |platform "foo/barbaz"
            |    requires { Model } { main : {} }
            |    exposes []
            |    packages { foo: "./foo" }
            |    imports []
            |    provides [mainForHost]
            |
            "#
        ));
    }

    #[test]
    fn test_one_def_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |x = 5
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_when_with_records_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            | { y } -> 2
            | { z, w } -> 4
            |
            "#
        ));
    }

    #[test]
    fn test_var_minus_two_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x-2
            "#
        ));
    }

    #[test]
    fn test_unary_negation_arg_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |whee 12 -foo
            "#
        ));
    }

    #[test]
    fn test_nonempty_package_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |package "foo/barbaz"
            |    exposes [Foo, Bar]
            |    packages { foo: "./foo" }
            "#
        ));
    }

    #[test]
    fn test_apply_two_args_expr() {
        snapshot_test!(block_indentify(
            r#"
            |whee  12  34
            "#
        ));
    }

    #[test]
    fn test_sub_with_spaces_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |1 - 2
            "#
        ));
    }

    #[test]
    fn test_where_ident_expr() {
        snapshot_test!(block_indentify(
            r#"
            |where : {where: I32}
            |where = {where: 1}
            |
            |where.where
            |
            "#
        ));
    }

    #[test]
    fn test_negate_multiline_string_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-""""""
            "#
        ));
    }

    #[test]
    fn test_pos_inf_float_expr() {
        snapshot_test!(block_indentify(
            r#"
            |inf
            "#
        ));
    }

    #[test]
    fn test_when_in_parens_indented_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(when x is
            |    Ok -> 3
            |     )
            |
            "#
        ));
    }

    #[test]
    fn test_multi_backpassing_with_apply_expr() {
        snapshot_test!(block_indentify(
            r#"
            |F 1, r <- a
            |W
            "#
        ));
    }

    #[test]
    fn test_apply_two_args_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |whee 12 34
            "#
        ));
    }

    #[test]
    fn test_annotated_tuple_destructure_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |(x, y) : Foo
            |(x, y) = ("foo", 3.14)
            |
            |x
            "#
        ));
    }

    #[test]
    fn test_standalone_module_defs_moduledefs_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |# comment 1
            |foo = 1
            |
            |# comment 2
            |bar = "hi"
            |baz = "stuff"
            |# comment n
            |
            "#
        ));
    }

    #[test]
    fn test_fn_with_record_arg_expr() {
        snapshot_test!(block_indentify(
            r#"
            |table : {
            |    height : Pixels
            |    } -> Table
            |table = \{height} -> crash "not implemented"
            |table
            "#
        ));
    }

    #[test]
    fn test_comment_after_annotation_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |F : e #
            |
            |q
            "#
        ));
    }

    #[test]
    fn test_space_before_colon_full() {
        snapshot_test!(block_indentify(
            r#"
            |app "example"
            |    packages { pf : "path" }
            |    imports [ pf.Stdout ]
            |    provides [ main ] to pf
            |
            |main = Stdout.line "Hello"
            "#
        ));
    }

    #[test]
    fn test_positive_float_expr() {
        snapshot_test!(block_indentify(
            r#"
            |42.9
            "#
        ));
    }

    #[test]
    fn test_expect_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |expect
            |    1 == 1
            |
            |4
            "#
        ));
    }

    #[test]
    fn test_record_type_with_function_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x : { init : {} -> Model, update : Model, Str -> Model, view : Model -> Str }
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_value_def_confusion_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a:F
            |F
            |:h
            |abc
            "#
        ));
    }

    #[test]
    fn test_negate_multiline_string_with_quote_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-""""<"""
            "#
        ));
    }

    #[test]
    fn test_single_arg_closure_expr() {
        snapshot_test!(block_indentify(
            r#"
            |\a -> 42
            "#
        ));
    }

    #[test]
    fn test_where_clause_function_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f : a -> (b -> c) where a implements A
            |
            |f
            |
            "#
        ));
    }

    #[test]
    fn test_empty_list_expr() {
        snapshot_test!(block_indentify(
            r#"
            |[]
            "#
        ));
    }

    #[test]
    fn test_number_literal_suffixes_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |{
            |    u8: 123u8,
            |    u16: 123u16,
            |    u32: 123u32,
            |    u64: 123u64,
            |    u128: 123u128,
            |    i8: 123i8,
            |    i16: 123i16,
            |    i32: 123i32,
            |    i64: 123i64,
            |    i128: 123i128,
            |    nat: 123nat,
            |    dec: 123dec,
            |    u8Neg: -123u8,
            |    u16Neg: -123u16,
            |    u32Neg: -123u32,
            |    u64Neg: -123u64,
            |    u128Neg: -123u128,
            |    i8Neg: -123i8,
            |    i16Neg: -123i16,
            |    i32Neg: -123i32,
            |    i64Neg: -123i64,
            |    i128Neg: -123i128,
            |    natNeg: -123nat,
            |    decNeg: -123dec,
            |    u8Bin: 0b101u8,
            |    u16Bin: 0b101u16,
            |    u32Bin: 0b101u32,
            |    u64Bin: 0b101u64,
            |    u128Bin: 0b101u128,
            |    i8Bin: 0b101i8,
            |    i16Bin: 0b101i16,
            |    i32Bin: 0b101i32,
            |    i64Bin: 0b101i64,
            |    i128Bin: 0b101i128,
            |    natBin: 0b101nat,
            |}
            "#
        ));
    }

    #[test]
    fn test_comment_before_op_expr() {
        snapshot_test!(block_indentify(
            r#"
            |3  # test!
            |+ 4
            "#
        ));
    }

    #[test]
    fn test_nested_if_expr() {
        snapshot_test!(block_indentify(
            r#"
            |if t1 then
            |  1
            |else if t2 then
            |  2
            |else
            |  3
            |
            "#
        ));
    }

    #[test]
    fn test_newline_after_sub_expr() {
        snapshot_test!(block_indentify(
            r#"
            |3  -
            |  4
            "#
        ));
    }

    #[test]
    fn test_fn_with_record_arg_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |table :
            |    {
            |        height : Pixels,
            |    }
            |    -> Table
            |table = \{ height } -> crash "not implemented"
            |table
            "#
        ));
    }

    #[test]
    fn test_ability_demand_signature_is_multiline_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Hash implements
            |  hash : a
            |         -> U64
            |
            |1
            |
            "#
        ));
    }

    #[test]
    fn test_annotated_tag_destructure_expr() {
        snapshot_test!(block_indentify(
            r#"
            |UserId x : [ UserId I64 ]
            |UserId x = UserId 42
            |
            |x
            |
            "#
        ));
    }

    #[test]
    fn test_negative_int_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-42
            "#
        ));
    }

    #[test]
    fn test_multiline_tuple_with_comments_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(
            |    #before 1
            |    1
            |    #after 1
            |    ,
            |    #before 2
            |    2
            |    #after 2
            |    ,
            |    #before 3
            |    3
            |    # after 3
            |)
            "#
        ));
    }

    #[test]
    fn test_when_with_tuples_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when (1, 2) is
            | (1, x) -> x
            | (_, b) -> 3 + b
            |
            "#
        ));
    }

    #[test]
    fn test_opaque_has_abilities_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |A := U8 implements [Eq, Hash]
            |
            |A := a where a implements Other
            |    implements [Eq, Hash]
            |
            |A := a where a implements Other
            |    implements [Eq, Hash]
            |
            |A := U8 implements [Eq { eq }, Hash { hash }]
            |
            |A := U8 implements [Eq { eq, eq1 }]
            |
            |A := U8 implements [Eq { eq, eq1 }, Hash]
            |
            |A := U8 implements [Hash, Eq { eq, eq1 }]
            |
            |A := U8 implements []
            |
            |A := a where a implements Other
            |    implements [Eq { eq }, Hash { hash }]
            |
            |A := U8 implements [Eq {}]
            |
            |0
            "#
        ));
    }

    #[test]
    fn test_negative_in_apply_def_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |a = A
            |    -g
            |    a
            |a
            "#
        ));
    }

    #[test]
    fn test_lowest_int_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-9223372036854775808
            "#
        ));
    }

    #[test]
    fn test_nested_if_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |if t1 then
            |    1
            |else if t2 then
            |    2
            |else
            |    3
            "#
        ));
    }

    #[test]
    fn test_opaque_reference_expr_with_arguments_expr() {
        snapshot_test!(block_indentify(
            r#"
            |@Age m n
            |
            "#
        ));
    }

    #[test]
    fn test_unary_not_expr() {
        snapshot_test!(block_indentify(
            r#"
            |!blah
            "#
        ));
    }

    #[test]
    fn test_not_multiline_string_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |!
            |"""
            |"""
            "#
        ));
    }

    #[test]
    fn test_expect_fx_moduledefs_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |# expecting some effects
            |expect-fx 5 == 2
            |
            "#
        ));
    }

    #[test]
    fn test_packed_singleton_list_expr() {
        snapshot_test!(block_indentify(
            r#"
            |[1]
            "#
        ));
    }

    #[test]
    fn test_parenthetical_apply_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(whee) 1
            "#
        ));
    }

    #[test]
    fn test_dbg_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |dbg
            |    1 == 1
            |
            |4
            "#
        ));
    }

    #[test]
    fn test_not_multiline_string_expr() {
        snapshot_test!(block_indentify(
            r#"
            |!""""""
            "#
        ));
    }

    #[test]
    fn test_def_without_newline_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |a : b
            |i
            "#
        ));
    }

    // This test no longer works on the new parser; h needs to be indented
    #[ignore]
    #[test]
    fn test_comment_after_tag_in_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Z#
            |h
            |:a
            |j
            "#
        ));
    }

    #[test]
    fn test_ten_times_eleven_expr() {
        snapshot_test!(block_indentify(
            r#"
            |10*11
            "#
        ));
    }

    #[test]
    fn test_multiline_type_signature_with_comment_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |f :
            |    # comment
            |    {}
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_apply_unary_not_expr() {
        snapshot_test!(block_indentify(
            r#"
            |!whee  12 foo
            "#
        ));
    }

    #[test]
    fn test_nonempty_package_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |package "foo/barbaz"
            |    exposes [Foo, Bar]
            |    packages { foo: "./foo" }
            |
            "#
        ));
    }

    #[test]
    fn test_empty_record_update_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |{ e &  }
            "#
        ));
    }

    #[test]
    fn test_spaced_singleton_list_expr() {
        snapshot_test!(block_indentify(
            r#"
            |[ 1 ]
            "#
        ));
    }

    #[test]
    fn test_ability_multi_line_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Hash implements
            |  hash : a -> U64
            |  hash2 : a -> U64
            |
            |1
            |
            "#
        ));
    }

    #[test]
    fn test_opaque_type_def_with_newline_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a:e
            |Na:=
            | e e0
            "#
        ));
    }

    #[test]
    fn test_parse_as_ann_expr() {
        snapshot_test!(block_indentify(
            r#"
            |foo : Foo.Bar.Baz x y as Blah a b
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_when_with_tuple_in_record_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when {foo: (1, 2)} is
            | {foo: (1, x)} -> x
            | {foo: (_, b)} -> 3 + b
            |
            "#
        ));
    }

    #[test]
    fn test_parens_in_value_def_annotation_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |i #
            |N : b
            |a
            "#
        ));
    }

    #[test]
    fn test_zero_int_expr() {
        snapshot_test!(block_indentify(
            r#"
            |0
            "#
        ));
    }

    #[test]
    fn test_plus_when_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |1
            |+
            |when Foo is
            |    Foo -> 2
            |    Bar -> 3
            "#
        ));
    }

    #[test]
    fn test_if_def_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |iffy = 5
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_add_with_spaces_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |1 + 2
            "#
        ));
    }

    #[test]
    fn test_comment_with_non_ascii_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |3 # 2 × 2
            |+ 4
            "#
        ));
    }

    #[test]
    fn test_full_app_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |app "quicksort"
            |    packages { pf: "./platform" }
            |    imports [foo.Bar.Baz]
            |    provides [quicksort] to pf
            |
            "#
        ));
    }

    #[test]
    fn test_not_docs_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |# ######
            |# ## not docs!
            |# #still not docs
            |# #####
            |x = 5
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_three_arg_closure_expr() {
        snapshot_test!(block_indentify(
            r#"
            |\a, b, c -> 42
            "#
        ));
    }

    #[test]
    fn test_when_in_function_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |func = \x ->
            |    when n is
            |        0 -> 0
            |42
            "#
        ));
    }

    #[test]
    fn test_list_closing_same_indent_with_trailing_comma_expr() {
        snapshot_test!(block_indentify(
            r#"
            |myList = [
            |    0,
            |    1,
            |]
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_extra_newline_in_parens_expr() {
        snapshot_test!(block_indentify(
            r#"
            |B:{}
            |
            |(
            |a)
            "#
        ));
    }

    #[test]
    fn test_newline_after_equals_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x =
            |    5
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_unary_not_with_parens_expr() {
        snapshot_test!(block_indentify(
            r#"
            |!(whee  12 foo)
            "#
        ));
    }

    #[test]
    fn test_two_branch_when_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            | "" -> 1
            | "mise" -> 2
            |
            "#
        ));
    }

    #[test]
    fn test_string_without_escape_expr() {
        snapshot_test!(block_indentify(
            r#"
            |"123 abc 456 def"
            "#
        ));
    }

    #[test]
    fn test_multi_backpassing_with_apply_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |(F 1), r <- a
            |W
            "#
        ));
    }

    #[test]
    fn test_multiline_tuple_with_comments_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |(
            |    # before 1
            |    1,
            |    # after 1
            |    # before 2
            |    2,
            |    # after 2
            |    # before 3
            |    3,
            |    # after 3
            |)
            "#
        ));
    }

    #[test]
    fn test_list_patterns_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when [] is
            |  [] -> {}
            |  [..] -> {}
            |  [_, .., _, ..] -> {}
            |  [a, b, c, d] -> {}
            |  [a, b, ..] -> {}
            |  [.., c, d] -> {}
            |  [[A], [..], [a]] -> {}
            |  [[[], []], [[], x]] -> {}
            |
            "#
        ));
    }

    #[test]
    fn test_when_in_parens_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    Ok ->
            |        3
            "#
        ));
    }

    #[test]
    fn test_unary_negation_with_parens_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-(whee  12 foo)
            "#
        ));
    }

    #[test]
    fn test_record_access_after_tuple_expr() {
        snapshot_test!(block_indentify(
            r#"
            |({a: 0}, {b: 1}).0.a
            "#
        ));
    }

    #[test]
    fn test_outdented_list_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a = [
            |  1, 2, 3
            |]
            |a
            |
            "#
        ));
    }

    #[test]
    fn test_multi_backpassing_in_def_moduledefs() {
        snapshot_test!(block_indentify(
            r#"
            |main =
            |    arg1, arg2 <- f {}
            |    "Roc <3 Zig!\n"
            |
            "#
        ));
    }

    #[test]
    fn test_plus_if_expr() {
        snapshot_test!(block_indentify(
            r#"
            |1 * if Bool.true then 1 else 1
            |
            "#
        ));
    }

    #[test]
    fn test_closure_in_binop_with_spaces_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |i
            |> (\s -> s
            |)
            |    -a
            "#
        ));
    }

    #[test]
    fn test_when_if_guard_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    _ ->
            |        1
            |
            |    _ ->
            |        2
            |
            |    Ok ->
            |        3
            |
            "#
        ));
    }

    #[test]
    fn test_opaque_reference_pattern_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when n is
            |  @Age -> 1
            |
            "#
        ));
    }

    #[test]
    fn test_basic_tuple_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(1, 2, 3)
            "#
        ));
    }

    #[test]
    fn test_standalone_module_defs_moduledefs() {
        snapshot_test!(block_indentify(
            r#"
            |# comment 1
            |foo = 1
            |
            |# comment 2
            |bar = "hi"
            |baz = "stuff"
            |# comment n
            |
            "#
        ));
    }

    #[test]
    fn test_newline_in_packages_full_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |app "hello"
            |    packages {
            |        pf:
            |        "https://github.com/roc-lang/basic-cli/releases/download/0.7.0/bkGby8jb0tmZYsy2hg1E_B2QrCgcSTxdUlHtETwm5m4.tar.br",
            |    }
            |    imports [pf.Stdout]
            |    provides [main] to pf
            |
            |main =
            |    Stdout.line "I'm a Roc application!"
            |
            "#
        ));
    }

    #[test]
    fn test_opaque_reference_pattern_with_arguments_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when n is
            |  @Add n m -> n + m
            |
            "#
        ));
    }

    #[test]
    fn test_tuple_access_after_ident_expr() {
        snapshot_test!(block_indentify(
            r#"
            |abc = (1, 2, 3)
            |abc.0
            "#
        ));
    }

    #[test]
    fn test_when_with_records_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    { y } -> 2
            |    { z, w } -> 4
            "#
        ));
    }

    #[test]
    fn test_neg_inf_float_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-inf
            "#
        ));
    }

    #[test]
    fn test_var_else_expr() {
        snapshot_test!(block_indentify(
            r#"
            |elsewhere
            "#
        ));
    }

    #[test]
    fn test_type_decl_with_underscore_expr() {
        snapshot_test!(block_indentify(
            r#"
            |doStuff : UserId -> Task Str _
            |42
            "#
        ));
    }

    #[test]
    fn test_equals_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x == y
            "#
        ));
    }

    #[test]
    fn test_one_spaced_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |x = 5
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_apply_unary_negation_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |-whee 12 foo
            "#
        ));
    }

    #[test]
    fn test_type_signature_function_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |foo : Int, Float -> Bool
            |foo = \x, _ -> 42
            |
            |42
            "#
        ));
    }

    #[test]
    fn test_qualified_var_expr() {
        snapshot_test!(block_indentify(
            r#"
            |One.Two.whee
            "#
        ));
    }

    #[test]
    fn test_nested_backpassing_no_newline_before_expr() {
        snapshot_test!(block_indentify(
            r#"
            |main =
            |    task = file <-
            |                foo
            |            bar
            |    task
            |42
            "#
        ));
    }

    #[test]
    fn test_record_with_if_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |{ x: if Bool.true then 1 else 2, y: 3 }
            "#
        ));
    }

    #[test]
    fn test_lowest_float_expr() {
        snapshot_test!(block_indentify(
            r#"
            |-179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0
            "#
        ));
    }

    #[test]
    fn test_destructure_tag_assignment_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Email str = Email "blah@example.com"
            |str
            |
            "#
        ));
    }

    #[test]
    fn test_not_docs_expr() {
        snapshot_test!(block_indentify(
            r#"
            |#######
            |### not docs!
            |##still not docs
            |######
            |x = 5
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_plus_when_expr() {
        snapshot_test!(block_indentify(
            r#"
            |1 +
            |    when Foo is
            |        Foo -> 2
            |        Bar -> 3
            |
            "#
        ));
    }

    #[test]
    fn test_closure_in_binop_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a && \x->x
            |8
            "#
        ));
    }

    #[test]
    fn test_multiline_string_in_apply_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |e
            |    """
            |    "\"
            |    """
            "#
        ));
    }

    #[test]
    fn test_comment_inside_empty_list_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |[ # comment
            |]
            "#
        ));
    }

    #[test]
    fn test_full_app_header_trailing_commas_header() {
        snapshot_test!(block_indentify(
            r#"
            |app "quicksort"
            |    packages { pf: "./platform", }
            |    imports [ foo.Bar.{
            |        Baz,
            |        FortyTwo,
            |        # I'm a happy comment
            |    } ]
            |    provides [ quicksort, ] to pf
            |
            "#
        ));
    }

    #[test]
    fn test_two_backpassing_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |x <- \y -> y
            |z <- {}
            |
            |x
            "#
        ));
    }

    #[test]
    fn test_newline_in_type_alias_application_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |A : A
            |    A
            |p
            "#
        ));
    }

    #[test]
    fn test_parenthesized_type_def_space_before_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(
            |A):b
            |a
            "#
        ));
    }

    #[test]
    fn test_when_with_alternative_patterns_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            | "blah" | "blop" -> 1
            | "foo" |
            |  "bar"
            | |"baz" -> 2
            | "stuff" -> 4
            |
            "#
        ));
    }

    #[test]
    fn test_negate_multiline_string_with_quote_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |-(
            |    """
            |    "<
            |    """)
            "#
        ));
    }

    #[test]
    fn test_control_characters_in_scalar_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |'\u(7)'
            "#
        ));
    }

    #[test]
    fn test_ten_times_eleven_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |10 * 11
            "#
        ));
    }

    #[test]
    fn test_minimal_app_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |app "test-app" provides [] to "./blah"
            |
            "#
        ));
    }

    #[test]
    fn test_nested_module_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |interface Foo.Bar.Baz exposes [] imports []
            |
            "#
        ));
    }

    #[test]
    fn test_when_in_function_python_style_indent_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |func = \x ->
            |    when n is
            |        0 -> 0
            |42
            "#
        ));
    }

    #[test]
    fn test_opaque_reference_pattern_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when n is
            |    @Age -> 1
            "#
        ));
    }

    #[test]
    fn test_comment_after_op_expr() {
        snapshot_test!(block_indentify(
            r#"
            |12  * # test!
            | 92
            "#
        ));
    }

    #[test]
    fn test_parenthesized_type_def_space_before_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |A : b
            |a
            "#
        ));
    }

    #[test]
    fn test_opaque_with_type_arguments_moduledefs_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |Bookmark a := { chapter : Str, stanza : Str, notes : a }
            |
            "#
        ));
    }

    #[test]
    fn test_list_closing_same_indent_no_trailing_comma_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |myList = [
            |    0,
            |    1,
            |]
            |42
            "#
        ));
    }

    #[test]
    fn test_parenthetical_var_expr() {
        snapshot_test!(block_indentify(
            r#"
            |(whee)
            "#
        ));
    }

    #[test]
    fn test_str_block_multiple_newlines_expr() {
        snapshot_test!(block_indentify(
            r###"
            |"""
            |
            |
            |#"""#
            "###
        ));
    }

    #[test]
    fn test_list_patterns_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when [] is
            |    [] -> {}
            |    [..] -> {}
            |    [_, .., _, ..] -> {}
            |    [a, b, c, d] -> {}
            |    [a, b, ..] -> {}
            |    [.., c, d] -> {}
            |    [[A], [..], [a]] -> {}
            |    [[[], []], [[], x]] -> {}
            "#
        ));
    }

    #[test]
    fn test_empty_platform_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |platform "rtfeldman/blah" requires {} { main : {} } exposes [] packages {} imports [] provides []
            |
            "#
        ));
    }

    #[test]
    fn test_interface_with_newline_header() {
        snapshot_test!(block_indentify(
            r#"
            |interface T exposes [] imports []
            |
            "#
        ));
    }

    #[test]
    fn test_newline_after_sub_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |3
            |-
            |4
            "#
        ));
    }

    #[test]
    fn test_newline_after_mul_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |3
            |*
            |4
            "#
        ));
    }

    #[test]
    fn test_provides_type_header() {
        snapshot_test!(block_indentify(
            r#"
            |app "test"
            |    packages { pf: "./platform" }
            |    imports [ foo.Bar.Baz ]
            |    provides [ quicksort ] { Flags, Model, } to pf
            |
            "#
        ));
    }

    #[test]
    fn test_ability_multi_line_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |Hash implements
            |    hash : a -> U64
            |    hash2 : a -> U64
            |
            |1
            "#
        ));
    }

    #[test]
    fn test_where_clause_non_function_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f : a where a implements A
            |
            |f
            |
            "#
        ));
    }

    #[test]
    fn test_ops_with_newlines_expr() {
        snapshot_test!(block_indentify(
            r#"
            |3
            |+
            |
            |  4
            "#
        ));
    }

    #[test]
    fn test_record_with_if_expr() {
        snapshot_test!(block_indentify(
            r#"
            |{x : if Bool.true then 1 else 2, y: 3 }
            "#
        ));
    }

    #[test]
    fn test_newline_after_paren_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |A
            "#
        ));
    }

    #[test]
    fn test_record_access_after_tuple_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |({ a: 0 }, { b: 1 }).0.a
            "#
        ));
    }

    #[test]
    fn test_single_underscore_closure_expr() {
        snapshot_test!(block_indentify(
            r#"
            |\\_ -> 42
            "#
        ));
    }

    #[test]
    fn test_empty_hosted_header_header() {
        snapshot_test!(block_indentify(
            r#"
            |hosted Foo exposes [] imports [] generates Bar with []
            |
            "#
        ));
    }

    #[test]
    fn test_multi_char_string_expr() {
        snapshot_test!(block_indentify(
            r#"
            |"foo"
            |
            "#
        ));
    }

    #[test]
    fn test_opaque_has_abilities_expr() {
        snapshot_test!(block_indentify(
            r#"
            |A := U8 implements [Eq, Hash]
            |
            |A := a where a implements Other implements [Eq, Hash]
            |
            |A := a where a implements Other
            |     implements [Eq, Hash]
            |
            |A := U8 implements [Eq {eq}, Hash {hash}]
            |
            |A := U8 implements [Eq {eq, eq1}]
            |
            |A := U8 implements [Eq {eq, eq1}, Hash]
            |
            |A := U8 implements [Hash, Eq {eq, eq1}]
            |
            |A := U8 implements []
            |
            |A := a where a implements Other
            |     implements [Eq {eq}, Hash {hash}]
            |
            |A := U8 implements [Eq {}]
            |
            |0
            |
            "#
        ));
    }

    #[test]
    fn test_outdented_app_with_record_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x = foo
            |    (
            |        baz {
            |            bar: blah,
            |        }
            |    )
            |x
            "#
        ));
    }

    #[test]
    fn test_multiline_string_expr() {
        snapshot_test!(block_indentify(
            r#"
            |a = "Hello,\n\nWorld!"
            |b = """Hello,\n\nWorld!"""
            |c =
            |    """
            |    Hello,
            |
            |    World!
            |    """
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_empty_package_header_header_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |package "rtfeldman/blah" exposes [] packages {}
            |
            "#
        ));
    }

    #[test]
    fn test_when_with_numbers_expr() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            | 1 -> 2
            | 3 -> 4
            |
            "#
        ));
    }

    #[test]
    fn test_one_def_expr() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |x=5
            |
            |42
            |
            "#
        ));
    }

    #[test]
    fn test_comment_after_expr_in_parens_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |i # abc
            "#
        ));
    }

    #[test]
    fn test_list_minus_newlines_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |[
            |    K,
            |]
            |- i
            "#
        ));
    }

    #[test]
    fn test_when_in_assignment_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x =
            |    when n is
            |        0 -> 0
            |42
            "#
        ));
    }

    #[test]
    fn test_nested_def_without_newline_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |x =
            |    a : n
            |    4
            |_
            "#
        ));
    }

    #[test]
    fn test_newline_in_packages_full() {
        snapshot_test!(block_indentify(
            r#"
            |app "hello"
            |    packages { pf:
            |"https://github.com/roc-lang/basic-cli/releases/download/0.7.0/bkGby8jb0tmZYsy2hg1E_B2QrCgcSTxdUlHtETwm5m4.tar.br"
            |}
            |    imports [pf.Stdout]
            |    provides [main] to pf
            |
            |main =
            |    Stdout.line "I'm a Roc application!"
            "#
        ));
    }

    #[test]
    fn test_outdented_list_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |a = [
            |    1,
            |    2,
            |    3,
            |]
            |a
            "#
        ));
    }

    #[test]
    fn test_one_char_string_expr() {
        snapshot_test!(block_indentify(
            r#"
            |"x"
            |
            "#
        ));
    }

    #[test]
    fn test_tuple_type_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |f : (Str, Str) -> (Str, Str)
            |f = \x -> x
            |
            |f (1, 2)
            "#
        ));
    }

    #[test]
    fn test_expect_expr() {
        snapshot_test!(block_indentify(
            r#"
            |expect 1 == 1
            |
            |4
            |
            "#
        ));
    }

    #[test]
    fn test_when_with_function_application_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    1 ->
            |        Num.neg
            |            2
            |
            |    _ -> 4
            "#
        ));
    }

    #[test]
    fn test_when_in_assignment_expr() {
        snapshot_test!(block_indentify(
            r#"
            |x = when n is
            |     0 -> 0
            |42
            "#
        ));
    }

    #[test]
    fn test_where_clause_multiple_bound_abilities_expr() {
        snapshot_test!(block_indentify(
            r#"
            |f : a -> b where a implements Hash & Eq, b implements Eq & Hash & Display
            |
            |f : a -> b
            |  where a implements Hash & Eq,
            |    b implements Hash & Display & Eq
            |
            |f
            |
            "#
        ));
    }

    #[test]
    fn test_two_backpassing_expr() {
        snapshot_test!(block_indentify(
            r#"
            |# leading comment
            |x <- (\y -> y)
            |z <- {}
            |
            |x
            |
            "#
        ));
    }

    #[test]
    fn test_two_branch_when_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |when x is
            |    "" -> 1
            |    "mise" -> 2
            "#
        ));
    }

    #[test]
    fn test_ability_two_in_a_row_expr() {
        snapshot_test!(block_indentify(
            r#"
            |Ab1 implements ab1 : a -> {} where a implements Ab1
            |
            |Ab2 implements ab2 : a -> {} where a implements Ab2
            |
            |1
            |
            "#
        ));
    }

    #[test]
    fn test_where_clause_multiple_bound_abilities_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |f : a -> b where a implements Hash & Eq, b implements Eq & Hash & Display
            |
            |f : a -> b where a implements Hash & Eq, b implements Hash & Display & Eq
            |
            |f
            "#
        ));
    }

    #[test]
    fn test_number_literal_suffixes_expr() {
        snapshot_test!(block_indentify(
            r#"
            |{
            |  u8:   123u8,
            |  u16:  123u16,
            |  u32:  123u32,
            |  u64:  123u64,
            |  u128: 123u128,
            |  i8:   123i8,
            |  i16:  123i16,
            |  i32:  123i32,
            |  i64:  123i64,
            |  i128: 123i128,
            |  nat:  123nat,
            |  dec:  123dec,
            |  u8Neg:   -123u8,
            |  u16Neg:  -123u16,
            |  u32Neg:  -123u32,
            |  u64Neg:  -123u64,
            |  u128Neg: -123u128,
            |  i8Neg:   -123i8,
            |  i16Neg:  -123i16,
            |  i32Neg:  -123i32,
            |  i64Neg:  -123i64,
            |  i128Neg: -123i128,
            |  natNeg:  -123nat,
            |  decNeg:  -123dec,
            |  u8Bin:   0b101u8,
            |  u16Bin:  0b101u16,
            |  u32Bin:  0b101u32,
            |  u64Bin:  0b101u64,
            |  u128Bin: 0b101u128,
            |  i8Bin:   0b101i8,
            |  i16Bin:  0b101i16,
            |  i32Bin:  0b101i32,
            |  i64Bin:  0b101i64,
            |  i128Bin: 0b101i128,
            |  natBin:  0b101nat,
            |}
            |
            "#
        ));
    }

    #[test]
    fn test_module_def_newline_moduledefs() {
        snapshot_test!(block_indentify(
            r#"
            |main =
            |    i = 64
            |
            |    i
            |
            "#
        ));
    }

    #[test]
    fn test_comment_before_op_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |3 # test!
            |+ 4
            "#
        ));
    }

    #[test]
    fn test_parens_in_type_def_apply_expr_formatted() {
        snapshot_test!(block_indentify(
            r#"
            |U (b a) : b
            |a
            "#
        ));
    }
}
