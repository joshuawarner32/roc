pub struct Tree {
    pub kinds: Vec<N>,
    pub indices: Vec<u32>,
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

    /// An opaque type/tag name, e.g. @Foo
    OpaqueName,

    TupleAccessFunction,
    FieldAccessFunction,

    /// The "crash" keyword
    Crash,

    /// The special dbg keyword, as in `dbg x`
    Dbg,

    /// Tag, e.g. `Foo`
    Tag,

    /// Reference to an opaque type, e.g. @Opaq
    OpaqueRef,

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

    /// As type, e.g. `List Foo a as a`
    InlineKwAs,
    EndTypeAs,

    /// As pattern, e.g. `when a is x as y`
    EndPatternAs,

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

    PatternAny, // special pattern that we use if the user didn't put anything after the ':' in a record pattern

    /// A pattern used in a list indicating missing elements, e.g. [.., a]
    PatternDoubleDot,

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
    InlineAbilityImplements,
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
    InlineTypeColonEqual,
    BeginTypeTagUnion,
    EndTypeTagUnion,
    TypeWildcard,

    BeginImplements,
    EndImplements,
    BeginAbilityMethod,
    EndAbilityMethod,
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
    EndRecordFieldPair,
    InlineKwIf,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NodeIndexKind {
    Begin,          // this is a begin node; the index points one past the corresponding end node
    End,            // this is an end node; the index points to the corresponding begin node
    EndOnly, // this is an end node; the index points to the first child and there is no corresponding begin node
    EndSingleToken, // this is an end node that only contains one item; the index points to the token
    Token,          // the index points to a token

    Unused, // we don't use the index for this node
}

impl N {
    pub fn is_decl(self) -> bool {
        match self {
            N::EndAssign | N::EndTypeOrTypeAlias | N::EndBackpassing | N::EndImplements => true,
            _ => false,
        }
    }

    pub fn index_kind(self) -> NodeIndexKind {
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
            | N::BeginAbilityMethod
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
            | N::EndAbilityMethod
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
            | N::InlineAbilityImplements
            | N::InlineKwIs
            | N::InlineKwIf
            | N::InlineLambdaArrow
            | N::InlineColon
            | N::InlineTypeColon
            | N::InlineTypeColonEqual
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
            | N::TupleAccessFunction
            | N::FieldAccessFunction => NodeIndexKind::Token,
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
            | N::EndPatternAs
            | N::EndWhereClause
            | N::EndRecordFieldPair
            | N::EndTypeApply => NodeIndexKind::EndOnly,
            N::DotModuleLowerIdent | N::DotModuleUpperIdent => NodeIndexKind::EndSingleToken,
            N::Float
            | N::SingleQuote
            | N::Underscore
            | N::TypeWildcard
            | N::Crash
            | N::Dbg
            | N::PatternAny
            | N::PatternDoubleDot
            | N::OpaqueName => NodeIndexKind::Token,
        }
    }
}

impl Tree {
    pub fn new() -> Tree {
        Tree {
            kinds: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn len(&self) -> u32 {
        self.kinds.len() as u32
    }
}
