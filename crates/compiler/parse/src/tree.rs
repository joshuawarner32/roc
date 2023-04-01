use bumpalo::{collections::Vec, Bump};

use crate::{token::{Token, BinOp}};

#[derive(Debug, Clone, Copy)]
pub struct TokenId(u32);

#[derive(Debug, Clone, Copy)]
pub struct ExposedName(pub TokenId);
#[derive(Debug, Clone, Copy)]
pub struct StrLiteral(pub TokenId);
#[derive(Debug, Clone, Copy)]
pub struct FloatLiteral(pub TokenId);
#[derive(Debug, Clone, Copy)]
pub struct IntLiteral(pub TokenId);
#[derive(Debug, Clone, Copy)]
pub struct SingleQuoteLiteral(pub TokenId);
#[derive(Debug, Clone, Copy)]
pub struct PackageName(pub StrLiteral);
#[derive(Debug, Clone, Copy)]
pub struct PackageShorthand(pub LowercaseIdent);
#[derive(Debug, Clone, Copy)]
pub struct UppercaseIdent(pub TokenId);
#[derive(Debug, Clone, Copy)]
pub struct LowercaseIdent(pub TokenId);
#[derive(Debug, Clone, Copy)]
pub struct FieldName(pub LowercaseIdent);
#[derive(Debug, Clone, Copy)]
pub struct ValueAccess(pub LowercaseIdent);
#[derive(Debug, Clone, Copy)]
pub struct ValueDef(pub LowercaseIdent);
#[derive(Debug, Clone, Copy)]
pub struct TupleIndex(pub IntLiteral);
#[derive(Debug, Clone, Copy)]
pub struct Underscore(pub TokenId);

#[derive(Debug, Clone, Copy)]
pub struct ModuleName {
    pub start: TokenId,
    pub name_count: u32, // might be 0!
}

#[derive(Debug, Clone, Copy)]
pub struct QualifiedTypeName {
    pub module: ModuleName,
    pub ty: TypeName,
}

#[derive(Debug, Clone, Copy)]
pub struct TypeName(pub UppercaseIdent);
#[derive(Debug, Clone, Copy)]
pub struct TypeVariableName(pub LowercaseIdent);
#[derive(Debug, Clone, Copy)]
pub struct TagName(pub UppercaseIdent);
#[derive(Debug, Clone, Copy)]
pub struct OpaqueName(pub TokenId);

#[derive(Debug, Clone, Copy)]
pub struct ListCurly<'a, T>(pub &'a [T]);
#[derive(Debug, Clone, Copy)]
pub struct ListSquare<'a, T>(pub &'a [T]);
#[derive(Debug, Clone, Copy)]
pub struct ListParen<'a, T>(pub &'a [T]);

#[derive(Debug, Clone, Copy)]
pub struct Root<'a> {
    pub header: Header<'a>,
    pub defs: Block<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum Header<'a> {
    Interface(InterfaceHeader<'a>),
    App(AppHeader<'a>),
    Package(PackageHeader<'a>),
    Platform(PlatformHeader<'a>),
    Hosted(HostedHeader<'a>),
}

#[derive(Debug, Clone, Copy)]
pub struct InterfaceHeader<'a> {
    pub name: ModuleName,
    pub exports: ListSquare<'a, ExposedName>,
    pub imports: ListSquare<'a, ImportsEntry<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub struct AppHeader<'a> {
    pub name: StrLiteral,
    pub packages: Option<ListCurly<'a, PackageEntry>>,
    pub imports: Option<ListSquare<'a, ImportsEntry<'a>>>,
    pub provides_entries: ListSquare<'a, ExposedName>,
    pub provides_types: ListCurly<'a, TypeName>,
    pub provides_to: ProvidesTarget,
}

#[derive(Debug, Clone, Copy)]
pub enum ProvidesTarget {
    Shorthand(PackageShorthand),
    Name(PackageName),
}

#[derive(Debug, Clone, Copy)]
pub struct PackageHeader<'a> {
    pub name: PackageName,
    pub exposes: ListSquare<'a, ModuleName>,
    pub packages: ListCurly<'a, PackageEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct PlatformHeader<'a> {
    pub name: PackageName,
    pub requires: PlatformRequires<'a>,
    pub exposes: ListSquare<'a, ModuleName>,
    pub packages: ListCurly<'a, PackageEntry>,
    pub imports: ListSquare<'a, ImportsEntry<'a>>,
    pub provides: ListSquare<'a, ExposedName>,
}

#[derive(Debug, Clone, Copy)]
pub struct HostedHeader<'a> {
    pub name: ModuleName,
    pub exposes: ListSquare<'a, ExposedName>,
    pub imports: ListSquare<'a, ImportsEntry<'a>>,
    pub generates: UppercaseIdent,
    pub generates_with: ListSquare<'a, ExposedName>,
}

#[derive(Debug, Clone, Copy)]
pub enum ImportsEntry<'a> {
    Module(ModuleName, ListCurly<'a, ExposedName>),
    Package(PackageShorthand, ModuleName, ListCurly<'a, ExposedName>),
}

#[derive(Debug, Clone, Copy)]
pub struct PackageEntry {
    pub shorthand: PackageShorthand,
    pub name: PackageName,
}

#[derive(Debug, Clone, Copy)]
pub struct PlatformRequires<'a> {
    pub rigids: ListCurly<'a, UppercaseIdent>,
    pub signature: ListCurly<'a, TypedIdent<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub struct TypedIdent<'a> {
    pub ident: LowercaseIdent,
    pub type_annotation: TypeAnnotation<'a>,
}

#[derive(Debug, Clone, Copy)]
pub enum TypeAnnotation<'a> {
    /// A function. The types of its arguments, then the type of its return value.
    Function(&'a [TypeAnnotation<'a>], &'a TypeAnnotation<'a>),

    /// Applying a type to some arguments (e.g. Map.Map String Int)
    /// Note that the arguments may be empty (e.g. Map.Map)
    Apply(QualifiedTypeName, &'a [TypeAnnotation<'a>]),

    /// A bound type variable, e.g. `a` in `(a -> a)`
    BoundVariable(TypeVariableName),

    /// Inline type alias, e.g. `as List a` in `[Cons a (List a), Nil] as List a`
    As(
        &'a TypeAnnotation<'a>,
        TypeHeader<'a>,
    ),

    Record {
        fields: ListCurly<'a, AssignedField<TypeAnnotation<'a>>>,
        /// The row type variable in an open record, e.g. the `r` in `{ name: Str }r`.
        /// This is None if it's a closed record annotation like `{ name: Str }`.
        ext: Option<&'a TypeAnnotation<'a>>,
    },

    Tuple {
        elems: ListParen<'a, TypeAnnotation<'a>>,
        /// The row type variable in an open tuple, e.g. the `r` in `( Str, Str )r`.
        /// This is None if it's a closed tuple annotation like `( Str, Str )`.
        ext: Option<&'a TypeAnnotation<'a>>,
    },

    /// A tag union, e.g. `[
    TagUnion {
        /// The row type variable in an open tag union, e.g. the `a` in `[Foo, Bar]a`.
        tags: ListSquare<'a, Tag<'a>>,
        /// This is None if it's a closed tag union like `[Foo, Bar]`.
        ext: Option<&'a TypeAnnotation<'a>>,
    },

    /// '_', indicating the compiler should infer the type
    Inferred,

    /// The `*` type variable, e.g. in (List *)
    Wildcard,

    /// A "where" clause demanding abilities designated by a `|`, e.g. `a -> U64 | a has Hash`
    Where(&'a TypeAnnotation<'a>, &'a [HasClause<'a>]),

}

#[derive(Debug, Clone, Copy)]
pub struct AbilityName(pub UppercaseIdent);

#[derive(Debug, Clone, Copy)]
pub struct Tag<'a> {
    pub name: TagName,
    pub args: &'a [TypeAnnotation<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct HasClause<'a> {
    pub var: LowercaseIdent,
    pub abilities: &'a [AbilityName],
}

#[derive(Debug, Clone, Copy)]
pub struct TypeHeader<'a> {
    pub name: TypeName,
    pub vars: &'a [Pattern<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct AssignedField<T> {
    pub name: FieldName,
    pub value: T,
}

#[derive(Debug, Clone, Copy)]
pub struct Block<'a> {
    pub items: &'a [Expr<'a>],
}

#[derive(Debug, Copy, Clone)]
pub enum Accessor {
    RecordField(FieldName),
    TupleIndex(TupleIndex),
}

#[derive(Debug, Copy, Clone)]
pub enum Expr<'a> {
    // Number Literals
    Float(FloatLiteral),

    /// Integer Literals, e.g. `42`
    Num(IntLiteral),

    /// String Literals, e.g. `"foo"`
    Str(StrLiteral),

    /// eg 'b'
    SingleQuote(SingleQuoteLiteral),

    /// Look up exactly one field on a record or tuple, e.g. `x.foo` or `x.0`.
    Access(&'a Expr<'a>, Accessor),

    /// e.g. `.foo` or `.0`
    AccessorFunction(Accessor),

    /// List literals, e.g. `[1, 2, 3]`
    List(ListSquare<'a, Expr<'a>>),

    /// Record updates (e.g. `{ x & y: 3 }`)
    RecordUpdate {
        update: &'a Expr<'a>,
        fields: ListCurly<'a, AssignedField<Expr<'a>>>,
    },

    /// Record literals, e.g. `{ x: 1, y: 2 }`
    Record(ListCurly<'a, AssignedField<Expr<'a>>>),

    /// Tuple literals, e.g. `(1, 2)`
    Tuple(ListParen<'a, Expr<'a>>),

    /// A variable, e.g. `x` or `SomeModule.x`
    Var {
        module_name: ModuleName, // Note: possibly empty
        ident: ValueAccess,
    },

    /// An underscore, e.g. `_` or `_x`
    Underscore(Underscore),

    /// The "crash" keyword
    Crash,

    /// Tag
    Tag(TagName),

    /// Reference to an opaque type, e.g. @Opaq
    OpaqueRef(OpaqueName),

    /// Closure, e.g. `\x -> x`
    Closure(&'a [Pattern<'a>], &'a Expr<'a>),

    /// Indented block of statements and expressions
    Block(Block<'a>),

    /// The special dbg function, e.g. `dbg x`
    Dbg,

    /// Function application, e.g. `f x`
    Apply(&'a Expr<'a>, &'a [Expr<'a>]),

    /// Binary operator, e.g. `x + y`
    BinOp(&'a Expr<'a>, BinOp, &'a Expr<'a>),

    /// Unary operator, e.g. `-x`
    UnaryOp(UnaryOp, &'a Expr<'a>),

    /// If expression, e.g. `if x then y else z`
    If(&'a [(Expr<'a>, Expr<'a>)], &'a Expr<'a>),

    /// A when expression, e.g. `when x is y -> z` (you need a newline after the 'is')
    When(
        /// The condition
        &'a Expr<'a>,
        /// A | B if bool -> expression
        /// <Pattern 1> | <Pattern 2> if <Guard> -> <Expr>
        /// Vec, because there may be many patterns, and the guard
        /// is Option<Expr> because each branch may be preceded by
        /// a guard (".. if ..").
        &'a [WhenBranch<'a>],
    ),


    // TODO: move these vvvvv to a Stmt type


    /// Type annotation, e.g. `x: Num *` or `f: Str, Str -> Str`
    TypeAnnotation(ValueDef, &'a TypeAnnotation<'a>),

    /// Assignment, e.g. `x = 1` or `Tag a b = foo 1 2 3`
    Assignment(&'a Pattern<'a>, &'a Expr<'a>),

    /// Expect, e.g. `expect 1 + 2 == 3`
    Expect(&'a Expr<'a>),

    TypeDef {
        header: TypeHeader<'a>,
        typ: TypeAnnotation<'a>,
    },

    OpaqueTypeDef {
        header: TypeHeader<'a>,
        typ: TypeAnnotation<'a>,
        abilities: Option<ListSquare<'a, Ability<'a>>>,
    },

    Ability {
        header: TypeHeader<'a>,
        members: &'a [AbilityMember<'a>],
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AbilityMember<'a> {
    pub name: ValueDef,
    pub typ: &'a TypeAnnotation<'a>,
}

#[derive(Debug, Copy, Clone)]
pub struct Ability<'a> {
    name: AbilityName,
    impls: Option<ListCurly<'a, LowercaseIdent>>, // TODO: might need to be more general than LowercaseIdent
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// (-), e.g. (-x)
    Negate,
    /// (!), e.g. (!x)
    Not,
}

#[derive(Clone, Copy, Debug)]
pub struct WhenBranch<'a> {
    pub patterns: &'a [Pattern<'a>],
    pub body: Expr<'a>,
    pub guard: Option<Expr<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub enum Pattern<'a> {
    /// Identifier, e.g. `x`
    Identifier(ValueDef),

    /// Tag, e.g. `Just` in `Just x`
    Tag(TagName),

    /// Opaque reference, e.g. `@Opaq`
    OpaqueRef(OpaqueName),

    /// Function application, e.g. `f x`
    Apply(&'a Pattern<'a>, &'a [Pattern<'a>]),

    /// A record pattern, e.g. `{ x, y }` or `{ x, y: Just z }`
    RecordDestructure(ListCurly<'a, FieldPattern<'a>>),

    /// An integer literal, e.g. `42`
    NumLiteral(IntLiteral),
    FloatLiteral(FloatLiteral),
    StrLiteral(StrLiteral),
    Underscore(Underscore),
    SingleQuote(SingleQuoteLiteral),

    /// A tuple pattern, e.g. (Just x, 1)
    Tuple(ListParen<'a, Pattern<'a>>),

    /// A list pattern like [_, x, ..]
    List(ListSquare<'a, ListItemPattern<'a>>),

    /// As, e.g. `x as y`
    As(&'a Pattern<'a>, ValueDef),
}

#[derive(Debug, Clone, Copy)]
pub enum ListItemPattern<'a> {
    /// A pattern, e.g. `x`
    Pattern(&'a Pattern<'a>),
    /// A rest pattern, e.g. `..` or `.. as xs`
    RestPattern(Option<ValueDef>),
}

#[derive(Debug, Clone, Copy)]
pub enum FieldPattern<'a> {
    /// Identifier, e.g. `x`
    Identifier(ValueDef),

    /// A required field pattern, e.g. { x: Just 0 } -> ...
    /// Can only occur inside of a RecordDestructure
    RequiredField(FieldName, &'a Pattern<'a>),

    /// An optional field pattern, e.g. { x ? Just 0 } -> ...
    /// Can only occur inside of a RecordDestructure
    OptionalField(FieldName, &'a Expr<'a>),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Context {
    Root,
    Ability,
    AbilityName,
    ListItemPattern,
    FieldPattern,
    Accessor,
    Header,
    InterfaceHeader,
    AppHeader,
    PackageHeader,
    StrLiteral,
    PackageName,
    UppercaseIdent,
    LowercaseIdent,
    PackageShorthand,
    ModuleName,
    TypeName,
    PlatformHeader,
    PlatformRequires,
    HostedHeader,
    PackageEntry,
    ImportsEntry,
    ExposedName,
    ListCurly,
    ListSquare,
    ListParen,
    TypedIdent,
    TypeAnnotation,
    ProvidesTarget,
    QualifiedTypeName,
    AssignedField,
    FieldName,
    Tag,
    Expr,
    Block,
    Pattern,
}

#[derive(Debug, Clone)]
pub struct Error {
    pub kind: ErrorKind,
    pub token_index: u32,
    pub context: std::vec::Vec<(Context, TokenId)>,
}

#[derive(Debug, Clone, Copy)]
pub enum ErrorKind {
    Expected(Token),
    ExpectedExposedName,
    ExpectedEndOfInput,
    ExtOnParens,
    ExpectedSingleTypeAnnotation,
    ExpectedProvidesTarget,
    ExpectedTagApply,
    HigherOrderTypeAnnotation,
    ExpectedTypeAnnotationEnd,
    ExpectedNewline,
    ExpectedNoModuleName,
    ExpectedAccessor,
    ExpectedPattern,
    ExpectedExpr,
}

pub trait Parse<'a>: Sized {
    const CONTEXT: Context;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error>;

    fn parse_seq_item<'t>(p: &mut Parser<'a, 't>, seq: &mut Vec<'a, Self>) -> Result<(), Error> {
        if p.consume(Token::OpenIndent).is_some() {
            let original_len = seq.len();

            if let Err(e) = Self::parse_seq_item(p, seq) {
                seq.truncate(original_len);
                return Err(e);
            }
            
            loop {
                if p.consume(Token::CloseIndent).is_some() {
                    break;
                }

                if p.consume(Token::Comma).is_none() {
                    seq.truncate(original_len);
                    return Err(p.error(ErrorKind::Expected(Token::Comma)));
                }

                // We shouldn't ever have closing indent immediately following a comma (it should come before, because of the lexer)
                // but we'll check for it anyway
                if p.consume(Token::CloseIndent).is_some() {
                    break;
                }

                if let Err(e) = Self::parse_seq_item(p, seq) {
                    seq.truncate(original_len);
                    return Err(e);
                }
            }

        } else {
            let item = Self::parse(p)?;
            seq.push(item);
        }
        Ok(())
    }
}

impl<'a> Parse<'a> for Root<'a> {
    const CONTEXT: Context = Context::Root;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let header = p.parse()?;
        let defs = p.parse_block_inner();
        Ok(Root { header, defs })
    }
}

impl<'a> Parse<'a> for Header<'a> {
    const CONTEXT: Context = Context::Header;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let header = match p.peek() {
            Some(Token::KwInterface) => Header::Interface(p.parse()?),
            Some(Token::KwApp) => Header::App(p.parse()?),
            Some(Token::KwPackage) => Header::Package(p.parse()?),
            Some(Token::KwPlatform) => Header::Platform(p.parse()?),
            Some(Token::KwHosted) => Header::Hosted(p.parse()?),
            _ => panic!("expected header"),
        };
        p.consume(Token::Newline);
        Ok(header)
    }
}

impl<'a> Parse<'a> for InterfaceHeader<'a> {
    const CONTEXT: Context = Context::InterfaceHeader;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        p.expect(Token::KwInterface)?;
        let name = p.parse()?;
        p.expect(Token::KwExposes)?;
        let exports = p.parse()?;
        p.expect(Token::KwImports)?;
        let imports = p.parse()?;

        Ok(InterfaceHeader { name, exports, imports })
    }
}

impl<'a> Parse<'a> for AppHeader<'a> {
    const CONTEXT: Context = Context::AppHeader;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        p.expect(Token::KwApp)?;
        let name = p.parse()?;
        
        let packages = if p.consume(Token::KwPackages).is_some() {
            Some(p.parse()?)
        } else {
            None
        };
        let imports = if p.consume(Token::KwImports).is_some() {
            Some(p.parse()?)
        } else {
            None
        };

        p.expect(Token::KwProvides)?;
        let provides_entries = p.parse()?;
        let provides_types = p.optional_if(Token::OpenCurly, |p| p.parse())?
            .unwrap_or_else(|| ListCurly(&[]));
        p.expect(Token::KwTo)?;
        let provides_to = p.parse()?;

        Ok(AppHeader { name, packages, imports, provides_entries, provides_types, provides_to })
    }
}

impl<'a> Parse<'a> for PackageHeader<'a> {
    const CONTEXT: Context = Context::PackageHeader;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        p.expect(Token::KwPackage)?;
        let name = p.parse()?;
        p.expect(Token::KwExposes)?;
        let exposes = p.parse()?;
        p.expect(Token::KwPackages)?;
        let packages = p.parse()?;

        Ok(PackageHeader { name, exposes, packages })
    }
}

impl<'a> Parse<'a> for StrLiteral {
    const CONTEXT: Context = Context::StrLiteral;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        Ok(StrLiteral(p.expect(Token::String)?))
    }
}

impl<'a> Parse<'a> for PackageName {
    const CONTEXT: Context = Context::PackageName;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        Ok(PackageName(p.parse()?))
    }
}

impl<'a> Parse<'a> for ProvidesTarget {
    const CONTEXT: Context = Context::ProvidesTarget;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        match p.peek() {
            Some(Token::LowerIdent) => Ok(ProvidesTarget::Shorthand(p.parse()?)),
            Some(Token::String) => Ok(ProvidesTarget::Name(p.parse()?)),
            _ => Err(p.error(ErrorKind::ExpectedProvidesTarget)),
        }
    }
}

impl<'a> Parse<'a> for UppercaseIdent {
    const CONTEXT: Context = Context::UppercaseIdent;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        Ok(UppercaseIdent(p.expect(Token::UpperIdent)?))
    }
}

impl<'a> Parse<'a> for LowercaseIdent {
    const CONTEXT: Context = Context::LowercaseIdent;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        Ok(LowercaseIdent(p.expect(Token::LowerIdent)?))
    }
}

impl<'a> Parse<'a> for PackageShorthand {
    const CONTEXT: Context = Context::PackageShorthand;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        Ok(PackageShorthand(LowercaseIdent(p.expect(Token::LowerIdent)?)))
    }
}

impl<'a> Parse<'a> for FieldName {
    const CONTEXT: Context = Context::FieldName;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        Ok(FieldName(LowercaseIdent(p.expect(Token::LowerIdent)?)))
    }
}

impl<'a> Parse<'a> for AbilityName {
    const CONTEXT: Context = Context::AbilityName;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        Ok(AbilityName(UppercaseIdent(p.expect(Token::UpperIdent)?)))
    }
}

impl<'a> Parse<'a> for Ability<'a> {
    const CONTEXT: Context = Context::Ability;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let name = p.parse()?;
        let impls = p.optional_if(Token::OpenCurly, |p| p.parse())?;
        Ok(Ability { name, impls })
    }
}

impl<'a> Parse<'a> for ModuleName {
    const CONTEXT: Context = Context::ModuleName;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        parse_module_name_in_header(p)
    }
}

impl<'a> Parse<'a> for QualifiedTypeName {
    const CONTEXT: Context = Context::QualifiedTypeName;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let initial = p.expect(Token::UpperIdent)?;
        let mut count = 1;
        let mut last = initial;
        while (p.peek() == Some(Token::DotNoLeadingWhitespace) || p.peek() == Some(Token::DotLeadingWhitespace)) && p.peek_ahead(1) == Some(Token::UpperIdent) {
            p.token_index += 1;
            last = TokenId(p.token_index as u32);
            p.token_index += 1;
            count += 1;
        }

        let module = ModuleName {
            start: initial,
            name_count: count - 1,
        };

        let ty = TypeName(UppercaseIdent(last));

        Ok(QualifiedTypeName {
            module,
            ty
        })
    }
}

                
impl<'a> Parse<'a> for TypeName {
    const CONTEXT: Context = Context::TypeName;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        Ok(TypeName(p.parse()?))
    }
}

impl<'a> Parse<'a> for PlatformHeader<'a> {
    const CONTEXT: Context = Context::PlatformHeader;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        p.expect(Token::KwPlatform)?;
        let name = p.parse()?;
        p.expect(Token::KwRequires)?;
        let requires = p.parse()?;
        p.expect(Token::KwExposes)?;
        let exposes = p.parse()?;
        p.expect(Token::KwPackages)?;
        let packages = p.parse()?;
        p.expect(Token::KwImports)?;
        let imports = p.parse()?;
        p.expect(Token::KwProvides)?;
        let provides = p.parse()?;

        Ok(PlatformHeader { name, requires, exposes, packages, imports, provides })
    }
}

impl<'a> Parse<'a> for PlatformRequires<'a> {
    const CONTEXT: Context = Context::PlatformRequires;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let rigids = p.parse()?;
        let signature = p.parse()?;

        Ok(PlatformRequires { rigids, signature })
    }
}

impl<'a> Parse<'a> for HostedHeader<'a> {
    const CONTEXT: Context = Context::HostedHeader;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        p.expect(Token::KwHosted)?;
        let name = p.parse()?;
        p.expect(Token::KwExposes)?;
        let exposes = p.parse()?;
        p.expect(Token::KwImports)?;
        let imports = p.parse()?;
        p.expect(Token::KwGenerates)?;
        let generates = p.parse()?;
        p.expect(Token::KwWith)?;
        let generates_with = p.parse()?;

        Ok(HostedHeader { name, exposes, imports, generates, generates_with })
    }
}

impl<'a> Parse<'a> for PackageEntry {
    const CONTEXT: Context = Context::PackageEntry;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let shorthand = p.parse()?;
        p.expect(Token::Colon)?;
        let name = p.parse()?;
        Ok(PackageEntry { shorthand, name })
    }
}

impl<'a> Parse<'a> for ImportsEntry<'a> {
    const CONTEXT: Context = Context::ImportsEntry;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let opt_package_name = if let Some(pn) = p.consume(Token::LowerIdent) {
            p.expect(Token::DotNoLeadingWhitespace)?;
            Some(PackageShorthand(LowercaseIdent(pn)))
        } else {
            None
        };
        
        let name = p.parse()?;
        
        let exposes = if p.consume(Token::DotLeadingWhitespace).is_some() || p.consume(Token::DotNoLeadingWhitespace).is_some() {
            p.parse()?
        } else {
            ListCurly(&[])
        };

        if let Some(pn) = opt_package_name {
            Ok(ImportsEntry::Package(pn, name, exposes))
        } else {
            Ok(ImportsEntry::Module(name, exposes))
        }
    }
}

fn parse_module_name_in_header(p: &mut Parser) -> Result<ModuleName, Error> {
    let initial = p.expect(Token::UpperIdent)?;
    let mut count = 1;
    while (p.peek() == Some(Token::DotNoLeadingWhitespace) || p.peek() == Some(Token::DotLeadingWhitespace)) && p.peek_ahead(1) != Some(Token::OpenCurly) {
        p.token_index += 1;
        p.expect(Token::UpperIdent)?;
        count += 1;
    }
    Ok(ModuleName {
        start: initial,
        name_count: count,
    })
}

impl<'a> Parse<'a> for ExposedName {
    const CONTEXT: Context = Context::ExposedName;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        if let Some(id) = p.consume(Token::LowerIdent).or_else(|| p.consume(Token::UpperIdent)) {
            Ok(ExposedName(id))
        } else {
            Err(p.error(ErrorKind::ExpectedExposedName))
        }
    }
}

impl<'a, T: Parse<'a>> Parse<'a> for ListCurly<'a, T> {
    const CONTEXT: Context = Context::ListCurly;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        p.expect(Token::OpenCurly)?;
        let items = p.parse_comma_sep_list()?;
        p.expect_masking_whitespace(Token::CloseCurly)?;
        Ok(ListCurly(items.into_bump_slice()))
    }
}

impl<'a, T: Parse<'a>> Parse<'a> for ListSquare<'a, T> {
    const CONTEXT: Context = Context::ListSquare;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        p.expect(Token::OpenSquare)?;
        let items = p.parse_comma_sep_list()?;
        p.expect_masking_whitespace(Token::CloseSquare)?;
        Ok(ListSquare(items.into_bump_slice()))
    }
}

impl<'a, T: Parse<'a>> Parse<'a> for ListParen<'a, T> {
    const CONTEXT: Context = Context::ListParen;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        p.expect(Token::OpenParen)?;
        let items = p.parse_comma_sep_list()?;
        p.expect_masking_whitespace(Token::CloseParen)?;
        Ok(ListParen(items.into_bump_slice()))
    }
}

impl<'a> Parse<'a> for TypedIdent<'a> {
    const CONTEXT: Context = Context::TypedIdent;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let ident = p.parse()?;
        p.expect(Token::Colon)?;
        let type_annotation = p.parse()?;

        Ok(TypedIdent { ident, type_annotation })
    }
}

impl<'a> Parse<'a> for Accessor {
    const CONTEXT: Context = Context::Accessor;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        debug_assert!(p.peek() == Some(Token::DotNoLeadingWhitespace) || p.peek() == Some(Token::DotLeadingWhitespace));
        p.token_index += 1;
        match p.peek() {
            Some(Token::LowerIdent) => {
                let name = p.expect(Token::LowerIdent)?;
                Ok(Accessor::RecordField(FieldName(LowercaseIdent(name))))
            }
            Some(Token::IntBase10) => {
                let num = p.expect(Token::IntBase10)?;
                Ok(Accessor::TupleIndex(TupleIndex(IntLiteral(num))))
            }
            _ => Err(p.error(ErrorKind::ExpectedAccessor)),
        }
    }
}

impl<'a> Parse<'a> for Block<'a> {
    const CONTEXT: Context = Context::Block;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let mut items = Vec::new_in(p.arena);
        loop {
            if p.peek_terminator() {
                break;
            }
            items.push(parse_stmt_or_expr(p)?);
            match p.peek() {
                Some(Token::Newline) => p.token_index += 1,
                None | Some(Token::CloseIndent) => break,
                _ => return Err(p.error(ErrorKind::ExpectedNewline)),
            }
        }

        Ok(Block { items: items.into_bump_slice() })
    }
}

impl<'a> Parse<'a> for Pattern<'a> {
    const CONTEXT: Context = Context::Pattern;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let pat = parse_pattern_atom(p)?;

        if p.consume(Token::KwAs).is_some() {
            let name = ValueDef(p.parse()?);
            Ok(Pattern::As(p.arena.alloc(pat), name))
        } else {
            Ok(pat)
        }
    }
}

impl<'a> Parse<'a> for ListItemPattern<'a> {
    const CONTEXT: Context = Context::ListItemPattern;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        // check for ..
        if p.peek() == Some(Token::DoubleDot) {
            p.token_index += 1;
            let name = p.optional_if(Token::KwAs, |p| {
                p.expect(Token::KwAs)?;
                Ok(ValueDef(p.parse()?))
            })?;
            Ok(ListItemPattern::RestPattern(name))
        } else {
            Ok(ListItemPattern::Pattern(p.arena.alloc(p.parse()?)))
        }
    }
}

fn parse_pattern_atom<'a, 't>(p: &mut Parser<'a, 't>) -> Result<Pattern<'a>, Error> {
    match p.peek() {
        Some(Token::OpenCurly) => {
            p.token_index += 1;
            let items = p.parse_comma_sep_list()?;
            p.expect_masking_whitespace(Token::CloseCurly)?;
            Ok(Pattern::RecordDestructure(ListCurly(items.into_bump_slice())))
        }
        Some(Token::OpenSquare) => {
            p.token_index += 1;
            let items = p.parse_comma_sep_list()?;
            p.expect_masking_whitespace(Token::CloseSquare)?;
            Ok(Pattern::List(ListSquare(items.into_bump_slice())))
        }
        Some(Token::OpenParen) => {
            p.token_index += 1;
            let items = p.parse_comma_sep_list()?;
            p.expect_masking_whitespace(Token::CloseParen)?;

            if items.len() == 1 {
                Ok(items[0])
            } else {
                Ok(Pattern::Tuple(ListParen(items.into_bump_slice())))
            }
        }
        Some(Token::LowerIdent) => {
            let name = p.parse()?;
            Ok(Pattern::Identifier(ValueDef(name)))
        }
        Some(Token::UpperIdent) => {
            let name = p.parse()?;
            Ok(Pattern::Tag(TagName(name)))
        }
        Some(Token::OpaqueName) => {
            let name = TokenId(p.token_index as u32);
            p.token_index += 1;
            Ok(Pattern::OpaqueRef(OpaqueName(name)))
        }
        Some(Token::IntBase10 | Token::IntNonBase10) => {
            let int = IntLiteral(TokenId(p.token_index as u32));
            p.token_index += 1;
            Ok(Pattern::NumLiteral(int))
        }
        Some(Token::String) => {
            let string = StrLiteral(TokenId(p.token_index as u32));
            p.token_index += 1;
            Ok(Pattern::StrLiteral(string))
        }
        Some(Token::Underscore | Token::NamedUnderscore) => {
            let token = TokenId(p.token_index as u32);
            p.token_index += 1;
            Ok(Pattern::Underscore(Underscore(token)))
        }
        // _ => todo!("{:?}", p.peek()),
        _ => Err(p.error(ErrorKind::ExpectedPattern)),
    }
}

impl<'a> Parse<'a> for FieldPattern<'a> {
    const CONTEXT: Context = Context::FieldPattern;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let name = p.parse()?;

        if p.consume(Token::Colon).is_some() {
            let pat = p.parse()?;
            Ok(FieldPattern::RequiredField(FieldName(name), p.arena.alloc(pat)))
        } else if p.consume(Token::Question).is_some() {
            let pat = p.parse()?;
            Ok(FieldPattern::OptionalField(FieldName(name), p.arena.alloc(pat)))
        } else {
            Ok(FieldPattern::Identifier(ValueDef(name)))
        }
    }
}

impl<'a> Parse<'a> for Expr<'a> {
    const CONTEXT: Context = Context::Expr;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        parse_expr(p, Prec::Pizza)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Prec {
    Pizza,
    AndOr, // BinOp::And, BinOp::Or,
    Compare, // BinOp::Equals, BinOp::NotEquals, BinOp::LessThan, BinOp::GreaterThan, BinOp::LessThanOrEq, BinOp::GreaterThanOrEq,
    Add, // BinOp::Plus, BinOp::Minus,
    Multiply, // BinOp::Star, BinOp::Slash, BinOp::DoubleSlash, BinOp::Percent,
    Exponent, // BinOp::Caret
}

impl Prec {
    fn next(self) -> Prec {
        match self {
            Prec::Pizza => Prec::AndOr,
            Prec::AndOr => Prec::Compare,
            Prec::Compare => Prec::Add,
            Prec::Add => Prec::Multiply,
            Prec::Multiply => Prec::Exponent,
            Prec::Exponent => Prec::Exponent,
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
            BinOp::Caret => Prec::Exponent,
            BinOp::Star | BinOp::Slash | BinOp::DoubleSlash | BinOp::Percent => Prec::Multiply,
            BinOp::Plus | BinOp::Minus => Prec::Add,
            BinOp::Equals | BinOp::NotEquals | BinOp::LessThan | BinOp::GreaterThan | BinOp::LessThanOrEq | BinOp::GreaterThanOrEq => Prec::Compare,
            BinOp::And | BinOp::Or => Prec::AndOr,
            BinOp::Pizza => Prec::Pizza,
        }
    }

    fn assoc(self) -> Assoc {
        match self {
            BinOp::Caret => Assoc::Right,
            BinOp::Star | BinOp::Slash | BinOp::DoubleSlash | BinOp::Percent => Assoc::Left,
            BinOp::Plus | BinOp::Minus => Assoc::Left,
            BinOp::Equals | BinOp::NotEquals | BinOp::LessThan | BinOp::GreaterThan | BinOp::LessThanOrEq | BinOp::GreaterThanOrEq => Assoc::NonAssociative,
            BinOp::And | BinOp::Or => Assoc::Left,
            BinOp::Pizza => Assoc::Left,
        }
    }
}

fn parse_stmt_or_expr<'a, 't>(p: &mut Parser<'a, 't>) -> Result<Expr<'a>, Error> {
    if p.peek() == Some(Token::KwExpect) {
        p.token_index += 1;
        let value = parse_expr(p, Prec::Pizza)?;
        return Ok(Expr::Expect(p.arena.alloc(value)));
    }

    let first = parse_expr(p, Prec::Pizza)?;

    match p.peek() {
        Some(Token::Colon) => {
            p.token_index += 1;

            match first {
                Expr::Var { module_name, ident } => {
                    if module_name.name_count > 0 {
                        return Err(p.error(ErrorKind::ExpectedNoModuleName));
                    }

                    let ty = p.parse()?;

                    Ok(Expr::TypeAnnotation(ValueDef(ident.0), p.arena.alloc(ty)))
                },
                Expr::Tag(name) => {
                    let ty = p.parse()?;

                    Ok(Expr::TypeDef { header: TypeHeader { name: TypeName(name.0), vars: &[] }, typ: ty })
                },
                _ => todo!("{:?}", first),
            }
        }
        Some(Token::Assignment) => {
            p.token_index += 1;

            let pat = expr_to_pattern(p.arena, first)
                .map_err(|kind| p.error(kind))?; // TODO: use error_at and an appropriate location

            let value = p.parse()?;

            Ok(Expr::Assignment(p.arena.alloc(pat), p.arena.alloc(value)))
        }
        Some(Token::ColonEqual) => {
            p.token_index += 1;

            let pat = expr_to_pattern(p.arena, first)
                .map_err(|kind| p.error(kind))?; // TODO: use error_at and an appropriate location

            let typ = p.parse()?;
            let abilities = p.optional_if(Token::KwHas, |p| {
                p.expect(Token::KwHas)?;
                p.parse()
            })?;

            let header = match pat {
                Pattern::Apply(&Pattern::Tag(name), alias_args) => {
                    TypeHeader {
                        name: TypeName(name.0),
                        vars: alias_args,
                    }
                }
                Pattern::Tag(name) => {
                    TypeHeader {
                        name: TypeName(name.0),
                        vars: &[],
                    }
                }
                _ => todo!(),
            };

            Ok(Expr::OpaqueTypeDef { header, typ, abilities })
        }
        Some(Token::BackArrow) => {
            todo!();
        }
        Some(Token::KwHas) => {
            p.token_index += 1;

            let pat = expr_to_pattern(p.arena, first)
                .map_err(|kind| p.error(kind))?; // TODO: use error_at and an appropriate location

            let header = match pat {
                Pattern::Apply(&Pattern::Tag(name), alias_args) => {
                    TypeHeader {
                        name: TypeName(name.0),
                        vars: alias_args,
                    }
                }
                Pattern::Tag(name) => {
                    TypeHeader {
                        name: TypeName(name.0),
                        vars: &[],
                    }
                }
                _ => todo!(),
            };

            p.expect(Token::OpenIndent)?;

            let mut members = Vec::new_in(p.arena);

            loop {
                let name = ValueDef(p.parse()?);
                let typ = p.arena.alloc(p.parse()?);

                members.push(AbilityMember { name, typ });

                if p.consume(Token::Newline).is_none() {
                    break;
                }
            }

            p.expect(Token::CloseIndent)?;

            Ok(Expr::Ability {
                header,
                members: members.into_bump_slice(),
            })
        }
        None | Some(Token::Newline) | Some(Token::CloseIndent) => Ok(first),
        _ => todo!("{:?}", p.peek()),
    }
}

fn expr_to_pattern<'a>(arena: &'a Bump, expr: Expr<'a>) -> Result<Pattern<'a>, ErrorKind> {
    match expr {
        Expr::Float(_) => todo!(),
        Expr::Num(_) => todo!(),
        Expr::Str(_) => todo!(),
        Expr::SingleQuote(_) => todo!(),
        Expr::Access(_, _) => todo!(),
        Expr::AccessorFunction(_) => todo!(),
        Expr::List(_) => todo!(),
        Expr::RecordUpdate { update, fields } => todo!(),
        Expr::Record(_) => todo!(),
        Expr::Tuple(_) => todo!(),
        Expr::Var { module_name, ident } => {
            if module_name.name_count > 0 {
                return Err(ErrorKind::ExpectedNoModuleName);
            }
            Ok(Pattern::Identifier(ValueDef(ident.0)))
        }
        Expr::Underscore(_) => todo!(),
        Expr::Crash => todo!(),
        Expr::Tag(name) => Ok(Pattern::Tag(name)),
        Expr::OpaqueRef(_) => todo!(),
        Expr::Closure(_, _) => todo!(),
        Expr::Block(_) => todo!(),
        Expr::Dbg => todo!(),
        Expr::Apply(func, args) => {
            let func = expr_to_pattern(arena, *func)?;
            let mut pat_args = Vec::new_in(arena);
            for arg in args {
                pat_args.push(expr_to_pattern(arena, *arg)?);
            }
            Ok(Pattern::Apply(arena.alloc(func), pat_args.into_bump_slice()))
        }
        Expr::BinOp(_, _, _) => todo!(),
        Expr::UnaryOp(_, _) => todo!(),
        Expr::If(_, _) => todo!(),
        Expr::When(_, _) => todo!(),
        Expr::TypeAnnotation(_, _) => todo!(),
        Expr::Assignment(_, _) => todo!(),

        Expr::Expect(_) => todo!(),
        Expr::OpaqueTypeDef { header: _, typ: _, abilities: _ } => todo!(),
        Expr::TypeDef { header: _, typ: _} => todo!(),
        Expr::Ability { header, members } => todo!(),
        
    }
}

fn parse_expr<'a, 't>(p: &mut Parser<'a, 't>, min_prec: Prec) -> Result<Expr<'a>, Error> {
    let mut result = parse_expr_apply(p)?;

    loop {
        let t = p.peek();
        if is_expr_end(t) {
            break;
        }

        let op = if let Some(op) = t.and_then(|t| t.to_binop()) {
            op
        } else {
            todo!("{:?}", t);
        };

        let op_prec = op.prec();
        let assoc = op.assoc();

        // TODO: check that the associativity is right here!!!!!!
        if op_prec < min_prec || (op_prec == min_prec && assoc == Assoc::Right) {
            break;
        }

        p.token_index += 1;

        let next_min_prec = if assoc == Assoc::Left {
            op_prec
        } else {
            op_prec.next()
        };

        let second = parse_expr(p, next_min_prec)?;

        result = Expr::BinOp(p.arena.alloc(result), op, p.arena.alloc(second));
    }
    
    Ok(result)
}

fn parse_expr_atom<'a, 't>(p: &mut Parser<'a, 't>) -> Result<Expr<'a>, Error> {
    let mut expr = match p.peek() {
        Some(Token::LowerIdent) => {
            let name = p.expect(Token::LowerIdent)?;

            Expr::Var { module_name: ModuleName { start: name, name_count: 0}, ident: ValueAccess(LowercaseIdent(name)) }
        }
        Some(Token::KwDbg) => {
            p.token_index += 1;
            Expr::Dbg
        }
        Some(Token::UpperIdent) => {
            let initial = p.expect(Token::UpperIdent)?;
            let mut count = 1;
            let mut last = initial;
            while p.peek() == Some(Token::DotNoLeadingWhitespace) {
                p.token_index += 1;

                match p.peek() {
                    Some(Token::UpperIdent) => {
                        last = TokenId(p.token_index as u32);
                        p.token_index += 1;
                        count += 1;
                    }
                    Some(Token::LowerIdent) => {
                        last = TokenId(p.token_index as u32);
                        p.token_index += 1;
                        // This must be a module.field access; start parsing any further field accesses
                        return parse_field_accesses(p, Expr::Var { module_name: ModuleName { start: initial, name_count: count - 1}, ident: ValueAccess(LowercaseIdent(last)) });
                    }
                    _ => todo!("{:?}", p.peek()),
                }
            }

            if count == 1 {
                Expr::Tag(TagName(UppercaseIdent(initial)))
            } else {
                let module = ModuleName {
                    start: initial,
                    name_count: count - 1,
                };

                let ty = TypeName(UppercaseIdent(last));

                // QualifiedTypeName {
                //     module,
                //     ty
                // }
                todo!();
            }
        }
        Some(Token::OpaqueName) => {
            let name = p.expect(Token::OpaqueName)?;

            Expr::OpaqueRef(OpaqueName(name))
        }
        Some(Token::IntBase10) => {
            let num = p.expect(Token::IntBase10)?;

            Expr::Num(IntLiteral(num))
        }
        Some(Token::IntNonBase10) => {
            let num = p.expect(Token::IntNonBase10)?;

            Expr::Num(IntLiteral(num))
        }
        Some(Token::String) => {
            let s = p.expect(Token::String)?;

            Expr::Str(StrLiteral(s))
        }
        Some(Token::SingleQuote) => {
            let s = p.expect(Token::SingleQuote)?;

            Expr::SingleQuote(SingleQuoteLiteral(s))
        }
        Some(Token::OpenParen) => {
            p.expect(Token::OpenParen)?;
            let items = p.parse_comma_sep_list()?;
            p.expect_masking_whitespace(Token::CloseParen)?;

            if items.len() == 1 {
                items[0]
            } else {
                Expr::Tuple(ListParen(items.into_bump_slice()))
            }
        }
        Some(Token::OpenSquare) => {
            p.expect(Token::OpenSquare)?;
            let items = p.parse_comma_sep_list()?;
            p.expect_masking_whitespace(Token::CloseSquare)?;

            Expr::List(ListSquare(items.into_bump_slice()))
        }
        Some(Token::OpenCurly) => {
            p.expect(Token::OpenCurly)?;

            if p.consume(Token::CloseCurly).is_some() || p.consume(Token::CloseCurlyNoTrailingWhitespace).is_some() {
                Expr::Record(ListCurly(&[]))
            } else {
                let first = p.parse()?;
                let mut items = Vec::new_in(p.arena);

                if p.consume(Token::Ampersand).is_some() {
                    // This is a record update
                } else {
                    if p.consume(Token::Comma).is_some() {
                        items.push(AssignedField { name: expr_to_field_name(first)?, value: first });
                    } else {
                        p.expect(Token::Colon)?;
                        let expr = p.parse()?;
                        items.push(AssignedField { name: expr_to_field_name(first)?, value: expr });
                    }
                }
                let items = p.append_comma_sep_list(items)?;
                p.expect_masking_whitespace(Token::CloseCurly)?;

                Expr::Record(ListCurly(items.into_bump_slice()))
            }
        }
        Some(Token::DotLeadingWhitespace) => Expr::AccessorFunction(p.parse()?),
        Some(Token::KwWhen) => parse_expr_when(p)?,
        Some(Token::KwIf) => parse_expr_if(p)?,
        Some(Token::Backslash) => parse_expr_closure(p)?,
        Some(Token::Underscore | Token::NamedUnderscore) => {
            let token = TokenId(p.token_index as u32);
            p.token_index += 1;
            Expr::Underscore(Underscore(token))
        }
        Some(Token::OpenIndent) => {
            p.expect(Token::OpenIndent)?;
            let block = p.parse()?;
            p.expect(Token::CloseIndent)?;
            Expr::Block(block)
        }
        Some(Token::MinusNoTrailingWhitespace) => {
            p.token_index += 1;
            let expr = parse_expr_atom(p)?;
            Expr::UnaryOp(UnaryOp::Negate, p.arena.alloc(expr))
        }
        Some(Token::Bang) => {
            p.token_index += 1;
            let expr = parse_expr_atom(p)?;
            Expr::UnaryOp(UnaryOp::Not, p.arena.alloc(expr))
        }
        // _ => todo!("{:?}", p.peek()),
        _ => return Err(p.error(ErrorKind::ExpectedExpr)),
    };

    while p.peek() == Some(Token::DotNoLeadingWhitespace) {
        let access = p.parse()?;
        expr = Expr::Access(p.arena.alloc(expr), access);
    }

    Ok(expr)
}

fn parse_expr_apply<'a, 't>(p: &mut Parser<'a, 't>) -> Result<Expr<'a>, Error> {
    let expr = parse_expr_atom(p)?;

    if is_expr_atom_end(p.peek()) {
        return Ok(expr);
    }

    // this is a function application
    let mut args = Vec::new_in(p.arena);
    args.push(p.parse()?);

    while !is_expr_atom_end(p.peek()) {
        args.push(p.parse()?);
    }

    Ok(Expr::Apply(p.arena.alloc(expr), args.into_bump_slice()))
}

fn is_expr_atom_end(token: Option<Token>) -> bool {
    if is_expr_end(token) {
        return true;
    }

    match token {
        Some(t) if t.to_binop().is_some() => true,
        _ => false,
    }
}

fn is_expr_end(token: Option<Token>) -> bool {
    match token {
        None => true,
        Some(t) if t.is_terminator() => true,
        Some(Token::Colon | Token::ColonEqual | Token::Assignment | Token::Newline | Token::KwIs | Token::KwThen | Token::KwElse) => true,
        Some(Token::Comma | Token::Ampersand | Token::BackArrow | Token::KwHas) => true,
        _ => false,
    }
}

fn parse_field_accesses<'a, 't>(p: &mut Parser<'a, 't>, mut expr: Expr<'a>) -> Result<Expr<'a>, Error> {
    while p.peek() == Some(Token::DotNoLeadingWhitespace) {
        p.token_index += 1;

        match p.peek() {
            Some(Token::LowerIdent) => {
                let last = TokenId(p.token_index as u32);
                p.token_index += 1;
                expr = Expr::Access(p.arena.alloc(expr), Accessor::RecordField(FieldName(LowercaseIdent(last))));
            }
            Some(Token::IntBase10) => {
                let last = TokenId(p.token_index as u32);
                p.token_index += 1;
                expr = Expr::Access(p.arena.alloc(expr), Accessor::TupleIndex(TupleIndex(IntLiteral(last))));
            }
            _ => todo!("{:?}", p.peek()),
        }
    }

    Ok(expr)
}

fn expr_to_field_name(expr: Expr) -> Result<FieldName, Error> {
    match expr {
        Expr::Var { module_name, ident } => {
            if module_name.name_count == 0 {
                match ident {
                    ValueAccess(name) => Ok(FieldName(name)),
                    _ => todo!(),
                }
            } else {
                todo!()
            }
        }
        _ => todo!(),
    }
}

fn parse_expr_when<'a, 't>(p: &mut Parser<'a, 't>) -> Result<Expr<'a>, Error> {
    p.expect(Token::KwWhen)?;
    let cond = p.parse()?;
    p.expect(Token::KwIs)?;
    p.expect(Token::OpenIndent)?;
    let mut branches = Vec::new_in(p.arena);
    while !p.peek_terminator() {
        let mut patterns = Vec::new_in(p.arena);
        patterns.push(p.parse()?);
        while p.consume(Token::Bar).is_some() {
            patterns.push(p.parse()?);
        }
        let guard = p.optional_if(Token::KwIf, |p| p.parse())?;
        p.expect(Token::ForwardArrow)?;
        let body = p.parse()?;
        p.consume(Token::Newline);
        branches.push(WhenBranch { patterns: patterns.into_bump_slice(), body, guard });
    }
    p.expect(Token::CloseIndent)?;

    Ok(Expr::When(p.arena.alloc(cond), branches.into_bump_slice()))
}

fn parse_expr_closure<'a, 't>(p: &mut Parser<'a, 't>) -> Result<Expr<'a>, Error> {
    p.expect(Token::Backslash)?;

    let mut args = Vec::new_in(p.arena);
    args.push(p.parse()?);
    while p.consume(Token::Comma).is_some() {
        args.push(p.parse()?);
    }
    p.expect(Token::ForwardArrow)?;

    let body = p.parse()?;

    Ok(Expr::Closure(args.into_bump_slice(), p.arena.alloc(body)))
}

fn parse_expr_if<'a, 't>(p: &mut Parser<'a, 't>) -> Result<Expr<'a>, Error> {
    let mut branches = Vec::new_in(p.arena);
    loop {
        p.expect(Token::KwIf)?;
        let cond = p.parse()?;
        p.expect(Token::KwThen)?;
        let then = p.parse()?;
        branches.push((cond, then));
        p.expect(Token::KwElse)?;

        if p.peek() == Some(Token::KwIf) {
            continue;
        } else {
            let else_ = p.parse()?;
            return Ok(Expr::If(branches.into_bump_slice(), p.arena.alloc(else_)));
        };
    }
}

impl<'a> Parse<'a> for TypeAnnotation<'a> {
    const CONTEXT: Context = Context::TypeAnnotation;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let mut items = Vec::new_in(p.arena);
        Self::parse_seq_item(p, &mut items)?;
        if items.len() == 1 {
            Ok(items.pop().unwrap())
        } else {
            Err(p.error(ErrorKind::ExpectedSingleTypeAnnotation))
        }
    }

    fn parse_seq_item<'t>(p: &mut Parser<'a, 't>, seq: &mut Vec<'a, Self>) -> Result<(), Error> {
        let start_index = seq.len();

        loop {
            let part = parse_type_annotation_part(p)?;

            seq.push(part);

            match p.peek() {
                Some(Token::Bar) => {
                    p.token_index += 1;

                    let mut items = Vec::new_in(p.arena);

                    let name = p.expect(Token::LowerIdent)?;
                    p.expect(Token::KwHas)?;
                    let abilities = parse_ability_list(p);

                    items.push((name, abilities));

                    while p.consume(Token::Comma).is_some() {
                        let name = p.expect(Token::LowerIdent)?;
                        p.expect(Token::KwHas)?;
                        let abilities = parse_ability_list(p);

                        items.push((name, abilities));
                    }
                    break;
                }
                Some(Token::ForwardArrow) => {
                    p.token_index += 1;
                    let res = p.parse()?;
                    let item = TypeAnnotation::Function(
                        p.arena.alloc_slice_copy(&seq[start_index..]), // TODO: if start_index == 0, we can just .into_bump_slice() instead
                        p.arena.alloc(res),
                    );
                    seq.truncate(start_index);
                    seq.push(item);
                    break;
                }
                Some(Token::Comma) => {
                    p.token_index += 1;
                    // continue; accumulate more items / args
                }
                Some(Token::CloseCurly | Token::CloseSquare | Token::CloseParen | Token::CloseIndent |
                    Token::CloseCurlyNoTrailingWhitespace | Token::CloseSquareNoTrailingWhitespace |
                    Token::CloseParenNoTrailingWhitespace | Token::Newline | Token::KwHas) => {
                        // panic!();
                    break;
                }
                _ => {
                    return Err(p.error(ErrorKind::ExpectedTypeAnnotationEnd));
                }
            }
        }

        Ok(())
    }
}

fn parse_ability_list<'a, 't>(p: &mut Parser<'a, 't>) -> Result<&'a [AbilityName], Error> {
    let mut abilities = Vec::new_in(p.arena);
    abilities.push(AbilityName(p.parse()?));
    while p.consume(Token::Ampersand).is_some() {
        let ability = p.parse()?;
        abilities.push(AbilityName(ability));
    }
    Ok(abilities.into_bump_slice())
}

fn is_type_annotation_seq_term(token: Option<Token>) -> bool {
    match token {
        None | Some(Token::ForwardArrow | Token::Comma | Token::CloseCurly | Token::CloseSquare | Token::CloseParen | Token::CloseIndent |
            Token::CloseCurlyNoTrailingWhitespace | Token::CloseSquareNoTrailingWhitespace |
            Token::CloseParenNoTrailingWhitespace) => true,
        _ => false
    }
}

fn parse_type_annotation_part<'a, 't>(p: &mut Parser<'a, 't>) -> Result<TypeAnnotation<'a>, Error> {
    let res = match p.peek() {
        Some(Token::OpenParen) => {
            p.expect(Token::OpenParen)?;
            let items = p.parse_comma_sep_list()?;
            let has_ext = p.peek() == Some(Token::CloseParenNoTrailingWhitespace);
            p.expect_masking_whitespace(Token::CloseParen)?;

            let elems = ListParen(items.into_bump_slice());

            let ext = if has_ext {
                Some(&*p.arena.alloc(p.parse()?))
            } else {
                None
            };

            if elems.0.len() == 1 {
                if ext.is_some() {
                    return Err(p.error(ErrorKind::ExtOnParens));
                }
                elems.0[0]
            } else {
                TypeAnnotation::Tuple { elems, ext}
            }
        }
        Some(Token::OpenCurly) => {
            p.expect(Token::OpenCurly)?;
            let items = p.parse_comma_sep_list()?;
            let has_ext = p.peek() == Some(Token::CloseCurlyNoTrailingWhitespace);
            p.expect_masking_whitespace(Token::CloseCurly)?;

            let fields = ListCurly(items.into_bump_slice());

            let ext = if has_ext {
                Some(&*p.arena.alloc(p.parse()?))
            } else {
                None
            };

            TypeAnnotation::Record { fields, ext}
        }
        Some(Token::OpenSquare) => {
            p.expect(Token::OpenSquare)?;
            let items = p.parse_comma_sep_list()?;
            let has_ext = p.peek() == Some(Token::CloseSquareNoTrailingWhitespace);
            p.expect_masking_whitespace(Token::CloseSquare)?;

            let mut tags = Vec::with_capacity_in(items.len(), p.arena);

            for item in items {
                match item {
                    TypeAnnotation::Apply(tag, args) => {
                        if tag.module.name_count > 0 {
                            return Err(p.error_at(ErrorKind::ExpectedTagApply, tag.module.start));
                        }
                        tags.push(Tag { name: TagName(tag.ty.0), args })
                    }
                    _ => {
                        // TODO: put the error at the right location
                        return Err(p.error(ErrorKind::ExpectedTagApply));
                    }
                }
            }

            let tags = ListSquare(tags.into_bump_slice());

            let ext = if has_ext {
                Some(&*p.arena.alloc(p.parse()?))
            } else {
                None
            };

            TypeAnnotation::TagUnion{ tags, ext}
        }
        Some(Token::Star) => {
            p.token_index += 1;
            TypeAnnotation::Wildcard
        }
        Some(Token::Underscore) => {
            p.token_index += 1;
            TypeAnnotation::Inferred
        }
        Some(Token::UpperIdent) => {
            let ty: QualifiedTypeName = p.parse()?;

            let mut args = Vec::new_in(p.arena);

            // parse type arguments immediately after (no parens, no delimeters)
            loop {
                match p.peek() {
                    Some(Token::OpenParen | Token::OpenCurly | Token::OpenSquare | Token::Star | Token::Underscore | Token::UpperIdent | Token::LowerIdent) => {
                        args.push(parse_type_annotation_part(p)?);
                    }
                    _ => break,
                }
            }

            TypeAnnotation::Apply(ty, args.into_bump_slice())
        }
        Some(Token::LowerIdent) => {
            let name = p.parse()?;
            TypeAnnotation::BoundVariable(TypeVariableName(name))
        }
        _ => todo!("{:?}", p.peek()),
    };

    Ok(res)
}

impl<'a, T: Parse<'a>> Parse<'a> for AssignedField<T> {
    const CONTEXT: Context = Context::AssignedField;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let name = p.parse()?;
        p.expect(Token::Colon)?;
        let value = p.parse()?;
        Ok(AssignedField { name, value })
    }
}

pub fn parse<'a, P: Parse<'a>>(tokens: &[Token], token_offsets: &[u32], arena: &'a Bump) -> Result<P, Error> {
    let mut p = Parser {
        arena,
        tokens,
        // token_offsets,
        token_index: 0,
        context: vec![],
    };
    let result = p.parse_all()?;
    if p.token_index != tokens.len() {
        return Err(p.error(ErrorKind::ExpectedEndOfInput));
    }
    Ok(result)
}

pub fn debug_parse_error(tokens: &[Token], token_offsets: &[u32], text: &str, error: Error) -> String {
    let mut s = String::new();
    let token_offset = if error.token_index as usize == token_offsets.len() {
        text.len()
    } else {
        token_offsets[error.token_index as usize] as usize
    };

    let mut highlight_next_line_offset = None;
    let mut last_line_start = 0;

    for (i, ch) in text.char_indices() {
        if i == token_offset {
            highlight_next_line_offset = Some(i - last_line_start);
        }
        s.push(ch);
        if ch == '\n' {
            maybe_highlight_error(&mut highlight_next_line_offset, &mut s, &error, tokens);
            last_line_start = i + 1;
        }
    }
    if !s.ends_with('\n') {
        s.push('\n');
    }
    maybe_highlight_error(&mut highlight_next_line_offset, &mut s, &error, tokens);

    if token_offset == text.len() {
        //blue
        s.push_str("\x1b[34m");
        s.push_str("<eof>\n");
        // reset color
        s.push_str("\x1b[0m");
        highlight_next_line_offset = Some(0);
        maybe_highlight_error(&mut highlight_next_line_offset, &mut s, &error, tokens);
    }

    s.push_str("Tokens:\n");
    for (i, token) in tokens.iter().enumerate() {
        s.push_str(&format!("{:?} ", token));
        if i == error.token_index as usize {
            // color red
            s.push_str("\x1b[31m");
            s.push_str("<<<<<\n");
            // reset color
            s.push_str("\x1b[0m");
        }
    }

    s
}

fn maybe_highlight_error(highlight_next_line_offset: &mut Option<usize>, s: &mut String, e: &Error, tokens: &[Token]) {
    if let Some(offset) = highlight_next_line_offset.take() {
        // color red
        s.push_str("\x1b[31m");
        s.push_str(&" ".repeat(offset));
        s.push_str("^ ");
        s.push_str(&format!("{:?}\n", e));
        s.push_str(&" ".repeat(offset));
        s.push_str(&format!("Tokens here: {:?}...\n", &tokens[e.token_index as usize..std::cmp::min(e.token_index as usize + 3, tokens.len())]));
        // reset color
        s.push_str("\x1b[0m");
    }
}

pub struct Parser<'a, 't> {
    arena: &'a Bump,
    tokens: &'t [Token],
    // token_offsets: &'a [u32],
    token_index: usize,
    context: std::vec::Vec<(Context, TokenId)>,
}

impl<'a, 't> Parser<'a, 't> {
    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.token_index).copied()
    }

    fn peek_terminator(&self) -> bool {
        self.peek().map(|t| t.is_terminator()).unwrap_or_default()
    }

    fn peek_ahead(&self, index: usize) -> Option<Token> {
        self.tokens.get(self.token_index + index).copied()
    }


    fn expect(&mut self, token: Token) -> Result<TokenId, Error> {
        if self.peek() != Some(token) {
            return Err(self.error(ErrorKind::Expected(token)));
        }
        let id = TokenId(self.token_index as u32);
        self.token_index += 1;
        Ok(id)
    }

    fn ws(&mut self) -> Result<(), Error> {
        fn is_whitespace(token: Option<Token>) -> bool {
            match token {
                Some(Token::Newline) | Some(Token::OpenIndent) | Some(Token::CloseIndent) => true,
                _ => false,
            }
        }

        while is_whitespace(self.peek()) {
            self.token_index += 1;
        }
        Ok(())
    }

    fn optional_if<T>(&mut self, token: Token, cond: impl FnOnce(&mut Parser<'a, 't>) -> Result<T, Error>) -> Result<Option<T>, Error> {
        if self.peek() != Some(token) {
            return Ok(None);
        }
        cond(self).map(Some)
    }

    fn expect_masking_whitespace(&mut self, token: Token) -> Result<TokenId, Error> {
        if self.peek().map(|t| t.mask_close_group_whitespace()) != Some(token) {
            return Err(self.error(ErrorKind::Expected(token)));
        }
        let id = TokenId(self.token_index as u32);
        self.token_index += 1;
        Ok(id)
    }

    fn error(&mut self, kind: ErrorKind) -> Error {
        Error {
            kind,
            token_index: self.token_index as u32,
            context: self.context.clone(),
        }
    }

    fn error_at(&mut self, kind: ErrorKind, token_index: TokenId) -> Error {
        Error {
            kind,
            token_index: token_index.0,
            context: self.context.clone(),
        }
    }

    fn consume(&mut self, token: Token) -> Option<TokenId> {
        if self.peek() == Some(token) {
            let id = TokenId(self.token_index as u32);
            self.token_index += 1;
            Some(id)
        } else {
            None
        }
    }

    fn parse<P: Parse<'a>>(&mut self) -> Result<P, Error> {
        let index = TokenId(self.token_index as u32);
        self.context.push((P::CONTEXT, index));
        let res = P::parse(self);
        let ctx = self.context.pop().unwrap();
        debug_assert!(ctx.0 == P::CONTEXT);
        res
    }

    fn parse_seq_item<P: Parse<'a>>(&mut self, seq: &mut Vec<'a, P>) -> Result<(), Error> {
        let index = TokenId(self.token_index as u32);
        self.context.push((P::CONTEXT, index));
        let res = P::parse_seq_item(self, seq);
        let ctx = self.context.pop().unwrap();
        debug_assert!(ctx.0 == P::CONTEXT);
        res
    }

    fn parse_all<P: Parse<'a>>(&mut self) -> Result<P, Error> {
        let res = P::parse(self)?;
        if self.token_index != self.tokens.len() {
            println!("token_index: {}, tokens.len(): {}", self.token_index, self.tokens.len());
            return Err(self.error(ErrorKind::ExpectedEndOfInput));
        }
        Ok(res)
    }

    fn parse_comma_sep_list<P: Parse<'a>>(&mut self) -> Result<Vec<'a, P>, Error> {
        let items = Vec::new_in(self.arena);
        self.append_comma_sep_list(items)
    }

    fn append_comma_sep_list<P: Parse<'a>>(&mut self, mut items: Vec<'a, P>) -> Result<Vec<'a, P>, Error> {
        loop {
            if self.peek_terminator() {
                break;
            }
            if items.len() > 0 {
                self.expect(Token::Comma)?;
            }
            if self.peek_terminator() {
                break;
            }
            self.parse_seq_item(&mut items)?;
        }
        Ok(items)
    }

    fn parse_block_inner(&mut self) -> Block<'a> {
        panic!("TODO")
    }
}