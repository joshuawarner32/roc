use bumpalo::{collections::Vec, Bump};

use crate::token::Token;

pub struct TokenId(u32);

pub struct ExposedName(pub TokenId);
pub struct StrLiteral(pub TokenId);
pub struct PackageName(pub StrLiteral);
pub struct PackageShorthand(pub LowercaseIdent);
pub struct UppercaseIdent(pub TokenId);
pub struct LowercaseIdent(pub TokenId);
pub struct FieldName(pub LowercaseIdent);
pub struct ModuleName(pub UppercaseIdent);
pub struct TypeName(pub UppercaseIdent);
pub struct TagName(pub UppercaseIdent);

pub struct Root<'a> {
    header: Header<'a>,
    defs: Block<'a>,
}

pub enum Header<'a> {
    Interface(InterfaceHeader<'a>),
    App(AppHeader<'a>),
    Package(PackageHeader<'a>),
    Platform(PlatformHeader<'a>),
    Hosted(HostedHeader<'a>),
}

pub struct InterfaceHeader<'a> {
    name: ModuleName,
    exports: Vec<'a, ExposedName>,
    imports: Vec<'a, ImportsEntry<'a>>,
}

pub struct AppHeader<'a> {
    name: StrLiteral,
    packages: Option<Vec<'a, PackageEntry>>,
    imports: Option<Vec<'a, ImportsEntry<'a>>>,
}

pub struct PackageHeader<'a> {
    name: PackageName,
    exposes: Vec<'a, ModuleName>,
    packages: Vec<'a, PackageEntry>,
}

pub struct PlatformHeader<'a> {
    name: PackageName,
    requires: PlatformRequires<'a>,
    exposes: Vec<'a, ModuleName>,
    packages: Vec<'a, PackageEntry>,
    imports: Vec<'a, ImportsEntry<'a>>,
    provides: Vec<'a, ExposedName>,
}

pub struct HostedHeader<'a> {
    name: ModuleName,
    exposes: Vec<'a, ExposedName>,
    imports: Vec<'a, ImportsEntry<'a>>,
    generates: UppercaseIdent,
    generates_with: Vec<'a, ExposedName>,
}

pub enum ImportsEntry<'a> {
    Module(ModuleName, Vec<'a, ExposedName>),
    Package(PackageName, ModuleName, Vec<'a, ExposedName>),
}

pub struct PackageEntry {
    shorthand: PackageShorthand,
    name: PackageName,
}

pub struct PlatformRequires<'a> {
    rigids: Vec<'a, UppercaseIdent>,
    signature: Vec<'a, TypedIdent<'a>>,
}

pub struct TypedIdent<'a> {
    ident: LowercaseIdent,
    type_annotation: TypeAnnotation<'a>,
}

pub enum TypeAnnotation<'a> {
    /// A function. The types of its arguments, then the type of its return value.
    Function(&'a [TypeAnnotation<'a>], &'a TypeAnnotation<'a>),

    /// Applying a type to some arguments (e.g. Map.Map String Int)
    Apply(&'a str, &'a str, &'a [TypeAnnotation<'a>]),

    /// A bound type variable, e.g. `a` in `(a -> a)`
    BoundVariable(&'a str),

    /// Inline type alias, e.g. `as List a` in `[Cons a (List a), Nil] as List a`
    As(
        &'a TypeAnnotation<'a>,
        TypeHeader<'a>,
    ),

    Record {
        fields: &'a [AssignedField<TypeAnnotation<'a>>],
        /// The row type variable in an open record, e.g. the `r` in `{ name: Str }r`.
        /// This is None if it's a closed record annotation like `{ name: Str }`.
        ext: Option<&'a TypeAnnotation<'a>>,
    },

    Tuple {
        elems: &'a [TypeAnnotation<'a>],
        /// The row type variable in an open tuple, e.g. the `r` in `( Str, Str )r`.
        /// This is None if it's a closed tuple annotation like `( Str, Str )`.
        ext: Option<&'a TypeAnnotation<'a>>,
    },

    /// A tag union, e.g. `[
    TagUnion {
        /// The row type variable in an open tag union, e.g. the `a` in `[Foo, Bar]a`.
        /// This is None if it's a closed tag union like `[Foo, Bar]`.
        ext: Option<&'a TypeAnnotation<'a>>,
        tags: &'a [Tag<'a>],
    },

    /// '_', indicating the compiler should infer the type
    Inferred,

    /// The `*` type variable, e.g. in (List *)
    Wildcard,

    /// A "where" clause demanding abilities designated by a `|`, e.g. `a -> U64 | a has Hash`
    Where(&'a TypeAnnotation<'a>, &'a [HasClause<'a>]),

}

/// Should always be a zero-argument `Apply`; we'll check this in canonicalization
pub type AbilityName<'a> = TypeAnnotation<'a>;

pub struct Tag<'a> {
    name: TypeName,
    args: &'a [TypeAnnotation<'a>],
}

pub struct HasClause<'a> {
    var: LowercaseIdent,
    abilities: &'a [AbilityName<'a>],
}

pub struct TypeHeader<'a> {
    name: TypeName,
    vars: &'a [Pattern<'a>],
}

pub struct AssignedField<T> {
    name: FieldName,
    value: T,
}

pub struct Block<'a> {
    items: Vec<'a, StmtOrExpr<'a>>,
}

pub enum StmtOrExpr<'a> {
    Stmt(Stmt),
    Expr(Expr<'a>),
}


pub enum Stmt {}
pub enum Expr<'a> {
    BinOp(BinOp, &'a Expr<'a>, &'a Expr<'a>),
}

pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    And,
    Or,
}

pub enum Pattern<'a> {
    Var(LowercaseIdent),
    Tag(TagName),
    Apply(&'a Pattern<'a>, &'a [Pattern<'a>]),
}






pub struct Parser<'a, 't> {
    arena: &'a Bump,
    tokens: &'t [Token],
    token_offsets: &'a [u32],
    token_index: usize,
}

impl<'a, 't> Parser<'a, 't> {
    fn peek(&self) -> Option<Token> {
        self.tokens.get(self.token_index).copied()
    }

    fn parse_root(&mut self) -> Root<'a> {
        let header = self.parse_header();
        let defs = self.parse_block_inner();
        self.ensure_end();
        Root { header, defs }
    }

    fn parse_header(&mut self) -> Header<'a> {
        match self.peek() {
            Some(Token::KwInterface) => self.parse_interface_header(),
            Some(Token::KwApp) => self.parse_app_header(),
            Some(Token::KwPackage) => self.parse_package_header(),
            Some(Token::KwPlatform) => self.parse_platform_header(),
            Some(Token::KwHosted) => self.parse_hosted_header(),
            _ => panic!("expected header"),
        }
    }
}