use bumpalo::{collections::Vec, Bump};

use crate::{token::Token};

#[derive(Debug, Clone, Copy)]
pub struct TokenId(u32);

#[derive(Debug, Clone, Copy)]
pub struct ExposedName(pub TokenId);
#[derive(Debug, Clone, Copy)]
pub struct StrLiteral(pub TokenId);
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

/// Should always be a zero-argument `Apply`; we'll check this in canonicalization
pub type AbilityName<'a> = TypeAnnotation<'a>;

#[derive(Debug, Clone, Copy)]
pub struct Tag<'a> {
    pub name: TagName,
    pub args: &'a [TypeAnnotation<'a>],
}

#[derive(Debug, Clone, Copy)]
pub struct HasClause<'a> {
    pub var: LowercaseIdent,
    pub abilities: &'a [AbilityName<'a>],
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
    pub items: &'a [StmtOrExpr<'a>],
}

#[derive(Debug, Clone, Copy)]
pub enum StmtOrExpr<'a> {
    Stmt(Stmt),
    Expr(Expr<'a>),
}


#[derive(Debug, Clone, Copy)]
pub enum Stmt {}

#[derive(Debug, Clone, Copy)]
pub enum Expr<'a> {
    BinOp(BinOp, &'a Expr<'a>, &'a Expr<'a>),
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
pub enum Pattern<'a> {
    Var(LowercaseIdent),
    Tag(TagName),
    Apply(&'a Pattern<'a>, &'a [Pattern<'a>]),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Context {
    Root,
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
        let name = dbg!(p.parse()?);
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
        if (p.peek() == Some(Token::DotNoLeadingWhitespace) || p.peek() == Some(Token::DotLeadingWhitespace)) && p.peek_ahead(1) == Some(Token::UpperIdent) {
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

// impl<'a> Parse<'a> for Tag<'a> {
//     const CONTEXT: Context = Context::Tag;
//     fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
//         let name = p.parse()?;
//         let mut args = Vec::new_in(p.arena);
//         while !p.peek_terminator() {}
//     }
// }

impl<'a> Parse<'a> for TypeAnnotation<'a> {
    const CONTEXT: Context = Context::TypeAnnotation;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let mut items = Vec::new_in(p.arena);
        Self::parse_seq_item(p, &mut items)?;
        if items.len() == 1 {
            Ok(items.pop().unwrap())
        } else {
            dbg!(items);
            Err(p.error(ErrorKind::ExpectedSingleTypeAnnotation))
        }
    }

    fn parse_seq_item<'t>(p: &mut Parser<'a, 't>, seq: &mut Vec<'a, Self>) -> Result<(), Error> {
        let start_index = seq.len();

        loop {
            let part = parse_type_annotation_part(p)?;

            seq.push(part);

            match p.peek() {
                Some(Token::ForwardArrow) => {
                    p.token_index += 1;
                    let res = p.parse()?;
                    let item = TypeAnnotation::Function(
                        p.arena.alloc_slice_copy(&seq[start_index..]), // TODO: if start_index == 0, we can just .into_bump_slice() instead
                        p.arena.alloc(res),
                    );
                    seq.truncate(start_index);
                    seq.push(item);
                }
                Some(Token::Comma) => {
                    p.token_index += 1;
                    // continue; accumulate more items / args
                }
                Some(Token::CloseCurly | Token::CloseSquare | Token::CloseParen | Token::CloseIndent |
                    Token::CloseCurlyNoTrailingWhitespace | Token::CloseSquareNoTrailingWhitespace |
                    Token::CloseParenNoTrailingWhitespace) => {
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

fn is_type_annotation_seq_term(token: Option<Token>) -> bool {
    match token {
        None | Some(Token::ForwardArrow | Token::Comma | Token::CloseCurly | Token::CloseSquare | Token::CloseParen | Token::CloseIndent |
            Token::CloseCurlyNoTrailingWhitespace | Token::CloseSquareNoTrailingWhitespace |
            Token::CloseParenNoTrailingWhitespace) => true,
        _ => false
    }
}

fn parse_type_annotation_part<'a, 't>(p: &mut Parser<'a, 't>) -> Result<TypeAnnotation<'a>, Error> {
    Ok(match p.peek() {
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
                    Some(Token::OpenParen | Token::OpenCurly | Token::OpenSquare | Token::Star | Token::Underscore | Token::UpperIdent) => {
                        args.push(parse_type_annotation_part(p)?);
                    }
                    _ => break,
                }
            }

            TypeAnnotation::Apply(ty, args.into_bump_slice())
        }
        _ => todo!("{:?}", p.peek()),
    })
}

impl<'a, T: Parse<'a>> Parse<'a> for AssignedField<T> {
    const CONTEXT: Context = Context::AssignedField;
    fn parse<'t>(p: &mut Parser<'a, 't>) -> Result<Self, Error> {
        let name = p.parse()?;
        p.expect(Token::Equals)?;
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
    let token_offset = token_offsets[error.token_index as usize] as usize;

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
        self.peek().map(|t| match t {
            Token::CloseCurly | Token::CloseSquare | Token::CloseParen | Token::CloseIndent |
            Token::CloseCurlyNoTrailingWhitespace | Token::CloseSquareNoTrailingWhitespace |
            Token::CloseParenNoTrailingWhitespace => true,
            _ => false,
        }).unwrap_or_default()
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
        let mut items = Vec::new_in(self.arena);
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