use std::fmt::Debug;

use crate::parse::BinOp;
use crate::parse::ParsedCtx;
use crate::tree::TreeWalker;
use crate::tree::{TreeStack, N};
use bumpalo::collections::vec::Vec;
use bumpalo::Bump;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expr<'a> {
    Crash,
    Ident(&'a str),
    OpaqueName(&'a str),
    Underscore(&'a str),
    UpperIdent(&'a str),
    IntBase10(&'a str),
    Float(&'a str),
    String(&'a str),
    DotNumber(&'a str),
    TupleAccessFunction(&'a str),
    FieldAccessFunction(&'a str),
    Apply(&'a Expr<'a>, &'a [Expr<'a>]),
    BinOp(&'a Expr<'a>, BinOp, &'a Expr<'a>),
    UnaryOp(UnaryOp, &'a Expr<'a>),
    Pizza(&'a [Expr<'a>]),
    Lambda(&'a [Expr<'a>], &'a Expr<'a>),
    If(&'a [(&'a Expr<'a>, &'a Expr<'a>)], &'a Expr<'a>),
    When(&'a Expr<'a>, &'a [Expr<'a>]),
    Block(&'a [Expr<'a>]),
    Record(&'a [Expr<'a>]),
    RecordAccess(&'a Expr<'a>, &'a str),
    TupleAccess(&'a Expr<'a>, &'a str),
    ModuleLowerName(&'a str, &'a str),
    ModuleUpperName(&'a str, &'a str),
    Tuple(&'a [Expr<'a>]),
    List(&'a [Expr<'a>]),

    // Not really expressions, but considering them as such to make the formatter as error tolerant as possible
    Assign(&'a Expr<'a>, &'a Expr<'a>),
    Backpassing(&'a [Expr<'a>], &'a Expr<'a>),
    Comment(&'a str),
    TypeAlias(&'a Expr<'a>, &'a Type<'a>),
    TypeAliasOpaque(&'a Expr<'a>, &'a Type<'a>),
    AbilityName(&'a str),
    Dbg(&'a Expr<'a>),
    Expect(&'a Expr<'a>),
    ExpectFx(&'a Expr<'a>),
    Ability(&'a str, &'a [(&'a str, &'a Type<'a>)]),
    RecordFieldPair(&'a str, &'a Expr<'a>),

    PatternList(&'a [Expr<'a>]),
    PatternRecord(&'a [(&'a str, &'a Expr<'a>)]),
    PatternAs(&'a Expr<'a>, &'a str),
    PatternAny,
    PatternDoubleDot,
    RecordUpdate(&'a [Expr<'a>]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Minus,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type<'a> {
    Wildcard,
    Infer,
    Name(&'a str),
    Record(&'a [(&'a str, &'a Type<'a>)]),
    Apply(&'a Type<'a>, &'a [Type<'a>]),
    Lambda(&'a [Type<'a>], &'a Type<'a>),
    WhereClause(&'a Type<'a>, &'a Type<'a>, &'a [&'a str]),
    Tuple(&'a [Type<'a>]),
    TagUnion(&'a [Type<'a>]),
    Adendum(&'a Type<'a>, &'a Type<'a>),
    ModuleType(&'a str, &'a str),
    As(&'a Type<'a>, &'a Type<'a>),
}

fn build_type<'a, 'b: 'a>(
    bump: &'a Bump,
    ctx: ParsedCtx<'b>,
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
            N::Underscore => stack.push(i, Type::Infer),
            N::Tag => stack.push(i, Type::Name(ctx.text(index))), // TODO
            N::AbilityName => stack.push(i, Type::Name(ctx.text(index))), // TODO
            N::DotModuleUpperIdent => {
                let last = stack.pop().unwrap();
                let name = match last {
                    Type::Name(_name) => ctx.text(index),
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
                let values = stack.drain_to_index(index);
                let values = values.collect::<std::vec::Vec<_>>();
                dbg!(&values);
                let mut values = values.into_iter();
                let mut pairs: Vec<(&'a str, &'a Type<'a>)> = Vec::new_in(bump);
                loop {
                    let name = match values.next() {
                        Some(Type::Name(name)) => name,
                        None => break,
                        name => panic!("Expected name, got {:?}", name),
                    };
                    let value = values.next().unwrap();
                    pairs.push((name, bump.alloc(value)));
                }
                drop(values);
                stack.push(i, Type::Record(pairs.into_bump_slice()));
            }
            N::EndWhereClause => {
                let mut values = stack.drain_to_index(index);
                let first = bump.alloc(dbg!(values.next().unwrap()));
                let second = bump.alloc(values.next().unwrap());
                let mut rest: Vec<&'a str> = Vec::new_in(bump);
                while let Some(a) = values.next() {
                    match a {
                        Type::Name(name) => rest.push(name),
                        _ => panic!("Expected ability name"),
                    }
                }
                drop(values);
                stack.push(i, Type::WhereClause(first, second, rest.into_bump_slice()));
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
                    let values = values.into_iter();
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Type::Tuple(args));
                }
            }
            N::EndTypeTagUnion => {
                let values = stack.drain_to_index(index);
                let args = bump.alloc_slice_fill_iter(values);
                stack.push(i, Type::TagUnion(args));
            }
            N::InlineLambdaArrow
            | N::InlineTypeWhere
            | N::InlineKwImplements
            | N::InlineTypeArrow
            | N::BeginTypeRecord
            | N::BeginParens
            | N::InlineKwAs
            | N::InlineTypeAs
            | N::BeginTypeTagUnion => {}
            N::EndTypeOrTypeAlias => {
                assert_eq!(stack.len(), 1, "{:?}", stack);

                return (stack.pop().unwrap(), i, index);
            }
            N::EndImplements | N::EndAbilityMethod => {
                assert_eq!(stack.len(), 1, "{:?}", stack);

                return (stack.pop().unwrap(), i, index);
            }
            _ => todo!("{:?}", node),
        }
    }

    panic!("didn't find EndTypeOrTypeAlias");
}

fn build_ability<'a, 'b: 'a>(
    bump: &'a Bump,
    ctx: ParsedCtx<'b>,
    w: &mut TreeWalker<'b>,
) -> (&'a [(&'a str, Type<'a>)], usize, u32) {
    let mut items = Vec::new_in(bump);
    loop {
        if w.cur() == Some(N::EndImplements) {
            let (_, i, index) = w.next_index().unwrap();
            return (items.into_bump_slice(), i, index);
        }
        assert_eq!(w.next(), Some(N::BeginAbilityMethod));
        if w.cur() == Some(N::Ident) {
            let name = ctx.text(w.next_index().unwrap().2);
            let (ty, _, _) = build_type(bump, ctx, w);
            items.push((name, ty));
        } else {
            todo!("{:?}", w.cur());
        }
    }
}

pub fn build<'a, 'b: 'a>(bump: &'a Bump, ctx: ParsedCtx<'b>) -> &'a [Expr<'a>] {
    // let mut ce = CommentExtractor::new(ctx.text, ctx.toks);

    let mut w = TreeWalker::new(&ctx.tree);

    if w.cur() == Some(N::BeginFile) {
        w.next();
    }

    if matches!(
        w.cur(),
        Some(
            N::BeginHeaderApp
                | N::BeginHeaderHosted
                | N::BeginHeaderInterface
                | N::BeginHeaderPackage
                | N::BeginHeaderPlatform
        )
    ) {
        w.next();
        while let Some((node, _i, _index)) = w.cur_index() {
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
                }
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
            N::OpaqueName => stack.push(i, Expr::OpaqueName(ctx.text(index))),
            N::DotNumber => stack.push(i, Expr::DotNumber(ctx.text(index))),
            N::Crash => stack.push(i, Expr::Crash),
            N::Underscore => stack.push(i, Expr::Underscore(ctx.text(index))),
            N::UpperIdent => stack.push(i, Expr::UpperIdent(ctx.text(index))),
            N::Num => stack.push(i, Expr::IntBase10(ctx.text(index))),
            N::Float => stack.push(i, Expr::Float(ctx.text(index))),
            N::String => stack.push(i, Expr::String(ctx.text(index))),
            N::DotIdent => stack.push(i, Expr::Ident(ctx.text(index))),
            N::TupleAccessFunction => stack.push(i, Expr::TupleAccessFunction(ctx.text(index))),
            N::FieldAccessFunction => stack.push(i, Expr::FieldAccessFunction(ctx.text(index))),
            N::ModuleName => {
                // stack.push(i, Expr::ModuleName(ctx.text(index)))
                if let Some((N::DotModuleLowerIdent, _, index2)) = w.cur_index() {
                    w.next();
                    // if the next node is also a module name, then this is a module name
                    stack.push(i, Expr::ModuleLowerName(ctx.text(index), ctx.text(index2)));
                } else if let Some((N::DotModuleUpperIdent, _, index2)) = w.cur_index() {
                    w.next();
                    // if the next node is also a module name, then this is a module name
                    stack.push(i, Expr::ModuleUpperName(ctx.text(index), ctx.text(index2)));
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
                let values = stack.drain_to_index(index);
                let args = bump.alloc_slice_fill_iter(values);
                stack.push(i, Expr::List(args));
            }
            N::EndPatternList => {
                let values = stack.drain_to_index(index);
                let args = bump.alloc_slice_fill_iter(values);
                stack.push(i, Expr::PatternList(args));
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
            N::EndBinOpLessThan => binop!(BinOp::LessThan),
            N::EndBinOpGreaterThan => binop!(BinOp::GreaterThan),
            N::EndBinOpLessThanOrEq => binop!(BinOp::LessThanOrEq),
            N::EndBinOpGreaterThanOrEq => binop!(BinOp::GreaterThanOrEq),
            N::EndLambda => {
                let mut values = stack.drain_to_index(index);
                let count = values.len() - 1;
                let args = bump.alloc_slice_fill_iter(values.by_ref().take(count));
                let body = values.next().unwrap();
                drop(values);
                stack.push(i, Expr::Lambda(args, bump.alloc(body)));
            }
            N::EndAssignDecl => {
                let mut values = stack.drain_to_index(index);
                assert_eq!(
                    values.len(),
                    2,
                    "{:?}",
                    values.collect::<std::vec::Vec<_>>()
                );
                let name = values.next().unwrap();
                let value = values.next().unwrap();
                drop(values);
                stack.push(i, Expr::Assign(bump.alloc(name), bump.alloc(value)));
            }
            N::EndBackpassing => {
                let mut values = stack.drain_to_index(index);
                let name_count = values.len() - 1;
                let names = bump.alloc_slice_fill_iter(values.by_ref().take(name_count));
                let value = values.next().unwrap();
                assert_eq!(values.next(), None);
                drop(values);
                stack.push(i, Expr::Backpassing(names, bump.alloc(value)));
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
            N::InlineTypeColonEqual => {
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
                stack.push(i, Expr::TypeAliasOpaque(bump.alloc(name), bump.alloc(ty)));
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
                let values = stack.drain_to_index(index);
                let args = bump.alloc_slice_fill_iter(values);
                stack.push(i, Expr::Record(args));
            }
            N::EndRecordUpdate => {
                let values = stack.drain_to_index(index);
                let args = bump.alloc_slice_fill_iter(values);
                stack.push(i, Expr::RecordUpdate(args)); // TODO: add the original record as a separate field
            }
            N::EndRecordFieldPair => {
                let mut values = stack.drain_to_index(index);
                let name = match values.next().unwrap() {
                    Expr::Ident(name) => name,
                    n => panic!("Expected ident, found {:?}", n),
                };
                let value = values.next().unwrap();
                drop(values);
                stack.push(i, Expr::RecordFieldPair(name, bump.alloc(value)));
            }
            N::PatternAny => stack.push(i, Expr::PatternAny),
            N::PatternDoubleDot => stack.push(i, Expr::PatternDoubleDot),
            N::EndPatternRecord => {
                let mut values = stack.drain_to_index(index);
                let mut pairs: Vec<(&'a str, &'a Expr<'a>)> = Vec::new_in(bump);
                loop {
                    let name = match values.next() {
                        Some(Expr::Ident(name)) => name,
                        None => break,
                        a => panic!("Expected ident, found {:?}", a),
                    };
                    let value = values.next().unwrap();
                    // TODO: allow optional :/? for field patterns
                    pairs.push((name, bump.alloc(value)));
                }
                drop(values);
                stack.push(i, Expr::PatternRecord(pairs.into_bump_slice()));
            }
            N::EndParens => {
                let mut values = stack.drain_to_index(index);
                if values.len() == 1 {
                    // we don't create a tuple for a single element
                    let value = values.next().unwrap();
                    drop(values);
                    stack.push(i, value);
                } else {
                    let values = values.into_iter();
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
                    let values = values.into_iter();
                    let args = bump.alloc_slice_fill_iter(values);
                    stack.push(i, Expr::Tuple(args));
                }
            }
            N::EndPatternAs => {
                let mut values = stack.drain_to_index(index);
                let pat = values.next().unwrap();
                let name = match values.next().unwrap() {
                    Expr::Ident(name) => name,
                    _ => panic!("Expected ident"),
                };
                drop(values);
                stack.push(i, Expr::PatternAs(bump.alloc(pat), bump.alloc(name)));
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
            N::InlineAbilityImplements => {
                let (_body, _i, index) = build_ability(bump, ctx, &mut w);
                let values = stack.drain_to_index(index);
                assert_eq!(
                    values.len(),
                    1,
                    "{:?}",
                    values.collect::<std::vec::Vec<_>>()
                );
            }
            N::BeginFile
            | N::InlineKwImplements
            | N::EndFile
            | N::InlineApply
            | N::InlineAssign
            | N::InlinePizza
            | N::InlineColon
            | N::InlineBackArrow
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
            | N::InlineBinOpLessThan
            | N::InlineBinOpGreaterThan
            | N::InlineBinOpLessThanOrEq
            | N::InlineBinOpGreaterThanOrEq
            | N::InlineLambdaArrow
            | N::InlineTypeWhere
            | N::BeginBlock
            | N::BeginParens
            | N::BeginRecord
            | N::BeginRecordUpdate
            | N::BeginBackpassing
            | N::BeginTypeOrTypeAlias
            | N::BeginWhen
            | N::BeginList
            | N::InlineKwIs
            | N::InlineWhenArrow
            | N::BeginIf
            | N::InlineKwThen
            | N::InlineKwElse
            | N::BeginLambda
            | N::BeginAssignDecl
            | N::BeginTopLevelDecls
            | N::EndTopLevelDecls
            | N::BeginDbg
            | N::BeginExpect
            | N::BeginExpectFx
            | N::BeginPatternRecord
            | N::BeginPatternParens
            | N::BeginPatternList
            | N::BeginImplements
            | N::EndMultiBackpassingArgs
            | N::InlineRecordUpdateAmpersand
            | N::HintExpr
            | N::InlineMultiBackpassingComma => {}
            _ => todo!("{:?}", node),
        }
    }
    bump.alloc_slice_fill_iter(stack.drain_to_index(0))
}
