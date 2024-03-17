use std::fmt::Display;

use crate::{
    parse::ParsedCtx,
    token::{Comment, TokenenizedBuffer, T},
    tree::{TreeStack, TreeWalker, N},
};

#[derive(Debug)]
pub enum Doc<'a> {
    Newline,
    Space,
    Copy(&'a str),
    Literal(&'static str),
    Comment(&'a str),
    Concat(Vec<Doc<'a>>),
    Group(Vec<Doc<'a>>),
}

impl<'a> Doc<'a> {
    fn contains_comment(&self) -> bool {
        match self {
            Doc::Newline | Doc::Space | Doc::Copy(_) | Doc::Literal(_) => false,
            Doc::Comment(_) => true,
            Doc::Concat(v) | Doc::Group(v) => v.iter().any(|d| d.contains_comment()),
        }
    }

    fn render(&self, max_width: usize, honor_newlines: bool, text: &mut String) {
        match self {
            Doc::Newline => {
                if honor_newlines {
                    text.push('\n');
                } else {
                    text.push(' ');
                }
            }
            Doc::Space => text.push(' '),
            Doc::Copy(s) => text.push_str(s),
            Doc::Literal(s) => text.push_str(s),
            Doc::Comment(s) => {
                assert!(!honor_newlines);
                text.push_str(s)
            }
            Doc::Concat(v) => {
                for d in v {
                    d.render(max_width, honor_newlines, text);
                }
            }
            Doc::Group(v) => {
                let mut honor_newlines = honor_newlines;
                if honor_newlines
                    && !self.contains_comment()
                    && self.width_without_newlines() <= max_width
                {
                    honor_newlines = false;
                }

                for d in v {
                    d.render(max_width, honor_newlines, text);
                }
            }
        }
    }

    fn width_without_newlines(&self) -> usize {
        match self {
            Doc::Newline => 1,
            Doc::Space => 1,
            Doc::Copy(s) => s.len(),
            Doc::Literal(s) => s.len(),
            Doc::Comment(s) => s.len(),
            Doc::Concat(v) => v.iter().map(|d| d.width_without_newlines()).sum(),
            Doc::Group(v) => v.iter().map(|d| d.width_without_newlines()).sum(),
        }
    }
}

impl Display for Doc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut text = String::new();
        self.render(10, true, &mut text);
        write!(f, "{}", text)
    }
}

fn op(text: &'static str) -> Doc {
    Doc::Concat(vec![Doc::Newline, Doc::Literal(text), Doc::Space])
}

pub fn fmt<'b>(ctx: ParsedCtx<'b>) -> Doc {
    let mut w = TreeWalker::new(&ctx.tree);
    let mut stack = TreeStack::new();
    let mut ca = CommentAligner::new(ctx);

    while let Some((node, i, index)) = w.next_index() {
        let comments = ca.consume(node);
        for comment in comments {
            stack.push(i, Doc::Comment(ctx.comment_text(*comment)));
        }
        match node {
            N::Ident
            | N::Num
            | N::Float
            | N::String
            | N::UpperIdent
            | N::TypeName
            | N::ModuleName
            | N::OpaqueName
            | N::Tag
            | N::Underscore => stack.push(i, Doc::Copy(ctx.text(index))),
            N::InlineAssign => stack.push(i, Doc::Literal("=")),
            N::BeginWhen => stack.push(i, Doc::Literal("when")),
            N::EndWhen => {}
            N::InlineKwIs => stack.push(i, Doc::Literal("is")),
            N::InlineWhenArrow => stack.push(i, Doc::Literal("->")),
            N::BeginIf => stack.push(i, Doc::Literal("if")),
            N::EndIf => {}
            N::InlineKwThen => stack.push(i, Doc::Literal("then")),
            N::InlineKwElse => stack.push(i, Doc::Literal("else")),
            N::Crash => stack.push(i, Doc::Literal("crash")),
            N::BeginDbg => stack.push(i, Doc::Literal("dbg")),
            N::EndDbg => {}
            N::BeginLambda => stack.push(i, Doc::Literal("\\")),
            N::EndLambda => {}
            N::BeginParens => stack.push(i, Doc::Literal("(")),
            N::EndParens => stack.push(i, Doc::Literal(")")),
            N::BeginFile | N::EndFile => {}
            N::BeginTopLevelDecls | N::EndTopLevelDecls => {}
            N::BeginAssignDecl | N::EndAssignDecl => {}
            N::HintExpr => {}
            N::BeginTypeOrTypeAlias | N::EndTypeOrTypeAlias => {}
            N::BeginRecord | N::BeginTypeRecord | N::BeginPatternRecord => {
                stack.push(i, Doc::Literal("{"))
            }
            N::EndRecord | N::EndTypeRecord | N::EndPatternRecord => {
                stack.push(i, Doc::Literal("}"))
            }
            N::BeginList | N::BeginTypeTagUnion => stack.push(i, Doc::Literal("[")),
            N::EndList | N::EndTypeTagUnion => stack.push(i, Doc::Literal("]")),
            N::InlineLambdaArrow | N::InlineTypeArrow => stack.push(i, Doc::Literal("->")),
            N::InlineTypeWhere => stack.push(i, Doc::Literal("where")),

            N::InlinePizza => stack.push(i, op("|>")),
            N::InlineBinOpAnd => stack.push(i, op("&&")),
            N::InlineBinOpCaret => stack.push(i, op("^")),
            N::InlineBinOpDoubleSlash => stack.push(i, op("//")),
            N::InlineBinOpEquals => stack.push(i, op("==")),
            N::InlineBinOpGreaterThan => stack.push(i, op(">")),
            N::InlineBinOpGreaterThanOrEq => stack.push(i, op(">=")),
            N::InlineBinOpLessThan => stack.push(i, op("<")),
            N::InlineBinOpLessThanOrEq => stack.push(i, op("<=")),
            N::InlineBinOpMinus => stack.push(i, op("-")),
            N::InlineBinOpNotEquals => stack.push(i, op("!=")),
            N::InlineBinOpOr => stack.push(i, op("||")),
            N::InlineBinOpPercent => stack.push(i, op("%")),
            N::InlineBinOpPlus => stack.push(i, op("+")),
            N::InlineBinOpSlash => stack.push(i, op("/")),
            N::InlineBinOpStar => stack.push(i, op("*")),

            N::InlineColon | N::InlineTypeColon => stack.push(i, Doc::Literal(":")),

            N::InlineTypeColonEqual => stack.push(i, Doc::Literal(":=")),

            N::EndPizza => {
                let items = stack.drain_to_index(index);
                let res = Doc::Group(items.collect());
                stack.push(i, res);
            }
            N::EndBinOpAnd
            | N::EndBinOpCaret
            | N::EndBinOpDoubleSlash
            | N::EndBinOpEquals
            | N::EndBinOpGreaterThan
            | N::EndBinOpGreaterThanOrEq
            | N::EndBinOpLessThan
            | N::EndBinOpLessThanOrEq
            | N::EndBinOpMinus
            | N::EndBinOpNotEquals
            | N::EndBinOpOr
            | N::EndBinOpPercent
            | N::EndBinOpPlus
            | N::EndBinOpSlash
            | N::EndBinOpStar => {}

            N::BeginBlock | N::EndBlock => {}
            N::InlineApply | N::EndApply => {}
            node => todo!("{:?}", node),
        }
    }

    let res: Vec<Doc> = stack.drain_to_index(0).collect();
    Doc::Concat(res)
}

pub struct CommentAligner<'a> {
    toks: &'a TokenenizedBuffer,
    pos: usize,
    line_index: usize,
    comments: Vec<Vec<Comment>>,
}

impl<'a> CommentAligner<'a> {
    pub fn new(ctx: ParsedCtx<'a>) -> Self {
        Self {
            toks: &ctx.toks,
            pos: 0,
            line_index: 0,
            comments: ctx.extract_comments(),
        }
    }

    fn check_next(&mut self, tok: T, is_expected: impl FnOnce(T) -> bool) -> &[Comment] {
        // HACK! `tok` is only used for debug printing. Remove + refactor.

        while self.line_index < self.toks.lines.len() - 1
            && (self.toks.lines[self.line_index + 1].0 as usize) < self.pos
        {
            self.line_index += 1;
        }

        let res = if self.line_index == self.toks.lines.len() && self.pos == self.toks.kinds.len() {
            self.comments[self.line_index].as_slice()
        } else if self.line_index < self.toks.lines.len() - 1
            && self.toks.lines[self.line_index + 1].0 as usize == self.pos
        {
            self.comments[self.line_index].as_slice()
        } else {
            &[]
        };

        if !is_expected(self.toks.kind(self.pos).unwrap()) {
            panic!(
                "programming error: misaligned token stream when formatting.\n\
                Expected {:?} at position {}, found {:?} instead.",
                tok,
                self.pos,
                self.toks.kind(self.pos)
            );
        }

        self.pos += 1;

        res
    }

    fn check_next_token(&mut self, tok: T) -> &[Comment] {
        self.check_next(tok, |t| t == tok)
    }

    fn consume(&mut self, node: N) -> &[Comment] {
        match node {
            N::BeginAssignDecl
            | N::BeginTopLevelDecls
            | N::HintExpr
            | N::EndIf
            | N::EndWhen
            | N::InlineApply
            | N::EndLambda
            | N::EndBlock
            | N::EndApply
            | N::EndTypeApply
            | N::EndBinOpPlus
            | N::EndBinOpStar
            | N::EndBinOpMinus
            | N::EndPizza
            | N::BeginBlock
            | N::EndTopLevelDecls
            | N::BeginFile
            | N::EndFile
            | N::EndAssign
            | N::EndAssignDecl
            | N::EndBinOpAnd
            | N::EndBinOpCaret
            | N::EndBinOpDoubleSlash
            | N::EndBinOpEquals
            | N::EndBinOpGreaterThan
            | N::EndBinOpGreaterThanOrEq
            | N::EndBinOpLessThan
            | N::EndBinOpLessThanOrEq
            | N::EndBinOpNotEquals
            | N::EndBinOpOr
            | N::EndBinOpPercent
            | N::EndBinOpSlash
            | N::EndDbg
            | N::EndTypeLambda => &[],

            N::BeginTypeOrTypeAlias | N::EndTypeOrTypeAlias => &[],
            N::InlineColon | N::InlineTypeColon => self.check_next_token(T::OpColon),

            N::InlineTypeColonEqual => self.check_next_token(T::OpColonEqual),

            N::Ident => self.check_next(T::LowerIdent, |t| t == T::LowerIdent || t.is_keyword()),
            N::UpperIdent | N::TypeName | N::ModuleName | N::Tag => {
                self.check_next_token(T::UpperIdent)
            }
            N::OpaqueName | N::OpaqueRef => self.check_next_token(T::OpaqueName),
            N::Num => self.check_next_token(T::Int),
            N::Underscore => self.check_next(T::Underscore, |t| {
                t == T::Underscore || t == T::NamedUnderscore
            }),
            N::Float => self.check_next_token(T::Float),
            N::String => self.check_next_token(T::String),
            N::InlineAssign => self.check_next_token(T::OpAssign),

            N::Crash => self.check_next_token(T::KwCrash),

            N::BeginDbg => self.check_next_token(T::KwDbg),

            N::BeginIf => self.check_next_token(T::KwIf),
            N::InlineKwThen => self.check_next_token(T::KwThen),
            N::InlineKwElse => self.check_next_token(T::KwElse),

            N::InlineKwImplements => self.check_next_token(T::KwImplements),

            N::BeginWhen => self.check_next_token(T::KwWhen),
            N::InlineKwIs => self.check_next_token(T::KwIs),
            N::InlineWhenArrow => self.check_next_token(T::OpArrow),

            N::BeginLambda => self.check_next_token(T::OpBackslash),
            N::InlineLambdaArrow => self.check_next_token(T::OpArrow),
            N::InlineTypeArrow => self.check_next_token(T::OpArrow),
            N::InlineTypeWhere => self.check_next_token(T::KwWhere),

            N::BeginRecord | N::BeginTypeRecord | N::BeginPatternRecord => {
                self.check_next_token(T::OpenCurly)
            }
            N::EndRecord | N::EndTypeRecord | N::EndPatternRecord => {
                self.check_next_token(T::CloseCurly)
            }
            N::BeginList | N::BeginPatternList | N::BeginTypeTagUnion => {
                self.check_next_token(T::OpenSquare)
            }
            N::EndList | N::EndPatternList | N::EndTypeTagUnion => {
                self.check_next_token(T::CloseSquare)
            }
            N::BeginParens => self.check_next_token(T::OpenRound),
            N::EndParens => self.check_next_token(T::CloseRound),

            N::InlinePizza => self.check_next_token(T::OpPizza),
            N::InlineBinOpAnd => self.check_next_token(T::OpAnd),
            N::InlineBinOpCaret => self.check_next_token(T::OpCaret),
            N::InlineBinOpDoubleSlash => self.check_next_token(T::OpDoubleSlash),
            N::InlineBinOpEquals => self.check_next_token(T::OpEquals),
            N::InlineBinOpGreaterThan => self.check_next_token(T::OpGreaterThan),
            N::InlineBinOpGreaterThanOrEq => self.check_next_token(T::OpGreaterThanOrEq),
            N::InlineBinOpLessThan => self.check_next_token(T::OpLessThan),
            N::InlineBinOpLessThanOrEq => self.check_next_token(T::OpLessThanOrEq),
            N::InlineBinOpMinus => self.check_next_token(T::OpBinaryMinus),
            N::InlineBinOpNotEquals => self.check_next_token(T::OpNotEquals),
            N::InlineBinOpOr => self.check_next_token(T::OpOr),
            N::InlineBinOpPercent => self.check_next_token(T::OpPercent),
            N::InlineBinOpPlus => self.check_next_token(T::OpPlus),
            N::InlineBinOpSlash => self.check_next_token(T::OpSlash),
            N::InlineBinOpStar => self.check_next_token(T::OpStar),

            _ => todo!("comment extract {:?}", node),
        }
    }
}
