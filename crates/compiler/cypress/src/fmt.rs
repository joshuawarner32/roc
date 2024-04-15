use std::fmt::Display;

use crate::{
    parse::ParsedCtx,
    token::{Comment, TokenenizedBuffer, T},
    tree::{TreeStack, TreeWalker, N},
};

#[derive(Debug)]
pub enum Doc<'a> {
    OptionalNewline,
    ForcedNewline,
    Space,
    Copy(&'a str),
    Literal(&'static str),
    Comment(&'a str),
    Concat(Vec<Doc<'a>>),
    Group(Vec<Doc<'a>>),
    Indent(Box<Doc<'a>>),
}

impl<'a> Doc<'a> {
    fn must_be_multiline(&self) -> bool {
        match self {
            Doc::OptionalNewline | Doc::Space | Doc::Copy(_) | Doc::Literal(_) => false,
            Doc::ForcedNewline | Doc::Comment(_) => true,
            Doc::Concat(v) | Doc::Group(v) => v.iter().any(|d| d.must_be_multiline()),
            Doc::Indent(d) => d.must_be_multiline(),
        }
    }

    fn render(&self, indent: usize, max_width: usize, honor_newlines: bool, text: &mut String) {
        match self {
            Doc::OptionalNewline => {
                if honor_newlines {
                    text.push('\n');
                    for _ in 0..indent {
                        text.push(' ');
                    }
                } else {
                    text.push(' ');
                }
            }
            Doc::Space => text.push(' '),
            Doc::Copy(s) => text.push_str(s),
            Doc::Literal(s) => text.push_str(s),
            Doc::ForcedNewline => {
                assert!(!honor_newlines);
                text.push('\n')
            }
            Doc::Comment(s) => {
                assert!(!honor_newlines);
                text.push_str(s)
            }
            Doc::Concat(v) => {
                for d in v {
                    d.render(indent, max_width, honor_newlines, text);
                }
            }
            Doc::Group(v) => {
                let mut honor_newlines = honor_newlines;
                if honor_newlines
                    && !self.must_be_multiline()
                    && self.width_without_newlines() <= max_width
                {
                    honor_newlines = false;
                }

                for d in v {
                    d.render(indent, max_width, honor_newlines, text);
                }
            }
            Doc::Indent(d) => {
                d.render(indent + 1, max_width, honor_newlines, text);
            }
        }
    }

    fn width_without_newlines(&self) -> usize {
        match self {
            Doc::OptionalNewline => 1,
            Doc::ForcedNewline => panic!("should not be called"),
            Doc::Space => 1,
            Doc::Copy(s) => s.len(),
            Doc::Literal(s) => s.len(),
            Doc::Comment(s) => s.len(),
            Doc::Concat(v) => v.iter().map(|d| d.width_without_newlines()).sum(),
            Doc::Group(v) => v.iter().map(|d| d.width_without_newlines()).sum(),
            Doc::Indent(d) => d.width_without_newlines(),
        }
    }
}

impl Display for Doc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut text = String::new();
        self.render(0, 10, true, &mut text);
        write!(f, "{}", text)
    }
}

fn op(text: &'static str) -> Doc {
    Doc::Concat(vec![Doc::OptionalNewline, Doc::Literal(text), Doc::Space])
}

fn spaced(text: &'static str) -> Doc {
    Doc::Concat(vec![Doc::Space, Doc::Literal(text), Doc::Space])
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
            | N::Underscore
            | N::AbilityName
            | N::DotIdent
            | N::DotModuleUpperIdent
            | N::DotModuleLowerIdent
            | N::TupleAccessFunction
            | N::FieldAccessFunction => stack.push(i, Doc::Copy(ctx.text(index))),
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
            N::PatternAny => {}
            N::EndPatternAs => {}
            N::BeginParens | N::BeginPatternParens => stack.push(i, Doc::Literal("(")),
            N::EndParens | N::EndPatternParens => stack.push(i, Doc::Literal(")")),
            N::BeginFile | N::EndFile => {}
            N::BeginTopLevelDecls | N::EndTopLevelDecls => {}
            N::BeginAssignDecl | N::EndAssignDecl => {}
            N::HintExpr => {}
            N::BeginTypeOrTypeAlias | N::EndTypeOrTypeAlias => {}
            N::BeginRecord | N::BeginTypeRecord | N::BeginPatternRecord | N::BeginRecordUpdate => {
                stack.push(i, Doc::Literal("{"))
            }
            N::EndRecord | N::EndTypeRecord | N::EndPatternRecord | N::EndRecordUpdate => {
                stack.push(i, Doc::Literal("}"))
            }
            N::BeginList | N::BeginTypeTagUnion | N::BeginPatternList => {
                stack.push(i, Doc::Literal("["))
            }
            N::EndList | N::EndTypeTagUnion | N::EndPatternList => stack.push(i, Doc::Literal("]")),
            N::InlineLambdaArrow | N::InlineTypeArrow => stack.push(i, spaced("->")),
            N::InlineBackArrow => stack.push(i, spaced("<-")),
            N::InlineTypeWhere => stack.push(i, Doc::Literal("where")),

            N::InlineAbilityImplements | N::InlineKwImplements => {
                stack.push(i, Doc::Literal("implements"))
            }

            N::InlineRecordUpdateAmpersand => stack.push(i, Doc::Literal("&")),

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

            N::PatternDoubleDot => stack.push(i, op("..")),
            N::InlineMultiBackpassingComma => stack.push(i, op(",")),

            N::InlineColon | N::InlineTypeColon => stack.push(i, Doc::Literal(":")),

            N::InlineTypeColonEqual => stack.push(i, Doc::Literal(":=")),

            N::EndPizza
            | N::EndBinOpAnd
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
            | N::EndBinOpStar => {
                let items = stack.drain_to_index(index);
                let res = Doc::Group(items.collect());
                stack.push(i, res);
            }

            N::EndTypeApply => {}
            N::BeginBlock | N::EndBlock => {}
            N::BeginBackpassing | N::EndBackpassing => {}
            N::BeginExpect | N::EndExpect => {}
            N::BeginExpectFx | N::EndExpectFx => {}
            N::BeginAbilityMethod | N::EndAbilityMethod => {}
            N::BeginImplements | N::EndImplements => {}
            N::InlineApply | N::EndApply => {}
            N::EndRecordFieldPair => {}
            N::EndAssign => {}
            N::EndWhereClause => {}
            N::EndTypeLambda => {}

            N::BeginHeaderApp => stack.push(i, Doc::Literal("app")),
            N::BeginHeaderPlatform => stack.push(i, Doc::Literal("platform")),
            N::BeginHeaderPackage => stack.push(i, Doc::Literal("package")),
            N::BeginHeaderInterface => stack.push(i, Doc::Literal("interface")),
            N::BeginHeaderHosted => stack.push(i, Doc::Literal("hosted")),

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
            | N::EndTypeLambda
            | N::BeginImplements
            | N::EndImplements
            | N::EndRecordFieldPair
            | N::BeginExpect
            | N::EndExpect
            | N::BeginExpectFx
            | N::EndExpectFx
            | N::BeginBackpassing
            | N::EndBackpassing
            | N::BeginAbilityMethod
            | N::EndAbilityMethod
            | N::EndPatternAs
            | N::EndWhereClause
            | N::PatternAny
            | N::EndMultiBackpassingArgs => &[],

            N::BeginTypeOrTypeAlias | N::EndTypeOrTypeAlias => &[],
            N::InlineColon | N::InlineTypeColon => self.check_next_token(T::OpColon),

            N::TupleAccessFunction => self.check_next_token(T::DotNumber),
            N::FieldAccessFunction => self.check_next_token(T::DotLowerIdent),
            N::InlineTypeColonEqual => self.check_next_token(T::OpColonEqual),

            N::Ident => self.check_next(T::LowerIdent, |t| t == T::LowerIdent || t.is_keyword()),
            N::UpperIdent | N::TypeName | N::ModuleName | N::Tag | N::AbilityName => {
                self.check_next_token(T::UpperIdent)
            }
            N::OpaqueName | N::OpaqueRef => self.check_next_token(T::OpaqueName),
            N::Num => self.check_next_token(T::Int),
            N::DotIdent | N::DotModuleLowerIdent => self.check_next_token(T::DotLowerIdent),
            N::DotModuleUpperIdent => self.check_next_token(T::DotUpperIdent),
            N::Underscore => self.check_next(T::Underscore, |t| {
                t == T::Underscore || t == T::NamedUnderscore
            }),
            N::Float => self.check_next_token(T::Float),
            N::String => self.check_next_token(T::String),
            N::InlineAssign => self.check_next_token(T::OpAssign),

            N::Crash => self.check_next_token(T::KwCrash),

            N::BeginDbg => self.check_next_token(T::KwDbg),

            N::BeginHeaderApp => self.check_next_token(T::KwApp),
            N::BeginHeaderPlatform => self.check_next_token(T::KwPlatform),
            N::BeginHeaderPackage => self.check_next_token(T::KwPackage),
            N::BeginHeaderInterface => self.check_next_token(T::KwInterface),
            N::BeginHeaderHosted => self.check_next_token(T::KwHosted),

            N::BeginIf => self.check_next_token(T::KwIf),
            N::InlineKwThen => self.check_next_token(T::KwThen),
            N::InlineKwElse => self.check_next_token(T::KwElse),

            N::InlineAbilityImplements | N::InlineKwImplements => {
                self.check_next_token(T::KwImplements)
            }

            N::BeginWhen => self.check_next_token(T::KwWhen),
            N::InlineKwIs => self.check_next_token(T::KwIs),
            N::InlineWhenArrow => self.check_next_token(T::OpArrow),

            N::BeginLambda => self.check_next_token(T::OpBackslash),
            N::InlineLambdaArrow => self.check_next_token(T::OpArrow),
            N::InlineTypeArrow => self.check_next_token(T::OpArrow),
            N::InlineTypeWhere => self.check_next_token(T::KwWhere),

            N::BeginRecord | N::BeginTypeRecord | N::BeginPatternRecord | N::BeginRecordUpdate => {
                self.check_next_token(T::OpenCurly)
            }
            N::EndRecord | N::EndTypeRecord | N::EndPatternRecord | N::EndRecordUpdate => {
                self.check_next_token(T::CloseCurly)
            }
            N::BeginList | N::BeginPatternList | N::BeginTypeTagUnion => {
                self.check_next_token(T::OpenSquare)
            }
            N::EndList | N::EndPatternList | N::EndTypeTagUnion => {
                self.check_next_token(T::CloseSquare)
            }
            N::BeginParens | N::BeginPatternParens => self.check_next_token(T::OpenRound),
            N::EndParens | N::EndPatternParens => self.check_next_token(T::CloseRound),

            N::InlineMultiBackpassingComma => self.check_next_token(T::Comma),

            N::InlineRecordUpdateAmpersand => self.check_next_token(T::OpAmpersand),

            N::InlineBackArrow => self.check_next_token(T::OpBackArrow),
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
            N::PatternDoubleDot => self.check_next_token(T::DoubleDot),

            _ => todo!("comment extract {:?}", node),
        }
    }
}
