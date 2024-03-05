use std::fmt::Display;

use crate::{
    parse::ParsedCtx,
    token::{Comment, TokenenizedBuffer, T},
    tree::{TreeStack, TreeWalker, N},
};

#[derive(Debug)]
pub enum Doc<'a> {
    Copy(&'a str),
    Comment(&'a str),
    Concat(Vec<Doc<'a>>),
}

impl Display for Doc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Doc::Copy(s) => f.write_str(s),
            Doc::Comment(s) => f.write_str(s),
            Doc::Concat(v) => {
                for d in v {
                    write!(f, "{}\n", d)?;
                }
                Ok(())
            }
        }
    }
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
            N::Ident | N::Num => stack.push(i, Doc::Copy(ctx.text(index))),
            N::BeginFile | N::EndFile => {}
            N::BeginTopLevelDecls | N::EndTopLevelDecls => {}
            N::HintExpr => {}
            node => todo!("{:?}", node),
        }
    }

    let res: Vec<Doc> = stack.drain_to_index(0).collect();
    Doc::Concat(res)
}

pub struct CommentAligner<'a> {
    text: &'a str,
    toks: &'a TokenenizedBuffer,
    pos: usize,
    next_line_index: usize,
    comments: Vec<Vec<Comment>>,
}

impl<'a> CommentAligner<'a> {
    pub fn new(ctx: ParsedCtx<'a>) -> Self {
        Self {
            text: ctx.text,
            toks: &ctx.toks,
            pos: 0,
            next_line_index: 0,
            comments: ctx.extract_comments(),
        }
    }

    fn check_next_token(&mut self, tok: T) -> &[Comment] {
        while self.next_line_index < self.toks.lines.len() - 1
            && (self.toks.lines[self.next_line_index + 1].0 as usize) < self.pos
        {
            self.next_line_index += 1;
        }

        let res =
            if self.next_line_index == self.toks.lines.len() && self.pos == self.toks.kinds.len() {
                self.comments[self.next_line_index].as_slice()
            } else if self.toks.lines[self.next_line_index].0 as usize == self.pos {
                self.comments[self.next_line_index].as_slice()
            } else {
                &[]
            };

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

        res
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
            | N::EndBinOpPlus
            | N::EndBinOpStar
            | N::EndBinOpMinus
            | N::EndPizza
            | N::BeginBlock
            | N::EndTopLevelDecls
            | N::BeginFile
            | N::EndFile
            | N::EndAssign => &[],

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
    }
}
