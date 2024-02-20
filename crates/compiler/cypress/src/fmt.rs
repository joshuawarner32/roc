use std::fmt::Display;

use crate::{
    parse::ParsedCtx,
    tree::{TreeStack, TreeWalker, N},
};

pub enum Doc<'a> {
    Copy(&'a str),
}

impl Display for Doc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Doc::Copy(s) => f.write_str(s),
        }
    }
}

pub fn fmt<'b>(ctx: ParsedCtx<'b>) -> Doc {
    let mut w = TreeWalker::new(&ctx.tree);
    let mut stack = TreeStack::new();

    while let Some((node, i, index)) = w.next_index() {
        match node {
            N::Ident | N::Num => stack.push(i, Doc::Copy(ctx.text(index))),
            N::BeginFile | N::EndFile => {}
            N::BeginTopLevelDecls | N::EndTopLevelDecls => {}
            N::HintExpr => {}
            node => todo!("{:?}", node),
        }
    }

    let mut it = stack.drain_to_index(0);
    let res = it.next().unwrap();
    assert!(it.next().is_none());
    res
}
