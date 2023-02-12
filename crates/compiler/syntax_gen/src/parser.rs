use crate::libparser::prelude::*;

struct Expr<'a> {
    seq: Seq<'a>,
    second: Vec<((), Seq<'a>)>,
}
type Grammar<'a> = Vec<Rule<'a>>;
enum Item<'a> {
    Name(Name<'a>),
    Ws(Ws<'a>),
    Literal(Literal<'a>),
    Parens(Parens<'a>),
    Optional(&'a Optional<'a>),
}
type Literal<'a> = StrLiteral<'a>;
type Name<'a> = ();
struct Optional<'a> {
    item: Item<'a>,
    question: (),
}
struct Parens<'a> {
    open_paren: (),
    expr: Expr<'a>,
    close_paren: (),
}
struct Rule<'a> {
    name: Name<'a>,
    equals: (),
    expr: Expr<'a>,
    semicolon: (),
}
type Seq<'a> = Vec<Item<'a>>;
type Ws<'a> = ();
