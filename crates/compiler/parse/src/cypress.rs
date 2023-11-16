#![allow(dead_code)]
#![allow(unused)]

use std::collections::VecDeque;

pub struct TokenenizedBuffer {
    kinds: Vec<T>,
    offsets: Vec<u32>,
    indents: Vec<u32>,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum T {
    Newline,
    
    Float,
    Num,
    String,
    SingleQuote,
    Number,

    UpperIdent,
    LowerIdent,
    Underscore,

    OpenRound,
    CloseRound,
    OpenSquare,
    CloseSquare,
    OpenCurly,
    CloseCurly,

    OpPlus,
    OpStar,
    OpPizza,
    OpAssign,

    Comma,
    Dot,
    Colon,
}

pub struct Tree {
    kinds: Vec<N>,
    tokens: Vec<Token>,
    subtree_start_positions: Vec<u32>,
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
    Str,

    /// eg 'b'
    SingleQuote,

    /// Look up exactly one field on a record or tuple, e.g. `x.foo` or `x.0`.
    Access,

    /// e.g. `.foo` or `.0`
    AccessorFunction,

    /// List literals, e.g. `[1, 2, 3]`
    List,

    /// Record literals, e.g. `{ x: 1, y: 2 }`
    Record,

    /// Record updates (e.g. `{ x & y: 3 }`)
    RecordUpdate,

    /// Tuple literals, e.g. `(1, 2)`
    Tuple,

    Ident,

    /// An underscore, e.g. `_` or `_x`
    Underscore,

    /// The "crash" keyword
    Crash,

    /// Tag
    Tag,

    /// Reference to an opaque type, e.g. @Opaq
    OpaqueRef,

    /// Closure, e.g. `\x -> x`
    Closure,

    /// Indented block of statements and expressions
    Block,

    /// The special dbg function, e.g. `dbg x`
    Dbg,

    /// Function application, e.g. `f x`
    Apply,

    /// Pizza operator, e.g. `x |> f`
    Pizza,

    /// Assignment declaration, e.g. `x = 1`
    Assign,

    /// A sequence of declarations, e.g. `x = 1 <newline> y = 2`
    DeclSeq,

    /// Binary operators, e.g. `x + y`
    BinOpPlus,
    BinOpStar,

    /// Unary operator, e.g. `-x`
    UnaryOp,

    /// If expression, e.g. `if x then y else z`
    If,

    /// A when expression, e.g. `when x is y -> z` (you need a newline after the 'is')
    When,
}

impl N {
    fn is_decl(self) -> bool {
        match self {
            N::Assign => true,
            _ => false,
        }
    }
}



impl TokenenizedBuffer {
    fn new(text: String) -> TokenenizedBuffer {
        todo!()
    }

    fn kind(&self, pos: usize) -> Option<T> {
        self.kinds.get(pos).copied()
    }

    fn from_tokens(kinds: &[T]) -> TokenenizedBuffer {
        TokenenizedBuffer { kinds:  kinds.to_owned(), offsets: Vec::new(), indents: Vec::new() }
    }
}


impl Tree {
    fn new() -> Tree {
        Tree {
            kinds: Vec::new(),
            tokens: Vec::new(),
            subtree_start_positions: Vec::new(),
        }
    }

    fn len(&self) -> u32 {
        self.kinds.len() as u32
    }

    fn debug_vis_grouping(&self) -> String {
        let mut stack = Vec::<(usize, String)>::new();

        for (i, (&kind, &subtree_start_pos)) in self.kinds.iter().zip(self.subtree_start_positions.iter()).enumerate() {

            if (subtree_start_pos as usize) < i {
                let mut items = VecDeque::new();
                while let Some((j, item)) = stack.pop() {
                    if j >= subtree_start_pos as usize {
                        items.push_front(item);
                    } else {
                        stack.push((j, item));
                        break;
                    }
                }

                // (a b c)
                let mut s = String::new();
                s.push('(');
                for (i, item) in items.into_iter().enumerate() {
                    if i > 0 {
                        s.push(' ');
                    }
                    s.push_str(&item);
                }
                s.push(')');

                s.push(' ');
                s.push_str(&format!("{:?}", kind));

                stack.push((i, s));
            } else {
                stack.push((i, format!("{:?}", kind)));
            }
        }
        
        // (a b c)
        let mut s = String::new();
        for (i, (_, item)) in stack.into_iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&item);
        }

        s
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Prec {
    Outer,
    DeclSeq, // BinOp::DeclSeq,
    Decl, // BinOp::Assign, BinOp::Backpassing,
    Pizza, // BinOp::Pizza,
    AndOr, // BinOp::And, BinOp::Or,
    Compare, // BinOp::Equals, BinOp::NotEquals, BinOp::LessThan, BinOp::GreaterThan, BinOp::LessThanOrEq, BinOp::GreaterThanOrEq,
    Add, // BinOp::Plus, BinOp::Minus,
    Multiply, // BinOp::Star, BinOp::Slash, BinOp::DoubleSlash, BinOp::Percent,
    Exponent, // BinOp::Caret
    Apply,
    Atom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    AssignBlock,
    DeclSeq,
    Assign,
    Backpassing,
    Pizza,
    And,
    Or,
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    LessThanOrEq,
    GreaterThanOrEq,
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    Percent,
    Caret,
    Apply,
}

impl Prec {
    fn next(self) -> Prec {
        match self {
            Prec::Outer => Prec::DeclSeq,
            Prec::DeclSeq => Prec::Decl,
            Prec::Decl => Prec::Pizza,
            Prec::Pizza => Prec::AndOr,
            Prec::AndOr => Prec::Compare,
            Prec::Compare => Prec::Add,
            Prec::Add => Prec::Multiply,
            Prec::Multiply => Prec::Exponent,
            Prec::Exponent => Prec::Apply,
            Prec::Apply => Prec::Atom,
            Prec::Atom => Prec::Atom,
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
            BinOp::AssignBlock => Prec::Outer,
            BinOp::DeclSeq => Prec::DeclSeq,
            BinOp::Assign | BinOp::Backpassing => Prec::Decl,
            BinOp::Apply => Prec::Apply,
            BinOp::Caret => Prec::Exponent,
            BinOp::Star | BinOp::Slash | BinOp::DoubleSlash | BinOp::Percent => Prec::Multiply,
            BinOp::Plus | BinOp::Minus => Prec::Add,
            BinOp::Equals | BinOp::NotEquals | BinOp::LessThan | BinOp::GreaterThan | BinOp::LessThanOrEq | BinOp::GreaterThanOrEq => Prec::Compare,
            BinOp::And | BinOp::Or => Prec::AndOr,
            BinOp::Pizza => Prec::Pizza,
        }
    }

    fn grouping_assoc(self) -> Assoc {
        match self {
            BinOp::AssignBlock => Assoc::Right,
            BinOp::DeclSeq => Assoc::Right,
            BinOp::Assign | BinOp::Backpassing => Assoc::Right,
            BinOp::Apply => Assoc::Left,
            BinOp::Caret => Assoc::Right,
            BinOp::Star | BinOp::Slash | BinOp::DoubleSlash | BinOp::Percent => Assoc::Left,
            BinOp::Plus | BinOp::Minus => Assoc::Left,
            BinOp::Equals | BinOp::NotEquals | BinOp::LessThan | BinOp::GreaterThan | BinOp::LessThanOrEq | BinOp::GreaterThanOrEq => Assoc::NonAssociative,
            BinOp::And | BinOp::Or => Assoc::Left,
            BinOp::Pizza => Assoc::Left,
        }
    }

    fn matching_assoc(self) -> Assoc {
        if self == BinOp::AssignBlock {
            return Assoc::Left;
        } else {
            self.grouping_assoc()
        }
    }

    fn n_arity(self) -> bool {
        match self {
            BinOp::Apply | BinOp::Pizza | BinOp::DeclSeq => true,
            _ => false,
        }
    }
}

impl From<BinOp> for N {
    fn from(op: BinOp) -> Self {
        match op {
            BinOp::Pizza => N::Pizza,
            BinOp::Apply => N::Apply,
            BinOp::Plus => N::BinOpPlus,
            BinOp::Minus => N::BinOpStar,
            BinOp::Assign => N::Assign,
            BinOp::DeclSeq => N::DeclSeq,
            BinOp::AssignBlock => N::Assign,
            _ => todo!("binop to node {:?}", op),
        }
    }
}

#[derive(Debug)]
enum Frame {
    StartExpr { min_prec: Prec },
    ContinueExpr { min_prec: Prec, subtree_start: u32, cur_op: Option<BinOp>, num_found: usize },
    FinishParen,
}

impl Frame {
    fn start_expr() -> Frame {
        Frame::StartExpr { min_prec: Prec::Outer }
    }
}

struct State {
    frames: Vec<Frame>,
    buf: TokenenizedBuffer,
    pos: usize,

    tree: Tree,
}

impl State {
    fn new(text: String) -> Self {
        State {
            frames: vec![Frame::start_expr()],
            buf: TokenenizedBuffer::new(text),
            pos: 0,
            tree: Tree::new(),
        }
    }

    fn from_tokens(tokens: &[T]) -> Self {
        State {
            frames: vec![Frame::start_expr()],
            buf: TokenenizedBuffer::from_tokens(tokens),
            pos: 0,
            tree: Tree::new(),
        }
    }

    fn push_node(&mut self, kind: N, subtree_start: Option<u32>) {
        eprintln!("{:indent$}pushing kind {:?} starting at {:?}", "", kind, subtree_start, indent = 2 * self.frames.len() + 2);
        self.tree.kinds.push(kind);
        let pos = subtree_start.unwrap_or(self.tree.subtree_start_positions.len() as u32);
        self.tree.subtree_start_positions.push(pos);
        eprintln!("{:indent$}tree: {:?}", "", self.tree.debug_vis_grouping(), indent = 2 * self.frames.len() + 4);
    }

    fn push_next_frame(&mut self, frame: Frame) {
        eprintln!("{:indent$}pushing frame {:?}", "", frame, indent = 2 * self.frames.len() + 2);
        self.frames.push(frame);
    }

    fn pump(&mut self) {
        while let Some(frame) = self.frames.pop() {
            eprintln!("{:indent$}pumping frame {:?}", "", frame, indent = 2 * self.frames.len());
            match frame {
                Frame::StartExpr { min_prec } => self.pump_start_expr(min_prec),
                Frame::FinishParen => self.pump_finish_paren(),
                Frame::ContinueExpr { min_prec, subtree_start, cur_op, num_found } => self.pump_continue_expr(min_prec, subtree_start, cur_op, num_found),
            }
        }
    }

    fn pump_start_expr(&mut self, mut min_prec: Prec) {
        loop {
            let subtree_start = self.tree.len();
            match self.buf.kind(self.pos) {
                Some(T::OpenRound) => {
                    self.pos += 1;
                    self.push_next_frame(Frame::ContinueExpr { min_prec, subtree_start, cur_op: None, num_found: 1 });
                    self.push_next_frame(Frame::FinishParen);
                    min_prec = Prec::Outer;
                    continue;
                }
                Some(T::LowerIdent) => {
                    self.pos += 1;
                    self.push_next_frame(Frame::ContinueExpr { min_prec, subtree_start, cur_op: None, num_found: 1 });
                    self.push_node(N::Ident, None);
                    return;
                }
                _ => todo!(),
            }
        }
    }

    fn pump_continue_expr(&mut self, min_prec: Prec, subtree_start: u32, cur_op: Option<BinOp>, mut num_found: usize) {
        if let Some(op) = self.next_op(min_prec, cur_op) {
            if let Some(cur_op) = cur_op {
                if op != cur_op || !op.n_arity() {
                    self.push_node(cur_op.into(), Some(subtree_start));
                }
            }

            eprintln!("{:indent$}next op {:?}", "", op, indent = 2 * self.frames.len() + 2);

            let op_prec = op.prec();
            let assoc = op.matching_assoc();

            let next_min_prec = if assoc == Assoc::Left {
                op_prec
            } else {
                op_prec.next()
            };

            self.push_next_frame(Frame::ContinueExpr { min_prec, subtree_start, cur_op: Some(op), num_found });
            self.push_next_frame(Frame::StartExpr { min_prec: next_min_prec });
            return;
        } else if let Some(cur_op) = cur_op {
            self.push_node(cur_op.into(), Some(subtree_start));
        }

    }

    fn pump_finish_paren(&mut self) {
        match self.buf.kind(self.pos) {
            Some(T::CloseRound) => {
                self.pos += 1;
            }
            _ => todo!(),
        }
    }

    fn next_op(&mut self, min_prec: Prec, cur_op: Option<BinOp>) -> Option<BinOp> {
        let k = self.buf.kind(self.pos);

        let (op, width) = match k {
            Some(T::LowerIdent) => (BinOp::Apply, 0),
            Some(T::OpPlus) => (BinOp::Plus, 1),
            Some(T::OpStar) => (BinOp::Star, 1),
            Some(T::OpPizza) => (BinOp::Pizza, 1),
            Some(T::OpAssign) => {
                if self.buf.kind(self.pos + 1) == Some(T::Newline) {
                    (BinOp::AssignBlock, 2)
                } else {
                    (BinOp::Assign, 1)
                }
            },
            Some(T::Newline) => {
                dbg!(cur_op);
                if matches!(cur_op, Some(BinOp::Assign | BinOp::AssignBlock)) {
                    (BinOp::DeclSeq, 1)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        if op.prec() < min_prec || (op.prec() == min_prec && op.grouping_assoc() == Assoc::Left) {
            return None;
        }

        self.pos += width;

        Some(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum ExpectAtom {
        Seq(Vec<ExpectAtom>),
        Unit(N),
    }

    struct ExpectBuilder {
        kinds: Vec<N>,
        subtree_start_positions: Vec<u32>,
    }

    impl ExpectBuilder {
        fn new() -> ExpectBuilder {
            ExpectBuilder {
                kinds: Vec::new(),
                subtree_start_positions: Vec::new(),
            }
        }

        fn consume_items(&mut self, items: Vec<ExpectAtom>) {
            let mut last_start_pos = None;

            for item in items {
                match item {
                    ExpectAtom::Seq(items) => {
                        assert!(last_start_pos.is_none());
                        last_start_pos = Some(self.kinds.len() as u32);
                        self.consume_items(items);
                    }
                    ExpectAtom::Unit(kind) => {
                        self.kinds.push(kind);
                        let pos = last_start_pos.take().unwrap_or(self.subtree_start_positions.len() as u32);
                        self.subtree_start_positions.push(pos);
                    }
                }
            }
        }

        fn finish(mut self) -> Tree {
            Tree {
                kinds: self.kinds,
                subtree_start_positions: self.subtree_start_positions,
                tokens: Vec::new(), // TODO
            }
        }
    }

    fn build_expect(items: Vec<ExpectAtom>) -> Tree {
        let mut b = ExpectBuilder::new();
        
        b.consume_items(items);

        b.finish()
    }

    macro_rules! cvt_item {
        ($item:ident) => {
            ExpectAtom::Unit(N::$item)
        };

        (($($items:tt)*)) => {
            ExpectAtom::Seq(vec![$(cvt_item!($items)),*])
        };
    }

    macro_rules! expect {
        ($($items:tt)*) => {{
            build_expect(vec![$(cvt_item!($items)),*])
        }};
    }

    #[test]
    fn simple_tests() {
        #[track_caller]
        fn test(kinds: &[T], expected: Tree) {
            let mut state = State::from_tokens(kinds);

            state.pump();

            assert_eq!(&state.tree.kinds, &expected.kinds);
            assert_eq!(&state.tree.subtree_start_positions, &expected.subtree_start_positions);
        }
        
        test(
            &[T::LowerIdent],
            expect!(Ident),
        );
        
        test(
            &[T::LowerIdent, T::LowerIdent],
            expect!((Ident Ident) Apply),
        );
        
        // plus
        test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!((Ident Ident) BinOpPlus),
        );

        // a b + c
        test(
            &[T::LowerIdent, T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!(((Ident Ident) Apply Ident) BinOpPlus),
        );

        // a + b c
        test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent, T::LowerIdent],
            expect!((Ident (Ident Ident) Apply) BinOpPlus),
        );

        // a + b + c
        test(
            &[T::LowerIdent, T::OpPlus, T::LowerIdent, T::OpPlus, T::LowerIdent],
            expect!(((Ident Ident) BinOpPlus Ident) BinOpPlus),
        );

        // a b c
        test(
            &[T::LowerIdent, T::LowerIdent, T::LowerIdent],
            expect!((Ident Ident Ident) Apply) // only one apply covering all three
        );

        // a |> b |> c
        test(
            &[T::LowerIdent, T::OpPizza, T::LowerIdent, T::OpPizza, T::LowerIdent],
            expect!((Ident Ident Ident) Pizza),
        );
        
    }

    #[test]
    fn decl_tests() {
        #[track_caller]
        fn test(kinds: &[T], expected: Tree) {
            let mut state = State::from_tokens(kinds);

            state.pump();

            // assert_eq!(&state.tree.kinds, &expected.kinds);
            // eprintln!("{:?}", state.tree.subtree_start_positions);
            // eprintln!("{:?}", expected.subtree_start_positions);
            assert_eq!(&state.tree.debug_vis_grouping(), &expected.debug_vis_grouping());
        }

        // a = b
        test(
            &[T::LowerIdent, T::OpAssign, T::LowerIdent],
            expect!((Ident Ident) Assign),
        );

        // a = b; c = d
        test(
            &[T::LowerIdent, T::OpAssign, T::LowerIdent, T::Newline, T::LowerIdent, T::OpAssign, T::LowerIdent],
            expect!(((Ident Ident) Assign (Ident Ident) Assign) DeclSeq),
        );

        // a =; b = c; d
        test(
            &[T::LowerIdent, T::OpAssign, T::Newline, T::LowerIdent, T::OpAssign, T::LowerIdent, T::Newline, T::LowerIdent],
            expect!((Ident ((Ident Ident) Assign Ident) DeclSeq) Assign),
        );

        // a=; b; d
        test(
            &[T::LowerIdent, T::OpAssign, T::Newline, T::LowerIdent, T::Newline, T::LowerIdent],
            expect!(((Ident Ident) Assign Ident) DeclSeq),
        );

        // a =; b =; c; d
        test(
            &[T::LowerIdent, T::OpAssign, T::Newline, T::LowerIdent, T::OpAssign, T::Newline, T::LowerIdent, T::Newline, T::LowerIdent],
            expect!((Ident ((Ident Ident) Assign Ident) DeclSeq) Assign),
        );
    }
}