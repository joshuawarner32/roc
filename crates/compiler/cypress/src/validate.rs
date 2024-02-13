use std::collections::VecDeque;

use crate::{
    parse::State,
    token::{TokenenizedBuffer, T},
    tree::{NodeIndexKind, Tree, N},
};

pub fn state_to_expect_atom(state: &State) -> ExpectAtom {
    tree_to_expect_atom(&state.tree, &state.buf)
}

fn tree_to_expect_atom(tree: &Tree, buf: &TokenenizedBuffer) -> ExpectAtom {
    let (i, atom) = tree_expect_atom(tree, tree.kinds.len(), buf);
    assert_eq!(i, 0);
    atom
}

fn tree_expect_atom(tree: &Tree, end: usize, buf: &TokenenizedBuffer) -> (usize, ExpectAtom) {
    let node = tree.kinds[end - 1];
    let index = tree.indices[end - 1];

    let has_begin = match node.index_kind() {
        NodeIndexKind::Begin => {
            return (end - 1, ExpectAtom::Unit(node));
        }

        NodeIndexKind::Token => {
            if let Some(token) = buf.kind(index as usize) {
                return (end - 1, ExpectAtom::Token(node, token, index as usize));
            } else {
                return (end - 1, ExpectAtom::BrokenToken(node, index as usize));
            }
        }
        NodeIndexKind::Unused => {
            return (end - 1, ExpectAtom::Unit(node));
        }
        NodeIndexKind::End => true,
        NodeIndexKind::EndOnly => false,

        NodeIndexKind::EndSingleToken => {
            let mut res = VecDeque::new();
            res.push_front(ExpectAtom::Unit(node));
            let (new_i, atom) = tree_expect_atom(tree, end - 1, buf);
            assert!(new_i < end - 1);
            res.push_front(atom);
            res.push_front(ExpectAtom::Empty);
            return (new_i, ExpectAtom::Seq(res.into()));
        }
    };

    let mut res = VecDeque::new();
    res.push_front(ExpectAtom::Unit(node));
    let begin = index as usize;

    let mut i = end - 1;

    while i > begin {
        let (new_i, atom) = tree_expect_atom(tree, i, buf);
        assert!(new_i < i);
        assert!(
            new_i >= begin,
            "new_i={}, begin={}, node={:?}",
            new_i,
            begin,
            node
        );
        res.push_front(atom);
        i = new_i;
    }

    if has_begin {
        assert_eq!(
            tree.indices[begin], end as u32,
            "begin/end mismatch at {}->{}, with node {:?}",
            begin, end, node
        );
        // if tree.indices[begin] != end as u32 {
        //     panic!("begin/end mismatch")
        // }
    } else {
        res.push_front(ExpectAtom::Empty);
    }

    (begin, ExpectAtom::Seq(res.into()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectAtom {
    Seq(Vec<ExpectAtom>),
    Unit(N),
    Token(N, T, usize),
    Empty,
    BrokenToken(N, usize),
}

impl ExpectAtom {
    pub fn debug_vis(&self) -> String {
        self.debug_vis_indent(0)
    }

    fn debug_vis_indent(&self, indent: usize) -> String {
        match self {
            ExpectAtom::Seq(items) => {
                // format!("({})", items.iter().map(|i| i.debug_vis()).collect::<Vec<_>>().join(" "))

                // first let's build up a list of items that have been formatted:
                let formatted_items = items
                    .iter()
                    .map(|i| i.debug_vis_indent(indent + 4))
                    .collect::<Vec<_>>();

                // now, if the total length of the formatted items is less than 80, we can just return them as a list
                let total_len = formatted_items.iter().map(|s| s.len()).sum::<usize>()
                    + formatted_items.len()
                    - 1
                    + 2;
                if total_len < 80 {
                    return format!("({})", formatted_items.join(" "));
                }

                // otherwise, we need to format them as an indented block
                // somewhat strangely, we format like this:
                // (first
                //     second
                //     ...
                // last)

                let mut res = String::new();
                res.push_str(&format!("({}", formatted_items[0]));
                for item in &formatted_items[1..formatted_items.len() - 1] {
                    res.push_str(&format!("\n{:indent$}{}", "", item, indent = indent + 4));
                }
                res.push_str(&format!(
                    "\n{:indent$}{})",
                    "",
                    formatted_items[formatted_items.len() - 1],
                    indent = indent
                ));

                res
            }
            ExpectAtom::Unit(kind) => format!("{:?}", kind),
            ExpectAtom::Token(kind, token, token_index) => {
                format!("{:?}=>{:?}@{}", kind, token, token_index)
            }
            ExpectAtom::BrokenToken(kind, token_index) => {
                format!("{:?}=>?broken?@{}", kind, token_index)
            }
            ExpectAtom::Empty => format!("*"),
        }
    }
}
