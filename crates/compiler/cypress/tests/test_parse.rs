use std::collections::VecDeque;

use roc_cypress::canfmt;
use roc_cypress::parse::*;
use roc_cypress::token::TokenenizedBuffer;
use roc_cypress::token::Tokenizer;
use roc_cypress::token::T;
use roc_cypress::tree::NodeIndexKind;
use roc_cypress::tree::Tree;
use roc_cypress::tree::N;

fn state_to_expect_atom(state: &State) -> ExpectAtom {
    tree_to_expect_atom(&state.tree, &state.buf)
}

fn tree_to_expect_atom(tree: &Tree, buf: &TokenenizedBuffer) -> ExpectAtom {
    let (i, atom) = tree_expect_atom(tree, tree.kinds.len(), buf);
    assert_eq!(i, 0);
    atom
}

fn tree_expect_atom(tree: &Tree, end: usize, buf: &TokenenizedBuffer) -> (usize, ExpectAtom) {
    let node = tree.kinds[end - 1];
    let index = tree.paird_group_ends[end - 1];

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
            tree.paird_group_ends[begin], end as u32,
            "begin/end mismatch at {}->{}, with node {:?}",
            begin, end, node
        );
    } else {
        res.push_front(ExpectAtom::Empty);
    }

    (begin, ExpectAtom::Seq(res.into()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ExpectAtom {
    Seq(Vec<ExpectAtom>),
    Unit(N),
    Token(N, T, usize),
    Empty,
    BrokenToken(N, usize),
}

impl ExpectAtom {
    fn debug_vis(&self) -> String {
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

fn format_message(text: &str, buf: &TokenenizedBuffer, msg: &Message) -> String {
    // binary search to find the line of msg.pos
    let (line_start, line_end) = match buf
        .lines
        .binary_search_by_key(&msg.pos, |(offset, _indent)| *offset)
    {
        Ok(i) => (buf.lines[i].0, buf.lines[i].0),
        Err(i) => {
            if i > 0 {
                if i < buf.lines.len() {
                    (buf.lines[i - 1].0, buf.lines[i].0)
                } else {
                    (buf.lines[i - 1].0, buf.kinds.len() as u32)
                }
            } else {
                (0, buf.lines[i].0)
            }
        }
    };

    debug_assert!(line_start <= msg.pos && msg.pos <= line_end);

    let mut res = String::new();

    res.push_str(&format!(
        "Error at token {} (offset {}):\n",
        msg.pos,
        buf.offset(msg.pos)
    ));

    // print the first line (tokens)
    let mut pointer_offset = 0;
    let mut pointer_len = 0;
    for (i, kind) in buf
        .kinds
        .iter()
        .enumerate()
        .skip(line_start as usize)
        .take((line_end - line_start) as usize)
    {
        let text = format!("{:?} ", kind);
        if i < msg.pos as usize {
            pointer_offset += text.len();
        } else if i == msg.pos as usize {
            pointer_len = text.len();
        }
        res.push_str(&text);
    }
    res.push('\n');

    // print the pointer
    for _ in 0..pointer_offset {
        res.push(' ');
    }
    for _ in 0..pointer_len {
        res.push('^');
    }
    res.push('\n');

    // print the text
    res.push_str(&text[buf.offset(line_start)..buf.offset(line_end)]);
    res.push('\n');

    let pointer_offset =
        buf.offsets[msg.pos as usize] as usize - buf.offsets[line_start as usize] as usize;
    let pointer_len = buf
        .lengths
        .get(msg.pos as usize)
        .map(|o| *o as usize)
        .unwrap_or(0);

    // print the pointer
    for _ in 0..pointer_offset {
        res.push(' ');
    }
    for _ in 0..pointer_len {
        res.push('^');
    }
    res.push('\n');

    res.push_str(&format!("{:?}", msg.kind));
    for frame in &msg.frames {
        res.push_str(&format!("\n  in {:?}", frame));
    }

    res
}

fn run_snapshot_test(text: &str) -> String {
    eprintln!("text:\n{}", text);
    let mut tokenizer = Tokenizer::new(text);
    tokenizer.tokenize();
    let (messages, tb) = tokenizer.finish();
    assert_eq!(messages, vec![]);
    eprint!("tokens:");
    let mut last = 0;
    for (i, (begin, indent)) in tb.lines.iter().enumerate() {
        for tok in &tb.kinds[last as usize..*begin as usize] {
            eprint!(" {:?}", tok);
        }
        eprint!(
            "\n{}: {:?} {}.{}:",
            i, begin, indent.num_spaces, indent.num_tabs
        );
        last = *begin;
    }
    for tok in &tb.kinds[last as usize..] {
        eprint!(" {:?}", tok);
    }
    eprintln!();

    let mut state = State::from_buf(tb);
    state.start_file();
    state.pump();
    state.assert_end();
    if state.messages.len() > 0 {
        for msg in state.messages.iter().take(3) {
            eprintln!("{}", format_message(text, &state.buf, msg));
        }

        panic!("unexpected messages: {:?}", state.messages);
    }

    eprintln!("raw tree: {:?}", state.tree.kinds);

    let tree_output = state_to_expect_atom(&state).debug_vis();

    let canfmt_output = {
        let bump = bumpalo::Bump::new();
        let canfmt = canfmt::build(
            &bump,
            ParsedCtx {
                tree: &state.tree,
                toks: &state.buf,
                text,
            },
        );
        canfmt
            .iter()
            .map(|i| format!("{:?}", i))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // let format_output = pretty(&state.tree, &state.buf, text).text;
    let format_output = String::new(); // TODO!

    format!(
        "{}\n\n[=== canfmt below ===]\n{}\n\n[=== formatted below ===]\n{}",
        tree_output, canfmt_output, format_output
    )
}

macro_rules! snapshot_test {
    ($text:expr) => {{
        let text = $text;
        let text: &str = text.as_ref();
        let output = run_snapshot_test(text);

        insta::with_settings!({
            // info => &ctx, // not sure what to put here?
            description => text, // the template source code
            omit_expression => true // do not include the default expression
        }, {
            insta::assert_display_snapshot!(output);
        });

        // Now let's verify that we can replace all the newlines in the original text with comments
        // and everything still works.

        // Iterate thru text and replace newlines with "# <line number>\n"
        // let mut new_text = String::with_capacity(text.len());
        // let mut line_num = 0;
        // for (i, c) in text.chars().enumerate() {
        //     if c == '\n' {
        //         line_num += 1;
        //         new_text.push_str(&format!("# {}\n", line_num));
        //     } else {
        //         new_text.push(c);
        //     }
        // }
        // let text = new_text;
        // eprintln!("commentified text {:?}", text);

        // let mut tokenizer = Tokenizer::new(&text);
        // tokenizer.tokenize();
        // let tb = tokenizer.finish();

        // let mut state = State::from_buf(tb);
        // state.start_top_level_decls();
        // state.pump();
        // state.assert_end();

        // let tree_output = state.to_expect_atom().debug_vis();

        // let bump = bumpalo::Bump::new();
        // let canfmt = canfmt::build(&bump, FormatCtx {
        //     tree: &state.tree,
        //     toks: &state.buf,
        //     text: &text,
        // });

    }};
}

#[test]
fn test_ident() {
    snapshot_test!("abc");
}

#[test]
fn test_apply() {
    snapshot_test!("abc def");
}

#[test]
fn test_simple_binop_plus() {
    snapshot_test!("abc + def");
}

#[test]
fn test_complex_apply() {
    snapshot_test!("abc def + ghi");
}

#[test]
fn test_complex_binop_plus() {
    snapshot_test!("abc + def ghi");
}

#[test]
fn test_nested_binop_plus() {
    snapshot_test!("abc + def + ghi");
}

#[test]
fn test_multiple_ident() {
    snapshot_test!("abc def ghi");
}

#[test]
fn test_pizza_operator() {
    snapshot_test!("abc |> def |> ghi |> jkl");
}

#[test]
fn test_lambda_expr() {
    snapshot_test!("\\abc -> def");
}

#[test]
fn test_if() {
    snapshot_test!("if abc then def else ghi");
}

#[test]
fn test_when() {
    snapshot_test!("when abc is def -> ghi");
}

fn block_indentify(text: &str) -> String {
    // remove the leading | from each line, along with any whitespace before that.
    // if the line is completely whitespace, remove it entirely.

    assert_eq!(text.chars().next(), Some('\n'));
    let mut res = String::new();
    let mut saw_newline = true;
    for ch in text.chars().skip(1) {
        if ch == '\n' {
            res.push(ch);
            saw_newline = true;
        } else if saw_newline {
            if ch.is_ascii_whitespace() {
                continue;
            } else if ch == '|' {
                saw_newline = false;
            }
        } else {
            res.push(ch);
        }
    }

    res
}

#[test]
fn test_block_indentify() {
    assert_eq!(
        block_indentify(
            r#"
    |abc
    |def
    |ghi
    "#
        ),
        "abc\ndef\nghi\n"
    );
}

#[test]
fn test_nested_when() {
    snapshot_test!(block_indentify(
        r#"
    |when abc is def ->
    |    when ghi is jkl ->
    |        mno
    "#
    ));
}

#[test]
fn test_weird_when_in_expr() {
    snapshot_test!(block_indentify(
        r#"
        |c=
        |when e is
        |                S->c
        | a=e
        |e
    "#
    ));
}

#[test]
fn test_simple_assign_decl() {
    snapshot_test!(block_indentify(
        r#"
    |abc = def
    "#
    ))
}

#[test]
fn test_double_assign_decl() {
    snapshot_test!(block_indentify(
        r#"
    |abc = def
    |ghi = jkl
    "#
    ))
}

#[test]
fn test_simple_nested_assign_decl() {
    snapshot_test!(block_indentify(
        r#"
    |abc =
    |    def = ghi
    |    jkl
    "#
    ))
}

#[test]
fn test_decl_then_top_level_expr() {
    snapshot_test!(block_indentify(
        r#"
    |abc =
    |    def
    |ghi
    "#
    ))
}

#[test]
fn test_double_nested_decl() {
    snapshot_test!(block_indentify(
        r#"
    |a =
    |    b =
    |        c
    |    d
    "#
    ))
}

#[test]
fn test_double_assign_block_decl() {
    snapshot_test!(block_indentify(
        r#"
    |abc =
    |    def
    |ghi =
    |    jkl
    "#
    ))
}

#[test]
fn test_lambda_decl() {
    snapshot_test!(block_indentify(
        r#"
    |abc = \def ->
    |    ghi
    "#
    ))
}

#[test]
fn test_leading_comment() {
    snapshot_test!(block_indentify(
        r#"
    |# hello
    |abc
    "#
    ))
}

// #[test]
// fn test_parse_all_files() {
//     // list all .roc files under ../test_syntax/tests/snapshots/pass
//     let files = std::fs::read_dir("../test_syntax/tests/snapshots/pass")
//         .unwrap()
//         .map(|res| res.map(|e| e.path()))
//         .collect::<Result<Vec<_>, std::io::Error>>()
//         .unwrap();

//     assert!(files.len() > 0, "no files found in ../test_syntax/tests/snapshots/pass");

//     for file in files {
//         // if the extension is not .roc, continue
//         if file.extension().map(|e| e != "roc").unwrap_or(true) {
//             continue;
//         }

//         eprintln!("parsing {:?}", file);
//         let text = std::fs::read_to_string(&file).unwrap();
//         eprintln!("---------------------\n{}\n---------------------", text);
//         let mut tokenizer = Tokenizer::new(&text);
//         tokenizer.tokenize(); // make sure we don't panic!
//         let tb = tokenizer.finish();
//         eprintln!("tokens: {:?}", tb.kinds);
//         let mut state = State::from_buf(tb);
//         state.start_file();
//         state.pump();
//         state.assert_end();
//     }
// }

#[test]
fn test_where_clause_on_newline_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f : a -> U64
        |    where a implements Hash
        |
        |f
        |
        "#
    ));
}

#[test]
fn test_one_plus_two_expr() {
    snapshot_test!(block_indentify(
        r#"
        |1+2
        "#
    ));
}

#[test]
fn test_nested_backpassing_no_newline_before_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |main =
        |    task =
        |        file <-
        |            foo
        |        bar
        |    task
        |42
        "#
    ));
}

#[test]
fn test_tag_pattern_expr() {
    snapshot_test!(block_indentify(
        r#"
        |\Thing -> 42
        "#
    ));
}

#[test]
fn test_newline_and_spaces_before_less_than_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x =
        |    1
        |    < 2
        |
        |42
        "#
    ));
}

#[test]
fn test_outdented_record_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x = foo {
        |    bar: blah,
        |}
        |x
        "#
    ));
}

#[test]
fn test_minimal_app_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |app "test-app" provides [] to "./blah"
        |
        "#
    ));
}

#[test]
fn test_newline_after_paren_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(
        |A)
        "#
    ));
}

#[test]
fn test_unary_not_with_parens_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |!(whee 12 foo)
        "#
    ));
}

#[test]
fn test_when_with_tuple_in_record_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when { foo: (1, 2) } is
        |    { foo: (1, x) } -> x
        |    { foo: (_, b) } -> 3 + b
        "#
    ));
}

#[test]
fn test_unary_negation_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-foo
        "#
    ));
}

#[test]
fn test_type_signature_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |foo : Int
        |foo = 4
        |
        |42
        "#
    ));
}

#[test]
fn test_function_with_tuple_ext_type_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f : (Str)a -> (Str)a
        |f = \x -> x
        |
        |f ("Str", 42)
        |
        "#
    ));
}

#[test]
fn test_multi_backpassing_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x, y <- List.map2 [] []
        |
        |x + y
        |
        "#
    ));
}

#[test]
fn test_opaque_simple_moduledefs() {
    snapshot_test!(block_indentify(
        r#"
        |Age := U8
        |
        "#
    ));
}

#[test]
fn test_list_closing_same_indent_no_trailing_comma_expr() {
    snapshot_test!(block_indentify(
        r#"
        |myList = [
        |    0,
        |    1
        |]
        |42
        |
        "#
    ));
}

#[test]
fn test_call_with_newlines_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |f
        |    -5
        |    2
        "#
    ));
}

#[test]
fn test_list_closing_indent_not_enough_expr() {
    snapshot_test!(block_indentify(
        r#"
        |myList = [
        |    0,
        |    [
        |        a,
        |        b,
        |],
        |    1,
        |]
        |42
        |
        "#
    ));
}

// This test no longer works on the new parser; h needs to be indented
#[ignore]
#[test]
fn test_comment_after_tag_in_def_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |Z #
        |h
        | : a
        |j
        "#
    ));
}

#[test]
fn test_lambda_in_chain_expr() {
    snapshot_test!(block_indentify(
        r#"
        |"a string"
        ||> Str.toUtf8
        ||> List.map \byte -> byte + 1
        ||> List.reverse
        "#
    ));
}

#[test]
fn test_minus_twelve_minus_five_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |-12 - 5
        "#
    ));
}

#[test]
fn test_newline_in_type_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |R
        |:D
        |a
        "#
    ));
}

#[test]
fn test_tuple_type_ext_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f: (Str, Str)a -> (Str, Str)a
        |f = \x -> x
        |
        |f (1, 2)
        "#
    ));
}

#[test]
fn test_closure_with_underscores_expr() {
    snapshot_test!(block_indentify(
        r#"
        |\_, _name -> 42
        "#
    ));
}

#[test]
fn test_negative_in_apply_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a=A
        | -g a
        |a
        "#
    ));
}

#[test]
fn test_ann_open_union_expr() {
    snapshot_test!(block_indentify(
        r#"
        |foo : [True, Perhaps Thing]*
        |foo = True
        |
        |42
        "#
    ));
}

#[test]
fn test_newline_before_sub_expr() {
    snapshot_test!(block_indentify(
        r#"
        |3
        |- 4
        "#
    ));
}

#[test]
fn test_annotated_tag_destructure_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |UserId x : [UserId I64]
        |(UserId x) = UserId 42
        |
        |x
        "#
    ));
}

#[test]
fn test_unary_negation_arg_expr() {
    snapshot_test!(block_indentify(
        r#"
        |whee  12 -foo
        "#
    ));
}

#[test]
fn test_parenthetical_var_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |whee
        "#
    ));
}

#[test]
fn test_one_backpassing_expr() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |x <- (\y -> y)
        |
        |x
        |
        "#
    ));
}

#[test]
fn test_list_pattern_weird_indent_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when [] is
        |    [1, 2, 3] -> ""
        "#
    ));
}

#[test]
fn test_negative_float_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-42.9
        "#
    ));
}

#[test]
fn test_value_def_confusion_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |a : F
        |F : h
        |abc
        "#
    ));
}

#[test]
fn test_ability_demand_signature_is_multiline_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |Hash implements
        |    hash : a
        |        -> U64
        |
        |1
        "#
    ));
}

#[test]
fn test_opaque_with_type_arguments_moduledefs() {
    snapshot_test!(block_indentify(
        r#"
        |Bookmark a := { chapter: Str, stanza: Str, notes: a }
        |
        "#
    ));
}

#[test]
fn test_var_when_expr() {
    snapshot_test!(block_indentify(
        r#"
        |whenever
        "#
    ));
}

#[test]
fn test_function_effect_types_header() {
    snapshot_test!(block_indentify(
        r#"
        |platform "cli"
        |    requires {}{ main : Task {} [] } # TODO FIXME
        |    exposes []
        |    packages {}
        |    imports [ Task.{ Task } ]
        |    provides [ mainForHost ]
        |
        "#
    ));
}

#[test]
fn test_apply_unary_not_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |!whee 12 foo
        "#
    ));
}

#[test]
fn test_basic_apply_expr() {
    snapshot_test!(block_indentify(
        r#"
        |whee 1
        "#
    ));
}

#[test]
fn test_list_minus_newlines_expr() {
    snapshot_test!(block_indentify(
        r#"
        |[K,
        |]-i
        "#
    ));
}

#[test]
fn test_pattern_with_space_in_parens_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when Delmin (Del rx) 0 is
        |    Delmin (Del ry) _ -> Node Black 0 Bool.false ry
        "#
    ));
}

#[test]
fn test_list_closing_indent_not_enough_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |myList = [
        |    0,
        |    [
        |        a,
        |        b,
        |    ],
        |    1,
        |]
        |42
        "#
    ));
}

#[test]
fn test_underscore_backpassing_expr() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |_ <- (\y -> y)
        |
        |4
        |
        "#
    ));
}

#[test]
fn test_ann_closed_union_expr() {
    snapshot_test!(block_indentify(
        r#"
        |foo : [True, Perhaps Thing]
        |foo = True
        |
        |42
        "#
    ));
}

#[test]
fn test_outdented_record_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x = foo {
        |  bar: blah
        |}
        |x
        |
        "#
    ));
}

#[test]
fn test_record_destructure_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |{ x, y } = 5
        |y = 6
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_space_only_after_minus_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x- y
        "#
    ));
}

#[test]
fn test_minus_twelve_minus_five_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-12-5
        "#
    ));
}

#[test]
fn test_basic_field_expr() {
    snapshot_test!(block_indentify(
        r#"
        |rec.field
        "#
    ));
}

#[test]
fn test_add_var_with_spaces_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x + 2
        "#
    ));
}

#[test]
fn test_annotated_record_destructure_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |{ x, y } : Foo
        |{ x, y } = { x: "foo", y: 3.14 }
        |
        |x
        "#
    ));
}

#[test]
fn test_empty_string_expr() {
    snapshot_test!(block_indentify(
        r#"
        |""
        |
        "#
    ));
}

#[test]
fn test_where_ident_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |where : { where : I32 }
        |where = { where: 1 }
        |
        |where.where
        "#
    ));
}

#[test]
fn test_apply_parenthetical_tag_args_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Whee (12) (34)
        "#
    ));
}

#[test]
fn test_when_with_tuples_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when (1, 2) is
        |    (1, x) -> x
        |    (_, b) -> 3 + b
        "#
    ));
}

#[test]
fn test_comment_after_annotation_expr() {
    snapshot_test!(block_indentify(
        r#"
        |F:e#
        |
        |
        |q
        "#
    ));
}

#[test]
fn test_when_with_negative_numbers_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        | 1 -> 2
        | -3 -> 4
        |
        "#
    ));
}

#[test]
fn test_newline_in_type_def_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |R : D
        |a
        "#
    ));
}

#[test]
fn test_var_minus_two_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x - 2
        "#
    ));
}

#[test]
fn test_opaque_reference_expr_expr() {
    snapshot_test!(block_indentify(
        r#"
        |@Age
        |
        "#
    ));
}

#[test]
fn test_equals_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x==y
        "#
    ));
}

#[test]
fn test_var_is_expr() {
    snapshot_test!(block_indentify(
        r#"
        |isnt
        "#
    ));
}

#[test]
fn test_function_effect_types_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |platform "cli"
        |    requires {} { main : Task {} [] } # TODO FIXME
        |    exposes []
        |    packages {}
        |    imports [Task.{ Task }]
        |    provides [mainForHost]
        |
        "#
    ));
}

#[test]
fn test_list_pattern_weird_indent_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when [] is
        |    [1, 2,
        |3] -> ""
        |
        "#
    ));
}

#[test]
fn test_multiple_fields_expr() {
    snapshot_test!(block_indentify(
        r#"
        |rec.abc.def.ghi
        "#
    ));
}

#[test]
fn test_nonempty_hosted_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |hosted Foo
        |    exposes
        |        [
        |            Stuff,
        |            Things,
        |            somethingElse,
        |        ]
        |    imports
        |        [
        |            Blah,
        |            Baz.{ stuff, things },
        |        ]
        |    generates Bar with
        |        [
        |            map,
        |            after,
        |            loop,
        |        ]
        |
        "#
    ));
}

#[test]
fn test_basic_tag_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Whee
        "#
    ));
}

#[test]
#[ignore] // parens don't introduce a block, so this is no longer valid
fn test_newline_before_operator_with_defs_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |7
        |== (
        |    Q : c
        |    42
        |)
        "#
    ));
}

#[test]
fn test_requires_type_header() {
    snapshot_test!(block_indentify(
        r#"
        |platform "test/types"
        |    requires { Flags, Model, } { main : App Flags Model }
        |    exposes []
        |    packages {}
        |    imports []
        |    provides [ mainForHost ]
        |
        |mainForHost : App Flags Model
        |mainForHost = main
        |
        "#
    ));
}

#[test]
fn test_empty_hosted_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |hosted Foo exposes [] imports [] generates Bar with []
        |
        "#
    ));
}

#[test]
fn test_multiline_string_in_apply_expr() {
    snapshot_test!(block_indentify(
        r#"
        |e""""\""""
        "#
    ));
}

#[test]
fn test_empty_interface_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |interface Foo exposes [] imports []
        |
        "#
    ));
}

#[test]
fn test_pattern_as_spaces_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when 0 is
        |    0 # foobar
        |        as # barfoo
        |        n -> {}
        |
        "#
    ));
}

#[test]
fn test_where_clause_on_newline_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |f : a -> U64 where a implements Hash
        |
        |f
        "#
    ));
}

#[test]
fn test_tuple_type_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f: (Str, Str) -> (Str, Str)
        |f = \x -> x
        |
        |f (1, 2)
        "#
    ));
}

#[test]
fn test_underscore_in_assignment_pattern_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Pair x _ = Pair 0 1
        |Pair _ y = Pair 0 1
        |Pair _ _ = Pair 0 1
        |_ = Pair 0 1
        |Pair (Pair x _) (Pair _ y) = Pair (Pair 0 1) (Pair 2 3)
        |
        |0
        |
        "#
    ));
}

#[test]
fn test_multiline_type_signature_with_comment_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f :# comment
        |    {}
        |
        |42
        "#
    ));
}

#[test]
fn test_basic_docs_expr() {
    snapshot_test!(block_indentify(
        r#"
        |## first line of docs
        |##     second line
        |##  third line
        |## fourth line
        |##
        |## sixth line after doc new line
        |x = 5
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_lambda_indent_expr() {
    snapshot_test!(block_indentify(
        r#"
        |\x ->
        |  1
        "#
    ));
}

#[test]
fn test_space_only_after_minus_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x - y
        "#
    ));
}

#[test]
fn test_parse_alias_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Blah a b : Foo.Bar.Baz x y
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_two_arg_closure_expr() {
    snapshot_test!(block_indentify(
        r#"
        |\a, b -> 42
        "#
    ));
}

#[test]
fn test_opaque_type_def_with_newline_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |a : e
        |Na :=
        |    e
        |e0
        "#
    ));
}

#[test]
fn test_newline_in_type_alias_application_expr() {
    snapshot_test!(block_indentify(
        r#"
        |A:A
        | A
        |p
        "#
    ));
}

#[test]
fn test_two_spaced_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |x = 5
        |y = 6
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_record_func_type_decl_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |f : {
        |    getLine : Effect Str,
        |    putLine : Str -> Effect Int,
        |    text : Str,
        |    value : Int *,
        |}
        |
        |42
        "#
    ));
}

#[test]
fn test_comment_before_equals_def_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |t #
        | = 3
        |e
        "#
    ));
}

#[test]
fn test_apply_three_args_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a b c d
        "#
    ));
}

#[test]
fn test_module_def_newline_moduledefs_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |main =
        |    i = 64
        |
        |    i
        |
        "#
    ));
}

#[test]
fn test_var_if_expr() {
    snapshot_test!(block_indentify(
        r#"
        |iffy
        "#
    ));
}

#[test]
fn test_empty_interface_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |interface Foo exposes [] imports []
        |
        "#
    ));
}

#[test]
fn test_closure_in_binop_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |a
        |&& (\x -> x)
        |    8
        "#
    ));
}

#[test]
fn test_float_with_underscores_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-1_23_456.0_1_23_456
        "#
    ));
}

#[test]
fn test_nested_def_annotation_moduledefs() {
    snapshot_test!(block_indentify(
        r#"
        |main =
        |    wrappedNotEq : a, a -> Bool
        |    wrappedNotEq = \num1, num2 ->
        |        num1 != num2
        |
        |    wrappedNotEq 2 3
        |
        "#
    ));
}

#[test]
fn test_crash_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |_ = crash ""
        |_ = crash "" ""
        |_ = crash 15 123
        |_ = try foo (\_ -> crash "")
        |_ =
        |    _ = crash ""
        |    crash
        |
        |{ f: crash "" }
        "#
    ));
}

#[test]
fn test_parenthesized_type_def_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |D : b
        |a
        "#
    ));
}

#[test]
fn test_apply_tag_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Whee 12 34
        "#
    ));
}

#[test]
fn test_tuple_access_after_record_expr() {
    snapshot_test!(block_indentify(
        r#"
        |{ a: (1, 2) }.a.0
        "#
    ));
}

#[test]
fn test_positive_int_expr() {
    snapshot_test!(block_indentify(
        r#"
        |42
        "#
    ));
}

#[test]
fn test_parenthesized_type_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(D):b
        |a
        "#
    ));
}

#[test]
fn test_space_before_colon_full_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |app "example"
        |    packages { pf: "path" }
        |    imports [pf.Stdout]
        |    provides [main] to pf
        |
        |main = Stdout.line "Hello"
        |
        "#
    ));
}

#[test]
fn test_comment_after_def_moduledefs_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |foo = 1 # comment after
        |
        "#
    ));
}

#[test]
fn test_qualified_field_expr() {
    snapshot_test!(block_indentify(
        r#"
        |One.Two.rec.abc.def.ghi
        "#
    ));
}

#[test]
fn test_comment_before_equals_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |t#
        |=3
        |e
        "#
    ));
}

#[test]
fn test_empty_record_update_expr() {
    snapshot_test!(block_indentify(
        r#"
        |{e&}
        "#
    ));
}

#[test]
fn test_when_in_parens_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(when x is
        |    Ok ->
        |        3)
        |
        "#
    ));
}

#[test]
fn test_multi_backpassing_in_def_moduledefs_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |main =
        |    arg1, arg2 <- f {}
        |    "Roc <3 Zig!\n"
        |
        "#
    ));
}

#[test]
fn test_parens_in_value_def_annotation_expr() {
    snapshot_test!(block_indentify(
        r#"
        |i
        |(#
        |N):b
        |a
        "#
    ));
}

#[test]
fn test_record_update_expr() {
    snapshot_test!(block_indentify(
        r#"
        |{ Foo.Bar.baz & x: 5, y: 0 }
        "#
    ));
}

#[test]
fn test_newline_and_spaces_before_less_than_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x = 1
        |    < 2
        |
        |42
        "#
    ));
}

#[test]
fn test_underscore_in_assignment_pattern_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |(Pair x _) = Pair 0 1
        |(Pair _ y) = Pair 0 1
        |(Pair _ _) = Pair 0 1
        |_ = Pair 0 1
        |(Pair (Pair x _) (Pair _ y)) = Pair (Pair 0 1) (Pair 2 3)
        |
        |0
        "#
    ));
}

#[test]
fn test_parens_in_type_def_apply_expr() {
    snapshot_test!(block_indentify(
        r#"
        |U(b a):b
        |a
        "#
    ));
}

#[test]
fn test_newline_inside_empty_list_expr() {
    snapshot_test!(block_indentify(
        r#"
        |[
        |]
        "#
    ));
}

#[test]
fn test_function_with_tuple_type_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f : I64 -> (I64, I64)
        |f = \x -> (x, x + 1)
        |
        |f 42
        |
        "#
    ));
}

#[test]
fn test_control_characters_in_scalar_expr() {
    snapshot_test!(block_indentify(
        r#"
        |''
        "#
    ));
}

#[test]
fn test_multiple_operators_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |31 * 42 + 534
        "#
    ));
}

#[test]
fn test_nested_module_header() {
    snapshot_test!(block_indentify(
        r#"
        |interface Foo.Bar.Baz exposes [] imports []
        |
        "#
    ));
}

#[test]
fn test_multiple_operators_expr() {
    snapshot_test!(block_indentify(
        r#"
        |31*42+534
        "#
    ));
}

#[test]
fn test_call_with_newlines_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f
        |-5
        |2
        "#
    ));
}

#[test]
fn test_newline_singleton_list_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |[
        |    1,
        |]
        "#
    ));
}

#[test]
fn test_basic_var_expr() {
    snapshot_test!(block_indentify(
        r#"
        |whee
        "#
    ));
}

#[test]
fn test_tuple_type_ext_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |f : (Str, Str)a -> (Str, Str)a
        |f = \x -> x
        |
        |f (1, 2)
        "#
    ));
}

#[test]
fn test_full_app_header_trailing_commas_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |app "quicksort"
        |    packages { pf: "./platform" }
        |    imports [foo.Bar.{
        |        Baz,
        |        FortyTwo,
        |        # I'm a happy comment
        |    }]
        |    provides [quicksort] to pf
        |
        "#
    ));
}

#[test]
fn test_negate_multiline_string_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |-(
        |    """
        |    """)
        "#
    ));
}

#[test]
#[ignore]
fn test_comment_before_colon_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |w#
        |:n
        |Q
        "#
    ));
}

#[test]
fn test_provides_type_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |app "test"
        |    packages { pf: "./platform" }
        |    imports [foo.Bar.Baz]
        |    provides [quicksort] { Flags, Model } to pf
        |
        "#
    ));
}

#[test]
fn test_outdented_colon_in_record_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x = foo {
        |    bar
        |    : blah,
        |}
        |x
        "#
    ));
}

#[test]
fn test_opaque_destructure_first_item_in_body_expr() {
    snapshot_test!(block_indentify(
        r#"
        |@Thunk it = id (@A {})
        |it {}
        |
        "#
    ));
}

#[test]
fn test_annotated_tuple_destructure_expr() {
    snapshot_test!(block_indentify(
        r#"
        |( x, y ) : Foo
        |( x, y ) = ( "foo", 3.14 )
        |
        |x
        |
        "#
    ));
}

#[test]
fn test_record_func_type_decl_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f :
        |    {
        |        getLine : Effect Str,
        |        putLine : Str -> Effect Int,
        |        text: Str,
        |        value: Int *
        |    }
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_requires_type_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |platform "test/types"
        |    requires { Flags, Model } { main : App Flags Model }
        |    exposes []
        |    packages {}
        |    imports []
        |    provides [mainForHost]
        |
        "#
    ));
}

#[test]
fn test_comment_before_colon_def_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |w #
        | : n
        |Q
        "#
    ));
}

#[test]
fn test_pattern_as_list_rest_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when myList is
        |    [first, .. as rest] -> 0
        |
        "#
    ));
}

#[test]
fn test_when_with_numbers_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    1 -> 2
        |    3 -> 4
        "#
    ));
}

#[test]
fn test_equals_with_spaces_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x == y
        "#
    ));
}

#[test]
fn test_nested_def_annotation_moduledefs_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |main =
        |    wrappedNotEq : a, a -> Bool
        |    wrappedNotEq = \num1, num2 ->
        |        num1 != num2
        |
        |    wrappedNotEq 2 3
        |
        "#
    ));
}

#[test]
fn test_nonempty_hosted_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |hosted Foo
        |    exposes
        |    [
        |        Stuff,
        |        Things,
        |        somethingElse,
        |    ]
        |    imports
        |    [
        |        Blah,
        |        Baz.{ stuff, things },
        |    ]
        |    generates Bar with
        |    [
        |        map,
        |        after,
        |        loop,
        |    ]
        |
        "#
    ));
}

#[test]
fn test_sub_with_spaces_expr() {
    snapshot_test!(block_indentify(
        r#"
        |1  -   2
        "#
    ));
}

#[test]
fn test_highest_float_expr() {
    snapshot_test!(block_indentify(
        r#"
        |179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0
        "#
    ));
}

#[test]
fn test_comment_with_non_ascii_expr() {
    snapshot_test!(block_indentify(
        r#"
        |3  # 2 × 2
        |+ 4
        "#
    ));
}

#[test]
fn test_newline_after_mul_expr() {
    snapshot_test!(block_indentify(
        r#"
        |3  *
        |  4
        "#
    ));
}

#[test]
#[ignore] // not supported anymore
fn test_nested_def_without_newline_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x=a:n 4
        |_
        "#
    ));
}

#[test]
fn test_one_backpassing_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |x <- \y -> y
        |
        |x
        "#
    ));
}

#[test]
fn test_outdented_app_with_record_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x = foo (baz {
        |  bar: blah
        |})
        |x
        |
        "#
    ));
}

#[test]
fn test_when_with_alternative_patterns_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    "blah" | "blop" -> 1
        |    "foo"
        |    | "bar"
        |    | "baz" -> 2
        |
        |    "stuff" -> 4
        "#
    ));
}

#[test]
fn test_comment_after_def_moduledefs() {
    snapshot_test!(block_indentify(
        r#"
        |foo = 1 # comment after
        |
        "#
    ));
}

#[test]
fn test_interface_with_newline_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |interface T exposes [] imports []
        |
        "#
    ));
}

#[test]
fn test_when_with_function_application_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    1 -> Num.neg
        |     2
        |    _ -> 4
        |
        "#
    ));
}

#[test]
fn test_zero_float_expr() {
    snapshot_test!(block_indentify(
        r#"
        |0.0
        "#
    ));
}

#[test]
fn test_opaque_simple_moduledefs_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |Age := U8
        |
        "#
    ));
}

#[test]
fn test_opaque_destructure_first_item_in_body_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |(@Thunk it) = id (@A {})
        |it {}
        "#
    ));
}

#[test]
fn test_when_in_parens_indented_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    Ok -> 3
        "#
    ));
}

#[test]
fn test_when_with_weird_trailing_if() {
    snapshot_test!(block_indentify(
        r#"
        |p=when d is
        |F->s\a->""
        |_->""if e then
        |e
        |else
        |e
        "#
    ));
}

#[test]
fn test_when_upper_bar() {
    snapshot_test!(block_indentify(
        r#"
        |c=when e is
        |                U|R->f
        |e
        "#
    ));
}

#[test]
fn test_when_field_access() {
    snapshot_test!(block_indentify(
        r#"
        |p=when s is H->s .d
        |e
        "#
    ));
}

#[test]
fn test_empty_app_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |app "test-app" packages {} imports [] provides [] to blah
        |
        "#
    ));
}

#[test]
fn test_extra_newline_in_parens_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |B : {}
        |
        |a
        "#
    ));
}

#[test]
fn test_highest_int_expr() {
    snapshot_test!(block_indentify(
        r#"
        |9223372036854775807
        "#
    ));
}

#[test]
fn test_str_block_multiple_newlines_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |"""
        |
        |
        |#
        |""" #
        "#
    ));
}

#[test]
fn test_apply_unary_negation_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-whee  12 foo
        "#
    ));
}

#[test]
fn test_nonempty_platform_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |platform "foo/barbaz"
        |    requires {Model} { main : {} }
        |    exposes []
        |    packages { foo: "./foo" }
        |    imports []
        |    provides [ mainForHost ]
        |
        "#
    ));
}

#[test]
fn test_multiline_type_signature_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f :
        |    {}
        |
        |42
        "#
    ));
}

#[test]
fn test_if_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |iffy=5
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_one_plus_two_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |1 + 2
        "#
    ));
}

#[test]
fn test_dbg_expr() {
    snapshot_test!(block_indentify(
        r#"
        |dbg 1 == 1
        |
        |4
        |
        "#
    ));
}

#[test]
#[ignore] // parens don't introduce a block, so this is no longer valid
fn test_newline_before_operator_with_defs_expr() {
    snapshot_test!(block_indentify(
        r#"
        |7
        |==(Q:c 42)
        "#
    ));
}

#[test]
fn test_int_with_underscore_expr() {
    snapshot_test!(block_indentify(
        r#"
        |1__23
        "#
    ));
}

#[test]
fn test_bound_variable_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a:
        |c 0
        "#
    ));
}

#[test]
fn test_when_in_function_python_style_indent_expr() {
    snapshot_test!(block_indentify(
        r#"
        |func = \x -> when n is
        |    0 -> 0
        |42
        "#
    ));
}

#[test]
fn test_ability_single_line_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Hash implements hash : a -> U64 where a implements Hash
        |
        |1
        |
        "#
    ));
}

#[test]
fn test_add_with_spaces_expr() {
    snapshot_test!(block_indentify(
        r#"
        |1  +   2
        "#
    ));
}

#[test]
fn test_comment_after_op_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |12
        |* # test!
        |92
        "#
    ));
}

#[test]
fn test_spaces_inside_empty_list_expr() {
    snapshot_test!(block_indentify(
        r#"
        |[  ]
        "#
    ));
}

#[test]
fn test_spaces_inside_empty_list_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |[]
        "#
    ));
}

#[test]
fn test_comment_after_expr_in_parens_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(i#abc
        |)
        "#
    ));
}

#[test]
fn test_unary_negation_with_parens_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |-(whee 12 foo)
        "#
    ));
}

#[test]
fn test_where_clause_multiple_has_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f : a -> (b -> c) where a implements A, b implements Eq, c implements Ord
        |
        |f
        |
        "#
    ));
}

#[test]
fn test_empty_record_expr() {
    snapshot_test!(block_indentify(
        r#"
        |{}
        "#
    ));
}

#[test]
fn test_where_clause_multiple_has_across_newlines_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |f : a -> (b -> c) where a implements Hash, b implements Eq, c implements Ord
        |
        |f
        "#
    ));
}

#[test]
fn test_var_then_expr() {
    snapshot_test!(block_indentify(
        r#"
        |thenever
        "#
    ));
}

#[test]
fn test_parenthetical_basic_field_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(rec).field
        "#
    ));
}

#[test]
fn test_crash_expr() {
    snapshot_test!(block_indentify(
        r#"
        |_ = crash ""
        |_ = crash "" ""
        |_ = crash 15 123
        |_ = try foo (\_ -> crash "")
        |_ =
        |  _ = crash ""
        |  crash
        |
        |{ f: crash "" }
        |
        "#
    ));
}

#[test]
fn test_opaque_reference_pattern_with_arguments_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when n is
        |    @Add n m -> n + m
        "#
    ));
}

#[test]
fn test_full_app_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |app "quicksort"
        |    packages { pf: "./platform" }
        |    imports [ foo.Bar.Baz ]
        |    provides [ quicksort ] to pf
        |
        "#
    ));
}

#[test]
fn test_empty_platform_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |platform "rtfeldman/blah" requires {} { main : {} } exposes [] packages {} imports [] provides []
        |
        "#
    ));
}

#[test]
fn test_expect_fx_moduledefs() {
    snapshot_test!(block_indentify(
        r#"
        |# expecting some effects
        |expect-fx 5 == 2
        |
        "#
    ));
}

#[test]
fn test_unary_negation_access_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-rec1.field
        "#
    ));
}

#[test]
fn test_pattern_as_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when 0 is
        |    _ as n -> n
        |
        "#
    ));
}

#[test]
fn test_pattern_with_space_in_parens_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when Delmin (Del rx) 0 is
        |    Delmin (Del ry ) _ -> Node Black 0 Bool.false ry
        |
        "#
    ));
}

#[test]
fn test_def_without_newline_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a:b i
        "#
    ));
}

#[test]
fn test_multiline_string_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |a = "Hello,\n\nWorld!"
        |b =
        |    """
        |    Hello,\n\nWorld!
        |    """
        |c =
        |    """
        |    Hello,
        |
        |    World!
        |    """
        |42
        "#
    ));
}

#[test]
fn test_empty_app_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |app "test-app" packages {} imports [] provides [] to blah
        |
        "#
    ));
}

#[test]
fn test_closure_in_binop_with_spaces_expr() {
    snapshot_test!(block_indentify(
        r#"
        |i>\s->s
        |-a
        "#
    ));
}

#[test]
fn test_when_with_negative_numbers_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    1 -> 2
        |    -3 -> 4
        "#
    ));
}

#[test]
fn test_newline_before_add_expr() {
    snapshot_test!(block_indentify(
        r#"
        |3
        |+ 4
        "#
    ));
}

#[test]
fn test_one_minus_two_expr() {
    snapshot_test!(block_indentify(
        r#"
        |1-2
        "#
    ));
}

#[test]
fn test_lambda_indent_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |\x ->
        |    1
        "#
    ));
}

#[test]
fn test_tuple_accessor_function_expr() {
    snapshot_test!(block_indentify(
        r#"
        |.1 (1, 2, 3)
        "#
    ));
}

#[test]
fn test_sub_var_with_spaces_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x - 2
        "#
    ));
}

#[test]
fn test_destructure_tag_assignment_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |(Email str) = Email "blah@example.com"
        |str
        "#
    ));
}

#[test]
fn test_ops_with_newlines_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |3
        |+
        |
        |4
        "#
    ));
}

#[test]
fn test_comment_inside_empty_list_expr() {
    snapshot_test!(block_indentify(
        r#"
        |[#comment
        |]
        "#
    ));
}

#[test]
fn test_outdented_colon_in_record_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x = foo {
        |bar
        |:
        |blah
        |}
        |x
        |
        "#
    ));
}

#[test]
fn test_underscore_backpassing_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |_ <- \y -> y
        |
        |4
        "#
    ));
}

#[test]
fn test_where_clause_multiple_has_across_newlines_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f : a -> (b -> c)
        |    where a implements Hash,
        |      b implements Eq,
        |      c implements Ord
        |
        |f
        |
        "#
    ));
}

#[test]
fn test_when_in_function_expr() {
    snapshot_test!(block_indentify(
        r#"
        |func = \x -> when n is
        |              0 -> 0
        |42
        "#
    ));
}

#[test]
fn test_spaced_singleton_list_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |[1]
        "#
    ));
}

#[test]
fn test_mixed_docs_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |# ## not docs!
        |## docs, but with a problem
        |## (namely that this is a mix of docs and regular comments)
        |# not docs
        |x = 5
        |
        |42
        "#
    ));
}

#[test]
fn test_parenthetical_field_qualified_var_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(One.Two.rec).field
        "#
    ));
}

#[test]
fn test_one_minus_two_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |1 - 2
        "#
    ));
}

#[test]
fn test_newline_singleton_list_expr() {
    snapshot_test!(block_indentify(
        r#"
        |[
        |1
        |]
        "#
    ));
}

#[test]
fn test_annotated_record_destructure_expr() {
    snapshot_test!(block_indentify(
        r#"
        |{ x, y } : Foo
        |{ x, y } = { x : "foo", y : 3.14 }
        |
        |x
        |
        "#
    ));
}

#[test]
fn test_mixed_docs_expr() {
    snapshot_test!(block_indentify(
        r#"
        |### not docs!
        |## docs, but with a problem
        |## (namely that this is a mix of docs and regular comments)
        |# not docs
        |x = 5
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_empty_package_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |package "rtfeldman/blah" exposes [] packages {}
        "#
    ));
}

#[test]
fn test_bound_variable_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |a :
        |    c
        |0
        "#
    ));
}

#[test]
fn test_nonempty_platform_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |platform "foo/barbaz"
        |    requires { Model } { main : {} }
        |    exposes []
        |    packages { foo: "./foo" }
        |    imports []
        |    provides [mainForHost]
        |
        "#
    ));
}

#[test]
fn test_one_def_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |x = 5
        |
        |42
        "#
    ));
}

#[test]
fn test_when_with_records_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        | { y } -> 2
        | { z, w } -> 4
        |
        "#
    ));
}

#[test]
fn test_var_minus_two_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x-2
        "#
    ));
}

#[test]
fn test_unary_negation_arg_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |whee 12 -foo
        "#
    ));
}

#[test]
fn test_nonempty_package_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |package "foo/barbaz"
        |    exposes [Foo, Bar]
        |    packages { foo: "./foo" }
        "#
    ));
}

#[test]
fn test_apply_two_args_expr() {
    snapshot_test!(block_indentify(
        r#"
        |whee  12  34
        "#
    ));
}

#[test]
fn test_sub_with_spaces_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |1 - 2
        "#
    ));
}

#[test]
fn test_where_ident_expr() {
    snapshot_test!(block_indentify(
        r#"
        |where : {where: I32}
        |where = {where: 1}
        |
        |where.where
        |
        "#
    ));
}

#[test]
fn test_negate_multiline_string_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-""""""
        "#
    ));
}

#[test]
fn test_pos_inf_float_expr() {
    snapshot_test!(block_indentify(
        r#"
        |inf
        "#
    ));
}

#[test]
fn test_when_in_parens_indented_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(when x is
        |    Ok -> 3
        |     )
        |
        "#
    ));
}

#[test]
fn test_multi_backpassing_with_apply_expr() {
    snapshot_test!(block_indentify(
        r#"
        |F 1, r <- a
        |W
        "#
    ));
}

#[test]
fn test_apply_two_args_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |whee 12 34
        "#
    ));
}

#[test]
fn test_annotated_tuple_destructure_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |(x, y) : Foo
        |(x, y) = ("foo", 3.14)
        |
        |x
        "#
    ));
}

#[test]
fn test_standalone_module_defs_moduledefs_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |# comment 1
        |foo = 1
        |
        |# comment 2
        |bar = "hi"
        |baz = "stuff"
        |# comment n
        |
        "#
    ));
}

#[test]
fn test_fn_with_record_arg_expr() {
    snapshot_test!(block_indentify(
        r#"
        |table : {
        |    height : Pixels
        |    } -> Table
        |table = \{height} -> crash "not implemented"
        |table
        "#
    ));
}

#[test]
fn test_comment_after_annotation_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |F : e #
        |
        |q
        "#
    ));
}

#[test]
fn test_space_before_colon_full() {
    snapshot_test!(block_indentify(
        r#"
        |app "example"
        |    packages { pf : "path" }
        |    imports [ pf.Stdout ]
        |    provides [ main ] to pf
        |
        |main = Stdout.line "Hello"
        "#
    ));
}

#[test]
fn test_positive_float_expr() {
    snapshot_test!(block_indentify(
        r#"
        |42.9
        "#
    ));
}

#[test]
fn test_expect_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |expect
        |    1 == 1
        |
        |4
        "#
    ));
}

#[test]
fn test_record_type_with_function_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x : { init : {} -> Model, update : Model, Str -> Model, view : Model -> Str }
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_value_def_confusion_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a:F
        |F
        |:h
        |abc
        "#
    ));
}

#[test]
fn test_negate_multiline_string_with_quote_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-""""<"""
        "#
    ));
}

#[test]
fn test_single_arg_closure_expr() {
    snapshot_test!(block_indentify(
        r#"
        |\a -> 42
        "#
    ));
}

#[test]
fn test_where_clause_function_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f : a -> (b -> c) where a implements A
        |
        |f
        |
        "#
    ));
}

#[test]
fn test_empty_list_expr() {
    snapshot_test!(block_indentify(
        r#"
        |[]
        "#
    ));
}

#[test]
fn test_number_literal_suffixes_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |{
        |    u8: 123u8,
        |    u16: 123u16,
        |    u32: 123u32,
        |    u64: 123u64,
        |    u128: 123u128,
        |    i8: 123i8,
        |    i16: 123i16,
        |    i32: 123i32,
        |    i64: 123i64,
        |    i128: 123i128,
        |    nat: 123nat,
        |    dec: 123dec,
        |    u8Neg: -123u8,
        |    u16Neg: -123u16,
        |    u32Neg: -123u32,
        |    u64Neg: -123u64,
        |    u128Neg: -123u128,
        |    i8Neg: -123i8,
        |    i16Neg: -123i16,
        |    i32Neg: -123i32,
        |    i64Neg: -123i64,
        |    i128Neg: -123i128,
        |    natNeg: -123nat,
        |    decNeg: -123dec,
        |    u8Bin: 0b101u8,
        |    u16Bin: 0b101u16,
        |    u32Bin: 0b101u32,
        |    u64Bin: 0b101u64,
        |    u128Bin: 0b101u128,
        |    i8Bin: 0b101i8,
        |    i16Bin: 0b101i16,
        |    i32Bin: 0b101i32,
        |    i64Bin: 0b101i64,
        |    i128Bin: 0b101i128,
        |    natBin: 0b101nat,
        |}
        "#
    ));
}

#[test]
fn test_comment_before_op_expr() {
    snapshot_test!(block_indentify(
        r#"
        |3  # test!
        |+ 4
        "#
    ));
}

#[test]
fn test_nested_if_expr() {
    snapshot_test!(block_indentify(
        r#"
        |if t1 then
        |  1
        |else if t2 then
        |  2
        |else
        |  3
        |
        "#
    ));
}

#[test]
fn test_newline_after_sub_expr() {
    snapshot_test!(block_indentify(
        r#"
        |3  -
        |  4
        "#
    ));
}

#[test]
fn test_fn_with_record_arg_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |table :
        |    {
        |        height : Pixels,
        |    }
        |    -> Table
        |table = \{ height } -> crash "not implemented"
        |table
        "#
    ));
}

#[test]
fn test_ability_demand_signature_is_multiline_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Hash implements
        |  hash : a
        |         -> U64
        |
        |1
        |
        "#
    ));
}

#[test]
fn test_annotated_tag_destructure_expr() {
    snapshot_test!(block_indentify(
        r#"
        |UserId x : [ UserId I64 ]
        |UserId x = UserId 42
        |
        |x
        |
        "#
    ));
}

#[test]
fn test_negative_int_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-42
        "#
    ));
}

#[test]
fn test_multiline_tuple_with_comments_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(
        |    #before 1
        |    1
        |    #after 1
        |    ,
        |    #before 2
        |    2
        |    #after 2
        |    ,
        |    #before 3
        |    3
        |    # after 3
        |)
        "#
    ));
}

#[test]
fn test_when_with_tuples_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when (1, 2) is
        | (1, x) -> x
        | (_, b) -> 3 + b
        |
        "#
    ));
}

#[test]
fn test_opaque_has_abilities_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |A := U8 implements [Eq, Hash]
        |
        |A := a where a implements Other
        |    implements [Eq, Hash]
        |
        |A := a where a implements Other
        |    implements [Eq, Hash]
        |
        |A := U8 implements [Eq { eq }, Hash { hash }]
        |
        |A := U8 implements [Eq { eq, eq1 }]
        |
        |A := U8 implements [Eq { eq, eq1 }, Hash]
        |
        |A := U8 implements [Hash, Eq { eq, eq1 }]
        |
        |A := U8 implements []
        |
        |A := a where a implements Other
        |    implements [Eq { eq }, Hash { hash }]
        |
        |A := U8 implements [Eq {}]
        |
        |0
        "#
    ));
}

#[test]
fn test_negative_in_apply_def_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |a = A
        |    -g
        |    a
        |a
        "#
    ));
}

#[test]
fn test_lowest_int_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-9223372036854775808
        "#
    ));
}

#[test]
fn test_nested_if_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |if t1 then
        |    1
        |else if t2 then
        |    2
        |else
        |    3
        "#
    ));
}

#[test]
fn test_opaque_reference_expr_with_arguments_expr() {
    snapshot_test!(block_indentify(
        r#"
        |@Age m n
        |
        "#
    ));
}

#[test]
fn test_unary_not_expr() {
    snapshot_test!(block_indentify(
        r#"
        |!blah
        "#
    ));
}

#[test]
fn test_not_multiline_string_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |!
        |"""
        |"""
        "#
    ));
}

#[test]
fn test_expect_fx_moduledefs_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |# expecting some effects
        |expect-fx 5 == 2
        |
        "#
    ));
}

#[test]
fn test_packed_singleton_list_expr() {
    snapshot_test!(block_indentify(
        r#"
        |[1]
        "#
    ));
}

#[test]
fn test_parenthetical_apply_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(whee) 1
        "#
    ));
}

#[test]
fn test_dbg_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |dbg
        |    1 == 1
        |
        |4
        "#
    ));
}

#[test]
fn test_not_multiline_string_expr() {
    snapshot_test!(block_indentify(
        r#"
        |!""""""
        "#
    ));
}

#[test]
fn test_def_without_newline_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |a : b
        |i
        "#
    ));
}

// This test no longer works on the new parser; h needs to be indented
#[ignore]
#[test]
fn test_comment_after_tag_in_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Z#
        |h
        |:a
        |j
        "#
    ));
}

#[test]
fn test_ten_times_eleven_expr() {
    snapshot_test!(block_indentify(
        r#"
        |10*11
        "#
    ));
}

#[test]
fn test_multiline_type_signature_with_comment_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |f :
        |    # comment
        |    {}
        |
        |42
        "#
    ));
}

#[test]
fn test_apply_unary_not_expr() {
    snapshot_test!(block_indentify(
        r#"
        |!whee  12 foo
        "#
    ));
}

#[test]
fn test_nonempty_package_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |package "foo/barbaz"
        |    exposes [Foo, Bar]
        |    packages { foo: "./foo" }
        |
        "#
    ));
}

#[test]
fn test_empty_record_update_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |{ e &  }
        "#
    ));
}

#[test]
fn test_spaced_singleton_list_expr() {
    snapshot_test!(block_indentify(
        r#"
        |[ 1 ]
        "#
    ));
}

#[test]
fn test_ability_multi_line_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Hash implements
        |  hash : a -> U64
        |  hash2 : a -> U64
        |
        |1
        |
        "#
    ));
}

#[test]
fn test_opaque_type_def_with_newline_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a:e
        |Na:=
        | e e0
        "#
    ));
}

#[test]
fn test_parse_as_ann_expr() {
    snapshot_test!(block_indentify(
        r#"
        |foo : Foo.Bar.Baz x y as Blah a b
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_when_with_tuple_in_record_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when {foo: (1, 2)} is
        | {foo: (1, x)} -> x
        | {foo: (_, b)} -> 3 + b
        |
        "#
    ));
}

#[test]
fn test_parens_in_value_def_annotation_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |i #
        |N : b
        |a
        "#
    ));
}

#[test]
fn test_zero_int_expr() {
    snapshot_test!(block_indentify(
        r#"
        |0
        "#
    ));
}

#[test]
fn test_plus_when_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |1
        |+
        |when Foo is
        |    Foo -> 2
        |    Bar -> 3
        "#
    ));
}

#[test]
fn test_if_def_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |iffy = 5
        |
        |42
        "#
    ));
}

#[test]
fn test_add_with_spaces_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |1 + 2
        "#
    ));
}

#[test]
fn test_comment_with_non_ascii_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |3 # 2 × 2
        |+ 4
        "#
    ));
}

#[test]
fn test_full_app_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |app "quicksort"
        |    packages { pf: "./platform" }
        |    imports [foo.Bar.Baz]
        |    provides [quicksort] to pf
        |
        "#
    ));
}

#[test]
fn test_not_docs_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |# ######
        |# ## not docs!
        |# #still not docs
        |# #####
        |x = 5
        |
        |42
        "#
    ));
}

#[test]
fn test_three_arg_closure_expr() {
    snapshot_test!(block_indentify(
        r#"
        |\a, b, c -> 42
        "#
    ));
}

#[test]
fn test_when_in_function_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |func = \x ->
        |    when n is
        |        0 -> 0
        |42
        "#
    ));
}

#[test]
fn test_list_closing_same_indent_with_trailing_comma_expr() {
    snapshot_test!(block_indentify(
        r#"
        |myList = [
        |    0,
        |    1,
        |]
        |42
        |
        "#
    ));
}

#[test]
fn test_extra_newline_in_parens_expr() {
    snapshot_test!(block_indentify(
        r#"
        |B:{}
        |
        |(
        |a)
        "#
    ));
}

#[test]
fn test_newline_after_equals_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x =
        |    5
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_unary_not_with_parens_expr() {
    snapshot_test!(block_indentify(
        r#"
        |!(whee  12 foo)
        "#
    ));
}

#[test]
fn test_two_branch_when_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        | "" -> 1
        | "mise" -> 2
        |
        "#
    ));
}

#[test]
fn test_string_without_escape_expr() {
    snapshot_test!(block_indentify(
        r#"
        |"123 abc 456 def"
        "#
    ));
}

#[test]
fn test_multi_backpassing_with_apply_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |(F 1), r <- a
        |W
        "#
    ));
}

#[test]
fn test_multiline_tuple_with_comments_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |(
        |    # before 1
        |    1,
        |    # after 1
        |    # before 2
        |    2,
        |    # after 2
        |    # before 3
        |    3,
        |    # after 3
        |)
        "#
    ));
}

#[test]
fn test_list_patterns_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when [] is
        |  [] -> {}
        |  [..] -> {}
        |  [_, .., _, ..] -> {}
        |  [a, b, c, d] -> {}
        |  [a, b, ..] -> {}
        |  [.., c, d] -> {}
        |  [[A], [..], [a]] -> {}
        |  [[[], []], [[], x]] -> {}
        |
        "#
    ));
}

#[test]
fn test_when_in_parens_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    Ok ->
        |        3
        "#
    ));
}

#[test]
fn test_unary_negation_with_parens_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-(whee  12 foo)
        "#
    ));
}

#[test]
fn test_record_access_after_tuple_expr() {
    snapshot_test!(block_indentify(
        r#"
        |({a: 0}, {b: 1}).0.a
        "#
    ));
}

#[test]
fn test_outdented_list_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a = [
        |  1, 2, 3
        |]
        |a
        |
        "#
    ));
}

#[test]
fn test_multi_backpassing_in_def_moduledefs() {
    snapshot_test!(block_indentify(
        r#"
        |main =
        |    arg1, arg2 <- f {}
        |    "Roc <3 Zig!\n"
        |
        "#
    ));
}

#[test]
fn test_plus_if_expr() {
    snapshot_test!(block_indentify(
        r#"
        |1 * if Bool.true then 1 else 1
        |
        "#
    ));
}

#[test]
fn test_closure_in_binop_with_spaces_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |i
        |> (\s -> s
        |)
        |    -a
        "#
    ));
}

#[test]
fn test_when_if_guard_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    _ ->
        |        1
        |
        |    _ ->
        |        2
        |
        |    Ok ->
        |        3
        |
        "#
    ));
}

#[test]
fn test_opaque_reference_pattern_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when n is
        |  @Age -> 1
        |
        "#
    ));
}

#[test]
fn test_basic_tuple_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(1, 2, 3)
        "#
    ));
}

#[test]
fn test_standalone_module_defs_moduledefs() {
    snapshot_test!(block_indentify(
        r#"
        |# comment 1
        |foo = 1
        |
        |# comment 2
        |bar = "hi"
        |baz = "stuff"
        |# comment n
        |
        "#
    ));
}

#[test]
fn test_newline_in_packages_full_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |app "hello"
        |    packages {
        |        pf:
        |        "https://github.com/roc-lang/basic-cli/releases/download/0.7.0/bkGby8jb0tmZYsy2hg1E_B2QrCgcSTxdUlHtETwm5m4.tar.br",
        |    }
        |    imports [pf.Stdout]
        |    provides [main] to pf
        |
        |main =
        |    Stdout.line "I'm a Roc application!"
        |
        "#
    ));
}

#[test]
fn test_opaque_reference_pattern_with_arguments_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when n is
        |  @Add n m -> n + m
        |
        "#
    ));
}

#[test]
fn test_tuple_access_after_ident_expr() {
    snapshot_test!(block_indentify(
        r#"
        |abc = (1, 2, 3)
        |abc.0
        "#
    ));
}

#[test]
fn test_when_with_records_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    { y } -> 2
        |    { z, w } -> 4
        "#
    ));
}

#[test]
fn test_neg_inf_float_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-inf
        "#
    ));
}

#[test]
fn test_var_else_expr() {
    snapshot_test!(block_indentify(
        r#"
        |elsewhere
        "#
    ));
}

#[test]
fn test_type_decl_with_underscore_expr() {
    snapshot_test!(block_indentify(
        r#"
        |doStuff : UserId -> Task Str _
        |42
        "#
    ));
}

#[test]
fn test_equals_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x == y
        "#
    ));
}

#[test]
fn test_one_spaced_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |x = 5
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_apply_unary_negation_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |-whee 12 foo
        "#
    ));
}

#[test]
fn test_type_signature_function_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |foo : Int, Float -> Bool
        |foo = \x, _ -> 42
        |
        |42
        "#
    ));
}

#[test]
fn test_qualified_var_expr() {
    snapshot_test!(block_indentify(
        r#"
        |One.Two.whee
        "#
    ));
}

#[test]
fn test_nested_backpassing_no_newline_before_expr() {
    snapshot_test!(block_indentify(
        r#"
        |main =
        |    task = file <-
        |                foo
        |            bar
        |    task
        |42
        "#
    ));
}

#[test]
fn test_record_with_if_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |{ x: if Bool.true then 1 else 2, y: 3 }
        "#
    ));
}

#[test]
fn test_lowest_float_expr() {
    snapshot_test!(block_indentify(
        r#"
        |-179769313486231570000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0
        "#
    ));
}

#[test]
fn test_destructure_tag_assignment_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Email str = Email "blah@example.com"
        |str
        |
        "#
    ));
}

#[test]
fn test_not_docs_expr() {
    snapshot_test!(block_indentify(
        r#"
        |#######
        |### not docs!
        |##still not docs
        |######
        |x = 5
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_plus_when_expr() {
    snapshot_test!(block_indentify(
        r#"
        |1 +
        |    when Foo is
        |        Foo -> 2
        |        Bar -> 3
        |
        "#
    ));
}

#[test]
fn test_closure_in_binop_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a && \x->x
        |8
        "#
    ));
}

#[test]
fn test_multiline_string_in_apply_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |e
        |    """
        |    "\"
        |    """
        "#
    ));
}

#[test]
fn test_comment_inside_empty_list_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |[ # comment
        |]
        "#
    ));
}

#[test]
fn test_full_app_header_trailing_commas_header() {
    snapshot_test!(block_indentify(
        r#"
        |app "quicksort"
        |    packages { pf: "./platform", }
        |    imports [ foo.Bar.{
        |        Baz,
        |        FortyTwo,
        |        # I'm a happy comment
        |    } ]
        |    provides [ quicksort, ] to pf
        |
        "#
    ));
}

#[test]
fn test_two_backpassing_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |x <- \y -> y
        |z <- {}
        |
        |x
        "#
    ));
}

#[test]
fn test_newline_in_type_alias_application_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |A : A
        |    A
        |p
        "#
    ));
}

#[test]
fn test_parenthesized_type_def_space_before_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(
        |A):b
        |a
        "#
    ));
}

#[test]
fn test_when_with_alternative_patterns_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        | "blah" | "blop" -> 1
        | "foo" |
        |  "bar"
        | |"baz" -> 2
        | "stuff" -> 4
        |
        "#
    ));
}

#[test]
fn test_negate_multiline_string_with_quote_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |-(
        |    """
        |    "<
        |    """)
        "#
    ));
}

#[test]
fn test_control_characters_in_scalar_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |'\u(7)'
        "#
    ));
}

#[test]
fn test_ten_times_eleven_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |10 * 11
        "#
    ));
}

#[test]
fn test_minimal_app_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |app "test-app" provides [] to "./blah"
        |
        "#
    ));
}

#[test]
fn test_nested_module_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |interface Foo.Bar.Baz exposes [] imports []
        |
        "#
    ));
}

#[test]
fn test_when_in_function_python_style_indent_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |func = \x ->
        |    when n is
        |        0 -> 0
        |42
        "#
    ));
}

#[test]
fn test_opaque_reference_pattern_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when n is
        |    @Age -> 1
        "#
    ));
}

#[test]
fn test_comment_after_op_expr() {
    snapshot_test!(block_indentify(
        r#"
        |12  * # test!
        | 92
        "#
    ));
}

#[test]
fn test_parenthesized_type_def_space_before_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |A : b
        |a
        "#
    ));
}

#[test]
fn test_opaque_with_type_arguments_moduledefs_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |Bookmark a := { chapter : Str, stanza : Str, notes : a }
        |
        "#
    ));
}

#[test]
fn test_list_closing_same_indent_no_trailing_comma_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |myList = [
        |    0,
        |    1,
        |]
        |42
        "#
    ));
}

#[test]
fn test_parenthetical_var_expr() {
    snapshot_test!(block_indentify(
        r#"
        |(whee)
        "#
    ));
}

#[test]
fn test_str_block_multiple_newlines_expr() {
    snapshot_test!(block_indentify(
        r###"
        |"""
        |
        |
        |#"""#
        "###
    ));
}

#[test]
fn test_list_patterns_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when [] is
        |    [] -> {}
        |    [..] -> {}
        |    [_, .., _, ..] -> {}
        |    [a, b, c, d] -> {}
        |    [a, b, ..] -> {}
        |    [.., c, d] -> {}
        |    [[A], [..], [a]] -> {}
        |    [[[], []], [[], x]] -> {}
        "#
    ));
}

#[test]
fn test_empty_platform_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |platform "rtfeldman/blah" requires {} { main : {} } exposes [] packages {} imports [] provides []
        |
        "#
    ));
}

#[test]
fn test_interface_with_newline_header() {
    snapshot_test!(block_indentify(
        r#"
        |interface T exposes [] imports []
        |
        "#
    ));
}

#[test]
fn test_newline_after_sub_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |3
        |-
        |4
        "#
    ));
}

#[test]
fn test_newline_after_mul_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |3
        |*
        |4
        "#
    ));
}

#[test]
fn test_provides_type_header() {
    snapshot_test!(block_indentify(
        r#"
        |app "test"
        |    packages { pf: "./platform" }
        |    imports [ foo.Bar.Baz ]
        |    provides [ quicksort ] { Flags, Model, } to pf
        |
        "#
    ));
}

#[test]
fn test_ability_multi_line_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |Hash implements
        |    hash : a -> U64
        |    hash2 : a -> U64
        |
        |1
        "#
    ));
}

#[test]
fn test_where_clause_non_function_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f : a where a implements A
        |
        |f
        |
        "#
    ));
}

#[test]
fn test_ops_with_newlines_expr() {
    snapshot_test!(block_indentify(
        r#"
        |3
        |+
        |
        |  4
        "#
    ));
}

#[test]
fn test_record_with_if_expr() {
    snapshot_test!(block_indentify(
        r#"
        |{x : if Bool.true then 1 else 2, y: 3 }
        "#
    ));
}

#[test]
fn test_newline_after_paren_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |A
        "#
    ));
}

#[test]
fn test_record_access_after_tuple_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |({ a: 0 }, { b: 1 }).0.a
        "#
    ));
}

#[test]
fn test_single_underscore_closure_expr() {
    snapshot_test!(block_indentify(
        r#"
        |\_ -> 42
        "#
    ));
}

#[test]
fn test_empty_hosted_header_header() {
    snapshot_test!(block_indentify(
        r#"
        |hosted Foo exposes [] imports [] generates Bar with []
        |
        "#
    ));
}

#[test]
fn test_multi_char_string_expr() {
    snapshot_test!(block_indentify(
        r#"
        |"foo"
        |
        "#
    ));
}

#[test]
fn test_opaque_has_abilities_expr() {
    snapshot_test!(block_indentify(
        r#"
        |A := U8 implements [Eq, Hash]
        |
        |A := a where a implements Other implements [Eq, Hash]
        |
        |A := a where a implements Other
        |     implements [Eq, Hash]
        |
        |A := U8 implements [Eq {eq}, Hash {hash}]
        |
        |A := U8 implements [Eq {eq, eq1}]
        |
        |A := U8 implements [Eq {eq, eq1}, Hash]
        |
        |A := U8 implements [Hash, Eq {eq, eq1}]
        |
        |A := U8 implements []
        |
        |A := a where a implements Other
        |     implements [Eq {eq}, Hash {hash}]
        |
        |A := U8 implements [Eq {}]
        |
        |0
        |
        "#
    ));
}

#[test]
fn test_outdented_app_with_record_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x = foo
        |    (
        |        baz {
        |            bar: blah,
        |        }
        |    )
        |x
        "#
    ));
}

#[test]
fn test_multiline_string_expr() {
    snapshot_test!(block_indentify(
        r#"
        |a = "Hello,\n\nWorld!"
        |b = """Hello,\n\nWorld!"""
        |c =
        |    """
        |    Hello,
        |
        |    World!
        |    """
        |42
        |
        "#
    ));
}

#[test]
fn test_empty_package_header_header_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |package "rtfeldman/blah" exposes [] packages {}
        |
        "#
    ));
}

#[test]
fn test_when_with_numbers_expr() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        | 1 -> 2
        | 3 -> 4
        |
        "#
    ));
}

#[test]
fn test_one_def_expr() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |x=5
        |
        |42
        |
        "#
    ));
}

#[test]
fn test_comment_after_expr_in_parens_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |i # abc
        "#
    ));
}

#[test]
fn test_list_minus_newlines_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |[
        |    K,
        |]
        |- i
        "#
    ));
}

#[test]
fn test_when_in_assignment_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x =
        |    when n is
        |        0 -> 0
        |42
        "#
    ));
}

#[test]
fn test_nested_def_without_newline_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |x =
        |    a : n
        |    4
        |_
        "#
    ));
}

#[test]
fn test_newline_in_packages_full() {
    snapshot_test!(block_indentify(
        r#"
        |app "hello"
        |    packages { pf:
        |"https://github.com/roc-lang/basic-cli/releases/download/0.7.0/bkGby8jb0tmZYsy2hg1E_B2QrCgcSTxdUlHtETwm5m4.tar.br"
        |}
        |    imports [pf.Stdout]
        |    provides [main] to pf
        |
        |main =
        |    Stdout.line "I'm a Roc application!"
        "#
    ));
}

#[test]
fn test_outdented_list_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |a = [
        |    1,
        |    2,
        |    3,
        |]
        |a
        "#
    ));
}

#[test]
fn test_one_char_string_expr() {
    snapshot_test!(block_indentify(
        r#"
        |"x"
        |
        "#
    ));
}

#[test]
fn test_tuple_type_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |f : (Str, Str) -> (Str, Str)
        |f = \x -> x
        |
        |f (1, 2)
        "#
    ));
}

#[test]
fn test_expect_expr() {
    snapshot_test!(block_indentify(
        r#"
        |expect 1 == 1
        |
        |4
        |
        "#
    ));
}

#[test]
fn test_when_with_function_application_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    1 ->
        |        Num.neg
        |            2
        |
        |    _ -> 4
        "#
    ));
}

#[test]
fn test_when_in_assignment_expr() {
    snapshot_test!(block_indentify(
        r#"
        |x = when n is
        |     0 -> 0
        |42
        "#
    ));
}

#[test]
fn test_where_clause_multiple_bound_abilities_expr() {
    snapshot_test!(block_indentify(
        r#"
        |f : a -> b where a implements Hash & Eq, b implements Eq & Hash & Display
        |
        |f : a -> b
        |  where a implements Hash & Eq,
        |    b implements Hash & Display & Eq
        |
        |f
        |
        "#
    ));
}

#[test]
fn test_two_backpassing_expr() {
    snapshot_test!(block_indentify(
        r#"
        |# leading comment
        |x <- (\y -> y)
        |z <- {}
        |
        |x
        |
        "#
    ));
}

#[test]
fn test_two_branch_when_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |when x is
        |    "" -> 1
        |    "mise" -> 2
        "#
    ));
}

#[test]
fn test_ability_two_in_a_row_expr() {
    snapshot_test!(block_indentify(
        r#"
        |Ab1 implements ab1 : a -> {} where a implements Ab1
        |
        |Ab2 implements ab2 : a -> {} where a implements Ab2
        |
        |1
        |
        "#
    ));
}

#[test]
fn test_where_clause_multiple_bound_abilities_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |f : a -> b where a implements Hash & Eq, b implements Eq & Hash & Display
        |
        |f : a -> b where a implements Hash & Eq, b implements Hash & Display & Eq
        |
        |f
        "#
    ));
}

#[test]
fn test_number_literal_suffixes_expr() {
    snapshot_test!(block_indentify(
        r#"
        |{
        |  u8:   123u8,
        |  u16:  123u16,
        |  u32:  123u32,
        |  u64:  123u64,
        |  u128: 123u128,
        |  i8:   123i8,
        |  i16:  123i16,
        |  i32:  123i32,
        |  i64:  123i64,
        |  i128: 123i128,
        |  nat:  123nat,
        |  dec:  123dec,
        |  u8Neg:   -123u8,
        |  u16Neg:  -123u16,
        |  u32Neg:  -123u32,
        |  u64Neg:  -123u64,
        |  u128Neg: -123u128,
        |  i8Neg:   -123i8,
        |  i16Neg:  -123i16,
        |  i32Neg:  -123i32,
        |  i64Neg:  -123i64,
        |  i128Neg: -123i128,
        |  natNeg:  -123nat,
        |  decNeg:  -123dec,
        |  u8Bin:   0b101u8,
        |  u16Bin:  0b101u16,
        |  u32Bin:  0b101u32,
        |  u64Bin:  0b101u64,
        |  u128Bin: 0b101u128,
        |  i8Bin:   0b101i8,
        |  i16Bin:  0b101i16,
        |  i32Bin:  0b101i32,
        |  i64Bin:  0b101i64,
        |  i128Bin: 0b101i128,
        |  natBin:  0b101nat,
        |}
        |
        "#
    ));
}

#[test]
fn test_module_def_newline_moduledefs() {
    snapshot_test!(block_indentify(
        r#"
        |main =
        |    i = 64
        |
        |    i
        |
        "#
    ));
}

#[test]
fn test_comment_before_op_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |3 # test!
        |+ 4
        "#
    ));
}

#[test]
fn test_parens_in_type_def_apply_expr_formatted() {
    snapshot_test!(block_indentify(
        r#"
        |U (b a) : b
        |a
        "#
    ));
}

#[test]
fn test_type_annotated_function() {
    snapshot_test!(block_indentify(
        r#"
        |t:d,T->r
        |t=e
        "#
    ));
}

#[test]
fn test_if_after_when() {
    snapshot_test!(block_indentify(
        r#"
        |p=when d is
        |F->""
        |_->""if e then
        |e
        |else
        |e
        "#
    ));
}

#[test]
fn test_parse_large_file() {
    // print pwd for debugging
    println!("pwd: {:?}", std::env::current_dir().unwrap());

    let path = "../../../crates/glue/src/RustGlue.roc";
    assert!(std::path::Path::new(path).exists());

    let text = std::fs::read_to_string(path).unwrap();

    snapshot_test!(&text);
}
