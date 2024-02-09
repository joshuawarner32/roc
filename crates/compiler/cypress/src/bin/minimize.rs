use std::panic::catch_unwind;

use roc_cypress::token::Tokenizer;
use roc_cypress::parse::State;
use bumpalo::Bump;
use roc_parse::parser::Parser;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() != 2 {
        eprintln!("Usage: {} <input>", args[0]);
        std::process::exit(1);
    }

    let text = std::fs::read_to_string(&args[1]).unwrap();

    let Some(original_error) = parse_once_and_extract_error(&text) else {
        eprintln!("No error found");
        return;
    };

    eprintln!("Error found: {}", original_error);
    eprintln!("Proceeding with minimization");

    let mut s = text;

    loop {
        let mut found = false;
        for update in candidate_minimizations(s.clone()) {
            let mut new_s = String::with_capacity(s.len());
            let mut offset = 0;
            for (start, end, replacement) in update.replacements {
                new_s.push_str(&s[offset..start]);
                new_s.push_str(&replacement);
                offset = end;
            }
            new_s.push_str(&s[offset..]);

            if let Some(result) = parse_once_and_extract_error(&new_s) {
                if result == original_error {
                    eprintln!("Successfully minimized, new length: {}", new_s.len());
                    s = new_s;
                    found = true;
                    break;
                // } else {
                //     eprintln!("Failed to minimize: {}", result);
                }
            // } else {
            //     eprintln!("No error found");
            }
        }

        if !found {
            eprintln!("No more minimizations found");
            break;
        }
    }

    eprintln!("Final result:");
    println!("{}", s);
}

fn parse_once_and_extract_error(text: &str) -> Option<String> {
    if !check_legacy_parse(text) {
        return Some("Legacy parser failed".to_string());
    }

    let res = std::panic::catch_unwind(|| {
        parse_once(text)
    });

    match res {
        Ok(res) => res,
        Err(e) => {
            if let Some(s) = e.downcast_ref::<&'static str>() {
                return Some(s.to_string());
            }
            if let Some(s) = e.downcast_ref::<String>() {
                return Some(s.clone());
            }
            Some("Panic during parsing".to_string())
        },
    }
}

fn check_legacy_parse(text: &str) -> bool {
    std::panic::catch_unwind(|| {
        let bump = Bump::new();
        parse_legacy_full(&bump, text) || parse_legacy_expr(&bump, text)
    }).unwrap_or(false)
}

fn parse_legacy_full<'a>(bump: &'a Bump, text: &'a str) -> bool {
    let state = roc_parse::state::State::new(text.as_bytes());

    let min_indent = 0;
    let Ok((_, _, state)) = roc_parse::module::header()
        .parse(&bump, state.clone(), min_indent) else {
        return false;
    };

    roc_parse::module::module_defs()
        .parse(&bump, state, min_indent)
        .is_ok()
}

fn parse_legacy_expr<'a>(bump: &'a Bump, text: &str) -> bool {
    roc_parse::test_helpers::parse_expr_with(bump, text).is_ok()
}

fn parse_once(text: &str) -> Option<String> {
    let mut tokenizer = Tokenizer::new(text);
    tokenizer.tokenize();
    let (messages, tb) = tokenizer.finish();
    assert_eq!(messages, vec![]);
    // eprint!("tokens:");
    // let mut last = 0;
    // for (i, (begin, indent)) in tb.lines.iter().enumerate() {
    //     for tok in &tb.kinds[last as usize .. *begin as usize] {
    //         eprint!(" {:?}", tok);
    //     }
    //     eprint!("\n{}: {:?} {}.{}:", i, begin, indent.num_spaces, indent.num_tabs);
    //     last = *begin;
    // }
    // for tok in &tb.kinds[last as usize..] {
    //     eprint!(" {:?}", tok);
    // }
    // eprintln!();

    let mut state = State::from_buf(tb);
    state.start_file();
    state.pump();
    state.assert_end();

    if let Some(msg) = state.messages.first() {
        let top_frames = msg.frames.iter().rev().take(2);
        let formatted_frames = top_frames.map(|frame| {
            format!("{:?}", frame)
        }).collect::<Vec<_>>().join(" -> ");

        Some(format!("{:?}: at {}", msg.kind, formatted_frames))
    } else {
        None
    }
}

struct Update {
    replacements: Vec<(usize, usize, String)>,
}

fn candidate_minimizations(s: String) -> Box<dyn Iterator<Item = Update>> {
    let mut line_offsets = vec![0];
    line_offsets.extend(s.match_indices('\n').map(|(i, _)| i + 1));
    let line_count = line_offsets.len();
    let s_len = s.len();

    let line_indents = line_offsets.iter().map(|&offset| {
        s[offset..].chars().take_while(|&c| c == ' ').count()
    }).collect::<Vec<_>>();

    let line_offsets_clone = line_offsets.clone();

    // first, try to remove every group of 1, 2, 3, ... lines - in reverse order (so, trying removing n lines first, then n-1, etc)
    let line_removals = (1..=line_count).rev().map(move |n| {
        let line_offsets_clone = line_offsets.clone();
        (0..line_count - n).map(move |start| {
            let end = start + n;
            let start_offset = line_offsets_clone[start];
            let end_offset = line_offsets_clone[end];
            let replacement = String::new();
            let replacements = vec![(start_offset, end_offset, replacement)];
            Update { replacements }
        })
    }).flatten();

    let line_offsets = line_offsets_clone;

    // then, try to dedent every group of 1, 2, 3, ... lines - in reverse order (so, trying dedenting n lines first, then n-1, etc)
    // just remove one space at a time, for now
    let line_dedents = (1..=line_count).rev().map(move |n| {
        let line_offsets_clone = line_offsets.clone();
        let line_indents_clone = line_indents.clone();
        (0..line_count - n).filter_map(move |start| {
            // first check if all lines are either zero-width or have greater than zero indent
            let end = start + n;
            for i in start..end {
                if line_indents_clone[i] == 0 && line_offsets_clone[i] + 1 < line_offsets_clone.get(i + 1).cloned().unwrap_or(s_len) {
                    return None;
                }
            }

            let mut replacements = vec![];
            for i in start..end {
                let offset = line_offsets_clone[i];
                let indent = line_indents_clone[i];
                if indent > 0 {
                    replacements.push((offset, offset + 1, String::new()));
                }
            }
            Some(Update { replacements })
        })
    }).flatten();

    // then, try to remove every range of 1, 2, 3, ... characters - in reverse order (so, trying removing n characters first, then n-1, etc)
    let charseq_removals = (1..s.len()).rev().map(move |n| {
        (0..s.len() - n).map(move |start| {
            let end = start + n;
            let replacement = String::new();
            let replacements = vec![(start, end, replacement)];
            Update { replacements }
        })
    }).flatten();

    Box::new(line_removals.chain(line_dedents).chain(charseq_removals))
}
