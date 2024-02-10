


use roc_cypress::parse::State;
use roc_cypress::token::Tokenizer;


const DATA: &str = include_str!("../../../../../crates/glue/src/RustGlue.roc");

fn main() {
    let start = std::time::Instant::now();

    while start.elapsed().as_secs() < 10 {
        parse_once(&DATA);
    }
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
        let formatted_frames = top_frames
            .map(|frame| format!("{:?}", frame))
            .collect::<Vec<_>>()
            .join(" -> ");

        Some(format!("{:?}: at {}", msg.kind, formatted_frames))
    } else {
        None
    }
}
