use bumpalo::Bump;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::path::PathBuf;

pub fn parse_benchmark(c: &mut Criterion) {
    let mut path = PathBuf::from(std::env!("ROC_WORKSPACE_DIR"));
    path.push("crates");
    path.push("glue");
    path.push("src");
    path.push("RustGlue.roc");
    let src = std::fs::read_to_string(&path).unwrap();

    c.bench_function("legacy", |b| {
        use roc_parse::{module, module::module_defs, parser::Parser, state::State};
        b.iter(|| {
            let arena = Bump::new();

            let (_actual, state) =
                module::parse_header(&arena, State::new(src.as_bytes())).unwrap();

            let min_indent = 0;
            let res = module_defs()
                .parse(&arena, state, min_indent)
                .map(|tuple| tuple.1)
                .unwrap();

            black_box(res.len());
        })
    });

    c.bench_function("cypress", |b| {
        use roc_cypress::parse::State;
        use roc_cypress::token::Tokenizer;
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(&src);
            tokenizer.tokenize();
            let (messages, tb) = tokenizer.finish();
            assert_eq!(messages, vec![]);
            let mut state = State::from_buf(tb);
            state.start_file();
            state.pump();
            state.assert_end();
            black_box(state.tree.kinds.len());
        })
    });
}

criterion_group!(benches, parse_benchmark);
criterion_main!(benches);
