use generate::generate_syntax;
use ir::convert_syntax;
use parser::{parse_syntax, State, format_error};
use rust::{to_rust_syntax, Options};

mod ast;
mod parser;
mod generate;
mod rust;
mod ir;


fn main() {
    let input = std::env::args().nth(1).unwrap();
    let output = std::env::args().nth(2).unwrap();
    let text = std::fs::read_to_string(input).unwrap();

    let syntax = match parse_syntax(&mut State::new(&text)) {
        Ok(syntax) => syntax,
        Err(e) => {
            panic!("{}", format_error(&text, e))
        }
    };

    let ir = convert_syntax(syntax);

    let rust_syntax = to_rust_syntax(&ir, Options {
        generate_tokens: false,
        generate_trivia: false,
    });

    let gen = generate_syntax(&rust_syntax);

    std::fs::write(output, gen).unwrap();
}
