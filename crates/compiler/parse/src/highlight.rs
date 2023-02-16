use encode_unicode::CharExt;
use std::collections::HashSet;

use bumpalo::Bump;
use roc_region::all::{Loc, Region};

use crate::{
    ast::CommentOrNewline,
    blankspace::loc_spaces,
    keyword::KEYWORDS,
    number_literal::positive_number_literal,
    parser::{EExpr, ParseResult, Parser},
    state::State,
    string_literal::{parse_str_like_literal, StrLikeLiteral},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Token {
    LineComment,
    DocComment,
    Error,
    SingleQuote,
    String,
    UnicodeEscape,
    EscapedChar,
    Interpolated,
    Keyword,
    UpperIdent,
    LowerIdent,
    Number,
    Other,
}

pub fn highlight(text: &str) -> Vec<Loc<Token>> {
    let mut tokens = Vec::new();
    let state = State::new(text.as_bytes());

    let arena = Bump::new();

    let header_keywords = HEADER_KEYWORDS.iter().copied().collect::<HashSet<_>>();
    let body_keywords = KEYWORDS.iter().copied().collect::<HashSet<_>>();

    if let Ok((_prog, _, new_state)) = crate::module::header().parse(&arena, state.clone(), 0) {
        let inner_state =
            State::new(text[..state.bytes().len() - new_state.bytes().len()].as_bytes());
        highlight_inner(&arena, inner_state, &mut tokens, &header_keywords);
        highlight_inner(&arena, new_state, &mut tokens, &body_keywords);
    } else {
        highlight_inner(&arena, state, &mut tokens, &body_keywords);
    }

    tokens
}

fn highlight_inner<'a>(
    arena: &'a Bump,
    mut state: State<'a>,
    tokens: &mut Vec<Loc<Token>>,
    keywords: &HashSet<&str>,
) {
    loop {
        let start = state.pos();
        if let Ok((b, _width)) = char::from_utf8_slice_start(state.bytes()) {
            match b {
                ' ' | '\n' | '\t' | '\r' | '#' => {
                    let res: ParseResult<'a, _, EExpr<'a>> =
                        loc_spaces().parse(arena, state.clone(), 0);
                    if let Ok((_, spaces, new_state)) = res {
                        state = new_state;
                        for space in spaces {
                            let token = match space.value {
                                CommentOrNewline::Newline => {
                                    continue;
                                }
                                CommentOrNewline::LineComment(_) => Token::LineComment,
                                CommentOrNewline::DocComment(_) => Token::DocComment,
                            };
                            tokens.push(Loc::at(space.region, token));
                        }
                    } else {
                        fast_forward_to(&mut state, tokens, start, |c| c == b'\n');
                    }
                }
                '"' | '\'' => {
                    if let Ok((_, item, new_state)) =
                        parse_str_like_literal().parse(arena, state.clone(), 0)
                    {
                        state = new_state;
                        match item {
                            StrLikeLiteral::SingleQuote(_) => {
                                tokens.push(Loc::at(
                                    Region::between(start, state.pos()),
                                    Token::SingleQuote,
                                ));
                            }
                            StrLikeLiteral::Str(_) => {
                                tokens.push(Loc::at(
                                    Region::between(start, state.pos()),
                                    Token::String,
                                ));
                            }
                        }
                    } else {
                        fast_forward_to(&mut state, tokens, start, |c| c == b'\n');
                    }
                }
                c if c.is_alphabetic() => {
                    let buffer = state.bytes();
                    let mut chomped = 0;

                    let is_upper = c.is_uppercase();

                    while let Ok((ch, width)) = char::from_utf8_slice_start(&buffer[chomped..]) {
                        if ch.is_alphabetic() || ch.is_ascii_digit() {
                            chomped += width;
                        } else {
                            // we're done
                            break;
                        }
                    }

                    let ident = std::str::from_utf8(&buffer[..chomped]).unwrap();
                    state.advance_mut(chomped);

                    if keywords.contains(ident) {
                        tokens.push(Loc::at(Region::between(start, state.pos()), Token::Keyword));
                    } else {
                        tokens.push(Loc::at(
                            Region::between(start, state.pos()),
                            if is_upper {
                                Token::UpperIdent
                            } else {
                                Token::LowerIdent
                            },
                        ));
                    }
                }
                '0'..='9' => {
                    if let Ok((_, _item, new_state)) =
                        positive_number_literal().parse(arena, state.clone(), 0)
                    {
                        state = new_state;
                        tokens.push(Loc::at(Region::between(start, state.pos()), Token::Number));
                    } else {
                        fast_forward_to(&mut state, tokens, start, |b| !b.is_ascii_digit());
                    }
                }
                _ => {
                    state.advance_mut(1);
                    tokens.push(Loc::at(Region::between(start, state.pos()), Token::Other));
                }
            }
        } else {
            break;
        }
    }
}

fn fast_forward_to(
    state: &mut State,
    tokens: &mut Vec<Loc<Token>>,
    start: roc_region::all::Position,
    end: impl Fn(u8) -> bool,
) {
    while let Some(b) = state.bytes().first() {
        if end(*b) {
            break;
        }
        state.advance_mut(1);
    }
    tokens.push(Loc::at(Region::between(start, state.pos()), Token::Error));
}

pub const HEADER_KEYWORDS: [&str; 14] = [
    "interface",
    "app",
    "package",
    "platform",
    "hosted",
    "exposes",
    "imports",
    "with",
    "generates",
    "package",
    "packages",
    "requires",
    "provides",
    "to",
];

#[cfg(test)]
mod tests {
    use roc_region::all::Position;

    use super::*;

    #[test]
    fn test_highlight_comments() {
        let text = "# a\n#b\n#c";
        let tokens = highlight(text);
        assert_eq!(
            tokens,
            vec![
                Loc::at(
                    Region::between(Position::new(0), Position::new(3)),
                    Token::LineComment
                ),
                Loc::at(
                    Region::between(Position::new(4), Position::new(6)),
                    Token::LineComment
                ),
                Loc::at(
                    Region::between(Position::new(7), Position::new(9)),
                    Token::LineComment
                ),
            ]
        );
    }

    #[test]
    fn test_highlight_doc_comments() {
        let text = "## a\n##b\n##c";
        let tokens = highlight(text);
        assert_eq!(
            tokens,
            vec![
                Loc::at(
                    Region::between(Position::new(0), Position::new(4)),
                    Token::DocComment
                ),
                // the next two are line comments because there's not a space at the beginning
                Loc::at(
                    Region::between(Position::new(5), Position::new(8)),
                    Token::LineComment
                ),
                Loc::at(
                    Region::between(Position::new(9), Position::new(12)),
                    Token::LineComment
                ),
            ]
        );
    }

    #[test]
    fn test_highlight_strings() {
        let text = r#""a""#;
        let tokens = highlight(text);
        assert_eq!(
            tokens,
            vec![Loc::at(
                Region::between(Position::new(0), Position::new(3)),
                Token::String
            )]
        );
    }

    #[test]
    fn test_highlight_single_quotes() {
        let text = r#"'a'"#;
        let tokens = highlight(text);
        assert_eq!(
            tokens,
            vec![Loc::at(
                Region::between(Position::new(0), Position::new(3)),
                Token::SingleQuote
            )]
        );
    }

    #[test]
    fn test_highlight_header() {
        let text = r#"app "test-app" provides [] to "./blah""#;
        let tokens = highlight(text);
        assert_eq!(
            tokens,
            vec![
                Loc::at(
                    Region::between(Position::new(0), Position::new(3)),
                    Token::Keyword
                ),
                Loc::at(
                    Region::between(Position::new(4), Position::new(14)),
                    Token::String
                ),
                Loc::at(
                    Region::between(Position::new(15), Position::new(23)),
                    Token::Keyword
                ),
                Loc::at(
                    Region::between(Position::new(24), Position::new(25)),
                    Token::Other
                ),
                Loc::at(
                    Region::between(Position::new(25), Position::new(26)),
                    Token::Other
                ),
                Loc::at(
                    Region::between(Position::new(27), Position::new(29)),
                    Token::Keyword
                ),
                Loc::at(
                    Region::between(Position::new(30), Position::new(38)),
                    Token::String
                ),
            ]
        );
    }

    #[test]
    fn test_highlight_numbers() {
        let text = "123.0 123 123. 123.0e10 123e10 123e-10 0x123";
        let tokens = highlight(text);
        assert_eq!(
            tokens,
            vec![
                Loc::at(
                    Region::between(Position::new(0), Position::new(5)),
                    Token::Number
                ),
                Loc::at(
                    Region::between(Position::new(6), Position::new(9)),
                    Token::Number
                ),
                Loc::at(
                    Region::between(Position::new(10), Position::new(14)),
                    Token::Number
                ),
                Loc::at(
                    Region::between(Position::new(15), Position::new(23)),
                    Token::Number
                ),
                Loc::at(
                    Region::between(Position::new(24), Position::new(30)),
                    Token::Number
                ),
                Loc::at(
                    Region::between(Position::new(31), Position::new(38)),
                    Token::Number
                ),
                Loc::at(
                    Region::between(Position::new(39), Position::new(44)),
                    Token::Number
                ),
            ]
        );
    }
}
