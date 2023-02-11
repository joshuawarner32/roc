
pub mod prelude {
    pub use super::StrLiteral;
}

pub struct StrLiteral<'a> {
    text: &'a str,
}

pub struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {

}

pub enum Token<'a> {
    StrLiteral(StrLiteral<'a>),
    
}