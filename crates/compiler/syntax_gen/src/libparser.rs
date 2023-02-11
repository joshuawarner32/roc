
pub mod prelude {
    pub use super::StrLiteral;
}

pub struct StrLiteral<'a> {
    text: &'a str,
}
