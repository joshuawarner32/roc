
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Syntax {
    pub items: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    Struct(Struct),
    Enum(Enum),
    Typedef(Typedef),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Struct {
    pub name: String,
    pub generics: Generics,
    pub fields: Fields,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Enum {
    pub name: String,
    pub generics: Generics,
    pub variants: Vec<(String, Fields)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Typedef {
    pub name: String,
    pub generics: Generics,
    pub ty: Type,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Generic {
    Type(String),
    Dollar(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generics {
    pub params: Vec<Generic>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fields {
    Unit,
    Tuple(Vec<Type>),
    Named(Vec<(String, Type)>),
    Seq(Vec<Type>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Literal(String),
    Named(String),
    Ref(Box<Type>),
    Tuple(Vec<Type>),
    Array(Box<Type>),
    Generics(String, Vec<Type>),
    Field(String, Box<Type>),
    Option(Box<Type>),
    Repeat(Box<Type>),
    Seq(Vec<Type>),
    Unimportant(Box<Type>),
    Dollar(String),
    Whitespace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltIn {
    String,
    Int,
    Bool,
}
