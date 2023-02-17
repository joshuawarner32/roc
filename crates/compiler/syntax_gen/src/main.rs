


fn main() {
    let input = std::env::args().nth(1).unwrap();
    let output = std::env::args().nth(2).unwrap();
    let text = std::fs::read_to_string(input).unwrap();

    let syntax = match parse_syntax(&mut State::new(&text)) {
        Ok(syntax) => syntax,
        Err(e) => {
            let mut newline_offsets = vec![0];
            for (i, ch) in text.char_indices() {
                if ch == '\n' {
                    newline_offsets.push(i + 1);
                }
            }

            // This is horribly inefficient, but it's a one-off script so who cares
            let line = newline_offsets
                .iter()
                .enumerate()
                .filter(|(_, &offset)| offset <= e.pos)
                .map(|(i, _)| i + 1)
                .last()
                .unwrap();
        
            let col = e.pos - newline_offsets.get(line - 1).copied().unwrap_or(0) + 1;

            panic!("Error at offset {} ({}:{}): {}", e.pos, line, col, e.message);
        }
    };

    let gen = generate_syntax(&syntax);

    std::fs::write(output, gen).unwrap();
}

#[derive(Copy, Clone)]
struct State<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> State<'a> {
    fn new(text: &'a str) -> State<'a> {
        State { text, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.text[self.pos..].chars().next()
    }

    fn next(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn error(&self, message: &str) -> Error {
        Error {
            message: message.to_string(),
            pos: self.pos,
        }
    }

    fn peek_terminator(&self) -> bool {
        self.peek() == Some(']') || self.peek() == Some(')') || self.peek() == Some('}') ||
        self.peek() == Some(',') || self.peek() == Some(';') || self.peek() == Some('>')
    }

    fn peek_maybe(&self, s: &str) -> bool {
        self.text[self.pos..].starts_with(s)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.next();
            } else if ch == '/' && self.text[self.pos + 1..].starts_with('/') {
                while let Some(ch) = self.next() {
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn maybe(&mut self, s: &str) -> bool {
        if self.text[self.pos..].starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    fn require(&mut self, s: &str) -> Result<(), Error> {
        if self.text[self.pos..].starts_with(s) {
            self.pos += s.len();
            Ok(())
        } else {
            Err(self.error(&format!("expected `{}`", s)))
        }
    }

    fn peek_ident(&self) -> Option<&'a str> {
        let mut it = self.text[self.pos..].char_indices();
        if let Some((i, ch)) = it.next() {
            if ch.is_alphabetic() || ch == '_' {
                let mut last = i;
                while let Some((i, ch)) = it.next() {
                    if !ch.is_alphanumeric() && ch != '_' {
                        last = i;
                        break;
                    }
                }
                let ident = &self.text[self.pos..self.pos + last];
                Some(ident)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn consume_ident(&mut self) -> Option<&'a str> {
        let ident = self.peek_ident()?;
        self.pos += ident.len();
        Some(ident)
    }

    fn maybe_keyword(&mut self, s: &str) -> bool {
        if self.peek_ident() == Some(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }
}

struct Error {
    message: String,
    pos: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Syntax {
    items: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Item {
    Struct(Struct),
    Enum(Enum),
    Typedef(Typedef),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Struct {
    name: String,
    generics: Generics,
    fields: Fields,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Enum {
    name: String,
    generics: Generics,
    variants: Vec<(String, Fields)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Typedef {
    name: String,
    generics: Generics,
    ty: Type,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Generics {
    params: Vec<Generic>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Generic {
    Type(String),
    Dollar(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Fields {
    Unit,
    Tuple(Vec<Type>),
    Named(Vec<(String, Type)>),
    Seq(Vec<Type>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Type {
    Literal(String),
    BuiltIn(BuiltIn),
    Named(String),
    Ref(Box<Type>),
    Tuple(Vec<Type>),
    Array(Box<Type>),
    Generics(String, Vec<Type>),
    Field(String, Box<Type>),
    Dollar(String),
    Option(Box<Type>),
    Repeat(Box<Type>),
    Seq(Vec<Type>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltIn {
    String,
    Int,
    Bool,
}

fn generate_syntax(syntax: &Syntax) -> String {
    let mut res = String::new();
    for item in &syntax.items {
        match item {
            Item::Struct(s) => {
                res.push_str(&format!("pub struct {}", s.name));
                generate_generics(&mut res, &s.generics);
                generate_fields(&mut res, &s.fields, true, true);
                res.push('\n');
            }
            Item::Enum(e) => {
                res.push_str(&format!("pub enum {} {{\n", e.name));
                generate_generics(&mut res, &e.generics);
                for (name, fields) in &e.variants {
                    res.push_str(&format!("    {}", name));
                    generate_fields(&mut res, fields, false, false);
                    res.push_str(",\n");
                }
                res.push_str("}\n");
            }
            Item::Typedef(td) => {
                res.push_str(&format!("pub type {}", td.name));
                generate_generics(&mut res, &td.generics);
                res.push_str(" = ");
                res.push_str(&generate_ty(&td.ty));
                res.push_str(";\n");
            }
        }
    }
    res
}

fn generate_generics(res: &mut String, generics: &Generics) {
    if !generics.params.is_empty() {
        res.push_str("<");
        for (i, param) in generics.params.iter().enumerate() {
            if i > 0 {
                res.push_str(", ");
            }
            let param = match param {
                Generic::Type(s) => s,
                Generic::Dollar(_s) => panic!(),
            };
            res.push_str(param);
        }
        res.push_str(">");
    }
}

fn generate_fields(res: &mut String, fields: &Fields, in_struct: bool, newlines: bool) {
    let public = if in_struct { "pub " } else { "" };
    match fields {
        Fields::Unit => {
            if in_struct {
                res.push_str(";");
            } else {
                // Nothing; unit variants in enums are just the name
            }
        }
        Fields::Named(fields) => {
            res.push_str(" {\n");
            for (name, ty) in fields {
                res.push_str(&format!("    {}{}: {},\n", public, name, generate_ty(&ty)));
            }
            res.push_str("}");
        }
        Fields::Tuple(tys) => {
            res.push_str("(");
            for (i, ty) in tys.iter().enumerate() {
                if i > 0 {
                    res.push_str(", ");
                }
                res.push_str(&generate_ty(ty));
            }
            res.push_str(")");
        }
        Fields::Seq(..) => panic!(),
    }
}

fn generate_ty(ty: &Type) -> String {
    match ty {
        Type::BuiltIn(b) => match b {
            BuiltIn::String => "String".to_string(),
            BuiltIn::Int => "u32".to_string(),
            BuiltIn::Bool => "bool".to_string(),
        },
        Type::Named(name) => name.to_string(),
        Type::Ref(ty) => format!("&{}", generate_ty(&ty)),
        Type::Tuple(tys) => {
            let mut res = String::new();
            res.push('(');
            for (i, ty) in tys.iter().enumerate() {
                if i > 0 {
                    res.push_str(", ");
                }
                res.push_str(&generate_ty(ty));
            }
            res.push(')');
            res
        }
        Type::Array(ty) => format!("[{}]", generate_ty(&ty)),
        Type::Generics(ty, tys) => {
            let mut res = String::new();
            res.push_str(&ty);
            res.push('<');
            for (i, ty) in tys.iter().enumerate() {
                if i > 0 {
                    res.push_str(", ");
                }
                res.push_str(&generate_ty(ty));
            }
            res.push('>');
            res
        }
        _ => panic!(),
    }
}

fn parse_syntax(state: &mut State) -> Result<Syntax, Error> {
    let mut items = Vec::new();
    loop {
        state.skip_whitespace();
        if state.peek().is_none() {
            break;
        }
        let item = parse_item(state)?;
        items.push(item);
    }
    Ok(Syntax { items })
}

fn parse_item(state: &mut State) -> Result<Item, Error> {

    // struct:
    // <ident> = ( <ident> : <type> , ... ) ;

    // enum:
    // <ident> = <variant> | <variant> | ... ;

    // variant:
    // <ident> ( <ident> : <type> , ... )
    // or:
    // <ident> ( <type> )
    // or:
    // <ident>

    // Rust-style structs and enums
    // struct Foo { <ident> : <type> , ... }
    // struct Foo;
    // struct Foo(<type>);
    // enum Foo { <ident> ( <type> ) , ... }
    // enum Foo { <ident> { <ident> : <type> } , ... }

    if state.maybe_keyword("struct") {
        state.skip_whitespace();
        let name = state.consume_ident().ok_or_else(|| state.error("expected struct name"))?.to_string();
        let generics = parse_generics(state)?;
        let fields = parse_fields(state, Ok(";"))?;
        
        Ok(Item::Struct(Struct {
            name,
            generics,
            fields,
        }))
    } else if state.maybe_keyword("enum") {
        state.skip_whitespace();
        let name = state.consume_ident().ok_or_else(|| state.error("expected enum name"))?.to_string();
        let generics = parse_generics(state)?;
        state.skip_whitespace();

        state.require("{")?;
        state.skip_whitespace();

        let mut variants = Vec::new();
        loop {

            if state.peek() == Some('}') {
                break;
            }

            let variant_name = state.consume_ident().ok_or_else(|| state.error("expected variant name"))?.to_string();
            
            let fields = parse_fields(state, Err(","))?;

            variants.push((variant_name, fields));

            state.skip_whitespace();

            if state.maybe(",") {
                state.skip_whitespace();
                continue;
            }

            break;
        }

        state.skip_whitespace();
        state.require("}")?;

        Ok(Item::Enum(Enum { name, generics, variants }))
    } else if state.maybe_keyword("type") {
        state.skip_whitespace();
        let name = state.consume_ident().ok_or_else(|| state.error("expected typedef name"))?.to_string();
        state.skip_whitespace();
        let generics = parse_generics(state)?;
        state.skip_whitespace();
        state.require("=")?;
        state.skip_whitespace();
        let ty = parse_type(state)?;
        state.skip_whitespace();
        state.require(";")?;
        Ok(Item::Typedef(Typedef { name, generics, ty }))
    } else {
        Err(state.error("expected struct/enum/type"))
    }
}

fn parse_generics(state: &mut State) -> Result<Generics, Error> {
    state.skip_whitespace();
    if state.maybe("<") {
        state.skip_whitespace();
        let mut params = Vec::new();
        loop {
            let dollar = state.maybe("$");
            let name = state.consume_ident().ok_or_else(|| state.error("expected generic name"))?.to_string();
            params.push(if dollar {
                Generic::Dollar(name)
            } else {
                Generic::Type(name)
            });
            state.skip_whitespace();
            if state.maybe(",") {
                state.skip_whitespace();
                continue;
            } else {
                break;
            }
        }
        state.require(">")?;
        Ok(Generics { params })
    } else {
        Ok(Generics { params: Vec::new() })
    }
}

fn parse_fields(state: &mut State, delim: Result<&str, &str>) -> Result<Fields, Error> {
    state.skip_whitespace();
    if state.maybe("(") {
        state.skip_whitespace();
        let fields = parse_fields_unnamed(state)?;
        state.skip_whitespace();
        state.require(")")?;
        return Ok(fields)
    } else if state.maybe("{") {
        state.skip_whitespace();
        let fields = parse_fields_named(state)?;
        state.skip_whitespace();
        state.require("}")?;
        return Ok(fields)
    } else if state.maybe("[") {
        state.skip_whitespace();
        let seq = parse_type_seq(state)?;
        state.skip_whitespace();
        state.require("]")?;
        return Ok(Fields::Seq(seq))
    }
    
    let found_delim = match delim {
        Ok(delim) => state.maybe(delim),
        Err(delim) => state.peek_maybe(delim),
    };

    if found_delim {
        Ok(Fields::Unit)
    } else {
        Err(state.error("expected struct fields"))
    }
}

fn parse_fields_named(state: &mut State) -> Result<Fields, Error> {
    let mut fields = Vec::new();
    loop {
        if state.peek() == Some(')') || state.peek() == Some('}') {
            break;
        }

        let name = state.consume_ident().ok_or_else(|| state.error("expected field name"))?.to_string();
        state.skip_whitespace();
        state.require(":")?;
        state.skip_whitespace();
        let ty = parse_type(state)?;
        fields.push((name, ty));
        state.skip_whitespace();
        if state.maybe(",") {
            state.skip_whitespace();
            continue;
        } else {
            break;
        }
    }
    Ok(Fields::Named(fields))
}

fn parse_fields_unnamed(state: &mut State) -> Result<Fields, Error> {
    let mut fields = Vec::new();
    loop {
        if state.peek() == Some(')') || state.peek() == Some('}') {
            break;
        }

        let ty = parse_type(state)?;
        fields.push(ty);
        state.skip_whitespace();
        if state.maybe(",") {
            state.skip_whitespace();
            continue;
        } else {
            break;
        }
    }
    Ok(Fields::Tuple(fields))
}

fn parse_type_seq(state: &mut State) -> Result<Vec<Type>, Error> {
    let mut types = Vec::new();
    loop {
        state.skip_whitespace();
        if state.peek_terminator() {
            break;
        }

        let ty = parse_type(state)?;
        types.push(ty);
    }

    Ok(types)
}

fn parse_type(state: &mut State) -> Result<Type, Error> {
    let initial = parse_type_initial(state)?;
    state.skip_whitespace();
    if state.maybe("?") {
        Ok(Type::Option(Box::new(initial)))
    } else if state.maybe("*") {
        Ok(Type::Repeat(Box::new(initial)))
    } else {
        Ok(initial)
    }
}


fn parse_type_initial(state: &mut State) -> Result<Type, Error> {
    if let Some(name) = state.consume_ident() {
        if state.maybe("<") {
            state.skip_whitespace();
            let mut params = Vec::new();
            loop {
                let ty = parse_type(state)?;
                params.push(ty);
                state.skip_whitespace();
                if state.maybe(",") {
                    state.skip_whitespace();
                    continue;
                } else {
                    break;
                }
            }
            state.require(">")?;
            Ok(Type::Generics(name.to_string(), params))
        } else if state.maybe(":") {
            state.skip_whitespace();
            let ty = parse_type(state)?;
            Ok(Type::Field(name.to_string(), Box::new(ty)))
        } else {
            Ok(Type::Named(name.to_string()))
        }
    } else if state.maybe("'") {
        // string literal (converted to a token)
        let mut s = String::new();
        loop {
            let c = state.next().ok_or_else(|| state.error("expected string literal"))?;
            if c == '\'' {
                break;
            } else {
                s.push(c);
            }
        }
        Ok(Type::Literal(s))
    } else if state.maybe("(") {
        // tuple
        state.skip_whitespace();
        let mut params = Vec::new();
        loop {
            let mut seq = parse_type_seq(state)?;
            if seq.len() == 1 {
                params.push(seq.pop().unwrap());
            } else {
                params.push(Type::Seq(seq));
            }
            state.skip_whitespace();
            if state.maybe(",") {
                state.skip_whitespace();
                continue;
            } else {
                break;
            }
        }
        state.require(")")?;

        if params.len() == 1 {
            Ok(params.pop().unwrap())
        } else {
            Ok(Type::Tuple(params))
        }
    } else if state.maybe("$") {
        // seq variable
        let name = state.consume_ident().ok_or_else(|| state.error("expected seq variable name"))?.to_string();
        Ok(Type::Dollar(name))
    } else {
        Err(state.error("expected type name"))
    }
}