


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

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.next();
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

    fn consume_ident(&mut self) -> Option<&'a str> {
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
                println!("ident {}", ident);
                self.pos += last;
                Some(ident)
            } else {
                None
            }
        } else {
            None
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Struct {
    name: String,
    fields: Fields,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Enum {
    name: String,
    variants: Vec<(String, Fields)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Fields {
    Unit,
    Single(Box<Type>),
    Named(Vec<(String, Type)>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Type {
    BuiltIn(BuiltIn),
    Named(String),
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
                generate_fields(&mut res, &s.fields, true, true);
                res.push('\n');
            }
            Item::Enum(e) => {
                res.push_str(&format!("pub enum {} {{\n", e.name));
                for (name, fields) in &e.variants {
                    res.push_str(&format!("    {}", name));
                    generate_fields(&mut res, fields, false, false);
                    res.push_str(",\n");
                }
                res.push_str("}\n");
            }
        }
    }
    res
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
        Fields::Single(ty) => {
            res.push_str("(");
            res.push_str(&format!("    {}{},\n", public, generate_ty(&ty)));
            res.push_str(")");
        }
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

    if let Some(name) = state.consume_ident() {
        let name = name.to_string();
        state.skip_whitespace();
        state.require("=")?;
        state.skip_whitespace();
        if state.peek() == Some('(') {
            // struct
            let fields = parse_fields(state)?;

            state.skip_whitespace();
            state.require(")")?;
            state.skip_whitespace();
            state.require(";")?;
            Ok(Item::Struct(Struct { name, fields }))
        } else {
            // enum
            let mut variants = Vec::new();
            loop {
                let variant_name = state.consume_ident().ok_or_else(|| state.error("expected variant name"))?.to_string();
                let fields = parse_fields(state)?;
                variants.push((variant_name, fields));
                state.skip_whitespace();

                if state.maybe("|") {
                    state.skip_whitespace();
                    continue;
                } else {
                    state.require(";")?;
                    break;
                }
            }
            Ok(Item::Enum(Enum { name, variants }))
        }
    } else {
        Err(state.error("expected ident"))
    }

}

fn parse_fields(state: &mut State) -> Result<Fields, Error> {
    state.skip_whitespace();

    if state.maybe("(") {
        state.skip_whitespace();

        // First let's try to parse a <ident> : <type> list
        let state_clone = state.clone();

        match parse_fields_named(state) {
            Ok(fields) => {
                Ok(fields)
            }
            Err(e) => {
                *state = state_clone;

                match parse_fields_single(state) {
                    Ok(fields) => {
                        Ok(fields)
                    }
                    Err(_) => {
                        Err(e) // return the original error
                    }
                }
            }
        }

    } else {
        return Ok(Fields::Unit);
    }
}

fn parse_fields_single(state: &mut State) -> Result<Fields, Error> {
    let ty = parse_type(state)?;
    Ok(Fields::Single(Box::new(ty)))
}

fn parse_fields_named(state: &mut State) -> Result<Fields, Error> {
    let mut fields = Vec::new();
    loop {
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

fn parse_type(state: &mut State) -> Result<Type, Error> {
    if let Some(name) = state.consume_ident() {
        Ok(Type::Named(name.to_string()))
    } else {
        Err(state.error("expected type name"))
    }
}