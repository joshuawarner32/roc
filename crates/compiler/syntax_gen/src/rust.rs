use indexmap::IndexMap;

use crate::{ir, ast};


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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generics {
    pub params: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fields {
    Unit,
    Tuple(Vec<Type>),
    Named(Vec<(String, Type)>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Type {
    Named(String),
    Ref(Box<Type>),
    Tuple(Vec<Type>),
    Array(Box<Type>),
    Generics(String, Vec<Type>),
    Field(String, Box<Type>),
    Option(Box<Type>),
    Repeat(Box<Type>),
    Whitespace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltIn {
    String,
    Int,
    Bool,
}

pub fn to_rust_syntax(ir: &ir::Syntax, options: Options) -> Syntax {
    let mut ctx = Ctx {
        options,
        ir,
        extra_token_types: IndexMap::new(),
    };

    let mut items: Vec<Item> = ir.items.iter().map(|item| ctx.to_rust_item(item)).collect();

    items.extend(ctx.extra_token_types.drain(..).map(|(_, item)| item));

    Syntax {
        items,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub generate_loc: bool,
    pub generate_tokens: bool,
    pub generate_trivia: bool,
}

struct Ctx<'a> {
    options: Options,
    ir: &'a ir::Syntax,
    extra_token_types: IndexMap<String, Item>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum FieldKind {
    Named(String),
    Token(String),
    Whitespace(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HowMany {
    One,
    Option,
    Many,
}

struct FieldsAccum {
    names: IndexMap<FieldKind, (Type, HowMany)>,
}

impl FieldsAccum {
    fn insert(&mut self, kind: FieldKind, ty: Type, count: HowMany) {
        match self.names.entry(kind) {
            indexmap::map::Entry::Occupied(existing) => {
                let existing = existing.into_mut();
                existing.1 = HowMany::Many;
                merge_tys(&mut existing.0, ty);
            }
            indexmap::map::Entry::Vacant(entry) => {
                entry.insert((ty, count));
            }
        }
    }

    fn insert_whitespace(&mut self, count: HowMany) {
        let id = self.names.len();
        self.insert(FieldKind::Whitespace(id), Type::Whitespace, count);
    }
}

fn merge_tys(existing: &mut Type, ty: Type) {
    if existing == &ty {
        return;
    }
    todo!()
}

impl<'a> Ctx<'a> {
    fn to_rust_item(&mut self, item: &ir::Item) -> Item {
        match &item.kind {
            ir::ItemKind::Struct { fields } => {
                let fields = self.to_rust_fields(item, fields);
                Item::Struct(Struct {
                    name: item.name.clone(),
                    generics: self.to_rust_generics(&item.generics),
                    fields,
                })
            }
            ir::ItemKind::Enum { variants } => {
                Item::Enum(Enum {
                    name: item.name.clone(),
                    generics: self.to_rust_generics(&item.generics),
                    variants: variants.iter().map(|(name, fields)| {
                        let fields = self.to_rust_fields(item, fields);
                        (name.clone(), fields)
                    }).collect(),
                })
            }
            ir::ItemKind::Typedef(ty) => {

                let mut accum = FieldsAccum {
                    names: IndexMap::new(),
                };

                self.accum_field_types(item, &mut accum, HowMany::One, Some(ty).into_iter());

                let ty = if accum.names.len() == 1 {
                    let name = accum.names.keys().next().unwrap().clone();
                    let (ty, count) = accum.names.remove(&name).unwrap();
                    resolve_ty(ty, count)
                } else {
                    panic!();
                };

                Item::Typedef(Typedef {
                    name: item.name.clone(),
                    generics: self.to_rust_generics(&item.generics),
                    ty,
                })
            }
        }
    }

    fn to_rust_generics(&self, generics: &ast::Generics) -> Generics {
        Generics {
            params: generics.params.iter().filter_map(|g| match g {
                crate::ast::Generic::Type(name) => Some(name.clone()),
                crate::ast::Generic::Dollar(name) => {
                    if self.options.generate_tokens {
                        Some(name.clone())
                    } else {
                        None
                    }
                }
            }).collect(),
        }
    }

    fn to_rust_fields(&mut self, outer_item: &ir::Item, fields: &ir::Fields) -> Fields {
        match fields {
            ir::Fields::Unit => Fields::Unit,
            ir::Fields::Tuple(fields) => Fields::Tuple(fields.iter().filter_map(|f| self.to_rust_type(outer_item, f)).collect()),
            ir::Fields::Named(fields) => Fields::Named(fields.iter().filter_map(|(name, ty)| self.to_rust_type(outer_item, ty).map(|ty| (name.clone(), ty))).collect()),
            ir::Fields::Seq(items) => {
                let mut accum = FieldsAccum {
                    names: IndexMap::new(),
                };

                self.accum_field_types(outer_item, &mut accum, HowMany::One, items.iter());

                if accum.names.len() == 1 {
                    let name = accum.names.keys().next().unwrap().clone();
                    let (ty, count) = accum.names.remove(&name).unwrap();
                    Fields::Tuple(vec![resolve_ty(ty, count)])
                } else {
                    let field_names = accum.names.keys().map(|kind| {
                        match kind {
                            FieldKind::Named(name) => Some(name.clone()),
                            FieldKind::Token(name) => Some(rust_token_field_name(name.clone())),
                            FieldKind::Whitespace(_id) => None,
                        }
                    }).collect::<Vec<_>>();

                    let fields = accum.names.into_iter().enumerate().map(|(i, (name, (ty, count)))| {
                        let ty = resolve_ty(ty, count);
                        let name = match &field_names[i] {
                            Some(name) => name.clone(),
                            None => {
                                if i == 0 {
                                    field_names.get(1).and_then(|n| n.as_ref()).map(|name| format!("before_{}", name)).unwrap_or("before".to_string())
                                } else {
                                    field_names.get(i - 1).and_then(|n| n.as_ref()).map(|name| format!("after_{}", name)).unwrap_or("after".to_string())
                                }
                            },
                        };
                        (name, ty)
                    }).collect();

                    Fields::Named(fields)
                }
            }
        }
    }

    fn to_token_type(&mut self, text: &str) -> String {
        let name = if text.chars().all(|c| c.is_alphabetic()) {
            // This is a keyword; convert it to a form suitable for a Rust struct name.
            let mut s = String::new();
            let mut it = text.chars();
            s.extend(it.next().unwrap().to_uppercase());
            for c in it {
                s.push(c);
            }
            s.push_str("Kw");
            s
        } else {
            format!("{}Tok", capitalize_name(&name_op(text)))
        };

        // TODO: assert no name clashes
        self.extra_token_types.insert(name.clone(), Item::Struct(Struct {
            name: name.clone(),
            generics: Generics {
                params: Vec::new(),
            },
            fields: Fields::Unit,
        }));
        name

    }

    fn locify(&self, ty: Type) -> Type {
        if self.options.generate_loc {
            Type::Generics("Loc".to_string(), vec![ty])
        } else {
            ty
        }
    }

    fn to_rust_type(&mut self, outer_item: &ir::Item, ty: &ir::Type) -> Option<Type> {
        match ty {
            ir::Type::Literal(text) => {
                if self.options.generate_tokens {
                    Some(Type::Named(self.to_token_type(text)))
                } else {
                    None
                }
            },
            ir::Type::Named(id) => {
                let name = self.to_rust_item_ref(outer_item, id)?;
                Some(self.locify(Type::Named(name)))
            }
            ir::Type::Ref(_) => todo!(),
            ir::Type::Tuple(_) => todo!(),
            ir::Type::Array(_) => todo!(),
            ir::Type::Generics(base, args) => {
                let name = self.to_rust_item_ref(outer_item, base).unwrap();
                Some(Type::Generics(name, args.iter().filter_map(|arg| self.to_rust_type(outer_item, arg)).collect()))
            }
            ir::Type::Field(_, _) => todo!(),
            ir::Type::Option(ty) => {
                Some(Type::Option(Box::new(self.to_rust_type(outer_item, ty)?)))
            }
            ir::Type::Repeat(_) => todo!(),
            ir::Type::Seq(items) => {
                // transform this into a tuple
                let items: Vec<_> = items.iter().filter_map(|item| self.to_rust_type(outer_item, item)).collect();
                if items.len() == 1 {
                    Some(items.into_iter().next().unwrap())
                } else {
                    Some(Type::Tuple(items))
                }
            }
            ir::Type::Unimportant(_) => todo!(),
            ir::Type::Whitespace => {
                if self.options.generate_trivia {
                    Some(Type::Named("Trivia".to_string()))
                } else {
                    None
                }
            }
        }
    }

    fn to_rust_item_ref(&mut self, outer_item: &ir::Item, base: &ir::ItemRef) -> Option<String> {
        match base {
            ir::ItemRef::Id(id) => {
                let item = self.ir.item(*id);
                Some(item.name.clone())
            },
            ir::ItemRef::Generic(id) => {
                match outer_item.lookup_generic(*id) {
                    ast::Generic::Type(ty) => Some(ty.clone()),
                    ast::Generic::Dollar(ty) => {
                        if self.options.generate_tokens {
                            Some(ty.clone())
                        } else {
                            None
                        }
                    }
                }
            },
            ir::ItemRef::Builtin(b) => {
                Some(match b {
                    ir::BuiltinItem::String => "String".to_string(),
                    ir::BuiltinItem::Int => "i32".to_string(),
                    ir::BuiltinItem::Bool => "bool".to_string(),
                    ir::BuiltinItem::Vec => "Vec".to_string(),
                    ir::BuiltinItem::Ident => "Ident".to_string(),
                    ir::BuiltinItem::Box => "Box".to_string(),
                })
            }
        }
    }

    fn accum_field_types<'t, It>(&mut self, outer_item: &ir::Item, accum: &mut FieldsAccum, outer_modifiers: HowMany, tys: It)
        where It: Iterator<Item = &'t ir::Type>
    {
        for ty in tys {
            match ty {
                ir::Type::Literal(text) => {
                    if self.options.generate_tokens {
                        accum.insert(FieldKind::Token(text.clone()), Type::Named(self.to_token_type(text)), outer_modifiers);
                    }
                }
                ir::Type::Named(ty) => {
                    if let Some(ty) = self.to_rust_item_ref(outer_item, ty) {
                        accum.insert(FieldKind::Named(lowercase_name(&ty)), Type::Named(ty), outer_modifiers);
                    }
                }
                ir::Type::Ref(_) => todo!(),
                ir::Type::Tuple(_) => todo!(),
                ir::Type::Array(_) => todo!(),
                ir::Type::Generics(name, args) => {
                    let name = self.to_rust_item_ref(outer_item, name).unwrap();
                    let ty = self.to_rust_type(outer_item, ty).unwrap();
                    accum.insert(FieldKind::Named(lowercase_name(&name)), ty, outer_modifiers);
                }
                ir::Type::Field(name, ty) => {
                    if let Some(ty) = self.to_rust_type(outer_item, ty) {
                        accum.insert(FieldKind::Named(name.clone()), ty, outer_modifiers);
                    }
                }
                ir::Type::Option(ty) => {
                    let modifiers = match outer_modifiers {
                        HowMany::One | HowMany::Option => HowMany::Option,
                        HowMany::Many => HowMany::Many,
                    };
                    self.accum_field_types(outer_item, accum, modifiers, Some(ty.as_ref()).into_iter());
                }
                ir::Type::Repeat(ty) => {
                    self.accum_field_types(outer_item, accum, HowMany::Many, Some(ty.as_ref()).into_iter());
                }
                ir::Type::Seq(item) => {
                    self.accum_field_types(outer_item, accum, outer_modifiers, item.iter());
                }
                ir::Type::Unimportant(_) => {
                    // pass, we skip this field
                }
                ir::Type::Whitespace => {
                    if self.options.generate_trivia {
                        accum.insert_whitespace(outer_modifiers);
                    }
                }
            }
        }
    }
}

fn rust_token_field_name(name: String) -> String {
    if name.chars().all(|c| c.is_alphabetic()) {
        format!("{}_kw", name)
    } else {
        name_op(&name)
    }
}

fn lowercase_name(ty: &str) -> String {
    let mut s = String::new();
    let mut it = ty.chars();
    s.extend(it.next().unwrap().to_lowercase());
    for c in it {
        if c.is_uppercase() {
            s.push('_');
        }
        s.extend(c.to_lowercase());
    }
    s
}

fn capitalize_name(name: &str) -> String {
    let mut s = String::new();
    let mut it = name.chars();
    let mut next_cap = true;
    for c in it {
        if c == '_' {
            next_cap = true;
        } else if next_cap {
            s.extend(c.to_uppercase());
            next_cap = false;
        } else {
            s.push(c);
        }
    }
    s
}


fn name_op(op: &str) -> String {
    let mut text = String::new();
    for c in op.chars() {
        let name = match c {
            '(' => "lparen",
            ')' => "rparen",
            '[' => "lsquare",
            ']' => "rsquare",
            '{' => "lcurly",
            '}' => "rcurly",
            ',' => "comma",
            '.' => "dot",
            ':' => "colon",
            ';' => "semicolon",
            '+' => "plus",
            '-' => "minus",
            '*' => "star",
            '/' => "slash",
            '%' => "percent",
            '^' => "caret",
            '&' => "amp",
            '|' => "pipe",
            '~' => "tilde",
            '!' => "bang",
            '?' => "question",
            '=' => "equals",
            '<' => "lt",
            '>' => "gt",
            _ => todo!("Unknown operator character: {}", c),
        };

        if !text.is_empty() {
            text.push('_');
        }
        text.push_str(name);
    }
    text
}

fn resolve_ty(ty: Type, count: HowMany) -> Type {
    if ty == Type::Whitespace {
        // whitespace isn't affected by option/many (it's an array of CommentsOrNewlines, so we just append to the array)
        return ty;
    }
    match count {
        HowMany::One => ty,
        HowMany::Option => Type::Option(Box::new(ty)),
        HowMany::Many => Type::Array(Box::new(ty)),
    }
}