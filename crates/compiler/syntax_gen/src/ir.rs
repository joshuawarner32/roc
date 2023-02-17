use std::collections::HashMap;

use crate::ast;



#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Syntax {
    pub items: Vec<Item>,
}
impl Syntax {
    pub(crate) fn item(&self, id: ItemId) -> &Item {
        &self.items[id.0]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    pub name: String,
    pub generics: ast::Generics,
    pub kind: ItemKind,
}
impl Item {
    pub(crate) fn lookup_generic(&self, id: GenericId) -> String {
        self.generics.params[id.0].clone()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ItemKind {
    Struct {
        fields: Fields,
    },
    Enum {
        variants: Vec<(String, Fields)>,
    },
    Typedef(Type),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Generic {
    Type(String),
    Dollar(String),
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
    Named(ItemRef),
    Ref(Box<Type>),
    Tuple(Vec<Type>),
    Array(Box<Type>),
    Generics(ItemRef, Vec<Type>),
    Field(String, Box<Type>),
    Option(Box<Type>),
    Repeat(Box<Type>),
    Seq(Vec<Type>),
    Unimportant(Box<Type>),
}

pub fn convert_syntax(syntax: ast::Syntax) -> Syntax {

    let mut ctx = Ctx { sym_tab: HashMap::new(), generics_sym_tab: HashMap::new() };

    // Insert built-in types
    ctx.define_ty_name("String".to_string(), ItemRef::Builtin(BuiltinItem::String));
    // ctx.define_ty_name("Int".to_string(), ItemRef::Builtin(BuiltinItem::Int));
    ctx.define_ty_name("bool".to_string(), ItemRef::Builtin(BuiltinItem::Bool));
    ctx.define_ty_name("Vec".to_string(), ItemRef::Builtin(BuiltinItem::Vec));
    ctx.define_ty_name("Ident".to_string(), ItemRef::Builtin(BuiltinItem::Ident));
    ctx.define_ty_name("Box".to_string(), ItemRef::Builtin(BuiltinItem::Box));

    for (i, item) in syntax.items.iter().enumerate() {
        let name = match item {
            ast::Item::Struct(d) => &d.name,
            ast::Item::Enum(d) => &d.name,
            ast::Item::Typedef(d) => &d.name,
        };
        let id = ItemId(i);
        ctx.define_ty_name(name.clone(), ItemRef::Id(id));
    }

    let mut items = Vec::new();
    for item in syntax.items {
        let item = ctx.convert_item(item);
        items.push(item);
    }

    Syntax { items }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ItemId(usize);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ItemRef {
    Id(ItemId),
    Generic(GenericId),
    Builtin(BuiltinItem),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinItem {
    String,
    Int,
    Bool,
    Vec,
    Ident,
    Box,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GenericId(usize);

struct Ctx {
    sym_tab: HashMap<String, ItemRef>,
    generics_sym_tab: HashMap<String, GenericId>,
}

impl Ctx {
    fn convert_item(&mut self, item: ast::Item) -> Item {


        enum InProgress {
            Struct(ast::Fields),
            Enum(Vec<(String, ast::Fields)>),
            Typedef(ast::Type),
        }

        let (name, generics, in_progress) = match item {
            ast::Item::Struct(d) => {
                (d.name, d.generics, InProgress::Struct(d.fields))
            }
            ast::Item::Enum(d) => {
                (d.name, d.generics, InProgress::Enum(d.variants))
            },
            ast::Item::Typedef(d) => {
                (d.name, d.generics, InProgress::Typedef(d.ty))
            }
        };

        assert!(self.generics_sym_tab.is_empty());

        for (i, generic) in generics.params.iter().enumerate() {
            let id = GenericId(i);
            self.generics_sym_tab.insert(generic.clone(), id);
        }

        let kind = match in_progress {
            InProgress::Struct(fields) => {
                let fields = self.convert_fields(fields);
                ItemKind::Struct { fields }
            },
            InProgress::Enum(variants) => {
                let mut res = Vec::new();
                for (name, fields) in variants {
                    let fields = self.convert_fields(fields);
                    res.push((name, fields));
                }
                ItemKind::Enum { variants: res }
            },
            InProgress::Typedef(ty) => {
                let ty = self.convert_type(ty);
                ItemKind::Typedef(ty)
            }
        };

        self.generics_sym_tab.clear();

        Item { name, generics, kind }
    }

    fn convert_fields(&mut self, fields: ast::Fields) -> Fields {
        match fields {
            ast::Fields::Unit => Fields::Unit,
            ast::Fields::Tuple(tys) => {
                let mut res = Vec::new();
                for ty in tys {
                    let ty = self.convert_type(ty);
                    res.push(ty);
                }
                Fields::Tuple(res)
            },
            ast::Fields::Named(fields) => {
                let mut res = Vec::new();
                for (name, ty) in fields {
                    let ty = self.convert_type(ty);
                    res.push((name, ty));
                }
                Fields::Named(res)
            }
            ast::Fields::Seq(seq) => {
                let mut res = Vec::new();
                for ty in seq {
                    let ty = self.convert_type(ty);
                    res.push(ty);
                }
                Fields::Seq(res)
            }
        }
    }

    fn define_ty_name(&mut self, name: String, item: ItemRef) {
        assert!(self.sym_tab.insert(name, item).is_none());
    }

    fn lookup_ty_name(&self, name: &str) -> ItemRef {

        if let Some(item) = self.generics_sym_tab.get(name) {
            return ItemRef::Generic(*item);
        }

        *self.sym_tab.get(name).unwrap_or_else(|| {
            panic!("Unknown type name: {}", name);
        })
    }

    fn convert_type(&mut self, ty: ast::Type) -> Type {
        match ty {
            ast::Type::Literal(text) => Type::Literal(text),
            ast::Type::Named(name) => {
                let base = self.lookup_ty_name(&name);
                Type::Named(base)
            }
            ast::Type::Ref(inner) => {
                let inner = self.convert_type(*inner);
                Type::Ref(Box::new(inner))
            }
            ast::Type::Tuple(items) => {
                let items = items.into_iter().map(|ty| self.convert_type(ty)).collect();
                Type::Tuple(items)
            }
            ast::Type::Array(_) => todo!(),
            ast::Type::Generics(name, args) => {
                let base = self.lookup_ty_name(&name);
                let args = args.into_iter().map(|ty| self.convert_type(ty)).collect();
                Type::Generics(base, args)
            }
            ast::Type::Field(name, ty) => {
                let ty = self.convert_type(*ty);
                Type::Field(name, Box::new(ty))
            }
            ast::Type::Option(inner) => {
                let inner = self.convert_type(*inner);
                Type::Option(Box::new(inner))
            }
            ast::Type::Repeat(inner) => {
                let inner = self.convert_type(*inner);
                Type::Repeat(Box::new(inner))
            }
            ast::Type::Seq(seq) => {
                let seq = seq.into_iter().map(|ty| self.convert_type(ty)).collect();
                Type::Seq(seq)
            }
            ast::Type::Unimportant(inner) => {
                let inner = self.convert_type(*inner);
                Type::Unimportant(Box::new(inner))
            }
        }
    }
}