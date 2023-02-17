use crate::rust::*;


pub fn generate_syntax(syntax: &Syntax) -> String {
    let mut res = String::new();
    for item in &syntax.items {
        match item {
            Item::Struct(s) => {
                res.push_str(&format!("pub struct {}", s.name));
                generate_generics(&mut res, &s.generics);
                generate_fields(&mut res, &s.fields, true, 1, true);
                res.push('\n');
            }
            Item::Enum(e) => {
                res.push_str(&format!("pub enum {}", e.name));
                generate_generics(&mut res, &e.generics);
                res.push_str(" {\n");
                for (name, fields) in &e.variants {
                    res.push_str(&format!("    {}", name));
                    generate_fields(&mut res, fields, false, 2, false);
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
            res.push_str(param);
        }
        res.push_str(">");
    }
}

fn generate_fields(res: &mut String, fields: &Fields, in_struct: bool, indent: usize, newlines: bool) {
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
                res.push_str(&"    ".repeat(indent));
                res.push_str(&format!("{}{}: {},\n", public, name, generate_ty(&ty)));
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
        Type::Array(ty) => format!("Vec<{}>", generate_ty(&ty)),
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
        Type::Option(ty) => format!("Option<{}>", generate_ty(&ty)),
        _ => panic!("unhanlded type: {:?}", ty),
    }
}
