use std::{collections::VecDeque, fmt::Display};

use bumpalo::Bump;
use roc_module::called_via::{BinOp, UnaryOp};
use roc_parse::{
    ast::{
        AbilityImpls, AssignedField, Base, Collection, Defs, EscapedChar, Expr, ExtractSpaces,
        Header, ImplementsAbilities, ImplementsAbility, ImportAlias, ImportAsKeyword,
        ImportExposingKeyword, ImportedModuleName, Module, Pattern, Spaced, Spaces, StrLiteral,
        StrSegment, Tag, TypeAnnotation, TypeDef, TypeHeader, ValueDef,
    },
    header::{
        ExposedName, ExposesKeyword, GeneratesKeyword, ImportsEntry, ImportsKeyword, KeywordItem,
        ModuleName, PackageEntry, PackageName, PackagesKeyword, PlatformRequires, ProvidesKeyword,
        RequiresKeyword, TypedIdent, WithKeyword,
    },
    ident::{Accessor, UppercaseIdent},
};
use roc_region::all::Loc;

use crate::{collection::Braces, Buf};

pub struct Doc<'a> {
    nodes: Vec<Node<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NodeId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NodeRange {
    begin: NodeId,
    end: NodeId,
}

impl NodeRange {
    fn contains(&self, other: &NodeRange) -> bool {
        self.begin <= other.begin && other.end <= self.end
    }
}

#[derive(Debug, Copy, Clone)]
enum Node<'a> {
    Copy(&'a str),
    OptionalNewline,
    WhenMultiline(&'static str),
    ForcedNewline,
    Space,
    Literal(&'static str),
    Comment(&'a str),
    Group(NodeRange),
    Indent(NodeRange),
}

macro_rules! group {
    ($doc:ident, $body:expr) => {{
        let begin = $doc.begin();
        $body;
        $doc.group_to(begin)
    }};
}

macro_rules! indent {
    ($doc:ident, $body:expr) => {{
        let begin = $doc.begin();
        $body;
        $doc.indent_to(begin)
    }};
}

impl<'a> Doc<'a> {
    fn new() -> Doc<'a> {
        Doc { nodes: Vec::new() }
    }

    fn fully_contained(&self, range: NodeRange) -> bool {
        for i in range.begin.0..range.end.0 {
            match &self.nodes[i] {
                Node::Group(r) | Node::Indent(r) => {
                    if !range.contains(r) {
                        return false;
                    }
                }
                _ => {}
            }
        }
        true
    }

    fn push(&mut self, node: Node<'a>) {
        let a = self.nodes.len();
        self.nodes.push(node);
    }

    fn copy(&mut self, text: &'a str) {
        self.push(Node::Copy(text))
    }

    fn literal(&mut self, text: &'static str) {
        self.push(Node::Literal(text))
    }

    fn begin(&mut self) -> NodeId {
        NodeId(self.nodes.len())
    }

    fn indent_to(&mut self, begin: NodeId) {
        let end = self.nodes.len();
        self.push(Node::Indent(NodeRange {
            begin,
            end: NodeId(end),
        }));
        let end = NodeId(self.nodes.len());
        let res = NodeRange { begin, end };
        debug_assert!(self.fully_contained(res));
    }

    fn group_to(&mut self, begin: NodeId) {
        let end = self.nodes.len();
        self.push(Node::Group(NodeRange {
            begin,
            end: NodeId(end),
        }));
        let end = NodeId(self.nodes.len());
        let res = NodeRange { begin, end };
        debug_assert!(self.fully_contained(res));
    }

    fn space(&mut self) {
        self.push(Node::Space)
    }
}

// #[derive(Debug)]
// pub enum Doc<'a> {
//     Copy(&'a str),
//     OptionalNewline,
//     WhenMultiline(Box<Doc<'a>>),
//     ForcedNewline,
//     Space,
//     Literal(&'static str),
//     Comment(&'a str),
//     Concat(Vec<Doc<'a>>),
//     Group(Vec<Doc<'a>>),
//     Indent(Box<Doc<'a>>),
// }

enum Prec {
    TaskAwaitBang,
}

pub fn doc_fmt_module(header: Option<&Module>, defs: Option<&Defs>) -> String {
    let mut doc = Doc::new();
    if let Some(header) = header {
        header.docify(&mut doc);
    }
    doc.push(Node::ForcedNewline);
    doc.push(Node::ForcedNewline);
    if let Some(defs) = defs {
        defs.docify(&mut doc);
    }
    doc.render(50)
}

pub fn doc_fmt_expr(expr: &Expr) -> String {
    let mut doc = Doc::new();
    expr.docify(&mut doc);
    doc.render(50)
}

trait Docify<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>);
}

impl<'a> Docify<'a> for Module<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        self.header.docify(doc)
    }
}

impl<'a, T: Docify<'a>> Docify<'a> for Loc<T> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        self.value.docify(doc)
    }
}

impl<'a, T: Docify<'a>> Docify<'a> for Spaces<'a, T> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        self.item.docify(doc)
    }
}

impl<'a> Docify<'a> for PlatformRequires<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        docify_collection(&self.rigids, Braces::Curly, doc);
        doc.space();
        doc.literal("{");
        doc.space();
        self.signature.docify(doc);
        doc.space();
        doc.literal("}");
    }
}

impl<'a> Docify<'a> for TypedIdent<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        self.ident.docify(doc);
        doc.space();
        doc.literal(":");
        doc.space();
        self.ann.docify(doc);
    }
}

impl<'a> Docify<'a> for Header<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            Header::Module(h) => {
                doc.literal("module");
                doc.space();
                docify_collection(&h.exposes, Braces::Square, doc);
            }
            Header::App(h) => {
                doc.literal("app");
                doc.space();
                docify_collection(&h.provides, Braces::Square, doc);
                docify_collection(&h.packages.value, Braces::Curly, doc);
            }
            Header::Package(h) => {
                doc.literal("package");
                doc.space();
                docify_collection(&h.exposes, Braces::Square, doc);
                docify_collection(&h.packages.value, Braces::Curly, doc);
            }
            Header::Platform(h) => {
                group!(doc, {
                    doc.literal("platform");
                    doc.space();
                    doc.literal("\"");
                    h.name.docify(doc);
                    doc.literal("\"");
                    indent!(doc, {
                        doc.push(Node::OptionalNewline);
                        h.requires.docify(doc);
                        doc.push(Node::OptionalNewline);
                        h.exposes.keyword.docify(doc);
                        doc.space();
                        docify_collection(&h.exposes.item, Braces::Square, doc);
                        doc.push(Node::OptionalNewline);
                        h.packages.keyword.docify(doc);
                        doc.space();
                        docify_collection(&h.packages.item, Braces::Curly, doc);
                        doc.push(Node::OptionalNewline);
                        h.imports.keyword.docify(doc);
                        doc.space();
                        docify_collection(&h.imports.item, Braces::Square, doc);
                        doc.push(Node::OptionalNewline);
                        h.provides.keyword.docify(doc);
                        doc.space();
                        docify_collection(&h.provides.item, Braces::Square, doc);
                    });
                });
            }
            Header::Hosted(h) => {
                group!(doc, {
                    doc.literal("hosted");
                    doc.space();
                    h.name.docify(doc);
                    indent!(doc, {
                        doc.push(Node::OptionalNewline);
                        h.exposes.keyword.docify(doc);
                        doc.space();
                        docify_collection(&h.exposes.item, Braces::Square, doc);
                        h.imports.keyword.docify(doc);
                        doc.space();
                        docify_collection(&h.imports.item, Braces::Square, doc);
                        h.generates.docify(doc);
                        doc.space();
                        h.generates_with.keyword.docify(doc);
                        doc.space();
                        docify_collection(&h.generates_with.item, Braces::Square, doc);
                    });
                });
            }
        }
    }
}

fn docify_collection<'a, T: Docify<'a>>(
    items: &'a Collection<'a, T>,
    delim: Braces,
    doc: &mut Doc<'a>,
) {
    group!(doc, {
        match delim {
            Braces::Curly => doc.literal("{"),
            Braces::Square => doc.literal("["),
            Braces::Round => doc.literal("("),
        };
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                doc.literal(",");
                doc.space();
            }
            item.docify(doc);
        }
        match delim {
            Braces::Curly => doc.literal("}"),
            Braces::Square => doc.literal("]"),
            Braces::Round => doc.literal(")"),
        };
    })
}

impl<'a> Docify<'a> for ExposedName<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.copy(self.as_str())
    }
}

impl<'a> Docify<'a> for PackageName<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.copy(self.as_str())
    }
}

impl<'a> Docify<'a> for ModuleName<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.copy(self.as_str())
    }
}

impl<'a> Docify<'a> for UppercaseIdent<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.copy(self.into())
    }
}

impl<'a> Docify<'a> for ImportedModuleName<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        let begin = doc.begin();
        if let Some(package) = &self.package {
            package.docify(doc);
            doc.literal(".");
        }
        self.name.docify(doc);
    }
}

impl<'a> Docify<'a> for ImportsEntry<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        // old:
        //
        // use roc_parse::header::ImportsEntry::*;

        // buf.indent(indent);

        // match entry {
        //     Module(module, loc_exposes_entries) => {
        //         buf.push_str(module.as_str());

        //         if !loc_exposes_entries.is_empty() {
        //             buf.push('.');

        //             fmt_collection(
        //                 buf,
        //                 indent,
        //                 Braces::Curly,
        //                 *loc_exposes_entries,
        //                 Newlines::No,
        //             )
        //         }
        //     }

        //     Package(pkg, name, entries) => {
        //         buf.push_str(pkg);
        //         buf.push('.');
        //         buf.push_str(name.as_str());

        //         if !entries.is_empty() {
        //             buf.push('.');

        //             fmt_collection(buf, indent, Braces::Curly, *entries, Newlines::No)
        //         }
        //     }

        //     IngestedFile(file_name, typed_ident) => {
        //         fmt_str_literal(buf, *file_name, indent);
        //         buf.push_str_allow_spaces(" as ");
        //         typed_ident.format(buf, 0);
        //     }
        // }
        //

        match self {
            ImportsEntry::Module(module, entries) => {
                module.docify(doc);
                if !entries.is_empty() {
                    doc.literal(".");
                    docify_collection(entries, Braces::Curly, doc);
                }
            }
            ImportsEntry::Package(pkg, name, entries) => {
                pkg.docify(doc);
                doc.literal(".");
                name.docify(doc);
                if !entries.is_empty() {
                    doc.literal(".");
                    docify_collection(entries, Braces::Curly, doc);
                }
            }
            ImportsEntry::IngestedFile(file_name, typed_ident) => {
                docify_str(file_name, doc);
                doc.space();
                doc.literal("as");
                doc.space();
                typed_ident.docify(doc);
            }
        }
    }
}

impl<'a> Docify<'a> for PackageEntry<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        let begin = doc.begin();
        doc.copy(self.shorthand);
        doc.literal(":");
        doc.space();

        if let Some(_) = self.platform_marker {
            doc.literal("platform");
            doc.space();
        }
        doc.literal("\"");
        self.package_name.docify(doc);
        doc.literal("\"");
        doc.group_to(begin)
    }
}

impl<'a> Docify<'a> for Defs<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        let begin = doc.begin();
        for def in self.defs() {
            match def {
                Ok(ty) => ty.docify(doc),
                Err(val) => val.docify(doc),
            };
            doc.push(Node::ForcedNewline);
        }
        doc.group_to(begin)
    }
}

impl<'a> Docify<'a> for Expr<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            Expr::Float(text) | Expr::Num(text) => doc.copy(text),
            Expr::NonBase10Int {
                string,
                base,
                is_negative,
            } => {
                let begin = doc.begin();
                doc.literal(if *is_negative { "-" } else { "" });
                doc.literal(match base {
                    Base::Hex => "0x",
                    Base::Octal => "0o",
                    Base::Binary => "0b",
                    Base::Decimal => "",
                });
                doc.copy(string);
            }

            Expr::Str(lit) => docify_str(lit, doc),
            Expr::SingleQuote(_) => todo!(),

            Expr::RecordAccess(rec, field) | Expr::TupleAccess(rec, field) => {
                let begin = doc.begin();
                let rec = rec.docify(doc);
                doc.literal(".");
                let field = doc.copy(field);
            }
            Expr::AccessorFunction(Accessor::TupleIndex(name))
            | Expr::AccessorFunction(Accessor::RecordField(name)) => {
                let begin = doc.begin();
                doc.literal(".");
                let name = doc.copy(name);
            }

            Expr::TaskAwaitBang(inner) => {
                let begin = doc.begin();
                inner.docify(doc);
                doc.literal("!");
            }

            Expr::List(items) => {
                let begin = doc.begin();
                doc.literal("[");

                let begin_indent = doc.begin();

                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        doc.literal(",");
                    }
                    doc.push(Node::OptionalNewline);
                    item.value.docify(doc);
                }

                if !items.is_empty() {
                    doc.push(Node::WhenMultiline(","));
                    doc.push(Node::OptionalNewline);
                }

                doc.indent_to(begin_indent);

                doc.literal("]");
                doc.group_to(begin)
            }
            Expr::Underscore(name) => {
                let begin = doc.begin();
                doc.literal("_");
                doc.copy(name); // might be empty
            }

            Expr::Record(fields) => {
                let begin = doc.begin();
                doc.literal("{");

                let begin_indent = doc.begin();

                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        doc.literal(",");
                    }
                    doc.push(Node::OptionalNewline);
                    field.value.docify(doc);
                }

                if !fields.is_empty() {
                    doc.push(Node::WhenMultiline(","));
                    doc.push(Node::OptionalNewline);
                }

                doc.indent_to(begin_indent);

                doc.literal("}");
                doc.group_to(begin)
            }

            Expr::RecordUpdate { update, fields } => {
                let begin = doc.begin();
                doc.literal("{");

                let begin_indent = doc.begin();

                doc.push(Node::OptionalNewline);

                update.value.docify(doc);

                doc.space();
                doc.literal("&");

                doc.push(Node::OptionalNewline);

                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        doc.literal(",");
                    }
                    doc.push(Node::OptionalNewline);
                    field.value.docify(doc);
                }

                if !fields.is_empty() {
                    doc.push(Node::WhenMultiline(","));
                    doc.push(Node::OptionalNewline);
                }

                doc.indent_to(begin_indent);

                doc.literal("}");
                doc.group_to(begin)
            }
            Expr::Tuple(fields) => {
                let begin = doc.begin();
                doc.literal("(");

                let begin_indent = doc.begin();

                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        doc.literal(",");
                    }
                    doc.push(Node::OptionalNewline);
                    field.value.docify(doc);
                }

                if !fields.is_empty() {
                    doc.push(Node::WhenMultiline(","));
                    doc.push(Node::OptionalNewline);
                }

                doc.indent_to(begin_indent);

                doc.literal(")");
                doc.group_to(begin)
            }
            Expr::RecordBuilder(_) => todo!(),
            Expr::Var { module_name, ident } => {
                if module_name.is_empty() {
                    doc.copy(ident)
                } else {
                    let begin = doc.begin();
                    doc.copy(module_name);
                    doc.literal(".");
                    doc.copy(ident);
                }
            }
            Expr::Crash => doc.literal("crash"),
            Expr::Tag(name) => doc.copy(name),
            Expr::OpaqueRef(name) => doc.copy(name),
            Expr::Closure(args, body) => {
                let begin = doc.begin();
                doc.literal("\\");
                let arg_begin = doc.begin();
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        doc.literal(",");
                        doc.space();
                    }
                    arg.value.docify(doc);
                }
                doc.indent_to(arg_begin);
                doc.group_to(arg_begin);
                doc.space();
                doc.literal("->");
                let body_begin = doc.begin();
                doc.push(Node::OptionalNewline);
                body.value.docify(doc);
                doc.indent_to(body_begin);
                doc.group_to(begin)
            }
            Expr::Defs(defs, final_expr) => {
                let begin = doc.begin();
                for def in defs.defs() {
                    match def {
                        Ok(ty) => ty.docify(doc),
                        Err(val) => val.docify(doc),
                    };
                    doc.push(Node::ForcedNewline);
                }
                final_expr.value.docify(doc);
                doc.group_to(begin)
            }
            Expr::EmptyDefsFinal => {
                let begin = doc.begin();
            }
            Expr::Backpassing(pats, call, body) => {
                let begin = doc.begin();
                for (i, pat) in pats.iter().enumerate() {
                    if i > 0 {
                        doc.literal(",");
                        doc.space();
                    }
                    pat.value.docify(doc);
                }
                doc.literal("<-");
                doc.space();
                call.value.docify(doc);
                doc.push(Node::ForcedNewline);
                // no indent!
                body.value.docify(doc);
            }
            Expr::Expect(condition, continuation) => {
                group!(doc, {
                    doc.literal("expect");
                    doc.space();
                    condition.value.docify(doc);
                    doc.push(Node::ForcedNewline);
                    continuation.value.docify(doc);
                });
            }
            Expr::Dbg(condition, continuation) => {
                group!(doc, {
                    doc.literal("dbg");
                    doc.space();
                    condition.value.docify(doc);
                    doc.push(Node::ForcedNewline);
                    continuation.value.docify(doc);
                });
            }
            Expr::LowLevelDbg(_, _, _) => todo!(),
            Expr::Apply(func, args, _) => {
                let begin = doc.begin();
                func.value.docify(doc);
                doc.space();
                let begin_indent = doc.begin();
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        doc.space();
                    }
                    arg.value.docify(doc);
                }
                doc.indent_to(begin_indent);
                doc.group_to(begin)
            }
            Expr::BinOps(vals_ops, last) => {
                group!(doc, {
                    for (val, op) in *vals_ops {
                        val.value.docify(doc);
                        doc.space();
                        // doc.push(Node::OptionalNewline); // Doing this is more correct, but hides a bug!
                        // Need to handle how to inject parens around the lambda here:
                        // "a string" |> Str.toUtf8 |> List.map \byte -> byte +  1 |>  List.reverse
                        op.value.docify(doc);
                        doc.space();
                    }
                    doc.space();
                    last.value.docify(doc);
                });
            }
            Expr::UnaryOp(expr, op) => {
                let begin = doc.begin();
                op.value.docify(doc);
                // no space!
                expr.value.docify(doc);
                doc.group_to(begin)
            }
            Expr::If(conds_thens, otherwise) => {
                group!(doc, {
                    for (i, (cond, then_expr)) in conds_thens.iter().enumerate() {
                        if i > 0 {
                            doc.push(Node::OptionalNewline);
                            doc.literal("else");
                            doc.space();
                        }
                        doc.literal("if");
                        doc.space();
                        cond.value.docify(doc);
                        doc.space();
                        doc.literal("then");
                        doc.push(Node::OptionalNewline);
                        then_expr.value.docify(doc);
                    }

                    doc.push(Node::OptionalNewline);
                    doc.literal("else");
                    doc.push(Node::OptionalNewline);
                    otherwise.value.docify(doc);
                })
            }
            Expr::When(cond, branches) => {
                let begin = doc.begin();
                doc.literal("when");
                doc.space();
                cond.value.docify(doc);
                doc.space();
                doc.literal("is");
                let begin_indent = doc.begin();
                for branch in branches.iter() {
                    doc.push(Node::ForcedNewline);
                    let guard_begin = doc.begin();
                    let pats_begin = doc.begin();
                    for (i, pat) in branch.patterns.iter().enumerate() {
                        if i > 0 {
                            doc.push(Node::OptionalNewline);
                            doc.literal("|");
                            doc.space();
                        }
                        pat.value.docify(doc);
                    }
                    doc.group_to(pats_begin);
                    if let Some(guard) = &branch.guard {
                        doc.push(Node::OptionalNewline);
                        doc.literal("if");
                        doc.space();
                        guard.value.docify(doc);
                    }
                    doc.group_to(guard_begin);
                    doc.space();
                    doc.literal("->");
                    let branch_indent = doc.begin();
                    doc.push(Node::OptionalNewline);
                    branch.value.value.docify(doc);
                    doc.indent_to(branch_indent);
                    doc.group_to(branch_indent);
                }
                doc.indent_to(begin_indent);
                doc.push(Node::OptionalNewline);
                doc.group_to(begin)
            }
            Expr::SpaceBefore(expr, _comments) | Expr::SpaceAfter(expr, _comments) => {
                expr.docify(doc)
            }
            Expr::ParensAround(expr) => {
                let begin = doc.begin();
                doc.literal("(");
                expr.docify(doc);
                doc.literal(")");
                doc.group_to(begin)
            }
            Expr::MalformedIdent(_, _) => todo!(),
            Expr::MalformedClosure => todo!(),
            Expr::MalformedSuffixed(_) => todo!(),
            Expr::PrecedenceConflict(_) => todo!(),
            Expr::MultipleRecordBuilders(_) => todo!(),
            Expr::UnappliedRecordBuilder(_) => todo!(),
        }
    }
}

impl<'a> Docify<'a> for TypeDef<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            TypeDef::Alias { header, ann } => {
                header.docify(doc);
                doc.literal(":");
                doc.space();
                ann.value.docify(doc);
            }
            TypeDef::Opaque {
                header,
                typ,
                derived,
            } => {
                group!(doc, {
                    docify_general_def(header, ":=", &typ.value, doc);

                    if let Some(derived) = derived {
                        indent!(doc, {
                            doc.push(Node::OptionalNewline);
                            derived.docify(doc);
                        });
                    }
                });
            }
            TypeDef::Ability {
                header,
                loc_implements,
                members,
            } => {
                let newline = if members.len() > 1 {
                    Node::ForcedNewline
                } else {
                    Node::OptionalNewline
                };

                group!(doc, {
                    doc.copy(header.name.value);
                    for pat in header.vars {
                        doc.space();
                        pat.value.docify(doc);
                    }
                    doc.space();
                    doc.literal("implements");
                    indent!(doc, {
                        for member in members.iter() {
                            doc.push(newline);
                            member.name.value.docify(doc);
                            doc.literal(":");
                            doc.space();
                            member.typ.docify(doc);
                        }
                    });
                })
            }
        }
    }
}

fn docify_general_def<'a, T: Docify<'a>>(
    t: &'a T,
    op: &'static str,
    ty: &'a TypeAnnotation<'a>,
    doc: &mut Doc<'a>,
) {
    t.docify(doc);
    doc.space();
    doc.literal(op);
    doc.space();
    ty.docify(doc);
}

impl<'a> Docify<'a> for AbilityImpls<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            AbilityImpls::AbilityImpls(items) => docify_collection(items, Braces::Curly, doc),
            AbilityImpls::SpaceBefore(_, _) => todo!(),
            AbilityImpls::SpaceAfter(_, _) => todo!(),
        }
    }
}

impl<'a> Docify<'a> for ImplementsAbility<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            ImplementsAbility::ImplementsAbility { ability, impls } => {
                ability.docify(doc);
                if let Some(impls) = impls {
                    doc.space();
                    impls.value.docify(doc);
                }
            }
            ImplementsAbility::SpaceBefore(item, _comments)
            | ImplementsAbility::SpaceAfter(item, _comments) => item.docify(doc),
        }
    }
}

impl<'a> Docify<'a> for ImplementsAbilities<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            ImplementsAbilities::Implements(items) => {
                doc.literal("implements");
                doc.space();
                docify_collection(items, Braces::Square, doc)
            }
            ImplementsAbilities::SpaceBefore(item, _comments)
            | ImplementsAbilities::SpaceAfter(item, _comments) => item.docify(doc),
        }
    }
}

impl<'a> Docify<'a> for UnaryOp {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            UnaryOp::Negate => doc.literal("-"),
            UnaryOp::Not => doc.literal("!"),
        }
    }
}

impl<'a> Docify<'a> for BinOp {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            BinOp::Caret => doc.literal("^"),
            BinOp::Star => doc.literal("*"),
            BinOp::Slash => doc.literal("/"),
            BinOp::DoubleSlash => doc.literal("//"),
            BinOp::Percent => doc.literal("%"),
            BinOp::Plus => doc.literal("+"),
            BinOp::Minus => doc.literal("-"),
            BinOp::Pizza => doc.literal("|>"),
            BinOp::Equals => doc.literal("=="),
            BinOp::NotEquals => doc.literal("!="),
            BinOp::LessThan => doc.literal("<"),
            BinOp::GreaterThan => doc.literal(">"),
            BinOp::LessThanOrEq => doc.literal("<="),
            BinOp::GreaterThanOrEq => doc.literal(">="),
            BinOp::And => doc.literal("&&"),
            BinOp::Or => doc.literal("||"),
            BinOp::Assignment => doc.literal("="),
            BinOp::IsAliasType => doc.literal("is"),
            BinOp::IsOpaqueType => doc.literal("is"),
            BinOp::Backpassing => doc.literal("<-"),
        }
    }
}

impl<'a> Docify<'a> for TypeHeader<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.copy(self.name.value);
        for var in self.vars {
            doc.space();
            docify_pattern_parens(&var.value, doc);
        }
    }
}

impl<'a> Docify<'a> for Tag<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            Tag::Apply { name, args } => {
                group!(doc, {
                    name.value.docify(doc);
                    for arg in *args {
                        doc.space();
                        arg.value.docify(doc);
                    }
                })
            }
            Tag::SpaceBefore(tag, _) | Tag::SpaceAfter(tag, _) => tag.docify(doc),
            Tag::Malformed(_) => todo!(),
        }
    }
}

impl<'a> Docify<'a> for TypeAnnotation<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            TypeAnnotation::Function(args, ret) => {
                let begin = doc.begin();
                let begin_indent = doc.begin();
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        doc.literal(",");
                        doc.space();
                    }
                    arg.value.docify(doc);
                }
                doc.space();
                doc.literal("->");
                doc.space();
                docify_parens(&ret.value, doc);
                doc.indent_to(begin_indent);
                doc.group_to(begin)
            }
            TypeAnnotation::Apply(module, func, args) => {
                let begin = doc.begin();
                doc.copy(module);
                if !module.is_empty() {
                    doc.literal(".");
                }
                doc.copy(func);
                for arg in *args {
                    doc.space();
                    arg.value.docify(doc);
                }
            }
            TypeAnnotation::BoundVariable(name) => doc.copy(name),
            TypeAnnotation::As(ty, _comment, th) => {
                let begin = doc.begin();
                ty.value.docify(doc);
                doc.space();
                doc.literal("as");
                doc.space();
                th.docify(doc);
                doc.group_to(begin)
            }
            TypeAnnotation::Record { fields, ext } => {
                let begin = doc.begin();
                doc.literal("{");
                let begin_indent = doc.begin();
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        doc.literal(",");
                    }
                    field.value.docify(doc);
                }
                doc.indent_to(begin_indent);
                doc.literal("}");
                doc.group_to(begin)
            }
            TypeAnnotation::Tuple { elems, ext } => {
                group!(doc, {
                    doc.literal("(");
                    let begin_indent = doc.begin();
                    for (i, elem) in elems.iter().enumerate() {
                        if i > 0 {
                            doc.literal(",");
                        }
                        elem.value.docify(doc);
                    }
                    doc.indent_to(begin_indent);
                    doc.literal(")");
                    if let Some(ext) = ext {
                        ext.value.docify(doc);
                    }
                })
            }
            TypeAnnotation::TagUnion { ext, tags } => {
                group!(doc, {
                    doc.literal("[");
                    let begin_indent = doc.begin();
                    for (i, tag) in tags.iter().enumerate() {
                        if i > 0 {
                            doc.literal(",");
                        }
                        tag.value.docify(doc);
                    }
                    doc.indent_to(begin_indent);
                    doc.literal("]");
                    if let Some(ext) = ext {
                        ext.value.docify(doc);
                    }
                })
            }
            TypeAnnotation::Inferred => doc.literal("_"),
            TypeAnnotation::Wildcard => doc.literal("*"),
            TypeAnnotation::Where(ty, clauses) => {
                let begin = doc.begin();
                ty.value.docify(doc);
                for (i, clause) in clauses.iter().enumerate() {
                    if i == 0 {
                        doc.space();
                        doc.literal("where");
                    } else {
                        doc.literal(",");
                    }
                    doc.space();
                    clause.value.var.value.docify(doc);
                    doc.space();
                    doc.literal("implements");

                    doc.space();
                    for (i, ability) in clause.value.abilities.iter().enumerate() {
                        if i > 0 {
                            doc.space();
                            doc.literal("&");
                            doc.space();
                        }
                        ability.value.docify(doc);
                    }
                }
            }
            TypeAnnotation::SpaceBefore(ty, _comments)
            | TypeAnnotation::SpaceAfter(ty, _comments) => ty.docify(doc),
            TypeAnnotation::Malformed(_) => todo!(),
        }
    }
}

fn docify_parens<'a>(value: &'a TypeAnnotation, doc: &mut Doc<'a>) {
    let need_parens = match value.extract_spaces().item {
        TypeAnnotation::Function(_, _) => true,
        TypeAnnotation::Apply(_, _, _) => true,
        _ => false,
    };

    if need_parens {
        let begin = doc.begin();
        doc.literal("(");
        value.docify(doc);
        doc.literal(")");
        doc.group_to(begin)
    } else {
        value.docify(doc)
    }
}

impl<'a, T: Docify<'a>> Docify<'a> for Spaced<'a, T> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            Spaced::Item(t) => t.docify(doc),
            Spaced::SpaceBefore(inner, _comment) | Spaced::SpaceAfter(inner, _comment) => {
                inner.docify(doc)
            }
        }
    }
}

impl<'a> Docify<'a> for &'a str {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.copy(self)
    }
}

impl<'a> Docify<'a> for ValueDef<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            ValueDef::Annotation(pat, ty) => {
                let begin = doc.begin();
                pat.value.docify(doc);
                doc.literal(":");
                doc.space();
                ty.value.docify(doc);
            }
            ValueDef::Body(pat, body) => {
                let begin = doc.begin();
                pat.value.docify(doc);
                doc.space();
                doc.literal("=");
                let begin_indent = doc.begin();
                doc.push(Node::OptionalNewline);
                body.value.docify(doc);
                doc.indent_to(begin_indent);
                doc.group_to(begin)
            }
            ValueDef::AnnotatedBody {
                ann_pattern,
                ann_type,
                comment,
                body_pattern,
                body_expr,
            } => {
                let begin = doc.begin();
                ann_pattern.value.docify(doc);
                doc.literal(":");
                doc.space();
                ann_type.value.docify(doc);
                doc.push(Node::ForcedNewline);
                body_pattern.value.docify(doc);
                doc.space();
                doc.literal("=");
                doc.space();
                body_expr.value.docify(doc);
            }
            ValueDef::Dbg {
                condition,
                preceding_comment,
            } => todo!(),
            ValueDef::Expect {
                condition,
                preceding_comment,
            } => todo!(),
            ValueDef::ExpectFx {
                condition,
                preceding_comment,
            } => {
                doc.literal("expect-fx");
                doc.space();
                condition.value.docify(doc);
            }
            ValueDef::ModuleImport(import) => {
                doc.literal("import");

                doc.space();
                import.name.value.docify(doc);

                if let Some(alias) = &import.alias {
                    doc.space();
                    alias.docify(doc);
                }

                if let Some(exposed) = &import.exposed {
                    doc.space();
                    exposed.keyword.docify(doc);
                    doc.space();
                    docify_collection(&exposed.item, Braces::Square, doc);
                }
            }
            ValueDef::IngestedFileImport(import) => {
                doc.literal("import");
                doc.space();
                docify_str(&import.path.value, doc);
                doc.space();
                import.name.docify(doc);
                if let Some(ann) = &import.annotation {
                    doc.space();
                    doc.literal(":");
                    doc.space();
                    ann.annotation.docify(doc);
                }
            }
            ValueDef::Stmt(expr) => expr.value.docify(doc),
        }
    }
}

impl<'a> Docify<'a> for ImportAsKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("as")
    }
}

impl<'a> Docify<'a> for RequiresKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("requires")
    }
}

impl<'a> Docify<'a> for ExposesKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("exposes")
    }
}

impl<'a> Docify<'a> for PackagesKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("packages")
    }
}

impl<'a> Docify<'a> for ImportsKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("imports")
    }
}

impl<'a> Docify<'a> for WithKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("with")
    }
}

impl<'a> Docify<'a> for GeneratesKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("generates")
    }
}

impl<'a> Docify<'a> for ProvidesKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("provides")
    }
}

impl<'a> Docify<'a> for ImportExposingKeyword {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.literal("exposing")
    }
}

impl<'a> Docify<'a> for ImportAlias<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        doc.copy(self.as_str())
    }
}

impl<'a, K: Docify<'a>, T: Docify<'a>> Docify<'a> for KeywordItem<'a, K, T> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        self.keyword.item.docify(doc);
        doc.space();
        self.item.docify(doc)
    }
}

impl<'a, T: Docify<'a>> Docify<'a> for AssignedField<'a, T> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            AssignedField::RequiredValue(name, _comments, value) => {
                let begin = doc.begin();
                doc.push(Node::Copy(name.value));
                doc.literal(":");
                doc.space();
                doc.space();
                value.value.docify(doc);
            }
            AssignedField::OptionalValue(name, _comments, value) => {
                let begin = doc.begin();
                doc.push(Node::Copy(name.value));
                doc.literal("?");
                doc.space();
                value.value.docify(doc);
            }
            AssignedField::LabelOnly(name) => doc.push(Node::Copy(name.value)),
            AssignedField::SpaceBefore(item, _comment)
            | AssignedField::SpaceAfter(item, _comment) => item.docify(doc),
            AssignedField::Malformed(_) => todo!(),
        }
    }
}

impl<'a> Docify<'a> for Pattern<'a> {
    fn docify(&'a self, doc: &mut Doc<'a>) {
        match self {
            Pattern::Identifier { ident } => doc.copy(ident),
            Pattern::QualifiedIdentifier { module_name, ident } => {
                let begin = doc.begin();
                doc.copy(module_name);
                doc.literal(".");
                doc.copy(ident);
            }
            Pattern::Tag(name) => doc.copy(name),
            Pattern::OpaqueRef(name) => doc.copy(name),
            Pattern::Apply(func, args) => {
                let begin = doc.begin();
                func.value.docify(doc);
                for arg in *args {
                    doc.space();
                    docify_pattern_parens(&arg.value, doc);
                }
            }
            Pattern::RecordDestructure(items) => {
                let begin = doc.begin();
                doc.literal("{");
                doc.space();
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        doc.literal(",");
                        doc.space();
                    }
                    item.value.docify(doc);
                }
                doc.space();
                doc.literal("}");
            }
            Pattern::RequiredField(name, pat) => {
                let begin = doc.begin();
                doc.copy(name);
                doc.literal(":");
                doc.space();
                pat.value.docify(doc);
            }
            Pattern::OptionalField(name, pat) => {
                let begin = doc.begin();
                doc.copy(name);
                doc.literal("?");
                doc.space();
                pat.value.docify(doc);
            }
            Pattern::NumLiteral(text) => doc.copy(text),
            Pattern::NonBase10Literal {
                string,
                base,
                is_negative,
            } => todo!(),
            Pattern::FloatLiteral(_) => todo!(),
            Pattern::StrLiteral(lit) => docify_str(lit, doc),
            Pattern::Underscore(name) => {
                let begin = doc.begin();
                doc.literal("_");
                doc.copy(name);
            }
            Pattern::SingleQuote(_) => todo!(),
            Pattern::Tuple(items) => {
                let begin = doc.begin();
                doc.literal("(");
                doc.space();
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        doc.literal(",");
                        doc.space();
                    }
                    item.value.docify(doc);
                }
                doc.space();
                doc.literal(")");
            }
            Pattern::List(items) => {
                let begin = doc.begin();
                doc.literal("[");
                doc.space();
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        doc.literal(",");
                        doc.space();
                    }
                    item.value.docify(doc);
                }
                doc.space();
                doc.literal("]");
            }
            Pattern::ListRest(comment_as_pair) => {
                let begin = doc.begin();
                doc.literal("..");
                if let Some((_, name)) = comment_as_pair {
                    doc.space();
                    doc.literal("as");
                    doc.space();
                    doc.copy(name.identifier.value);
                }
            }
            Pattern::As(pat, pat_as) => {
                let begin = doc.begin();
                pat.value.docify(doc);
                doc.space();
                doc.literal("as");
                doc.space();
                doc.copy(pat_as.identifier.value);
            }
            Pattern::SpaceBefore(pat, _comment) | Pattern::SpaceAfter(pat, _comment) => {
                pat.docify(doc)
            }
            Pattern::Malformed(_) => todo!(),
            Pattern::MalformedIdent(_, _) => todo!(),
        }
    }
}

fn docify_str<'a>(lit: &'a StrLiteral, doc: &mut Doc<'a>) {
    match lit {
        StrLiteral::PlainLine(text) => {
            let begin = doc.begin();
            doc.literal("\"");
            doc.copy(text);
            doc.literal("\"");
        }
        StrLiteral::Line(segments) => {
            let begin = doc.begin();
            doc.literal("\"");
            for segment in *segments {
                match segment {
                    StrSegment::Plaintext(text) => {
                        doc.copy(text);
                    }
                    StrSegment::Unicode(text) => {
                        doc.literal("\\u(");
                        doc.copy(text.value);
                        doc.literal(")");
                    }
                    StrSegment::EscapedChar(escaped) => {
                        doc.literal("\\");
                        match escaped {
                            EscapedChar::Newline => doc.literal("n"),
                            EscapedChar::Tab => doc.literal("t"),
                            EscapedChar::DoubleQuote => doc.literal("\""),
                            EscapedChar::SingleQuote => doc.literal("'"),
                            EscapedChar::Backslash => doc.literal("\\"),
                            EscapedChar::CarriageReturn => doc.literal("r"),
                            EscapedChar::Dollar => doc.literal("$"),
                        };
                    }
                    StrSegment::DeprecatedInterpolated(expr) | StrSegment::Interpolated(expr) => {
                        doc.literal("$(");
                        expr.value.docify(doc);
                        doc.literal(")");
                    }
                }
            }
            doc.literal("\"");
        }
        StrLiteral::Block(_) => todo!(),
    }
}

fn docify_pattern_parens<'a>(value: &'a Pattern, doc: &mut Doc<'a>) {
    let needs_parens = match value.extract_spaces().item {
        Pattern::Apply(_, _) => true,
        _ => false,
    };

    if needs_parens {
        let begin = doc.begin();
        doc.literal("(");
        value.docify(doc);
        doc.literal(")");
    } else {
        value.docify(doc)
    }
}

impl<'a> Doc<'a> {
    fn bubble_up<T: Copy>(
        &self,
        map: impl Fn(usize, Node) -> T,
        aggregate: impl Fn(T, T) -> T,
    ) -> Vec<T> {
        let mut res = Vec::with_capacity(self.nodes.len());
        let mut stack = Vec::<(usize, T)>::new(); // todo: split into two parallel stacks

        for (i, node) in self.nodes.iter().enumerate() {
            let mut t = map(i, *node);

            match node {
                Node::Copy(_)
                | Node::Literal(_)
                | Node::Space
                | Node::OptionalNewline
                | Node::WhenMultiline(_)
                | Node::ForcedNewline
                | Node::Comment(_) => {}
                Node::Group(range) | Node::Indent(range) => {
                    while let Some((i, v)) = stack.pop() {
                        if i < range.begin.0 {
                            stack.push((i, v));
                            break;
                        }
                        t = aggregate(t, v);
                    }
                }
            };

            res.push(t);
            stack.push((i, t));
        }

        res
    }

    fn compute_must_be_multiline(&self) -> Vec<bool> {
        self.bubble_up(
            |_, node| match node {
                Node::Copy(_)
                | Node::Literal(_)
                | Node::Space
                | Node::OptionalNewline
                | Node::WhenMultiline(_)
                | Node::Group(_)
                | Node::Indent(_) => false,
                Node::ForcedNewline | Node::Comment(_) => true,
            },
            |a, b| a | b,
        )
    }

    fn compute_width_without_newlines(&self) -> Vec<usize> {
        self.bubble_up(
            |_, node| match node {
                Node::Copy(s) | Node::Literal(s) => s.len(),
                Node::Space | Node::OptionalNewline => 1,
                Node::WhenMultiline(_) => 0,
                Node::ForcedNewline => 0,
                Node::Comment(s) => s.len(),
                Node::Group(_) | Node::Indent(_) => 0,
            },
            |a, b| a + b,
        )
    }

    fn compute_indents(&self) -> VecDeque<u16> {
        let mut res = VecDeque::with_capacity(self.nodes.len());
        let mut stack = Vec::<usize>::new();

        for (i, node) in self.nodes.iter().enumerate().rev() {
            while let Some(top) = stack.pop() {
                if top <= i {
                    stack.push(top);
                    break;
                }
            }
            res.push_front(stack.len() as u16);
            if let Node::Indent(range) = node {
                stack.push(range.begin.0);
            }
        }

        res
    }

    fn compute_is_multiline(
        &self,
        must_be_multiline: &[bool],
        width_without_newlines: &[usize],
        _indents: &VecDeque<u16>, // TODO: use this in computing width?
        max_width: usize,
    ) -> Vec<bool> {
        self.bubble_up(
            |i, _| must_be_multiline[i] || width_without_newlines[i] > max_width,
            |a, b| a | b,
        )
    }

    fn compute_honor_newlines(&self, is_multiline: &[bool]) -> VecDeque<bool> {
        let mut stack = Vec::<(usize, bool)>::new();
        let mut res = VecDeque::with_capacity(self.nodes.len());
        let mut honor_newlines = false;

        for (i, node) in self.nodes.iter().enumerate().rev() {
            while let Some((top, hn)) = stack.pop() {
                if top <= i {
                    stack.push((top, hn));
                    break;
                }
                honor_newlines = hn;
            }
            res.push_front(honor_newlines || is_multiline[i]);
            if let Node::Group(range) = node {
                stack.push((range.begin.0, is_multiline[i]));
                honor_newlines = is_multiline[i];
            }
        }

        res
    }

    fn render(&self, max_width: usize) -> String {
        let arena = Bump::new();
        let mut buf = Buf::new_in(&arena);
        self.render_inner(max_width, &mut buf);
        buf.as_str().to_string()
    }

    fn render_inner(&self, max_width: usize, buf: &mut Buf<'_>) {
        let must_be_multiline = self.compute_must_be_multiline();
        let width_without_newlines = self.compute_width_without_newlines();
        let indents = self.compute_indents();
        let is_multiline = self.compute_is_multiline(
            &must_be_multiline,
            &width_without_newlines,
            &indents,
            max_width,
        );
        let honor_newlines = self.compute_honor_newlines(&is_multiline);

        for i in 0..must_be_multiline.len() {
            println!(
                "{:5} {:5} {:5} {:5} {:5} {:?}",
                must_be_multiline[i],
                width_without_newlines[i],
                is_multiline[i],
                indents[i],
                honor_newlines[i],
                self.nodes[i]
            );
        }

        for ((node, honor_newlines), indent) in self
            .nodes
            .iter()
            .zip(honor_newlines.into_iter())
            .zip(indents.into_iter())
        {
            let indent = indent * 4;
            match node {
                Node::OptionalNewline => {
                    if honor_newlines {
                        buf.newline();
                    } else {
                        buf.spaces(1)
                    }
                }
                Node::WhenMultiline(s) => {
                    if honor_newlines {
                        buf.indent(indent);
                        buf.push_str(s)
                    }
                }
                Node::Space => buf.spaces(1),
                Node::Copy(s) | Node::Literal(s) => {
                    buf.indent(indent);
                    buf.push_str(s)
                }
                Node::ForcedNewline => {
                    assert!(honor_newlines);
                    buf.newline()
                }
                Node::Comment(s) => {
                    buf.indent(indent);
                    buf.push_str(s)
                }
                Node::Group(_) | Node::Indent(_) => {}
            }
        }
    }
}

impl Display for Doc<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render(10))
    }
}
