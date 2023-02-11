mod libparser;
mod parser;

use std::{collections::{HashMap, HashSet}, fmt::{Display, self}};

#[derive(Debug, Clone)]
enum TextRule {
    Recur(String),
    Tok(Token),
    Sequence(Vec<TextRule>),
    Choice(Vec<TextRule>),
    Optional(Vec<TextRule>),
    Repeat(Vec<TextRule>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Token {
    Identifier,
    Keyword(String),
    Operator(String),

    LiteralString,
    LiteralSingleQuote,
    LiteralInteger,
    LiteralFloat,

    CommentsAndNewlines,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct TokenId(usize);
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct RuleId(usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Rule {
    Recur(RuleId),
    Tok(TokenId),
    Sequence(Vec<RuleId>),
    Choice(Vec<RuleId>),
    Optional(RuleId),
    Repeat(RuleId),
}

#[derive(Debug)]
struct TextGrammar {
    rules: HashMap<String, TextRule>,
}

#[derive(Debug)]
struct IdGrammar {
    tokens: Vec<Token>,
    rules: Vec<Rule>,
    names: HashMap<String, RuleId>,
    reverse_names: HashMap<RuleId, String>,

    root: RuleId,
}

impl TextGrammar {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    fn add_rule(&mut self, name: String, rule: TextRule) {
        self.rules.insert(name, rule);
    }

    fn to_id_grammar(&self) -> IdGrammar {
        let mut root = None;

        let mut tokens = Vec::new();
        let mut rules = Vec::new();

        let mut token_map = HashMap::new();
        let mut rule_map = HashMap::new();

        let mut names = HashMap::new();
        let mut reverse_names = HashMap::new();

        for (name, rule) in self.rules.iter() {
            let rule_id = RuleId(rules.len());
            rules.push(Rule::Recur(RuleId(0))); // Placeholder, will be overwritten later
            names.insert(name.clone(), rule_id);
            reverse_names.insert(rule_id, name.clone());
            rule_map.insert(name.clone(), rule_id);

            if name == "root" {
                root = Some(rule_id);
            }
        }

        for (name, rule) in self.rules.iter() {
            let rule = self.to_id_rule(rule, &mut tokens, &mut rules, &mut token_map, &mut rule_map);
            rules[rule_map[name].0] = rule;
        }

        assert!(tokens.len() < 127); // so we can use a u128 to represent a set of tokens, with room for a special "end of input" token

        IdGrammar {
            tokens,
            rules,
            names,
            reverse_names,
            root: root.unwrap(),
        }
    }

    fn to_rule_id(
        &self,
        rule: &TextRule,
        tokens: &mut Vec<Token>,
        rules: &mut Vec<Rule>,
        token_map: &mut HashMap<Token, TokenId>,
        rule_map: &mut HashMap<String, RuleId>,
    ) -> RuleId {
        let rule = self.to_id_rule(rule, tokens, rules, token_map, rule_map);
        let rule_id = RuleId(rules.len());
        rules.push(rule);
        rule_id
    }

    fn to_id_rule(
        &self,
        rule: &TextRule,
        tokens: &mut Vec<Token>,
        rules: &mut Vec<Rule>,
        token_map: &mut HashMap<Token, TokenId>,
        rule_map: &mut HashMap<String, RuleId>,
    ) -> Rule {
        match rule {
            TextRule::Recur(name) => {
                let rule_id = rule_map.get(name).unwrap();
                Rule::Recur(*rule_id)
            }
            TextRule::Tok(token) => {
                let next_id = TokenId(tokens.len());
                let token_id = *token_map.entry(token.clone()).or_insert_with(|| {
                    tokens.push(token.clone());
                    next_id
                });
                Rule::Tok(token_id)
            }
            TextRule::Sequence(inner_rules) => {
                let rule_ids = inner_rules.iter().map(|rule| self.to_rule_id(rule, tokens, rules, token_map, rule_map)).collect();
                Rule::Sequence(rule_ids)
            }
            TextRule::Choice(inner_rules) => {
                let rule_ids = inner_rules.iter().map(|rule| self.to_rule_id(rule, tokens, rules, token_map, rule_map)).collect();
                Rule::Choice(rule_ids)
            }
            TextRule::Optional(inner_rules) => {
                let inner_rule = self.singular_rule(inner_rules, tokens, rules, token_map, rule_map);
                Rule::Optional(inner_rule)
            }
            TextRule::Repeat(inner_rules) => {
                let inner_rule = self.singular_rule(inner_rules, tokens, rules, token_map, rule_map);
                Rule::Repeat(inner_rule)
            }
        }
    }

    fn singular_rule(&self, inner_rules: &Vec<TextRule>, tokens: &mut Vec<Token>, rules: &mut Vec<Rule>, token_map: &mut HashMap<Token, TokenId>, rule_map: &mut HashMap<String, RuleId>) -> RuleId {
        if inner_rules.len() > 1 {
            // turn this into a sequence of the inner rule and an empty rule
            self.to_rule_id(&TextRule::Sequence(inner_rules.clone()), tokens, rules, token_map, rule_map)
        } else if inner_rules.len() == 1 {
            self.to_rule_id(inner_rules.first().unwrap(), tokens, rules, token_map, rule_map)
        } else {
            panic!("Optional rule must have at least one inner rule");
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct TokenSet(u128);

impl TokenSet {
    fn update(&mut self, other: TokenSet) -> bool {
        let old = self.0;
        self.0 |= other.0;
        old != self.0
    }
}

impl IdGrammar {
    fn compute_nullable(&self) -> Vec<bool> {
        let mut nullable = vec![false; self.rules.len()];

        let mut changed = true;
        while changed {
            changed = false;

            for (rule_id, rule) in self.rules.iter().enumerate() {
                let rule_id = RuleId(rule_id);
                let rule_nullable = self.compute_rule_nullable(rule_id, &nullable) | nullable[rule_id.0];
                if rule_nullable != nullable[rule_id.0] {
                    changed = true;
                    nullable[rule_id.0] = rule_nullable;
                }
            }
        }

        nullable
    }

    fn compute_rule_nullable(&self, rule_id: RuleId, nullable: &Vec<bool>) -> bool {
        let rule = &self.rules[rule_id.0];
        match rule {
            Rule::Recur(rule_id) => nullable[rule_id.0],
            Rule::Tok(_) => false,
            Rule::Sequence(inner_rules) => inner_rules.iter().all(|rule_id| nullable[rule_id.0]),
            Rule::Choice(inner_rules) => inner_rules.iter().any(|rule_id| nullable[rule_id.0]),
            Rule::Optional(_inner_rule) => true,
            Rule::Repeat(_inner_rule) => true,
        }
    }

    fn compute_first_sets(&self, nullable: &[bool]) -> Vec<TokenSet> {
        let mut first_sets = vec![TokenSet(0); self.rules.len()];

        loop {
            let mut changed = false;

            for (rule_id, rule) in self.rules.iter().enumerate() {
                let rule_id = RuleId(rule_id);
                let first_set = self.compute_first_set(rule_id, nullable, &first_sets);
                
                changed |= first_sets[rule_id.0].update(first_set);
            }

            if !changed {
                break;
            }
        }

        first_sets
    }

    fn compute_first_set(&self, rule_id: RuleId, nullable: &[bool], first_sets: &[TokenSet]) -> TokenSet {
        let rule = &self.rules[rule_id.0];
        match rule {
            Rule::Recur(rule_id) => first_sets[rule_id.0],
            Rule::Tok(token_id) => TokenSet(1 << token_id.0),
            Rule::Sequence(inner_rules) => {
                let mut first_set = TokenSet(0);
                for rule_id in inner_rules {
                    let inner_first_set = first_sets[rule_id.0];
                    first_set.0 |= inner_first_set.0;
                    if !nullable[rule_id.0] {
                        break;
                    }
                }
                first_set
            }
            Rule::Choice(inner_rules) => {
                let mut first_set = TokenSet(0);
                for rule_id in inner_rules {
                    let inner_first_set = first_sets[rule_id.0];
                    first_set.0 |= inner_first_set.0;
                }
                first_set
            }
            Rule::Optional(inner_rules) => {
                let inner_first_set = first_sets[inner_rules.0];
                TokenSet(inner_first_set.0)
            }
            Rule::Repeat(inner_rules) => {
                let inner_first_set = first_sets[inner_rules.0];
                TokenSet(inner_first_set.0)
            }
        }
    }

    fn compute_follow_sets(&self, nullable: &[bool], first_sets: &[TokenSet]) -> Vec<TokenSet> {
        let mut follow_sets = vec![TokenSet(0); self.rules.len()];

        follow_sets[self.root.0] = TokenSet(1 << 127);

        loop {
            let mut changed = false;

            for rule in &self.rules {
                changed |= self.update_follow_set(rule, TokenSet(0), nullable, first_sets, &mut follow_sets);
            }

            if !changed {
                break;
            }
        }

        follow_sets
    }

    fn update_follow_set(&self, rule: &Rule, mut tokens: TokenSet, nullable: &[bool], first_sets: &[TokenSet], follow_sets: &mut [TokenSet]) -> bool {
        match rule {
            Rule::Recur(rule_id) => {
                follow_sets[rule_id.0].update(tokens)
            }
            Rule::Tok(_) => false,
            Rule::Sequence(inner_rules) => {
                let mut changed = false;
                for rule_id in inner_rules.iter().rev() {
                    changed |= follow_sets[rule_id.0].update(tokens);
                    if !nullable[rule_id.0] {
                        tokens = first_sets[rule_id.0];
                    } else {
                        tokens.update(first_sets[rule_id.0]);
                    }
                }
                changed
            }
            Rule::Choice(inner_rules) => {
                let mut changed = false;
                for rule_id in inner_rules {
                    changed |= follow_sets[rule_id.0].update(tokens);
                }
                changed
            }
            Rule::Optional(inner_rule) |
            Rule::Repeat(inner_rule) => {
                follow_sets[inner_rule.0].update(tokens)
            }
        }
    }
}

struct DbgTokens<'a>(&'a [Token], TokenSet);

impl std::fmt::Debug for DbgTokens<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (i, token) in self.0.iter().enumerate() {
            if self.1.0 & (1 << i) != 0 {
                if first {
                    first = false;
                } else {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", token)?;
            }
        }
        Ok(())
    }
}

struct DbgRule<'a>(&'a IdGrammar, &'a Rule);

impl std::fmt::Debug for DbgRule<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.1 {
            Rule::Recur(rule_id) => {
                self.dbg_rule_id(rule_id, f)
            }
            Rule::Tok(token_id) => write!(f, "{:?}", self.0.tokens[token_id.0]),
            Rule::Sequence(inner_rules) => {
                write!(f, "(")?;
                let mut first = true;
                for rule_id in inner_rules {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    self.dbg_rule_id(rule_id, f)?;
                }
                write!(f, ")")?;
                Ok(())
            }
            Rule::Choice(inner_rules) => {
                write!(f, "[")?;
                let mut first = true;
                for rule_id in inner_rules {
                    if first {
                        first = false;
                    } else {
                        write!(f, " | ")?;
                    }
                    self.dbg_rule_id(rule_id, f)?;
                }
                write!(f, "]")?;
                Ok(())
            }
            Rule::Optional(inner_rule) => {
                write!(f, "Optional(")?;
                self.dbg_rule_id(inner_rule, f)?;
                write!(f, ")")?;
                Ok(())
            }
            Rule::Repeat(inner_rule) => {
                write!(f, "Repeat(")?;
                self.dbg_rule_id(inner_rule, f)?;
                write!(f, ")")?;
                Ok(())
            }
        }
    }
}

impl<'a> DbgRule<'a> {
    fn dbg_rule_id(&self, rule_id: &RuleId, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        if let Some(name) = self.0.reverse_names.get(rule_id) {
            write!(f, "{:?}", name)
        } else {
            write!(f, "{:?}", rule_id.0)
        }
    }
}

macro_rules! grammar {
    ($($name:ident = $rule:expr),*) => {{
        let mut g = TextGrammar::new();

        $(
            #[allow(unused)]
            let $name = TextRule::Recur(stringify!($name).to_string());
        )*

        #[allow(unused_imports)]
        {

            use TextRule::*;
            use Token::*;
            $(
                g.add_rule(stringify!($name).to_string(), $rule);
            )*
        }

        dbg!(&g);

        g.to_id_grammar()
    }}
}


/// A "synthesizable" grammar - one that can be directly converted into a parser.
struct SynthGrammar {
    types: HashMap<String, SynthData>,
    needs_ref: HashSet<String>,
}

enum SynthData {
    Struct(Vec<(String, SynthType)>),
    Enum(Vec<(String, SynthFields)>),
    Typedef(SynthType),
}

enum SynthType {
    None,
    Type(String),
    Option(Box<SynthType>),
    Vec(Box<SynthType>),
    Tuple(Vec<SynthType>),
    Str,
    StrLiteral,
    SingleQuoteLiteral,
    IntegerLiteral,
    FloatLiteral,
}
impl SynthType {
    fn needs_lifetime(&self) -> bool {
        match self {
            SynthType::None => false,
            SynthType::Type(_) => true,
            SynthType::Option(inner) => inner.needs_lifetime(),
            SynthType::Vec(inner) => inner.needs_lifetime(),
            SynthType::Tuple(inners) => inners.iter().any(|inner| inner.needs_lifetime()),
            SynthType::Str => true,
            SynthType::StrLiteral => true,
            SynthType::SingleQuoteLiteral => true,
            SynthType::IntegerLiteral => true,
            SynthType::FloatLiteral => true,
        }
    }
}

impl SynthType {
    fn format(&self, needs_ref: &HashSet<String>) -> String {
        match self {
            SynthType::None => format!("()"),
            SynthType::Type(name) => {
                if needs_ref.contains(name) {
                    format!("&'a {}<'a>", name)
                } else {
                    format!("{}<'a>", name)
                }
            }
            SynthType::Option(inner) => format!("Option<{}>", inner.format(needs_ref)),
            SynthType::Vec(inner) => format!("Vec<{}>", inner.format(needs_ref)),
            SynthType::Tuple(inners) => {
                let mut res = String::new();
                res.push_str("(");
                let mut first = true;
                for inner in inners {
                    if first {
                        first = false;
                    } else {
                        res.push_str(", ");
                    }
                    res.push_str(&inner.format(needs_ref));
                }
                res.push_str(")");
                res
            }
            SynthType::Str => format!("&'a str"),
            SynthType::StrLiteral => format!("StrLiteral<'a>"),
            SynthType::SingleQuoteLiteral => format!("SingleQuoteLiteral<'a>"),
            SynthType::IntegerLiteral => format!("&'a str"),
            SynthType::FloatLiteral => format!("&'a str"),
        }
    }
}

const FALLBACK_FIELD_NAMES: [&str; 10] = [ "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth" ];

fn camel_case(underscore_name: &str) -> String {
    let mut camel = String::new();
    let mut capitalize_next = true;
    for c in underscore_name.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            camel.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            camel.push(c);
        }
    }
    camel
}

impl SynthGrammar {
    fn new() -> Self {
        Self {
            types: HashMap::new(),
            needs_ref: HashSet::new(),
        }
    }

    fn from_id_grammar(g: &IdGrammar) -> Self {
        let mut synth = Self::new();

        for (rule_id, rule) in g.rules.iter().enumerate() {
            let rule_id = RuleId(rule_id);
            let Some(rule_name) = g.reverse_names.get(&rule_id) else { continue; };
            let rule_name = rule_name.to_string();
            let camel_name = camel_case(&rule_name);

            match rule {
                Rule::Sequence(inner_rules) => {
                    let mut fields = Vec::new();
                    for (i, rule_id) in inner_rules.iter().enumerate() {
                        let field_frag = infer_fragment(g, *rule_id, i);
                        let field_name = frag_to_field_name(&field_frag, i);
                        let field_ty = field_frag.ty;
                        fields.push((field_name, field_ty));
                    } 
                    synth.types.insert(camel_name, SynthData::Struct(fields));
                }
                Rule::Choice(inner_rules) => {
                    let mut variants = Vec::new();
                    for (i, rule_id) in inner_rules.iter().enumerate() {
                        let field_frag = infer_fragment(g, *rule_id, i);
                        let field_name = frag_to_field_name(&field_frag, i);
                        let field_ty = field_frag.ty;
                        let variant_name = camel_case(&field_name);
                        variants.push((variant_name, SynthFields::Tuple(vec![field_ty])));
                    }
                    synth.types.insert(camel_name, SynthData::Enum(variants));
                }
                Rule::Optional(inner_rule) => {
                    // generate a typedef
                    let inner_frag = infer_fragment(g, *inner_rule, 0);
                    let inner_ty = inner_frag.ty;
                    let ty = SynthType::Option(Box::new(inner_ty));
                    synth.types.insert(camel_name, SynthData::Typedef(ty));
                }
                Rule::Repeat(inner_rule) => {
                    // generate a typedef
                    let inner_frag = infer_fragment(g, *inner_rule, 0);
                    let inner_ty = inner_frag.ty;
                    let ty = SynthType::Vec(Box::new(inner_ty));
                    synth.types.insert(camel_name, SynthData::Typedef(ty));
                }
                Rule::Recur(_) => {}
                Rule::Tok(tok) => {
                    // generate a typedef
                    let ty = frag_for_token(&g.tokens[tok.0]).ty;
                    synth.types.insert(camel_name, SynthData::Typedef(ty));
                }
            }
        }

        // Now we do a DFS to find all the types that need to be references
        let mut stack = HashSet::new();
        for (name, data) in &synth.types {
            dfs_for_refs_data(&synth.types, &mut synth.needs_ref, &g.rules, &g.reverse_names, &g.tokens, &mut stack, name, data);
        }

        synth
    }
}

fn dfs_for_refs_data<'a>(types: &'a HashMap<String, SynthData>, needs_ref: &mut HashSet<String>, rules: &[Rule], reverse_names: &HashMap<RuleId, String>, tokens: &[Token], stack: &mut HashSet<&'a str>, name: &'a str, data: &'a SynthData) {
    if stack.contains(name) {
        // We found a cycle, so we need to make this a reference
        needs_ref.insert(name.to_string());
        return;
    }
    if needs_ref.contains(name) {
        // We already know this needs to be a reference
        return;
    }
    stack.insert(name);
    match data {
        SynthData::Typedef(ty) => dfs_for_refs_type(types, needs_ref, rules, reverse_names, tokens, stack, name, ty),
        SynthData::Struct(fields) => {
            for (_, ty) in fields {
                dfs_for_refs_type(types, needs_ref, rules, reverse_names, tokens, stack, name, ty);
            }
        }
        SynthData::Enum(variants) => {
            for (_, fields) in variants {
                match fields {
                    SynthFields::Empty => {}
                    SynthFields::Struct(fields) => {
                        for (_, ty) in fields {
                            dfs_for_refs_type(types, needs_ref, rules, reverse_names, tokens, stack, name, ty);
                        }
                    }
                    SynthFields::Tuple(fields) => {
                        for ty in fields {
                            dfs_for_refs_type(types, needs_ref, rules, reverse_names, tokens, stack, name, ty);
                        }
                    }
                }
            }
        }
    }
    stack.remove(name);
}

fn dfs_for_refs_type<'a>(types: &'a HashMap<String, SynthData>, needs_ref: &mut HashSet<String>, rules: &[Rule], reverse_names: &HashMap<RuleId, String>, tokens: &[Token], stack: &mut HashSet<&'a str>, name: &'a str, ty: &'a SynthType) {
    match ty {
        SynthType::None => {}
        SynthType::Type(name) => {
            if let Some(data) = types.get(name) {
                dfs_for_refs_data(types, needs_ref, rules, reverse_names, tokens, stack, name, data);
            }
        }
        SynthType::Option(inner) => dfs_for_refs_type(types, needs_ref, rules, reverse_names, tokens, stack, name, inner),
        SynthType::Vec(inner) => {
            // vecs are always references
        }
        SynthType::Tuple(inner) => {
            for ty in inner {
                dfs_for_refs_type(types, needs_ref, rules, reverse_names, tokens, stack, name, ty);
            }
        }
        SynthType::Str |
        SynthType::StrLiteral |
        SynthType::SingleQuoteLiteral |
        SynthType::IntegerLiteral |
        SynthType::FloatLiteral => {}
        
    }
}

enum SynthFields {
    Empty,
    Struct(Vec<(String, SynthType)>),
    Tuple(Vec<SynthType>),
}

struct SynthFrag {
    name: Option<String>,
    ty: SynthType,
}

fn frag_to_field_name(frag: &SynthFrag, index: usize) -> String {
    if let Some(name) = &frag.name {
        name.to_string()
    } else {
        FALLBACK_FIELD_NAMES[index].to_string()
    }
}

fn type_to_field_name(ty: &SynthType) -> String {
    match ty {
        SynthType::None => "none".to_string(),
        SynthType::Type(name) => camel_case(name),
        SynthType::Option(_) => "option".to_string(),
        SynthType::Vec(_) => "vec".to_string(),
        SynthType::Tuple(_) => "tuple".to_string(),
        SynthType::Str => "str".to_string(),
        SynthType::StrLiteral => "str_literal".to_string(),
        SynthType::SingleQuoteLiteral => "single_quote_literal".to_string(),
        SynthType::IntegerLiteral => "integer_literal".to_string(),
        SynthType::FloatLiteral => "float_literal".to_string(),
    }
}

fn infer_fragment(g: &IdGrammar, rule_id: RuleId, i: usize) -> SynthFrag {
    if let Some(name) = g.reverse_names.get(&rule_id) {
        return SynthFrag { 
            name: Some(name.clone()),
            ty: SynthType::Type(camel_case(name))
        };
    }

    let rule = &g.rules[rule_id.0];
    match rule {
        Rule::Recur(rule) => infer_fragment(g, *rule, i),
        Rule::Tok(token) => frag_for_token(&g.tokens[token.0]),
        Rule::Sequence(seq) => {
            let mut fields = Vec::new();
            for (i, rule_id) in seq.iter().enumerate() {
                let field_frag = infer_fragment(g, *rule_id, i);
                let field_ty = field_frag.ty;
                fields.push(field_ty);
            } 
            SynthFrag {
                name: None,
                ty: SynthType::Tuple(fields)
            }
        }
        Rule::Choice(_) => todo!(),
        Rule::Optional(inner_rule) => {
            let inner_frag = infer_fragment(g, *inner_rule, i);

            SynthFrag {
                name: inner_frag.name,
                ty: SynthType::Option(Box::new(inner_frag.ty))
            }
        }
        Rule::Repeat(inner_rule) => {
            let inner_frag = infer_fragment(g, *inner_rule, i);

            // If the inner fragment is named, we want to pluralize it.
            // Also wrap the type in a vec
            let name = inner_frag.name.map(|name| {
                if name.ends_with('s') {
                    name + "es"
                } else {
                    name + "s"
                }
            });

            let ty = SynthType::Vec(Box::new(inner_frag.ty));
            SynthFrag {
                name,
                ty,
            }
        }
    }
}

fn infer_field_name(g: &IdGrammar, rule_id: &RuleId, i: usize) -> String {
    if let Some(name) = g.reverse_names.get(rule_id) {
        camel_case(name)
    } else {
        let rule = &g.rules[rule_id.0];
        match rule {
            Rule::Recur(rule) => return infer_field_name(g, rule, i),
            Rule::Tok(tok) => {
                panic!()
            }
            Rule::Sequence(_) => {}
            Rule::Choice(_) => {}
            Rule::Optional(_) => {}
            Rule::Repeat(_) => {}
        }
        FALLBACK_FIELD_NAMES[i].to_string()
    }
}

fn frag_for_token(token: &Token) -> SynthFrag {
    match token {
        Token::Keyword(name) => SynthFrag {
            name: Some(format!("{}_kw", name)),
            ty: SynthType::None,
        },
        Token::Identifier => {
            SynthFrag {
                name: Some("identifier".to_string()),
                ty: SynthType::None,
            }
        }
        Token::Operator(op) => {
            let ty = SynthType::None;
            let name = Some(name_op(op));

            SynthFrag {
                name,
                ty,
            }
        }
        Token::LiteralString => SynthFrag {
            name: Some("string".to_string()),
            ty: SynthType::StrLiteral
        },
        Token::LiteralSingleQuote => SynthFrag {
            name: Some("single_quote".to_string()),
            ty: SynthType::SingleQuoteLiteral
        },
        Token::LiteralInteger => SynthFrag {
            name: Some("integer".to_string()),
            ty: SynthType::IntegerLiteral
        },
        Token::LiteralFloat => SynthFrag {
            name: Some("float".to_string()),
            ty: SynthType::FloatLiteral
        },
        Token::CommentsAndNewlines => SynthFrag {
            name: Some("spaces".to_string()),
            ty: SynthType::Vec(Box::new(SynthType::Type("Space".to_string()))),
        },
    }
}

// fn name_token(token: &Token) -> String {
//     match token {
//         Token::Keyword(name) => return format!("{}_kw", name),
//         Token::Identifier => return "identifier".to_string(),
//         Token::Operator(op) => {
//             if let Some(value) = name_op(op) {
//                 return value;
//             }
//         }
//         Token::LiteralString => return "string".to_string(),
//         Token::LiteralSingleQuote => return "single_quote".to_string(),
//         Token::LiteralInteger => return "integer".to_string(),
//         Token::LiteralFloat => return "float".to_string(),
//     }
// }

fn name_op(op: &str) -> String {
    let mut text = String::new();
    for c in op.chars() {
        let name = match c {
            '(' => "open_paren",
            ')' => "close_paren",
            '[' => "open_square",
            ']' => "close_square",
            '{' => "open_curly",
            '}' => "close_curly",
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
            '&' => "ampersand",
            '|' => "pipe",
            '~' => "tilde",
            '!' => "bang",
            '?' => "question",
            '=' => "equals",
            '<' => "less_than",
            '>' => "greater_than",
            _ => todo!("Unknown operator character: {}", c),
        };

        if !text.is_empty() {
            text.push('_');
        }
        text.push_str(name);
    }
    text
}

fn generate_parser(grammar: &IdGrammar, nullables: &[bool], first_sets: &[TokenSet], follow_sets: &[TokenSet]) -> String {
    let mut s = String::new();

    // First, we generate a series of AST enums/structs for each rule.
    let synth = SynthGrammar::from_id_grammar(grammar);

    s.push_str("use crate::libparser::prelude::*;\n\n");

    let mut types_to_generate = synth.types.into_iter().collect::<Vec<_>>();
    types_to_generate.sort_by_key(|(name, _)| name.clone());
    
    for (name, data) in types_to_generate {
        match data {
            SynthData::Struct(fields) => {
                let needs_lifetime = fields.iter().any(|(_, ty)| ty.needs_lifetime());
                s.push_str(&format!("struct {}{} {{\n", name, if needs_lifetime { "<'a>" } else { "" } ));
                for (field_name, field_type) in fields {
                    s.push_str(&format!("    {}: {},\n", field_name, field_type.format(&synth.needs_ref)));
                }
                s.push_str("}\n");
            }
            SynthData::Enum(variants) => {
                s.push_str(&format!("enum {}<'a> {{\n", name));
                for (variant_name, variant_fields) in variants {
                    match variant_fields {
                        SynthFields::Empty => {
                            s.push_str(&format!("    {},\n", variant_name));
                        }
                        SynthFields::Struct(fields) => {
                            s.push_str(&format!("    {} {{\n", variant_name));
                            for (field_name, field_type) in fields {
                                s.push_str(&format!("        {}: {},\n", field_name, field_type.format(&synth.needs_ref)));
                            }
                            s.push_str("    },\n");
                        }
                        SynthFields::Tuple(fields) => {
                            s.push_str(&format!("    {}(", variant_name));
                            for (i, field_type) in fields.iter().enumerate() {
                                if i != 0 {
                                    s.push_str(", ");
                                }
                                s.push_str(&format!("{}", field_type.format(&synth.needs_ref)));
                            }
                            s.push_str("),\n");
                        }
                    }
                }
                s.push_str("}\n");
            }
            SynthData::Typedef(ty) => {
                s.push_str(&format!("type {}<'a> = {};\n", name, ty.format(&synth.needs_ref)));
            }
        }
    }
    
    s
}

fn main() {
    let g = grammar! {
        root = grammar.clone(),
        grammar = Repeat(vec![rule.clone()]),
        rule = Sequence(vec![name.clone(), Tok(Operator("=".to_string())), expr.clone(), Tok(Operator(";".to_string()))]),
        name = Tok(Identifier),
        expr = Sequence(vec![seq.clone(), Repeat(vec![Tok(Operator("|".to_string())), seq.clone()])]),
        seq = Repeat(vec![item.clone()]),
        item = Choice(vec![
            name.clone(),
            ws.clone(),
            literal.clone(),
            parens.clone(),
            optional.clone(),
        ]),
        ws = Tok(Operator('.'.to_string())),
        literal = Tok(LiteralString),
        parens = Sequence(vec![Tok(Operator("(".to_string())), expr.clone(), Tok(Operator(")".to_string()))]),
        optional = Sequence(vec![item.clone(), Tok(Operator("?".to_string()))])
    };

    let nullables = g.compute_nullable();
    let first_sets = g.compute_first_sets(&nullables);
    let follow_sets = g.compute_follow_sets(&nullables, &first_sets);

    dbg!(&g);

    for (rule_id, rule) in g.rules.iter().enumerate() {
        println!("{}: {:?}", rule_id, DbgRule(&g, rule));
        println!("  nullable: {}", nullables[rule_id]);
        println!("  first: {:?}", DbgTokens(&g.tokens, first_sets[rule_id]));
        println!("  follow: {:?}", DbgTokens(&g.tokens, follow_sets[rule_id]));
    }

    let parser = generate_parser(&g, &nullables, &first_sets, &follow_sets);
    println!("{}", parser);
    std::fs::write("src/parser.rs", parser.as_bytes()).unwrap();
}