use std::collections::HashMap;


#[derive(Debug, Clone)]
enum TextRule {
    Recur(String),
    Tok(Token),
    Sequence(Vec<TextRule>),
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
                let rule_nullable = self.compute_rule_nullable(rule_id, &nullable);
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

fn main() {
    let g = grammar! {
        root = expr.clone(),
        expr = Sequence(vec![term.clone(), Repeat(vec![op.clone(), expr.clone()])]),
        term = Tok(LiteralInteger),
        op = Tok(Operator("+".to_string()))
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
}