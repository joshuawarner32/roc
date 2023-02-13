use std::collections::HashSet;

use crate::{Token, IdGrammar, RuleId, Rule, TokenId, TokenSet};



#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstrId(usize);

pub enum Instr {
    MatchNextToken {
        branches: Vec<(TokenSet, Vec<Instr>)>,
        otherwise: Option<Vec<Instr>>,
    },
    RequireNextToken(TokenId),
    CallRule(RuleId),
    BuildNone,
    Break,
    Loop(Vec<Instr>),
}

pub fn compile_rule(g: &IdGrammar, rule: RuleId) -> Vec<Instr> {
    let mut instrs = Vec::new();

    compile_rule_inner(g, rule, &mut instrs);

    instrs
}

fn compile_rule_inner(g: &IdGrammar, rule_id: RuleId, instrs: &mut Vec<Instr>) {
    let rule = &g.rules[rule_id.0];
    match rule {
        Rule::Recur(inner_rule) => instrs.push(Instr::CallRule(*inner_rule)),
        Rule::Tok(token) => instrs.push(Instr::RequireNextToken(*token)),
        Rule::Sequence(seq) => {
            for rule in seq {
                // if g.nullables[rule.0] {
                //     todo!();
                // } else {
                    // compile_rule_inner(g, *rule, instrs);
                    instrs.push(Instr::CallRule(*rule));
                // }
            }
        }
        Rule::Choice(options) => {
            let mut cases = options.iter().copied().collect::<HashSet<_>>();

            let mut handle_later = HashSet::new();

            // First check if any of the options have this rule in the first rule set. If so, we need to defer processing the remainder of those rules until the end.
            for option in options {
                if g.first_rule_sets[option.0].contains(rule_id) {
                    cases.remove(option);
                    handle_later.insert(*option);
                }
            }

            // If there are any tokens that only belong to the first set of one option, let's split those out

            let union = options.iter().fold(g.first_sets[options[0].0], |mut acc, rule| {
                acc.update(g.first_sets[rule.0]);
                acc
            });

            let mut branches = Vec::new();

            for token in union {
                let mut case = None;

                for option in cases.iter() {
                    if g.first_sets[option.0].contains(token) {
                        // compile_rule_inner(g, *option, instrs);
                        if case.is_none() {
                            case = Some(*option);
                        } else {
                            case = None;
                            break;
                        }
                    }
                }

                if let Some(case) = case {
                    cases.remove(&case);
                    branches.push((TokenSet::single(token), vec![Instr::CallRule(case)]));
                }
            }

            if !cases.is_empty() {
                todo!();
            }

            if !handle_later.is_empty() {
                todo!();
            }
        }
        Rule::Optional(rule) => {
            if g.first_sets[rule.0].is_disjoint(g.follow_sets[rule_id.0]) {
                // Simple case, where the first set doesn't intersect with the follow set

                let branches = vec![
                    (g.first_sets[rule.0], vec![Instr::CallRule(*rule)]),
                ];

                instrs.push(Instr::MatchNextToken {
                    branches,
                    otherwise: Some(vec![Instr::BuildNone]),
                });
            } else {
                todo!();
            }
        }
        Rule::Repeat(rule) => {
            if g.first_sets[rule.0].is_disjoint(g.follow_sets[rule_id.0]) {
                // Simple case, where the first set doesn't intersect with the follow set

                let branches = vec![
                    (g.first_sets[rule.0], vec![Instr::CallRule(*rule)]),
                ];

                let loop_body = vec![Instr::MatchNextToken {
                    branches,
                    otherwise: Some(vec![Instr::Break]),
                }];

                instrs.push(Instr::Loop(loop_body));
            } else {
                todo!();
            }
        }
    }
}
