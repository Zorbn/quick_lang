use std::collections::HashMap;

pub struct SubEnvironment {
    variable_type_kinds: HashMap<String, usize>,
}

pub struct Environment {
    stack: Vec<SubEnvironment>,
}

impl Environment {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn push(&mut self) {
        self.stack.push(SubEnvironment {
            variable_type_kinds: HashMap::new(),
        });
    }

    pub fn pop(&mut self) {
        self.stack.pop();
    }

    pub fn insert(&mut self, name: String, type_kind: usize) {
        self.stack
            .last_mut()
            .unwrap()
            .variable_type_kinds
            .insert(name, type_kind);
    }

    pub fn get(&self, name: &str) -> Option<usize> {
        for sub_env in self.stack.iter().rev() {
            if let Some(type_kind) = sub_env.variable_type_kinds.get(name) {
                return Some(*type_kind);
            }
        }

        None
    }
}
