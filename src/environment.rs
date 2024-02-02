use std::{collections::HashMap, sync::Arc};

use crate::type_checker::Type;

pub struct SubEnvironment {
    name_types: HashMap<Arc<str>, Type>,
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
            name_types: HashMap::new(),
        });
    }

    pub fn pop(&mut self) {
        self.stack.pop();
    }

    pub fn insert(&mut self, name: Arc<str>, name_type: Type) {
        self.stack
            .last_mut()
            .unwrap()
            .name_types
            .insert(name, name_type);
    }

    pub fn get(&self, name: &str) -> Option<Type> {
        for sub_env in self.stack.iter().rev() {
            if let Some(name_type) = sub_env.name_types.get(name) {
                return Some(*name_type);
            }
        }

        None
    }
}
