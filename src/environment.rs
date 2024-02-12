use std::{collections::HashMap, sync::Arc};

pub struct SubEnvironment<T> {
    name_types: HashMap<Arc<str>, T>,
}

pub struct Environment<T: Clone> {
    stack: Vec<SubEnvironment<T>>,
}

impl<T: Clone> Environment<T> {
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

    pub fn insert(&mut self, name: Arc<str>, name_type: T, is_global: bool) {
        if is_global {
            self.stack
                .first_mut()
                .unwrap()
                .name_types
                .insert(name, name_type);
        } else {
            self.stack
                .last_mut()
                .unwrap()
                .name_types
                .insert(name, name_type);
        }
    }

    pub fn get(&self, name: &str) -> Option<T> {
        for sub_env in self.stack.iter().rev() {
            if let Some(name_type) = sub_env.name_types.get(name) {
                return Some(name_type.clone());
            }
        }

        None
    }
}
