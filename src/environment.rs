use std::{collections::HashMap, hash::Hash};

#[derive(Clone)]
pub struct SubEnvironment<K: Hash + PartialEq + Eq, V> {
    name_types: HashMap<K, V>,
    is_connected_to_previous: bool,
}

#[derive(Clone)]
pub struct Environment<K: Hash + PartialEq + Eq, V: Clone> {
    stack: Vec<SubEnvironment<K, V>>,
}

impl<K: Clone + Hash + PartialEq + Eq, V: Clone> Environment<K, V> {
    pub fn new() -> Self {
        let mut environment = Self { stack: Vec::new() };
        environment.push(false);

        environment
    }

    pub fn push(&mut self, is_connected_to_previous: bool) {
        self.stack.push(SubEnvironment {
            name_types: HashMap::new(),
            is_connected_to_previous,
        });
    }

    pub fn pop(&mut self) {
        self.stack.pop();
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.stack.last_mut().unwrap().name_types.insert(key, value);
    }

    pub fn get(&self, key: &K) -> Option<V> {
        for (i, sub_env) in self.stack.iter().enumerate().rev() {
            if let Some(value) = sub_env.name_types.get(key) {
                return Some(value.clone());
            }

            // If this sub environment doesn't connect to the previous one then the only other place to check is the global level.
            if !sub_env.is_connected_to_previous && i > 0 {
                if let Some(value) = self.stack.first().unwrap().name_types.get(key) {
                    return Some(value.clone());
                }

                return None;
            }
        }

        None
    }
}
