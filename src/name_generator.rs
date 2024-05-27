use std::{collections::HashSet, sync::Arc};

#[derive(Clone)]
pub struct NameGenerator {
    temp_count: usize,
    reusable_strings: HashSet<Arc<str>>,
}

impl NameGenerator {
    pub fn new() -> NameGenerator {
        NameGenerator {
            temp_count: 0,
            reusable_strings: HashSet::new(),
        }
    }

    pub fn temp_name(&mut self, prefix: &str) -> String {
        let temp_index = self.temp_count;
        self.temp_count += 1;

        format!("__{}{}", prefix, temp_index)
    }

    pub fn reuse(&mut self, string: &str) -> Arc<str> {
        if let Some(arc_string) = self.reusable_strings.get(string) {
            return arc_string.clone();
        }

        let arc_string: Arc<str> = string.into();

        self.reusable_strings.insert(arc_string.clone());

        arc_string
    }
}
