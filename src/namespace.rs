use std::{collections::HashMap, hash::Hash, sync::Arc};

use crate::{parser::NodeIndex, typer::Type};

#[derive(Debug)]
pub struct DefinitionIndices {
    indices: HashMap<Arc<str>, NodeIndex>,
}

impl DefinitionIndices {
    pub fn new() -> Self {
        Self {
            indices: HashMap::new(),
        }
    }

    // TODO: Error if entry already exists for name.
    pub fn insert(&mut self, name: Arc<str>, index: NodeIndex) {
        self.indices.insert(name, index);
    }

    pub fn get(&self, name: &str) -> Option<NodeIndex> {
        self.indices.get(name).copied()
    }

    // TODO: Error if entry already exists for any names.
    pub fn extend(&mut self, definition_indices: &DefinitionIndices) {
        for (name, index) in &definition_indices.indices {
            self.insert(name.clone(), *index);
        }
    }
}

#[derive(Clone, Debug)]
pub enum Definition {
    Function {
        type_kind_id: usize,
        is_extern: bool,
    },
    TypeKind {
        type_kind_id: usize,
    },
    Variable {
        variable_type: Type,
    },
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct Identifier {
    pub name: Arc<str>,
    pub generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
}

impl Identifier {
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            generic_arg_type_kind_ids: None,
        }
    }
}

pub enum LookupResult {
    DefinitionIndex(NodeIndex),
    Definition(Definition),
    None,
}

pub struct NamespaceGenericArg {
    pub param_name: Arc<str>,
    pub type_kind_id: usize,
}

pub struct Namespace {
    pub name: Arc<str>,
    pub associated_type_kind_id: Option<usize>,
    pub generic_args: Vec<NamespaceGenericArg>,
    pub parent_id: Option<usize>,
    pub child_ids: HashMap<Arc<str>, usize>,

    definition_indices: DefinitionIndices,
    definitions: HashMap<Identifier, Definition>,
}

impl Namespace {
    pub fn new(
        name: Arc<str>,
        associated_type_kind_id: Option<usize>,
        generic_args: Vec<NamespaceGenericArg>,
        parent_id: Option<usize>,
    ) -> Self {
        Self {
            name,
            associated_type_kind_id,
            generic_args,
            parent_id,
            child_ids: HashMap::new(),

            definition_indices: DefinitionIndices::new(),
            definitions: HashMap::new(),
        }
    }

    // Add a new name to the namespace, and ensure it isn't a duplicate.
    // TODO: Error if this name already exists in the namespace.
    pub fn insert(&mut self, name: Arc<str>, definition: Definition) {
        self.definitions.insert(Identifier::new(name), definition);
    }

    // Not adding a new name to the namespace, just adding a definition for an existing one.
    // There may be multiple definitions with the same name but different identifiers, eg.
    // in the case of multiple generic usages of the same type/function.
    pub fn define(&mut self, identifier: Identifier, definition: Definition) {
        self.definitions.insert(identifier, definition);
    }

    pub fn lookup(&self, identifier: &Identifier) -> LookupResult {
        if let Some(definition) = self.definitions.get(identifier) {
            return LookupResult::Definition(definition.clone());
        }

        if let Some(definition_index) = self.definition_indices.get(&identifier.name) {
            return LookupResult::DefinitionIndex(definition_index);
        }

        LookupResult::None
    }

    pub fn has_name(&mut self, name: Arc<str>) -> bool {
        self.definitions.contains_key(&Identifier::new(name))
    }

    // TODO: Error if entry already exists for any names.
    pub fn extend_definition_indices(&mut self, definition_indices: &DefinitionIndices) {
        self.definition_indices.extend(definition_indices);
    }
}
