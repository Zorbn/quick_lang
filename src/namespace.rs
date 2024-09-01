use std::{
    collections::{hash_map::Iter, HashMap},
    fmt::Display,
    hash::Hash,
    sync::Arc,
};

use crate::{parser::NodeIndex, typer::Type};

pub const DEFINITION_ERROR: &str = "a definition with this name already exists in this namespace";

#[derive(Debug)]
pub struct DefinitionIndexError(pub NodeIndex);

impl Display for DefinitionIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", DEFINITION_ERROR)
    }
}

#[derive(Debug)]
pub struct DefinitionError;

impl Display for DefinitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", DEFINITION_ERROR)
    }
}

#[derive(Clone, Debug)]
pub struct DefinitionIndices {
    indices: HashMap<Arc<str>, NodeIndex>,
}

impl DefinitionIndices {
    pub fn new() -> Self {
        Self {
            indices: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: Arc<str>, index: NodeIndex) -> Result<(), DefinitionIndexError> {
        if self.indices.insert(name, index).is_some() {
            return Err(DefinitionIndexError(index));
        }

        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<NodeIndex> {
        self.indices.get(name).copied()
    }

    pub fn extend(
        &mut self,
        definition_indices: &DefinitionIndices,
        errors: &mut Vec<DefinitionIndexError>,
    ) {
        for (name, index) in &definition_indices.indices {
            if let Err(error) = self.insert(name.clone(), *index) {
                errors.push(error);
            }
        }
    }

    pub fn extend_unchecked(&mut self, definition_indices: &DefinitionIndices) {
        for (name, index) in &definition_indices.indices {
            let _ = self.insert(name.clone(), *index);
        }
    }

    pub fn iter(&self) -> Iter<Arc<str>, NodeIndex> {
        self.indices.iter()
    }

    fn has_name(&self, name: &str) -> bool {
        self.indices.contains_key(name)
    }
}

#[derive(Clone, Debug)]
pub enum Definition {
    Function {
        type_kind_id: usize,
        is_extern: bool,
        default_args: Arc<Vec<NodeIndex>>,
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

pub enum NamespaceLookupResult {
    DefinitionIndex(NodeIndex),
    Definition(Definition),
    None,
}

#[derive(Clone)]
pub struct NamespaceGenericArg {
    pub param_name: Arc<str>,
    pub type_kind_id: usize,
}

#[derive(Clone)]
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
    pub fn insert(
        &mut self,
        name: Arc<str>,
        definition: Definition,
    ) -> Result<(), DefinitionError> {
        if self.has_name(name.clone()) {
            return Err(DefinitionError);
        }

        self.definitions.insert(Identifier::new(name), definition);

        Ok(())
    }

    // Not adding a new name to the namespace, just adding a definition for an existing one.
    // There may be multiple definitions with the same name but different identifiers, eg.
    // in the case of multiple generic usages of the same type/function.
    pub fn define(&mut self, identifier: Identifier, definition: Definition) {
        self.definitions.insert(identifier, definition);
    }

    pub fn lookup(&self, identifier: &Identifier) -> NamespaceLookupResult {
        if let Some(definition) = self.definitions.get(identifier) {
            return NamespaceLookupResult::Definition(definition.clone());
        }

        if let Some(definition_index) = self.definition_indices.get(&identifier.name) {
            return NamespaceLookupResult::DefinitionIndex(definition_index);
        }

        NamespaceLookupResult::None
    }

    pub fn is_name_defined(&self, name: Arc<str>) -> bool {
        self.definitions.contains_key(&Identifier::new(name))
    }

    pub fn extend_definition_indices(
        &mut self,
        definition_indices: &DefinitionIndices,
        errors: &mut Vec<DefinitionIndexError>,
    ) {
        self.definition_indices.extend(definition_indices, errors);
    }

    pub fn extend_definition_indices_unchecked(&mut self, definition_indices: &DefinitionIndices) {
        self.definition_indices.extend_unchecked(definition_indices);
    }

    fn has_name(&self, name: Arc<str>) -> bool {
        self.definition_indices.has_name(&name)
            || self.definitions.contains_key(&Identifier::new(name))
    }
}
