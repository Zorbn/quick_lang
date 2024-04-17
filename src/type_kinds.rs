use std::sync::Arc;

use crate::{
    assert_matches,
    parser::{NodeIndex, NodeKind},
    typer::TypedNode,
};

#[derive(Debug, PartialEq)]
pub struct Field {
    pub name: NodeIndex,
    pub type_kind_id: usize,
}

pub fn get_field_index_by_name(
    typed_nodes: &[TypedNode],
    name_text: &Arc<str>,
    fields: &Arc<Vec<Field>>,
) -> Option<usize> {
    let mut tag = None;
    for (i, field) in fields.iter().enumerate() {
        assert_matches!(
            NodeKind::Name {
                text: field_name_text,
            },
            &typed_nodes[field.name.node_index].node_kind
        );

        if *field_name_text == *name_text {
            tag = Some(i);
        }
    }

    tag
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    Int,
    Bool,
    Char,
    Void,
    UInt,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
    Tag,
    Pointer {
        inner_type_kind_id: usize,
        is_inner_mutable: bool,
    },
    Array {
        element_type_kind_id: usize,
        element_count: usize,
    },
    Struct {
        name: NodeIndex,
        fields: Arc<Vec<Field>>,
        is_union: bool,
        namespace_id: usize,
    },
    Placeholder,
    Function {
        param_type_kind_ids: Arc<Vec<usize>>,
        return_type_kind_id: usize,
    },
    Enum {
        name: NodeIndex,
        variant_names: Arc<Vec<NodeIndex>>,
    },
    Namespace {
        namespace_id: usize,
    },
}

impl TypeKind {
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            TypeKind::Int
                | TypeKind::UInt
                | TypeKind::Int8
                | TypeKind::UInt8
                | TypeKind::Int16
                | TypeKind::UInt16
                | TypeKind::Int32
                | TypeKind::UInt32
                | TypeKind::Int64
                | TypeKind::UInt64
        )
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            TypeKind::UInt
                | TypeKind::UInt8
                | TypeKind::UInt16
                | TypeKind::UInt32
                | TypeKind::UInt64
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, TypeKind::Float32 | TypeKind::Float64)
    }

    pub fn is_numeric(&self) -> bool {
        self.is_int() || self.is_float()
    }
}

#[derive(Clone)]
pub struct TypeKinds {
    type_kinds: Vec<TypeKind>,
}

impl TypeKinds {
    pub fn new() -> Self {
        Self {
            type_kinds: Vec::new(),
        }
    }

    pub fn add_or_get(&mut self, type_kind: TypeKind) -> usize {
        for id in 0..self.type_kinds.len() {
            if self.type_kinds[id] == type_kind {
                return id;
            }
        }

        let id = self.type_kinds.len();

        self.type_kinds.push(type_kind);

        id
    }

    pub fn get_by_id(&self, type_kind_id: usize) -> TypeKind {
        self.type_kinds[type_kind_id].clone()
    }

    pub fn add_placeholder(&mut self) -> usize {
        let type_kind_id = self.type_kinds.len();
        self.type_kinds.push(TypeKind::Placeholder);
        type_kind_id
    }

    pub fn replace_placeholder(&mut self, type_kind_id: usize, type_kind: TypeKind) {
        if self.type_kinds[type_kind_id] != TypeKind::Placeholder {
            panic!("tried to replace a non-placeholder type kind");
        }

        self.type_kinds[type_kind_id] = type_kind;
    }
}
