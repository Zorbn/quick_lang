use std::sync::Arc;

#[derive(Debug, PartialEq)]
pub struct Field {
    pub name: usize,
    pub type_kind_id: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    Int,
    String,
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
        name: usize,
        field_kinds: Arc<Vec<Field>>,
        generic_type_kind_ids: Arc<Vec<usize>>,
        generic_param_type_kind_ids: Arc<Vec<usize>>,
        is_union: bool,
    },
    Partial,
    Function {
        param_type_kind_ids: Arc<Vec<usize>>,
        generic_type_kind_ids: Arc<Vec<usize>>,
        return_type_kind_id: usize,
    },
    Enum {
        name: usize,
        variant_names: Arc<Vec<usize>>,
    },
}

impl TypeKind {
    pub fn is_numeric(&self) -> bool {
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
                | TypeKind::Float32
                | TypeKind::Float64
        )
    }
}

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

    pub fn get_by_id(&mut self, type_kind_id: usize) -> TypeKind {
        self.type_kinds[type_kind_id].clone()
    }
}
