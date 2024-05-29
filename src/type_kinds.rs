use std::sync::Arc;

use crate::{
    assert_matches,
    namespace::{Definition, Identifier, Namespace, NamespaceLookupResult},
    parser::{MethodKind, NodeIndex, NodeKind},
    typer::{InstanceKind, Type, TypedNode},
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
pub enum PrimitiveType {
    None,
    Int,
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
    Char,
    Bool,
}

pub const INT_TYPE_KIND_ID: usize = 0;
pub const UINT_TYPE_KIND_ID: usize = 1;
pub const INT8_TYPE_KIND_ID: usize = 2;
pub const UINT8_TYPE_KIND_ID: usize = 3;
pub const INT16_TYPE_KIND_ID: usize = 4;
pub const UINT16_TYPE_KIND_ID: usize = 5;
pub const INT32_TYPE_KIND_ID: usize = 6;
pub const UINT32_TYPE_KIND_ID: usize = 7;
pub const INT64_TYPE_KIND_ID: usize = 8;
pub const UINT64_TYPE_KIND_ID: usize = 9;
pub const FLOAT32_TYPE_KIND_ID: usize = 10;
pub const FLOAT64_TYPE_KIND_ID: usize = 11;
pub const CHAR_TYPE_KIND_ID: usize = 12;
pub const BOOL_TYPE_KIND_ID: usize = 13;

#[derive(Clone, Debug, PartialEq)]
pub enum TypeKind {
    Void,
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
        primitive_type: PrimitiveType,
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
        if let TypeKind::Struct { primitive_type, .. } = self {
            return matches!(
                primitive_type,
                PrimitiveType::Int
                    | PrimitiveType::UInt
                    | PrimitiveType::Int8
                    | PrimitiveType::UInt8
                    | PrimitiveType::Int16
                    | PrimitiveType::UInt16
                    | PrimitiveType::Int32
                    | PrimitiveType::UInt32
                    | PrimitiveType::Int64
                    | PrimitiveType::UInt64
            );
        };

        false
    }

    pub fn is_unsigned(&self) -> bool {
        if let TypeKind::Struct { primitive_type, .. } = self {
            return matches!(
                primitive_type,
                PrimitiveType::UInt
                    | PrimitiveType::UInt8
                    | PrimitiveType::UInt16
                    | PrimitiveType::UInt32
                    | PrimitiveType::UInt64
            );
        };

        false
    }

    pub fn is_float(&self) -> bool {
        if let TypeKind::Struct { primitive_type, .. } = self {
            return matches!(
                primitive_type,
                PrimitiveType::Float32 | PrimitiveType::Float64
            );
        };

        false
    }

    pub fn is_numeric(&self) -> bool {
        self.is_int() || self.is_float()
    }
}

#[derive(Clone)]
pub struct TypeKinds {
    type_kinds: Vec<TypeKind>,
    destructor_name: Arc<str>,
}

impl TypeKinds {
    pub fn new() -> Self {
        let mut type_kinds = Self {
            type_kinds: Vec::new(),
            destructor_name: "Destroy".into(),
        };

        type_kinds.add_placeholder(); // Int
        type_kinds.add_placeholder(); // UInt
        type_kinds.add_placeholder(); // Int8
        type_kinds.add_placeholder(); // UInt32
        type_kinds.add_placeholder(); // Int16
        type_kinds.add_placeholder(); // UInt16
        type_kinds.add_placeholder(); // Int32
        type_kinds.add_placeholder(); // UInt32
        type_kinds.add_placeholder(); // Int64
        type_kinds.add_placeholder(); // UInt64
        type_kinds.add_placeholder(); // Float32
        type_kinds.add_placeholder(); // Float64
        type_kinds.add_placeholder(); // Char
        type_kinds.add_placeholder(); // Bool

        type_kinds
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

    pub fn dereference_type_kind_id(&self, type_kind_id: usize) -> (usize, bool) {
        if let TypeKind::Pointer {
            inner_type_kind_id, ..
        } = self.get_by_id(type_kind_id)
        {
            (inner_type_kind_id, true)
        } else {
            (type_kind_id, false)
        }
    }

    pub fn is_assignment_valid(&self, to_type_kind_id: usize, from_type_kind_id: usize) -> bool {
        if to_type_kind_id == from_type_kind_id {
            return true;
        }

        match self.get_by_id(to_type_kind_id) {
            TypeKind::Array {
                element_type_kind_id: to_element_type_kind_id,
                element_count: to_element_count,
            } => {
                let TypeKind::Array {
                    element_type_kind_id: from_element_type_kind_id,
                    element_count: from_element_count,
                } = self.get_by_id(from_type_kind_id)
                else {
                    return false;
                };

                from_element_count == to_element_count
                    && self.is_assignment_valid(to_element_type_kind_id, from_element_type_kind_id)
            }
            // It's possible to assign either a val or var pointer to a val pointer, because assigning a var pointer to a val pointer
            // just reduces the number of things you can do with the type, it doesn't let you modify immutable values like going the other direction would.
            TypeKind::Pointer {
                inner_type_kind_id: to_inner_type_kind_id,
                is_inner_mutable: false,
            } => {
                let TypeKind::Pointer {
                    inner_type_kind_id: from_inner_type_kind_id,
                    ..
                } = self.get_by_id(from_type_kind_id)
                else {
                    return false;
                };

                from_inner_type_kind_id == to_inner_type_kind_id
            }
            _ => false,
        }
    }

    pub fn is_method_call_valid(
        &self,
        param_type_kind_id: usize,
        instance_type: &Type,
    ) -> Option<MethodKind> {
        if instance_type.instance_kind == InstanceKind::Name {
            return None;
        }

        if self.is_assignment_valid(param_type_kind_id, instance_type.type_kind_id) {
            return Some(MethodKind::ByValue);
        }

        if let TypeKind::Pointer {
            inner_type_kind_id,
            is_inner_mutable,
        } = self.get_by_id(param_type_kind_id)
        {
            if instance_type.instance_kind == InstanceKind::Literal {
                return None;
            }

            if is_inner_mutable && instance_type.instance_kind != InstanceKind::Var {
                return None;
            }

            if self.is_assignment_valid(inner_type_kind_id, instance_type.type_kind_id) {
                return Some(MethodKind::ByReference);
            }
        }

        if let TypeKind::Pointer {
            inner_type_kind_id, ..
        } = self.get_by_id(instance_type.type_kind_id)
        {
            if self.is_assignment_valid(param_type_kind_id, inner_type_kind_id) {
                return Some(MethodKind::ByDereference);
            }
        }

        None
    }

    pub fn get_method(
        &self,
        method_name: Arc<str>,
        type_kind_id: usize,
        namespaces: &[Namespace],
    ) -> Option<TypeKind> {
        let (dereferenced_type_kind_id, _) = self.dereference_type_kind_id(type_kind_id);

        let TypeKind::Struct { namespace_id, .. } = self.get_by_id(dereferenced_type_kind_id)
        else {
            return None;
        };

        let namespace = &namespaces[namespace_id];

        let NamespaceLookupResult::Definition(Definition::Function { type_kind_id, .. }) =
            namespace.lookup(&Identifier::new(method_name))
        else {
            return None;
        };

        let method = self.get_by_id(type_kind_id);

        let TypeKind::Function { .. } = &method else {
            return None;
        };

        Some(method)
    }

    pub fn is_destructor_call_valid(
        &self,
        instance_type: &Type,
        namespaces: &[Namespace],
    ) -> Option<MethodKind> {
        let Some(TypeKind::Function {
            param_type_kind_ids,
            return_type_kind_id,
        }) = self.get_method(
            self.destructor_name.clone(),
            instance_type.type_kind_id,
            namespaces,
        )
        else {
            return None;
        };

        if param_type_kind_ids.len() != 1 || self.get_by_id(return_type_kind_id) != TypeKind::Void {
            return None;
        }

        self.is_method_call_valid(param_type_kind_ids[0], instance_type)
    }
}
