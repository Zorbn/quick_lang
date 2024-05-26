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
    destructor_name: Arc<str>,
}

impl TypeKinds {
    pub fn new() -> Self {
        Self {
            type_kinds: Vec::new(),
            destructor_name: "Destroy".into(),
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
        instance_type: Type,
    ) -> Option<MethodKind> {
        if self.is_assignment_valid(param_type_kind_id, instance_type.type_kind_id) {
            return Some(MethodKind::ByValue);
        }

        if let TypeKind::Pointer {
            inner_type_kind_id,
            is_inner_mutable,
        } = self.get_by_id(param_type_kind_id)
        {
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

    pub fn is_destructor_call_valid(
        &mut self,
        instance_type: Type,
        namespaces: &[Namespace],
    ) -> Option<MethodKind> {
        let (dereferenced_type_kind_id, _) =
            self.dereference_type_kind_id(instance_type.type_kind_id);

        let TypeKind::Struct { namespace_id, .. } = self.get_by_id(dereferenced_type_kind_id)
        else {
            return None;
        };

        let namespace = &namespaces[namespace_id];

        let NamespaceLookupResult::Definition(Definition::Function { type_kind_id, .. }) =
            namespace.lookup(&Identifier::new(self.destructor_name.clone()))
        else {
            return None;
        };

        let TypeKind::Function {
            param_type_kind_ids,
            return_type_kind_id,
        } = self.get_by_id(type_kind_id)
        else {
            return None;
        };

        if param_type_kind_ids.len() != 1 || return_type_kind_id != self.add_or_get(TypeKind::Void)
        {
            return None;
        }

        self.is_method_call_valid(param_type_kind_ids[0], instance_type)
    }
}
