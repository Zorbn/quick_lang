use std::collections::HashMap;

use crate::{
    parser::{NodeKind, TypeKind},
    type_checker::TypedNode,
};

pub fn add_type(types: &mut Vec<TypeKind>, type_kind: TypeKind) -> usize {
    let index = types.len();
    types.push(type_kind);
    index
}

pub fn get_type_kind_as_pointer(
    types: &mut Vec<TypeKind>,
    pointer_type_kinds: &mut HashMap<usize, usize>,
    type_kind: usize,
) -> usize {
    if let Some(index) = pointer_type_kinds.get(&type_kind) {
        *index
    } else {
        let index = add_type(
            types,
            TypeKind::Pointer {
                inner_type_kind: type_kind,
            },
        );
        pointer_type_kinds.insert(type_kind, index);
        index
    }
}

pub fn is_type_kind_array(types: &[TypeKind], type_kind: usize) -> bool {
    let type_kind = &types[type_kind];

    matches!(type_kind, TypeKind::Array { .. })
}

// TODO: When we can tell between literals and variables in the type checker this shouldn't be needed.
pub fn is_typed_expression_array_literal(typed_nodes: &[TypedNode], expression: usize) -> bool {
    let TypedNode {
        node_kind: NodeKind::ArrayLiteral { .. },
        ..
    } = typed_nodes[expression]
    else {
        return false;
    };

    true
}
