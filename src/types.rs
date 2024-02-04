use std::{collections::HashMap, sync::Arc};

use crate::{
    parser::{ArrayLayout, FunctionLayout, NodeKind, TypeKind},
    type_checker::TypedNode,
};

pub fn add_type(types: &mut Vec<TypeKind>, type_kind: TypeKind) -> usize {
    let index = types.len();
    types.push(type_kind);
    index
}

pub fn generic_params_to_concrete(param_type_kinds: &Arc<Vec<usize>>, generic_type_kinds: &Arc<Vec<usize>>, type_names: &Arc<Vec<usize>>) -> Vec<usize> {
    let mut concrete_param_type_kinds = Vec::new();

    'param_loop: for param_type_kind in param_type_kinds.iter() {
        for (i, generic_type_kind) in generic_type_kinds.iter().enumerate() {
            if *param_type_kind != *generic_type_kind {
                continue;
            }

            concrete_param_type_kinds.push(type_names[i]);
            continue 'param_loop;
        }

        concrete_param_type_kinds.push(*param_type_kind);
    }

    concrete_param_type_kinds
}

pub fn get_function_type_kind(
    types: &mut Vec<TypeKind>,
    function_type_kinds: &mut HashMap<FunctionLayout, usize>,
    function_layout: FunctionLayout,
) -> usize {
    if let Some(index) = function_type_kinds.get(&function_layout)
    {
        *index
    } else {
        let index = {
            let function_layout = function_layout.clone();
            add_type(types, TypeKind::Function {
                param_type_kinds: function_layout.param_type_kinds,
                generic_type_kinds: function_layout.generic_type_kinds,
                return_type_kind: function_layout.return_type_kind,
            })
        };
        function_type_kinds
            .insert(function_layout, index);
        index
    }
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

pub fn get_type_kind_as_array(
    types: &mut Vec<TypeKind>,
    array_type_kinds: &mut HashMap<ArrayLayout, usize>,
    element_type_kind: usize,
    element_count: usize,
) -> usize {
    let layout = ArrayLayout {
        element_type_kind,
        element_count,
    };

    if let Some(index) = array_type_kinds.get(&layout) {
        *index
    } else {
        let index = add_type(
            types,
            TypeKind::Array {
                element_type_kind,
                element_count,
            },
        );
        array_type_kinds.insert(layout, index);
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
