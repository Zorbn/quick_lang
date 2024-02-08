use std::{collections::HashMap, sync::Arc};

use crate::{
    parser::{ArrayLayout, Field, FunctionLayout, NodeKind, StructLayout, TypeKind},
    type_checker::TypedNode,
};

pub fn add_type(type_kinds: &mut Vec<TypeKind>, type_kind: TypeKind) -> usize {
    let index = type_kinds.len();
    type_kinds.push(type_kind);
    index
}

pub fn generic_type_kind_to_concrete(
    type_kind: usize,
    generic_type_kinds: &[usize],
    generic_param_type_kinds: &[usize],
) -> usize {
    for (i, generic_type_kind) in generic_type_kinds.iter().enumerate() {
        if type_kind != *generic_type_kind {
            continue;
        }

        return generic_param_type_kinds[i];
    }

    type_kind
}

pub fn generic_params_to_concrete(
    param_type_kinds: &[usize],
    generic_type_kinds: &[usize],
    generic_param_type_kinds: &[usize],
) -> Vec<usize> {
    let mut concrete_param_type_kinds = Vec::new();

    for param_type_kind in param_type_kinds.iter() {
        let concrete_type_kind = generic_type_kind_to_concrete(
            *param_type_kind,
            generic_type_kinds,
            generic_param_type_kinds,
        );
        concrete_param_type_kinds.push(concrete_type_kind);
    }

    concrete_param_type_kinds
}

pub fn generic_function_to_concrete(
    type_kinds: &mut Vec<TypeKind>,
    function_type_kind: usize,
    function_type_kinds: &mut HashMap<FunctionLayout, usize>,
    type_names: &[usize],
) -> usize {
    let TypeKind::Function {
        param_type_kinds,
        generic_type_kinds,
        return_type_kind,
    } = &type_kinds[function_type_kind]
    else {
        panic!("type kind is not a function");
    };

    let concrete_param_type_kinds =
        generic_params_to_concrete(param_type_kinds, generic_type_kinds, type_names);
    let return_type_kind =
        generic_type_kind_to_concrete(*return_type_kind, generic_type_kinds, type_names);

    let concrete_function = FunctionLayout {
        param_type_kinds: Arc::new(concrete_param_type_kinds),
        generic_type_kinds: Arc::new(Vec::new()),
        return_type_kind,
    };

    get_function_type_kind(type_kinds, function_type_kinds, concrete_function)
}

pub fn generic_struct_to_concrete(
    struct_layout: StructLayout,
    type_kinds: &mut Vec<TypeKind>,
    struct_type_kind: usize,
    struct_type_kinds: &mut HashMap<StructLayout, usize>,
    function_type_kinds: &mut HashMap<FunctionLayout, usize>,
    generic_param_type_kinds: &Arc<Vec<usize>>,
) -> usize {
    let TypeKind::Struct {
        name,
        field_kinds,
        generic_type_kinds,
        is_union,
        ..
    } = type_kinds[struct_type_kind].clone()
    else {
        panic!("type kind is not a struct");
    };

    if let Some(concrete_type_kind) = struct_type_kinds.get(&struct_layout) {
        return *concrete_type_kind;
    }

    let mut concrete_field_kinds = Vec::new();

    for field_kind in field_kinds.iter() {
        match &type_kinds[field_kind.type_kind] {
            TypeKind::Function { .. } => {
                let concrete_type_kind = generic_function_to_concrete(
                    type_kinds,
                    field_kind.type_kind,
                    function_type_kinds,
                    generic_param_type_kinds,
                );

                concrete_field_kinds.push(Field {
                    name: field_kind.name,
                    type_kind: concrete_type_kind,
                });
            }
            _ => {
                let concrete_type_kind = generic_type_kind_to_concrete(
                    field_kind.type_kind,
                    &generic_type_kinds,
                    generic_param_type_kinds,
                );

                concrete_field_kinds.push(Field {
                    name: field_kind.name,
                    type_kind: concrete_type_kind,
                });
            }
        }
    }

    let concrete_type_kind = add_type(
        type_kinds,
        TypeKind::Struct {
            name,
            field_kinds: Arc::new(concrete_field_kinds),
            generic_type_kinds: Arc::new(Vec::new()),
            generic_param_type_kinds: generic_param_type_kinds.clone(),
            is_union,
        },
    );

    struct_type_kinds.insert(struct_layout, concrete_type_kind);

    concrete_type_kind
}

pub fn get_function_type_kind(
    type_kinds: &mut Vec<TypeKind>,
    function_type_kinds: &mut HashMap<FunctionLayout, usize>,
    function_layout: FunctionLayout,
) -> usize {
    if let Some(index) = function_type_kinds.get(&function_layout) {
        *index
    } else {
        let index = {
            let function_layout = function_layout.clone();
            add_type(
                type_kinds,
                TypeKind::Function {
                    param_type_kinds: function_layout.param_type_kinds,
                    generic_type_kinds: function_layout.generic_type_kinds,
                    return_type_kind: function_layout.return_type_kind,
                },
            )
        };
        function_type_kinds.insert(function_layout, index);
        index
    }
}

pub fn get_type_kind_as_pointer(
    type_kinds: &mut Vec<TypeKind>,
    pointer_type_kinds: &mut HashMap<usize, usize>,
    type_kind: usize,
) -> usize {
    if let Some(index) = pointer_type_kinds.get(&type_kind) {
        *index
    } else {
        let index = add_type(
            type_kinds,
            TypeKind::Pointer {
                inner_type_kind: type_kind,
            },
        );
        pointer_type_kinds.insert(type_kind, index);
        index
    }
}

pub fn get_type_kind_as_array(
    type_kinds: &mut Vec<TypeKind>,
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
            type_kinds,
            TypeKind::Array {
                element_type_kind,
                element_count,
            },
        );
        array_type_kinds.insert(layout, index);
        index
    }
}

pub fn is_type_kind_array(type_kinds: &[TypeKind], type_kind: usize) -> bool {
    let type_kind = &type_kinds[type_kind];

    matches!(type_kind, TypeKind::Array { .. })
}

pub fn is_typed_expression_array_literal(typed_nodes: &[TypedNode], expression: usize) -> bool {
    matches!(typed_nodes[expression].node_kind, NodeKind::ArrayLiteral { .. })
}

pub fn replace_generic_type_kinds(type_kinds: &mut [TypeKind], generic_type_kinds: &[usize], generic_param_type_kinds: &[usize]) {
    for (generic_param_type_kind, generic_type_kind) in
        generic_param_type_kinds.iter().zip(generic_type_kinds.iter())
    {
        type_kinds[*generic_type_kind] = TypeKind::Alias {
            inner_type_kind: *generic_param_type_kind,
        };
    }
}