use std::sync::Arc;

use crate::{
    as_option::AsOption, parser::NodeKind, type_checker::{InstanceKind, TypedNode}, type_kinds::Field
};

pub struct MethodSubject {
    pub node: usize,
    pub type_kind_id: usize,
}

// pub fn add_type(type_kinds: &mut Vec<TypeKind>, type_kind: TypeKind) -> usize {
//     let index = type_kinds.len();
//     type_kinds.push(type_kind);
//     index
// }
//
// pub fn is_type_kind_array(type_kinds: &[TypeKind], type_kind: usize) -> bool {
//     let type_kind = &type_kinds[type_kind];
//
//     matches!(type_kind, TypeKind::Array { .. })
// }

pub fn is_typed_expression_array_literal(typed_nodes: &[TypedNode], expression: usize) -> bool {
    matches!(
        typed_nodes[expression].node_kind,
        NodeKind::ArrayLiteral { .. }
    )
}

// pub fn replace_generic_type_kinds(
//     type_kinds: &mut [TypeKind],
//     generic_type_kinds: &[usize],
//     generic_param_type_kinds: &[usize],
// ) {
//     for (generic_param_type_kind, generic_type_kind) in generic_param_type_kinds
//         .iter()
//         .zip(generic_type_kinds.iter())
//     {
//         type_kinds[*generic_type_kind] = TypeKind::Alias {
//             inner_type_kind: *generic_param_type_kind,
//         };
//     }
// }

pub fn get_field_index_by_name(
    name_text: &Arc<str>,
    typed_nodes: &[TypedNode],
    field_kinds: &Arc<Vec<Field>>,
) -> Option<usize> {
    let mut tag = None;
    for (i, field) in field_kinds.iter().enumerate() {
        let NodeKind::Name {
            text: field_name_text,
        } = &typed_nodes[field.name].node_kind
        else {
            panic!("invalid field name on accessed struct");
        };

        if *field_name_text == *name_text {
            tag = Some(i);
        }
    }

    tag
}

pub fn get_method_subject(typed_nodes: &[impl AsOption<TypedNode>], left: usize) -> Option<MethodSubject> {
    let NodeKind::FieldAccess {
        left: field_access_left,
        ..
    } = &typed_nodes[left].as_option()?.node_kind else
    {
        return None;
    };

    let field_access_left_type = typed_nodes[*field_access_left]
        .as_option()?
        .node_type
        .as_ref()
        .unwrap();

    if field_access_left_type.instance_kind != InstanceKind::Var && field_access_left_type.instance_kind != InstanceKind::Val {
        return None;
    }

    Some(MethodSubject {
        node: *field_access_left,
        type_kind_id: field_access_left_type.type_kind_id,
    })
}