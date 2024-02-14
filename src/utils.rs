use std::sync::Arc;

use crate::{
    parser::NodeKind, typer::TypedNode, type_kinds::Field
};

// TODO: Get rid of this, could just use a matches! inline...
pub fn is_typed_expression_array_literal(typed_nodes: &[TypedNode], expression: usize) -> bool {
    matches!(
        typed_nodes[expression].node_kind,
        NodeKind::ArrayLiteral { .. }
    )
}

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