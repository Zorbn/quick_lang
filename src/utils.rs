use std::sync::Arc;

use crate::{
    parser::{NodeIndex, NodeKind},
    typer::TypedNode,
};

// TODO: Get rid of this, could just use a matches! inline...
pub fn is_typed_expression_array_literal(typed_nodes: &[TypedNode], expression: NodeIndex) -> bool {
    matches!(
        typed_nodes[expression.node_index].node_kind,
        NodeKind::ArrayLiteral { .. }
    )
}

pub fn is_first_typed_param_me(typed_nodes: &[TypedNode], typed_params: &Arc<Vec<NodeIndex>>) -> bool {
    if !typed_params.is_empty() {
        let NodeKind::Param { name: param_name, .. } = &typed_nodes[typed_params[0].node_index].node_kind else {
            panic!("invalid param");
        };

        let NodeKind::Name { text: param_name_text, .. } = &typed_nodes[param_name.node_index].node_kind else {
            panic!("invalid param name");
        };

        if param_name_text.as_ref() == "me" {
            return true;
        }
    }

    false
}