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