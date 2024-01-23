use crate::{parser::{NodeKind, TypeKind}, type_checker::TypedNode};

pub fn is_type_kind_array(types: &[TypeKind], type_kind: usize) -> bool {
    let type_kind = &types[type_kind];

    matches!(type_kind, TypeKind::Array { .. })
}

pub fn is_typed_expression_array_literal(typed_nodes: &[TypedNode], expression: usize) -> bool {
    let TypedNode { node_kind: NodeKind::Expression { term, trailing_terms }, .. } = &typed_nodes[expression] else {
        return false;
    };

    if trailing_terms.len() > 0 {
        return false;
    }

    let TypedNode { node_kind: NodeKind::Term { unary, trailing_unaries }, .. } = &typed_nodes[*term] else {
        return false;
    };

    if trailing_unaries.len() > 0 {
        return false;
    }

    let TypedNode { node_kind: NodeKind::Unary { op, primary }, .. } = typed_nodes[*unary] else {
        return false;
    };

    if op.is_some() {
        return false;
    }

    let TypedNode { node_kind: NodeKind::Primary { inner }, .. } = typed_nodes[primary] else {
        return false;
    };

    let TypedNode { node_kind: NodeKind::ArrayLiteral { .. }, .. } = typed_nodes[inner] else {
        return false;
    };

    true
}