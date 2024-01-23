use crate::parser::{NodeKind, TypeKind};

pub fn is_type_name_array(nodes: &[NodeKind], types: &[TypeKind], type_name: usize) -> bool {
    let NodeKind::TypeName { type_kind } = nodes[type_name] else {
        panic!("Tried to emit node that wasn't a type name");
    };
    let type_kind = &types[type_kind];

    matches!(type_kind, TypeKind::Array { .. })
}

pub fn is_expression_array_literal(nodes: &Vec<NodeKind>, expression: usize) -> bool {
    let NodeKind::Expression { term, trailing_terms } = &nodes[expression] else {
        return false;
    };

    if trailing_terms.len() > 0 {
        return false;
    }

    let NodeKind::Term { unary, trailing_unaries } = &nodes[*term] else {
        return false;
    };

    if trailing_unaries.len() > 0 {
        return false;
    }

    let NodeKind::Unary { op, primary } = nodes[*unary] else {
        return false;
    };

    if op.is_some() {
        return false;
    }

    let NodeKind::Primary { inner } = nodes[primary] else {
        return false;
    };

    let NodeKind::ArrayLiteral { .. } = nodes[inner] else {
        return false;
    };

    true
}