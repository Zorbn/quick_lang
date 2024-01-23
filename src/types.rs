use crate::parser::{NodeKind, TypeKind};

pub fn is_type_name_array(nodes: &[NodeKind], types: &[TypeKind], type_name: usize) -> bool {
    let NodeKind::TypeName { type_kind } = nodes[type_name] else {
        panic!("Tried to emit node that wasn't a type name");
    };
    let type_kind = &types[type_kind];

    matches!(type_kind, TypeKind::Array { .. })
}