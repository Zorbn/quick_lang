use crate::type_checker::TypedNode;

pub trait AsOption<T> {
    fn as_option(&self) -> Option<&T>;
}

impl AsOption<TypedNode> for TypedNode {
    fn as_option(&self) -> Option<&TypedNode> {
        Some(self)
    }
}

impl<T> AsOption<T> for Option<T> {
    fn as_option(&self) -> Option<&T> {
        self.as_ref()
    }
}