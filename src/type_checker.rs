/*

How the flow type checking needs to be implemented:

struct Gen<A> {
	func(self Gen.<A>) Void {
		var other = Gen.<A> {};
	}
}

func main() Int32 {
	var gen = Gen.<Int> {};
}

The following should be done for all types, generic or otherwise, starting at the body of main this should cause type checking of everything that was used.
However, it would be good to type check all functions/structs that haven't been checked yet, even if they are never used.

Environment for types and functions should have keys be name + optional generic args.

On encountering Gen.<Int>, look it up in the environment (Gen + Int), and realize that it's not there.
look up "Gen" to find generic struct.
Add Gen.<Int> to the environment with a placeholder TypeKind to get a TypeId. Because Gen.<Int> is a global type it should be added to the top level of the environment.
Type check with A as Int. Push environment with A having Int's TypeId. Because A is a scoped type it should be added to the current level of the environment.
Check each function (prototype only) and normal field to get all the field types for Gen.<Int>.
When "self Gen.<A>" is reached, look up A to find that it's TypeId is Int's TypeId, then look up Gen.<Int> to find the type id that was added earlier, use it.

When done with the prototypes of each field/function, replace the placeholder TypeKind for Gen.<Int> with the real TypeKind at the same TypeId using the information gathered.
Now loop over each function body in Gen.<Int> to check them. This might require recursing if more new types are encountered

Pop environment.

Loop over each function doing type checking.
When "A" is type checked look it up in the environment, and find Int

*/

use std::{
    collections::{HashMap, HashSet},
    mem,
    sync::Arc,
};

use crate::{
    const_value::ConstValue, environment::Environment, file_data::FileData, parser::{DeclarationKind, Node, NodeKind, Op}, type_kinds::{Field, TypeKind, TypeKinds}, utils::get_method_subject
};

// Used to search for the index of an array type by its layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayLayout {
    pub element_type_kind: usize,
    pub element_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionLayout {
    pub param_type_kinds: Arc<Vec<usize>>,
    pub generic_type_kinds: Arc<Vec<usize>>,
    pub return_type_kind: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructLayout {
    pub name: Arc<str>,
    pub generic_param_type_kinds: Arc<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PointerLayout {
    pub inner_type_kind: usize,
    pub is_inner_mutable: bool,
}

#[derive(Clone, Debug)]
pub struct TypedNode {
    pub node_kind: NodeKind,
    pub node_type: Option<Type>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InstanceKind {
    Var,
    Val,
    Literal,
    Name,
    Const(ConstValue),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Type {
    pub type_kind_id: usize,
    pub instance_kind: InstanceKind,
}

macro_rules! type_error {
    ($self:ident, $message:expr) => {{
        return $self.type_error($message);
    }};
}

struct PendingGenericUsage {
    index: usize,
    usage: Arc<Vec<usize>>,
}

pub struct TypeChecker {
    pub typed_nodes: Vec<Option<TypedNode>>,
    pub nodes: Vec<Node>,
    pub type_kinds: TypeKinds,
    pub definition_indices: HashMap<Vec<Arc<str>>, usize>,
    pub generic_usages: HashMap<usize, HashSet<Arc<Vec<usize>>>>,
    pub had_error: bool,
    files: Arc<Vec<FileData>>,
    environment: Environment<Type>,
    has_function_opened_block: bool,
    last_visited_index: usize,
    pending_generic_usages: Vec<PendingGenericUsage>,
}

impl TypeChecker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nodes: Vec<Node>,
        definition_indices: HashMap<Vec<Arc<str>>, usize>,
        files: Arc<Vec<FileData>>,
    ) -> Self {
        let node_count = nodes.len();

        let mut type_checker = Self {
            files,
            typed_nodes: Vec::new(),
            nodes,
            type_kinds: TypeKinds::new(),
            definition_indices,
            generic_usages: HashMap::new(),
            had_error: false,
            environment: Environment::new(),
            has_function_opened_block: false,
            last_visited_index: 0,
            pending_generic_usages: Vec::new(),
        };

        type_checker.typed_nodes.resize(node_count, None);
        type_checker.environment.push();

        type_checker.type_kinds.add_or_get(TypeKind::Int);
        type_checker.type_kinds.add_or_get(TypeKind::String);
        type_checker.type_kinds.add_or_get(TypeKind::Bool);
        type_checker.type_kinds.add_or_get(TypeKind::Char);
        type_checker.type_kinds.add_or_get(TypeKind::Void);
        type_checker.type_kinds.add_or_get(TypeKind::UInt);
        type_checker.type_kinds.add_or_get(TypeKind::Int8);
        type_checker.type_kinds.add_or_get(TypeKind::UInt8);
        type_checker.type_kinds.add_or_get(TypeKind::Int16);
        type_checker.type_kinds.add_or_get(TypeKind::UInt16);
        type_checker.type_kinds.add_or_get(TypeKind::Int32);
        type_checker.type_kinds.add_or_get(TypeKind::UInt32);
        type_checker.type_kinds.add_or_get(TypeKind::Int64);
        type_checker.type_kinds.add_or_get(TypeKind::UInt64);
        type_checker.type_kinds.add_or_get(TypeKind::Float32);
        type_checker.type_kinds.add_or_get(TypeKind::Float64);
        type_checker.type_kinds.add_or_get(TypeKind::Tag);

        type_checker
    }

    fn type_error(&mut self, message: &str) -> Option<Type> {
        self.had_error = true;
        self.nodes[self.last_visited_index]
            .start
            .error("Type", message, &self.files);

        None
    }

    pub fn check(&mut self, start_index: usize) {
        self.check_node(start_index);
    }

    fn check_node(&mut self, index: usize) -> Option<Type> {
        self.last_visited_index = index;

        let node_type = match self.nodes[index].kind.clone() {
            NodeKind::TopLevel {
                functions,
                structs,
                enums,
            } => self.top_level(functions, structs, enums),
            NodeKind::ExternFunction { declaration } => self.extern_function(declaration),
            NodeKind::Param { name, type_name } => self.param(name, type_name),
            NodeKind::Block { statements } => self.block(statements),
            NodeKind::Statement { inner } => self.statement(inner),
            NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                type_name,
                expression,
            } => self.variable_declaration(declaration_kind, name, type_name, expression),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression),
            NodeKind::DeferStatement { statement } => self.defer_statement(statement),
            NodeKind::IfStatement {
                expression,
                statement,
                next,
            } => self.if_statement(expression, statement, next),
            NodeKind::SwitchStatement {
                expression,
                case_statement,
            } => self.switch_statement(expression, case_statement),
            NodeKind::CaseStatement {
                expression,
                statement,
                next,
            } => self.case_statement(expression, statement, next),
            NodeKind::WhileLoop {
                expression,
                statement,
            } => self.while_loop(expression, statement),
            NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                statement,
            } => self.for_loop(iterator, op, from, to, by, statement),
            NodeKind::ConstExpression { inner } => self.const_expression(inner),
            NodeKind::Binary { left, op, right } => self.binary(left, op, right),
            NodeKind::UnaryPrefix { op, right } => self.unary_prefix(op, right),
            NodeKind::UnarySuffix { left, op } => self.unary_suffix(left, op),
            NodeKind::Call { left, args } => self.call(left, args),
            NodeKind::IndexAccess { left, expression } => self.index_access(left, expression),
            NodeKind::FieldAccess { left, name } => self.field_access(left, name),
            NodeKind::Cast { left, type_name } => self.cast(left, type_name),
            NodeKind::Name { text } => self.name(text),
            NodeKind::Identifier { name } => self.identifier(name),
            NodeKind::IntLiteral { text } => self.int_literal(text),
            NodeKind::Float32Literal { text } => self.float32_literal(text),
            NodeKind::StringLiteral { text } => self.string_literal(text),
            NodeKind::BoolLiteral { value } => self.bool_literal(value),
            NodeKind::CharLiteral { value } => self.char_literal(value),
            NodeKind::ArrayLiteral {
                elements,
                repeat_count_const_expression,
            } => self.array_literal(elements, repeat_count_const_expression),
            NodeKind::StructLiteral { left, fields } => self.struct_literal(left, fields),
            NodeKind::FieldLiteral { name, expression } => self.field_literal(name, expression),
            NodeKind::TypeSize { type_name } => self.type_size(type_name),
            NodeKind::Error => type_error!(self, "cannot generate error node"),
        };

        self.typed_nodes[index] = Some(TypedNode {
            node_kind: self.nodes[index].kind.clone(),
            node_type: node_type.clone(),
        });

        node_type
    }

    fn check_const_node(&mut self, index: usize) -> Option<Type> {
        self.last_visited_index = index;

        let node_type = match self.nodes[index].kind.clone() {
            NodeKind::Binary { left, op, right } => self.const_binary(left, op, right, index),
            NodeKind::UnaryPrefix { op, right } => self.const_unary_prefix(op, right, index),
            NodeKind::Cast { left, type_name } => self.const_cast(left, type_name, index),
            NodeKind::Identifier { name } => self.const_identifier(name, index),
            NodeKind::IntLiteral { text } => self.const_int_literal(text, index),
            NodeKind::Float32Literal { text } => self.const_float32_literal(text, index),
            NodeKind::StringLiteral { text } => self.const_string_literal(text, index),
            NodeKind::BoolLiteral { value } => self.const_bool_literal(value, index),
            NodeKind::CharLiteral { value } => self.const_char_literal(value, index),
            NodeKind::TypeSize { type_name } => self.const_type_size(type_name, index),
            _ => {
                self.type_error("non-constant in constant expression");
                None
            }
        };

        self.typed_nodes[index] = Some(TypedNode {
            node_kind: self.nodes[index].kind.clone(),
            node_type: node_type.clone(),
        });

        node_type
    }

    fn top_level(
        &mut self,
        functions: Arc<Vec<usize>>,
        structs: Arc<Vec<usize>>,
        enums: Arc<Vec<usize>>,
    ) -> Option<Type> {
        for struct_definition in structs.iter() {
            self.check_node(*struct_definition);
        }

        for enum_definition in enums.iter() {
            self.check_node(*enum_definition);
        }

        for function in functions.iter() {
            self.check_node(*function);
        }

        None
    }

//     fn struct_definition(
//         &mut self,
//         name: usize,
//         fields: Arc<Vec<usize>>,
//         generic_params: Arc<Vec<usize>>,
//         functions: Arc<Vec<usize>>,
//         generic_usage: Option<Arc<Vec<usize>>>,
//     ) -> Option<Type> {
//         let TypeKind::Struct {
//             generic_type_kinds, ..
//         } = self.type_kinds[type_kind].clone()
//         else {
//             type_error!(self, "invalid struct definition");
//         };
//
//         if !generic_type_kinds.is_empty() && generic_usage.is_none() {
//             return None;
//         }
//
//         if let Some(generic_usage) = generic_usage {
//             replace_generic_type_kinds(&mut self.type_kinds, &generic_type_kinds, &generic_usage);
//         }
//
//         self.check_node(name);
//
//         let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
//             type_error!(self, "invalid name in struct definition");
//         };
//
//         let struct_type = Type {
//             type_kind,
//             instance_kind: InstanceKind::Name,
//         };
//         self.environment.insert(name_text, struct_type.clone());
//
//         self.environment.push();
//
//         for generic_param in generic_params.iter() {
//             self.check_node(*generic_param);
//         }
//
//         for field in fields.iter() {
//             self.check_node(*field);
//         }
//
//         for function in functions.iter() {
//             self.check_node(*function);
//         }
//
//         self.environment.pop();
//
//         Some(struct_type)
//     }
//
//     fn enum_definition(
//         &mut self,
//         name: usize,
//         variant_names: Arc<Vec<usize>>,
//     ) -> Option<Type> {
//         self.check_node(name);
//
//         let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
//             type_error!(self, "invalid name in enum definition");
//         };
//
//         let type_kind = self.type_kinds.add_or_get(TypeKind::Enum { name, variant_names });
//
//         let enum_type = Type {
//             type_kind,
//             instance_kind: InstanceKind::Name,
//         };
//         self.environment.insert(name_text, enum_type.clone());
//
//         for variant_name in variant_names.iter() {
//             self.check_node(*variant_name);
//         }
//
//         Some(enum_type)
//     }
//
//     fn field(&mut self, name: usize, type_name: usize) -> Option<Type> {
//         self.check_node(name);
//         self.check_node(type_name)
//     }
//
//     fn function(
//         &mut self,
//         declaration: usize,
//         statement: usize,
//         generic_usage: Option<Arc<Vec<usize>>>,
//     ) -> Option<Type> {
//         self.environment.push();
//         let declaration_type = self.check_node(declaration)?;
//
//         let TypeKind::Function {
//             generic_type_kinds, ..
//         } = self.type_kinds[declaration_type.type_kind].clone()
//         else {
//             type_error!(self, "invalid function declaration");
//         };
//
//         if !generic_type_kinds.is_empty() && generic_usage.is_none() {
//             self.environment.pop();
//             return None;
//         }
//
//         self.has_function_opened_block = true;
//
//         if let Some(generic_usage) = generic_usage {
//             replace_generic_type_kinds(&mut self.type_kinds, &generic_type_kinds, &generic_usage);
//         }
//
//         self.check_node(statement);
//
//         Some(declaration_type)
//     }
//
//     fn function_declaration(
//         &mut self,
//         name: usize,
//         return_type_name: usize,
//         params: Arc<Vec<usize>>,
//         generic_params: Arc<Vec<usize>>,
//     ) -> Option<Type> {
//         self.check_node(name);
//
//         for param in params.iter() {
//             self.check_node(*param);
//         }
//
//         for generic_param in generic_params.iter() {
//             self.check_node(*generic_param);
//         }
//
//         self.check_node(return_type_name);
//
//         let mut resolved_type_kind = 0;
//         if let Some(error_type) = self.resolve_partial_types(type_kind, &mut resolved_type_kind) {
//             return Some(error_type);
//         }
//
//         Some(Type {
//             type_kind: resolved_type_kind,
//             instance_kind: InstanceKind::Literal,
//         })
//     }

    fn extern_function(&mut self, declaration: usize) -> Option<Type> {
        let NodeKind::FunctionDeclaration { generic_params, .. } = &self.nodes[declaration].kind
        else {
            type_error!(self, "invalid function declaration");
        };

        if !generic_params.is_empty() {
            type_error!(self, "extern function cannot be generic");
        }

        self.check_node(declaration)
    }

    fn param(&mut self, name: usize, type_name: usize) -> Option<Type> {
        self.check_node(name);

        let type_name_type = self.check_node(type_name)?;

        let Node {
            kind: NodeKind::Name { text: name_text },
            ..
        } = self.nodes[name].clone()
        else {
            type_error!(self, "invalid parameter name");
        };

        let param_type = Type {
            type_kind_id: type_name_type.type_kind_id,
            instance_kind: InstanceKind::Var,
        };
        self.environment.insert(name_text, param_type.clone(), false);

        Some(param_type)
    }

    fn block(&mut self, statements: Arc<Vec<usize>>) -> Option<Type> {
        if !self.has_function_opened_block {
            self.environment.push();
        } else {
            self.has_function_opened_block = false;
        }

        for statement in statements.iter() {
            self.check_node(*statement);
        }

        self.environment.pop();

        None
    }

    fn statement(&mut self, inner: Option<usize>) -> Option<Type> {
        self.check_node(inner?);

        None
    }

    fn variable_declaration(
        &mut self,
        declaration_kind: DeclarationKind,
        name: usize,
        type_name: Option<usize>,
        expression: usize,
    ) -> Option<Type> {
        self.check_node(name);

        let expression_type = self.check_node(expression)?;

        if expression_type.instance_kind == InstanceKind::Name {
            type_error!(self, "only instances of types can be stored in variables");
        }

        let mut variable_type = if let Some(type_name) = type_name {
            let variable_type = self.check_node(type_name)?;
            if variable_type.type_kind_id != expression_type.type_kind_id {
                type_error!(self, "mismatched types in variable declaration");
            }

            variable_type
        } else {
            expression_type
        };

        if declaration_kind == DeclarationKind::Const
            && !matches!(variable_type.instance_kind, InstanceKind::Const(..))
        {
            type_error!(self, "cannot declare a const with a non-const value");
        }

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(variable_type.type_kind_id) {
            type_error!(
                self,
                "variables can't have a function type, try a function pointer instead"
            );
        }

        let Node {
            kind: NodeKind::Name { text: name_text },
            ..
        } = self.nodes[name].clone()
        else {
            type_error!(self, "invalid variable name");
        };

        variable_type.instance_kind = match declaration_kind {
            DeclarationKind::Var => InstanceKind::Var,
            DeclarationKind::Val => InstanceKind::Val,
            DeclarationKind::Const => variable_type.instance_kind,
        };

        self.environment.insert(name_text, variable_type.clone(), false);

        Some(variable_type)
    }

    fn return_statement(&mut self, expression: Option<usize>) -> Option<Type> {
        self.check_node(expression?)
    }

    fn defer_statement(&mut self, statement: usize) -> Option<Type> {
        self.check_node(statement)
    }

    fn if_statement(
        &mut self,
        expression: usize,
        statement: usize,
        next: Option<usize>,
    ) -> Option<Type> {
        self.check_node(expression);
        self.check_node(statement);

        if let Some(next) = next {
            self.check_node(next);
        }

        None
    }

    fn switch_statement(&mut self, expression: usize, case_statement: usize) -> Option<Type> {
        self.check_node(expression);
        self.check_node(case_statement);

        None
    }

    fn case_statement(
        &mut self,
        expression: usize,
        statement: usize,
        next: Option<usize>,
    ) -> Option<Type> {
        self.check_node(expression);
        self.check_node(statement);

        if let Some(next) = next {
            self.check_node(next);
        }

        None
    }

    fn while_loop(&mut self, expression: usize, statement: usize) -> Option<Type> {
        self.check_node(expression);
        self.check_node(statement);

        None
    }

    fn for_loop(
        &mut self,
        iterator: usize,
        _op: Op,
        from: usize,
        to: usize,
        by: Option<usize>,
        statement: usize,
    ) -> Option<Type> {
        self.check_node(iterator);
        self.check_node(from);
        self.check_node(to);
        if let Some(by) = by {
            self.check_node(by);
        }
        self.check_node(statement);

        None
    }

    fn const_expression(&mut self, inner: usize) -> Option<Type> {
        self.check_const_node(inner)
    }

    fn binary(&mut self, left: usize, op: Op, right: usize) -> Option<Type> {
        let left_type = self.check_node(left)?;
        let right_type = self.check_node(right)?;

        if left_type.type_kind_id != right_type.type_kind_id {
            type_error!(self, "type mismatch");
        }

        if left_type.instance_kind == InstanceKind::Name
            || right_type.instance_kind == InstanceKind::Name
        {
            type_error!(self, "binary operators are only useable on instances");
        }

        if matches!(
            op,
            Op::Assign | Op::PlusAssign | Op::MinusAssign | Op::MultiplyAssign | Op::DivideAssign
        ) && left_type.instance_kind != InstanceKind::Var
        {
            type_error!(self, "only variables can be assigned to");
        }

        match op {
            Op::Plus
            | Op::Minus
            | Op::Multiply
            | Op::Divide
            | Op::PlusAssign
            | Op::MinusAssign
            | Op::MultiplyAssign
            | Op::DivideAssign => {
                if !self.type_kinds.get_by_id(left_type.type_kind_id).is_numeric() {
                    type_error!(self, "expected arithmetic types");
                }
            }
            Op::Less | Op::Greater | Op::LessEqual | Op::GreaterEqual => {
                if !self.type_kinds.get_by_id(left_type.type_kind_id).is_numeric() {
                    type_error!(self, "expected comparable types");
                }

                return Some(Type {
                    type_kind_id: self.type_kinds.add_or_get(TypeKind::Bool),
                    instance_kind: InstanceKind::Literal,
                });
            }
            Op::Equal | Op::NotEqual => {
                return Some(Type {
                    type_kind_id: self.type_kinds.add_or_get(TypeKind::Bool),
                    instance_kind: InstanceKind::Literal,
                });
            }
            Op::And | Op::Or => {
                if self.type_kinds.get_by_id(left_type.type_kind_id) != TypeKind::Bool {
                    type_error!(self, "expected bool");
                }
            }
            _ => {}
        }

        Some(Type {
            type_kind_id: left_type.type_kind_id,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_binary(&mut self, left: usize, op: Op, right: usize, index: usize) -> Option<Type> {
        let binary_type = self.check_node(index)?;

        let InstanceKind::Const(left_const_value) = self.check_const_node(left)?.instance_kind
        else {
            type_error!(self, "expected const operand")
        };
        let InstanceKind::Const(right_const_value) = self.check_const_node(right)?.instance_kind
        else {
            type_error!(self, "expected const operand")
        };

        let result_value = match op {
            Op::Plus => left_const_value.add(right_const_value),
            Op::Minus => left_const_value.subtract(right_const_value),
            Op::Multiply => left_const_value.multiply(right_const_value),
            Op::Divide => left_const_value.divide(right_const_value),
            Op::Equal => Some(ConstValue::Bool {
                value: left_const_value == right_const_value,
            }),
            Op::NotEqual => Some(ConstValue::Bool {
                value: left_const_value != right_const_value,
            }),
            Op::Less => left_const_value.less(right_const_value),
            Op::Greater => left_const_value.greater(right_const_value),
            Op::LessEqual => left_const_value.less_equal(right_const_value),
            Op::GreaterEqual => left_const_value.greater_equal(right_const_value),
            Op::And => left_const_value.and(right_const_value),
            Op::Or => left_const_value.or(right_const_value),
            _ => type_error!(self, "unexpected operator in constant binary expression"),
        };

        let Some(result_value) = result_value else {
            type_error!(self, "unexpected const types for operator");
        };

        Some(Type {
            type_kind_id: binary_type.type_kind_id,
            instance_kind: InstanceKind::Const(result_value),
        })
    }

    fn unary_prefix(&mut self, op: Op, right: usize) -> Option<Type> {
        let right_type = self.check_node(right)?;

        if right_type.instance_kind == InstanceKind::Name {
            type_error!(
                self,
                "unary prefix operators can only be applied to instances"
            );
        }

        match op {
            Op::Plus | Op::Minus => {
                if !self.type_kinds.get_by_id(right_type.type_kind_id).is_numeric() {
                    type_error!(self, "expected numeric type");
                }

                Some(right_type)
            }
            Op::Not => {
                if self.type_kinds.get_by_id(right_type.type_kind_id) != TypeKind::Bool {
                    type_error!(self, "expected bool");
                }

                Some(right_type)
            }
            Op::Reference => {
                let is_mutable = match right_type.instance_kind {
                    InstanceKind::Val => false,
                    InstanceKind::Var => true,
                    _ => type_error!(self, "references must refer to a variable"),
                };

                if let TypeKind::Function {
                    generic_type_kind_ids, ..
                } = &self.type_kinds.get_by_id(right_type.type_kind_id)
                {
                    if !generic_type_kind_ids.is_empty() {
                        type_error!(self, "cannot take a function pointer to a generic function, specify its types first");
                    }
                }

                let type_kind_id = self.type_kinds.add_or_get(TypeKind::Pointer { inner_type_kind_id: right_type.type_kind_id, is_inner_mutable: is_mutable });

                Some(Type {
                    type_kind_id,
                    instance_kind: InstanceKind::Literal,
                })
            }
            _ => type_error!(self, "unknown unary prefix operator"),
        }
    }

    fn const_unary_prefix(&mut self, op: Op, right: usize, index: usize) -> Option<Type> {
        let unary_type = self.check_node(index)?;

        let InstanceKind::Const(right_const_value) = self.check_const_node(right)?.instance_kind
        else {
            type_error!(self, "expected const operand")
        };

        let result_value = match op {
            Op::Plus => Some(right_const_value),
            Op::Minus => right_const_value.unary_prefix_minus(),
            Op::Not => right_const_value.unary_prefix_not(),
            _ => type_error!(
                self,
                "unexpected operator in constant unary prefix expression"
            ),
        };

        let Some(result_value) = result_value else {
            type_error!(self, "unexpected const types for operator");
        };

        Some(Type {
            type_kind_id: unary_type.type_kind_id,
            instance_kind: InstanceKind::Const(result_value),
        })
    }

    fn unary_suffix(&mut self, left: usize, op: Op) -> Option<Type> {
        let left_type = self.check_node(left)?;

        if let Op::Dereference = op {
            let TypeKind::Pointer {
                inner_type_kind_id,
                is_inner_mutable,
            } = &self.type_kinds.get_by_id(left_type.type_kind_id)
            else {
                type_error!(self, "only pointers can be dereferenced");
            };

            if left_type.instance_kind == InstanceKind::Name {
                type_error!(self, "only pointer instances can be dereferenced");
            }

            let instance_kind = if *is_inner_mutable {
                InstanceKind::Var
            } else {
                InstanceKind::Val
            };

            Some(Type {
                type_kind_id: *inner_type_kind_id,
                instance_kind,
            })
        } else {
            type_error!(self, "unknown unary suffix operator")
        }
    }

    fn call(&mut self, left: usize, args: Arc<Vec<usize>>) -> Option<Type> {
        let left_type = self.check_node(left)?;

        let TypeKind::Function {
            return_type_kind_id,
            generic_type_kind_ids,
            param_type_kind_ids,
            ..
        } = self.type_kinds.get_by_id(left_type.type_kind_id).clone()
        else {
            type_error!(self, "only functions can be called");
        };

        if !generic_type_kind_ids.is_empty() {
            type_error!(
                self,
                "cannot call generic function without generic specifier"
            );
        }

        let mut skip_param_count = 0;
        if let Some(method_subject) = get_method_subject(&self.typed_nodes, left) {
            skip_param_count += 1;

            if param_type_kind_ids.len() < 1 {
                type_error!(self, "function with no arguments cannot be called as an instance method");
            }

            let type_kind_as_pointer_id = if let TypeKind::Pointer { .. } = self.type_kinds.get_by_id(method_subject.type_kind_id) {
                method_subject.type_kind_id
            } else {
                self.type_kinds.add_or_get(TypeKind::Pointer { inner_type_kind_id: method_subject.type_kind_id, is_inner_mutable: true })
            };

            if type_kind_as_pointer_id != param_type_kind_ids[0] {
                type_error!(self, "incorrect type used as subject of method call");
            }
        }

        if args.len() + skip_param_count != param_type_kind_ids.len() {
            type_error!(self, "wrong number of arguments");
        }

        for (arg, param_type_kind) in args.iter().zip(param_type_kind_ids.iter().skip(skip_param_count)) {
            let arg_type = self.check_node(*arg)?;

            if arg_type.type_kind_id != *param_type_kind {
                type_error!(self, "incorrect argument type");
            }
        }

        Some(Type {
            type_kind_id: return_type_kind_id,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn index_access(&mut self, left: usize, expression: usize) -> Option<Type> {
        let left_type = self.check_node(left)?;

        let element_type_kind_id = if let TypeKind::Array {
            element_type_kind_id, ..
        } = &self.type_kinds.get_by_id(left_type.type_kind_id)
        {
            *element_type_kind_id
        } else {
            type_error!(self, "indexing is only allowed on arrays");
        };

        let expression_type = self.check_node(expression)?;

        let TypeKind::Int = &self.type_kinds.get_by_id(expression_type.type_kind_id) else {
            type_error!(self, "expected index to be of type int");
        };

        Some(Type {
            type_kind_id: element_type_kind_id,
            instance_kind: InstanceKind::Var,
        })
    }

    fn field_access(&mut self, left: usize, name: usize) -> Option<Type> {
        let parent_type = self.check_node(left)?;
        self.check_node(name);

        let NodeKind::Name { text: name_text } = &self.nodes[name].kind else {
            type_error!(self, "invalid field name");
        };

        let mut is_tag_access = false;
        let (struct_type_kind_id, field_instance_kind) = match &self.type_kinds.get_by_id(parent_type.type_kind_id)
        {
            TypeKind::Struct { is_union, .. } => {
                is_tag_access = parent_type.instance_kind == InstanceKind::Name && *is_union;
                (parent_type.type_kind_id, InstanceKind::Var)
            }
            TypeKind::Pointer {
                inner_type_kind_id,
                is_inner_mutable,
            } => {
                let field_instance_kind = if *is_inner_mutable {
                    InstanceKind::Var
                } else {
                    InstanceKind::Val
                };

                (*inner_type_kind_id, field_instance_kind)
            }
            TypeKind::Enum { variant_names, .. } => {
                for variant_name in variant_names.iter() {
                    let NodeKind::Name {
                        text: variant_name_text,
                    } = &self.nodes[*variant_name].kind
                    else {
                        type_error!(self, "invalid enum variant name");
                    };

                    if *variant_name_text == *name_text {
                        return Some(Type {
                            type_kind_id: parent_type.type_kind_id,
                            instance_kind: InstanceKind::Literal,
                        });
                    }
                }

                type_error!(self, "variant not found in enum");
            }
            TypeKind::Array { .. } => {
                if name_text.as_ref() != "count" {
                    type_error!(self, "field not found on array");
                }

                return Some(Type {
                    type_kind_id: self.type_kinds.add_or_get(TypeKind::UInt),
                    instance_kind: InstanceKind::Literal,
                });
            }
            _ => type_error!(
                self,
                "field access is only allowed on structs, enums, and pointers to structs"
            ),
        };

        let TypeKind::Struct { field_kinds, .. } = &self.type_kinds.get_by_id(struct_type_kind_id) else {
            type_error!(self, "field access is only allowed on struct types");
        };

        for Field {
            name: field_name,
            type_kind_id: field_kind_id,
        } in field_kinds.iter()
        {
            let NodeKind::Name {
                text: field_name_text,
            } = &self.nodes[*field_name].kind
            else {
                type_error!(self, "invalid field name on struct");
            };

            if *field_name_text != *name_text {
                continue;
            }

            if is_tag_access {
                return Some(Type {
                    type_kind_id: self.type_kinds.add_or_get(TypeKind::Tag),
                    instance_kind: InstanceKind::Literal,
                });
            }

            if let TypeKind::Function {
                param_type_kind_ids, ..
            } = &self.type_kinds.get_by_id(*field_kind_id)
            {
                if parent_type.instance_kind == InstanceKind::Literal {
                    type_error!(self, "method calls are not allowed on literals");
                }

                // A method is static if it's first parameter isn't a pointer to it's own struct's type.
                let mut is_method_static = true;
                if param_type_kind_ids.len() > 0 {
                    if let TypeKind::Pointer {
                        inner_type_kind_id, ..
                    } = self.type_kinds.get_by_id(param_type_kind_ids[0])
                    {
                        is_method_static = inner_type_kind_id != struct_type_kind_id;
                    }
                }

                if is_method_static && parent_type.instance_kind != InstanceKind::Name {
                    type_error!(self, "static method calls are not allowed on instances");
                }
            } else if parent_type.instance_kind == InstanceKind::Name {
                type_error!(self, "struct field access is only allowed on instances");
            }

            return Some(Type {
                type_kind_id: *field_kind_id,
                instance_kind: field_instance_kind,
            });
        }

        type_error!(self, "field doesn't exist in struct");
    }

    fn cast(&mut self, left: usize, type_name: usize) -> Option<Type> {
        self.check_node(left);
        let type_name_type = self.check_node(type_name)?;

        Some(Type {
            type_kind_id: type_name_type.type_kind_id,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_cast(&mut self, left: usize, _type_name: usize, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;
        let left_type = self.check_const_node(left)?;

        let InstanceKind::Const(left_const_value) = left_type.instance_kind else {
            type_error!(self, "cannot cast non const in const expression");
        };

        let result_value = match &self.type_kinds.get_by_id(const_type.type_kind_id) {
            TypeKind::Int
            | TypeKind::Int8
            | TypeKind::Int16
            | TypeKind::Int32
            | TypeKind::Int64 => left_const_value.cast_to_int(),
            TypeKind::UInt
            | TypeKind::UInt8
            | TypeKind::UInt16
            | TypeKind::UInt32
            | TypeKind::UInt64 => left_const_value.cast_to_uint(),
            TypeKind::Float32 | TypeKind::Float64 => left_const_value.cast_to_float(),
            TypeKind::Bool => left_const_value.cast_to_bool(),
            TypeKind::Char => left_const_value.cast_to_char(),
            _ => type_error!(self, "compile time casts to this type are not allowed"),
        };

        let Some(result_value) = result_value else {
            type_error!(self, "value cannot be cast at compile time");
        };

        Some(Type {
            type_kind_id: const_type.type_kind_id,
            instance_kind: InstanceKind::Const(result_value),
        })
    }

    fn get_namespaced_name(&mut self, node: usize) -> Option<Vec<Arc<str>>> {
        let mut namespaced_name = Vec::new();

        match self.nodes[node].kind.clone() {
            NodeKind::FieldAccess { left, name } => {
                let Some(Type { type_kind_id, .. }) =
                    &self.typed_nodes[left].as_ref().unwrap().node_type
                else {
                    return None;
                };

                let TypeKind::Struct {
                    name: left_name, ..
                } = &self.type_kinds.get_by_id(*type_kind_id)
                else {
                    return None;
                };

                let NodeKind::Name {
                    text: left_name_text,
                } = self.nodes[*left_name].kind.clone()
                else {
                    return None;
                };

                let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
                    return None;
                };

                namespaced_name.push(left_name_text);
                namespaced_name.push(name_text);
            }
            NodeKind::Identifier { name } => {
                let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
                    return None;
                };

                namespaced_name.push(name_text);
            }
            _ => return None,
        };

        Some(namespaced_name)
    }

/*     fn generic_specifier(
        &mut self,
        left: usize,
        generic_arg_type_names: Arc<Vec<usize>>,
    ) -> Option<Type> {
        let left_type = self.check_node(left)?;
        let (index, concrete_type_kind, instance_kind) =
            match self.type_kinds[left_type.type_kind].clone() {
                TypeKind::Function {
                    generic_type_kinds, ..
                } => {
                    if generic_type_kinds.is_empty() {
                        type_error!(
                            self,
                            "generic specifier cannot be applied to non-generic functions"
                        );
                    }

                    let Some(namespaced_name) = self.get_namespaced_name(left) else {
                        type_error!(self, "expected function name before generic specifier");
                    };

                    let Some(function_index) = self.definition_indices.get(&namespaced_name) else {
                        type_error!(self, "invalid function before generic specifier");
                    };

                    let concrete_type_kind = generic_function_to_concrete(
                        &mut self.type_kinds,
                        left_type.type_kind,
                        &mut self.function_type_kinds,
                        &generic_param_type_kinds,
                    );

                    (function_index, concrete_type_kind, InstanceKind::Val)
                }
                TypeKind::Struct {
                    generic_type_kinds, ..
                } => {
                    if generic_type_kinds.is_empty() {
                        type_error!(
                            self,
                            "generic specifier cannot be applied to non-generic functions"
                        );
                    }

                    let Some(namespaced_name) = self.get_namespaced_name(left) else {
                        type_error!(self, "expected struct name before generic specifier");
                    };

                    let Some(struct_index) = self.definition_indices.get(&namespaced_name) else {
                        type_error!(self, "invalid struct before generic specifier");
                    };

                    let struct_layout = StructLayout {
                        name: namespaced_name.last().unwrap().clone(),
                        generic_param_type_kinds: generic_param_type_kinds.clone(),
                    };

                    let concrete_type_kind = generic_struct_to_concrete(
                        struct_layout,
                        &mut self.type_kinds,
                        left_type.type_kind,
                        &mut self.struct_type_kinds,
                        &mut self.function_type_kinds,
                        &generic_param_type_kinds,
                    );

                    (struct_index, concrete_type_kind, InstanceKind::Name)
                }
                _ => type_error!(
                    self,
                    "generic specifier can only be applied to functions and structs"
                ),
            };

        self.pending_generic_usages.push(PendingGenericUsage {
            index: *index,
            usage: generic_param_type_kinds.clone(),
        });

        Some(Type {
            type_kind: concrete_type_kind,
            instance_kind,
        })
    } */

    fn name(&mut self, _text: Arc<str>) -> Option<Type> {
        None
    }

    fn identifier(&mut self, name: usize) -> Option<Type> {
        self.check_node(name);

        let NodeKind::Name { text } = &self.nodes[name].kind else {
            type_error!(self, "invalid identifier name");
        };

        let Some(identifier_type) = self.environment.get(text) else {
            type_error!(self, "undeclared identifier");
        };

        Some(identifier_type)
    }

    fn const_identifier(&mut self, _name: usize, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;

        if !matches!(const_type.instance_kind, InstanceKind::Const(..)) {
            type_error!(self, "expected identifier to refer to a const value");
        }

        Some(const_type)
    }

    fn int_literal(&mut self, _text: Arc<str>) -> Option<Type> {
        Some(Type {
            type_kind_id: self.type_kinds.add_or_get(TypeKind::Int),
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_int_literal(&mut self, text: Arc<str>, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;
        let Ok(value) = text.parse::<i64>() else {
            self.type_error("invalid value of int literal");
            return None;
        };

        Some(Type {
            type_kind_id: const_type.type_kind_id,
            instance_kind: InstanceKind::Const(ConstValue::Int { value }),
        })
    }

    fn float32_literal(&mut self, _text: Arc<str>) -> Option<Type> {
        Some(Type {
            type_kind_id: self.type_kinds.add_or_get(TypeKind::Float32),
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_float32_literal(&mut self, text: Arc<str>, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;
        let Ok(value) = text.parse::<f32>() else {
            self.type_error("invalid value of float32 literal");
            return None;
        };

        Some(Type {
            type_kind_id: const_type.type_kind_id,
            instance_kind: InstanceKind::Const(ConstValue::Float32 { value }),
        })
    }

    fn string_literal(&mut self, _text: Arc<str>) -> Option<Type> {
        Some(Type {
            type_kind_id: self.type_kinds.add_or_get(TypeKind::String),
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_string_literal(&mut self, text: Arc<str>, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;

        Some(Type {
            type_kind_id: const_type.type_kind_id,
            instance_kind: InstanceKind::Const(ConstValue::String { value: text }),
        })
    }

    fn bool_literal(&mut self, _value: bool) -> Option<Type> {
        Some(Type {
            type_kind_id: self.type_kinds.add_or_get(TypeKind::Bool),
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_bool_literal(&mut self, value: bool, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;

        Some(Type {
            type_kind_id: const_type.type_kind_id,
            instance_kind: InstanceKind::Const(ConstValue::Bool { value }),
        })
    }

    fn char_literal(&mut self, _value: char) -> Option<Type> {
        Some(Type {
            type_kind_id: self.type_kinds.add_or_get(TypeKind::Char),
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_char_literal(&mut self, value: char, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;

        Some(Type {
            type_kind_id: const_type.type_kind_id,
            instance_kind: InstanceKind::Const(ConstValue::Char { value }),
        })
    }

    fn array_literal(
        &mut self,
        elements: Arc<Vec<usize>>,
        repeat_count_const_expression: Option<usize>,
    ) -> Option<Type> {
        for element in elements.iter() {
            self.check_node(*element);
        }

        let repeat_count = if let Some(const_expression) = repeat_count_const_expression {
            let mut repeat_count = 0;
            if let Some(error_type) =
                self.const_expression_to_uint(const_expression, &mut repeat_count)
            {
                return Some(error_type);
            }

            repeat_count
        } else {
            1
        };

        let node_type = self.check_node(*elements.first()?)?;
        let type_kind_id = self.type_kinds.add_or_get(TypeKind::Array { element_type_kind_id: node_type.type_kind_id, element_count: elements.len() * repeat_count });

        Some(Type {
            type_kind_id,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn struct_literal(&mut self, left: usize, fields: Arc<Vec<usize>>) -> Option<Type> {
        let struct_type = self.check_node(left)?;

        for field in fields.iter() {
            self.check_node(*field);
        }

        if !matches!(
            self.type_kinds.get_by_id(struct_type.type_kind_id),
            TypeKind::Struct { .. }
        ) || struct_type.instance_kind != InstanceKind::Name
        {
            type_error!(
                self,
                "expected struct literal to start with the name of a struct type"
            );
        }

        Some(Type {
            type_kind_id: struct_type.type_kind_id,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn field_literal(&mut self, name: usize, expression: usize) -> Option<Type> {
        self.check_node(name);
        self.check_node(expression)
    }

    fn type_size(&mut self, type_name: usize) -> Option<Type> {
        self.check_node(type_name);

        Some(Type {
            type_kind_id: self.type_kinds.add_or_get(TypeKind::UInt),
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_type_size(&mut self, type_name: usize, index: usize) -> Option<Type> {
        let type_name_type = self.check_node(type_name)?;
        let const_type = self.check_node(index)?;

        let native_size = mem::size_of::<usize>() as u64;

        let value = match self.type_kinds.get_by_id(type_name_type.type_kind_id) {
            TypeKind::Int => native_size,
            TypeKind::String => native_size,
            TypeKind::Bool => 1,
            TypeKind::Char => 1,
            TypeKind::Void => 0,
            TypeKind::UInt => native_size,
            TypeKind::Int8 => 1,
            TypeKind::UInt8 => 1,
            TypeKind::Int16 => 2,
            TypeKind::UInt16 => 2,
            TypeKind::Int32 => 4,
            TypeKind::UInt32 => 4,
            TypeKind::Int64 => 8,
            TypeKind::UInt64 => 8,
            TypeKind::Float32 => 4,
            TypeKind::Float64 => 8,
            TypeKind::Tag => native_size,
            TypeKind::Pointer { .. } => native_size,
            _ => type_error!(self, "size unknown at compile time"),
        };

        Some(Type {
            type_kind_id: const_type.type_kind_id,
            instance_kind: InstanceKind::Const(ConstValue::UInt { value }),
        })
    }

    fn const_expression_to_uint(
        &mut self,
        const_expression: usize,
        result: &mut usize,
    ) -> Option<Type> {
        let NodeKind::ConstExpression { inner } = self.nodes[const_expression].kind else {
            return self.type_error("expected const expression");
        };

        let Some(Type {
            instance_kind: InstanceKind::Const(const_value),
            ..
        }) = self.const_expression(inner)
        else {
            return self.type_error("expected const value from const expression");
        };

        *result = match const_value {
            ConstValue::Int { value } => {
                if value < 0 {
                    return self.type_error("expected positive integer");
                } else {
                    value as usize
                }
            }
            ConstValue::UInt { value } => value as usize,
            _ => return self.type_error("expected integer"),
        };

        None
    }
}