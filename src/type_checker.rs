use std::{collections::{HashMap, HashSet}, sync::Arc};

use crate::{
    environment::Environment,
    parser::{
        ArrayLayout, Field, Node, NodeKind, Op, TypeKind, BOOL_INDEX, FLOAT32_INDEX, FLOAT64_INDEX, INT16_INDEX,
        INT32_INDEX, INT64_INDEX, INT8_INDEX, INT_INDEX, STRING_INDEX, UINT16_INDEX, UINT32_INDEX,
        UINT64_INDEX, UINT8_INDEX, UINT_INDEX,
    },
};

#[derive(Clone, Debug)]
pub struct TypedNode {
    pub node_kind: NodeKind,
    pub type_kind: Option<usize>,
}

macro_rules! type_error {
    ($self:ident, $message:expr) => {{
        $self.had_error = true;

        println!(
            "Type error at line {}, column {}: {}",
            $self.nodes[$self.last_visited_index].start.line,
            $self.nodes[$self.last_visited_index].start.column,
            $message,
        );

        return None;
    }};
}

pub struct TypeChecker {
    pub typed_nodes: Vec<Option<TypedNode>>,
    pub nodes: Vec<Node>,
    pub types: Vec<TypeKind>,
    pub function_declaration_indices: HashMap<String, usize>,
    pub struct_definition_indices: HashMap<String, usize>,
    pub array_type_kinds: HashMap<ArrayLayout, usize>,
    pub pointer_type_kinds: HashMap<usize, usize>,
    pub had_error: bool,
    pointer_type_kind_set: HashSet<usize>,
    environment: Environment,
    has_function_opened_block: bool,
    last_visited_index: usize,
}

impl TypeChecker {
    pub fn new(
        nodes: Vec<Node>,
        types: Vec<TypeKind>,
        function_declaration_indices: HashMap<String, usize>,
        struct_definition_indices: HashMap<String, usize>,
        array_type_kinds: HashMap<ArrayLayout, usize>,
        pointer_type_kinds: HashMap<usize, usize>,
    ) -> Self {
        let node_count = nodes.len();

        let pointer_type_kind_set = HashSet::from_iter(pointer_type_kinds.values().copied());

        let mut type_checker = Self {
            typed_nodes: Vec::new(),
            nodes,
            types,
            function_declaration_indices,
            struct_definition_indices,
            array_type_kinds,
            pointer_type_kinds,
            pointer_type_kind_set,
            had_error: false,
            environment: Environment::new(),
            has_function_opened_block: false,
            last_visited_index: 0,
        };

        type_checker.typed_nodes.resize(node_count, None);
        type_checker.environment.push();

        type_checker
    }

    pub fn check(&mut self, start_index: usize) {
        self.check_node(start_index);
    }

    fn check_node(&mut self, index: usize) -> Option<usize> {
        self.last_visited_index = index;

        let type_kind = match self.nodes[index].kind.clone() {
            NodeKind::TopLevel { functions, structs } => self.top_level(functions, structs),
            NodeKind::StructDefinition {
                name,
                fields,
                type_kind,
            } => self.struct_definition(name, fields, type_kind),
            NodeKind::Field { name, type_name } => self.field(name, type_name),
            NodeKind::Function { declaration, block } => self.function(declaration, block),
            NodeKind::FunctionDeclaration {
                name,
                return_type_name,
                params,
            } => self.function_declaration(name, return_type_name, params),
            NodeKind::ExternFunction { declaration } => self.extern_function(declaration),
            NodeKind::Param { name, type_name } => self.param(name, type_name),
            NodeKind::Block { statements } => self.block(statements),
            NodeKind::Statement { inner } => self.statement(inner),
            NodeKind::VariableDeclaration {
                is_mutable,
                name,
                type_name,
                expression,
            } => self.variable_declaration(is_mutable, name, type_name, expression),
            // NodeKind::VariableAssignment {
            //     dereference_count,
            //     variable,
            //     expression,
            // } => self.variable_assignment(dereference_count, variable, expression),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression),
            NodeKind::DeferStatement { statement } => self.defer_statement(statement),
            NodeKind::IfStatement { expression, block } => self.if_statement(expression, block),
            NodeKind::WhileLoop { expression, block } => self.while_loop(expression, block),
            NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                block,
            } => self.for_loop(iterator, op, from, to, by, block),
            // NodeKind::Expression {
            //     comparison,
            //     trailing_comparisons,
            // } => self.expression(comparison, trailing_comparisons),
            // NodeKind::Comparision {
            //     binary,
            //     trailing_binary,
            // } => self.comparison(binary, trailing_binary),
            // NodeKind::Binary {
            //     term,
            //     trailing_terms,
            // } => self.binary(term, trailing_terms),
            // NodeKind::Term {
            //     unary,
            //     trailing_unaries,
            // } => self.term(unary, trailing_unaries),
            // NodeKind::UnaryPrefix { op, primary } => self.unary(op, primary),
            // NodeKind::Primary { inner } => self.primary(inner),
            // NodeKind::ParenthesizedExpression { expression } => {
            //     self.parenthesized_expression(expression)
            // }
            // NodeKind::Variable { inner } => self.variable(inner),
            // NodeKind::VariableName { name } => self.variable_name(name),
            // NodeKind::VariableIndex { parent, expression } => {
            //     self.variable_index(parent, expression)
            // }
            // NodeKind::VariableField { parent, name } => self.variable_field(parent, name),
            // NodeKind::FunctionCall { name, args } => self.function_call(name, args),
            NodeKind::IntLiteral { text } => self.int_literal(text),
            NodeKind::Float32Literal { text } => self.float32_literal(text),
            NodeKind::StringLiteral { text } => self.string_literal(text),
            NodeKind::BoolLiteral { value } => self.bool_literal(value),
            NodeKind::ArrayLiteral {
                elements,
                repeat_count,
            } => self.array_literal(elements, repeat_count),
            NodeKind::StructLiteral { name, fields } => self.struct_literal(name, fields),
            NodeKind::FieldLiteral { name, expression } => self.field_literal(name, expression),
            NodeKind::TypeSize { type_name } => self.type_size(type_name),
            NodeKind::TypeName { type_kind } => self.type_name(type_kind),
            NodeKind::Error => type_error!(self, "cannot generate error node"),
        };

        self.typed_nodes[index] = Some(TypedNode {
            node_kind: self.nodes[index].kind.clone(),
            type_kind,
        });

        type_kind
    }

    fn top_level(&mut self, functions: Arc<Vec<usize>>, structs: Arc<Vec<usize>>) -> Option<usize> {
        for struct_definition in structs.iter() {
            self.check_node(*struct_definition);
        }

        for function in functions.iter() {
            self.check_node(*function);
        }

        None
    }

    fn struct_definition(
        &mut self,
        _name: String,
        fields: Arc<Vec<usize>>,
        type_kind: usize,
    ) -> Option<usize> {
        for field in fields.iter() {
            self.check_node(*field);
        }

        Some(type_kind)
    }

    fn field(&mut self, _name: String, type_name: usize) -> Option<usize> {
        self.check_node(type_name)
    }

    fn function(&mut self, declaration: usize, block: usize) -> Option<usize> {
        self.environment.push();
        self.has_function_opened_block = true;
        let type_kind = self.check_node(declaration);
        self.check_node(block);

        type_kind
    }

    fn function_declaration(
        &mut self,
        _name: String,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
    ) -> Option<usize> {
        for param in params.iter() {
            self.check_node(*param);
        }

        self.check_node(return_type_name)
    }

    fn extern_function(&mut self, declaration: usize) -> Option<usize> {
        self.check_node(declaration)
    }

    fn param(&mut self, name: String, type_name: usize) -> Option<usize> {
        let type_kind = self.check_node(type_name);
        self.environment.insert(name, type_kind.unwrap());

        type_kind
    }

    fn block(&mut self, statements: Arc<Vec<usize>>) -> Option<usize> {
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

    fn statement(&mut self, inner: usize) -> Option<usize> {
        self.check_node(inner);

        None
    }

    fn variable_declaration(
        &mut self,
        _is_mutable: bool,
        name: String,
        type_name: Option<usize>,
        expression: usize,
    ) -> Option<usize> {
        let expression_type_kind = self.check_node(expression);

        let type_kind = if let Some(type_name) = type_name {
            self.check_node(type_name)
        } else {
            expression_type_kind
        };

        if type_kind != expression_type_kind {
            type_error!(self, "mismatched types in variable declaration");
        }

        self.environment.insert(name, type_kind.unwrap());

        type_kind
    }

//     fn variable_assignment(
//         &mut self,
//         dereference_count: usize,
//         variable: usize,
//         expression: usize,
//     ) -> Option<usize> {
//         let Some(mut type_kind) = self.check_node(variable) else {
//             type_error!(self, "cannot assign to untyped variable");
//         };
//
//         for _ in 0..dereference_count {
//             let TypeKind::Pointer { inner_type_kind } = &self.types[type_kind] else {
//                 type_error!(self, "only pointers can be dereferenced");
//             };
//
//             type_kind = *inner_type_kind;
//         }
//
//         let Some(expression_type_kind) = self.check_node(expression) else {
//             type_error!(self, "cannot assign an untyped value to a variable");
//         };
//
//         if type_kind != expression_type_kind {
//             type_error!(self, "type mismatch in assignment");
//         }
//
//         Some(type_kind)
//     }

    fn return_statement(&mut self, expression: Option<usize>) -> Option<usize> {
        self.check_node(expression?)
    }

    fn defer_statement(&mut self, statement: usize) -> Option<usize> {
        self.check_node(statement)
    }

    fn if_statement(&mut self, expression: usize, block: usize) -> Option<usize> {
        self.check_node(expression);
        self.check_node(block)
    }

    fn while_loop(&mut self, expression: usize, block: usize) -> Option<usize> {
        self.check_node(expression);
        self.check_node(block)
    }

    fn for_loop(
        &mut self,
        _iterator: String,
        _op: Op,
        from: usize,
        to: usize,
        by: Option<usize>,
        block: usize,
    ) -> Option<usize> {
        self.check_node(from);
        self.check_node(to);
        if let Some(by) = by {
            self.check_node(by);
        }
        self.check_node(block)
    }

//     fn expression(
//         &mut self,
//         comparison: usize,
//         trailing_comparisons: Arc<Vec<TrailingComparison>>,
//     ) -> Option<usize> {
//         let type_kind = self.check_node(comparison);
//
//         for trailing_comparison in trailing_comparisons.iter() {
//             if self.check_node(trailing_comparison.comparison) != Some(BOOL_INDEX) {
//                 type_error!(self, "attempted to perform a boolean operation on a non-boolean");
//             }
//         }
//
//         type_kind
//     }
//
//     fn comparison(
//         &mut self,
//         binary: usize,
//         trailing_binary: Option<TrailingBinary>,
//     ) -> Option<usize> {
//         let type_kind = self.check_node(binary);
//
//         if let Some(trailing_binary) = trailing_binary {
//             if type_kind != self.check_node(trailing_binary.binary) {
//                 type_error!(self, "cannot compare values with different types");
//             }
//
//             if matches!(trailing_binary.op, Op::Less | Op::Greater | Op::LessEqual | Op::GreaterEqual) && !TypeChecker::is_type_numeric(type_kind) {
//                 type_error!(self, "only numeric types can be compared by magnitude");
//             }
//
//             return Some(BOOL_INDEX);
//         }
//
//         type_kind
//     }
//
//     fn binary(&mut self, term: usize, trailing_terms: Arc<Vec<TrailingTerm>>) -> Option<usize> {
//         let type_kind = self.check_node(term);
//
//         for trailing_term in trailing_terms.iter() {
//             if type_kind != self.check_node(trailing_term.term) {
//                 type_error!(self, "cannot perform binary operation on values with different types");
//             }
//
//             if !TypeChecker::is_type_numeric(type_kind) {
//                 type_error!(self, "binary operations can only be performed on numeric types");
//             }
//         }
//
//         type_kind
//     }
//
//     fn term(&mut self, unary: usize, trailing_unaries: Arc<Vec<TrailingUnary>>) -> Option<usize> {
//         let type_kind = self.check_node(unary);
//
//         for trailing_unary in trailing_unaries.iter() {
//             if type_kind != self.check_node(trailing_unary.unary) {
//                 type_error!(self, "cannot perform binary operation on values with different types");
//             }
//
//             if !TypeChecker::is_type_numeric(type_kind) {
//                 type_error!(self, "binary operations can only be performed on numeric types");
//             }
//         }
//
//         type_kind
//     }
//
//     fn unary(&mut self, op: Option<Op>, primary: usize) -> Option<usize> {
//         let type_kind = self.check_node(primary);
//
//         if let Some(op) = op {
//             let Some(type_kind) = type_kind else {
//                 type_error!(self, "cannot apply unary operator to untyped value");
//             };
//
//             match op {
//                 Op::Plus | Op::Minus => if !TypeChecker::is_type_numeric(Some(type_kind)) {
//                     type_error!(self, "numeric operations can only be performed on numeric types");
//                 },
//                 Op::Not => if type_kind != BOOL_INDEX {
//                     type_error!(self, "not operator can only be used on booleans");
//                 },
//                 Op::Reference => if !matches!(self.typed_nodes[primary], Some(TypedNode { node_kind: NodeKind::Variable { .. }, .. })) {
//                     type_error!(self, "references must refer to a variable");
//                 }
//                 Op::Dereference => if !self.pointer_type_kind_set.contains(&type_kind) {
//                     type_error!(self, "cannot dereference non-pointer type");
//                 },
//                 _ => type_error!(self, "unexpected operator in unary expression")
//             }
//         }
//
//         type_kind
//     }
//
//     fn primary(&mut self, inner: usize) -> Option<usize> {
//         self.check_node(inner)
//     }

//     fn parenthesized_expression(&mut self, expression: usize) -> Option<usize> {
//         self.check_node(expression)
//     }
//
//     fn variable(&mut self, inner: usize) -> Option<usize> {
//         self.check_node(inner)
//     }
//
//     fn variable_name(&mut self, name: String) -> Option<usize> {
//         self.environment.get(&name)
//     }
//
//     fn variable_index(&mut self, parent: usize, expression: usize) -> Option<usize> {
//         let parent_type = self.check_node(parent).unwrap();
//         let element_type_kind = if let TypeKind::Array {
//             element_type_kind, ..
//         } = &self.types[parent_type]
//         {
//             *element_type_kind
//         } else {
//             type_error!(self, "indexing is only allowed on arrays");
//         };
//
//         self.check_node(expression);
//
//         Some(element_type_kind)
//     }
//
//     fn variable_field(&mut self, parent: usize, name: String) -> Option<usize> {
//         let parent_type = self.check_node(parent).unwrap();
//         let field_kinds = match &self.types[parent_type] {
//             TypeKind::Struct { field_kinds, .. } => field_kinds,
//             TypeKind::Pointer { inner_type_kind } => {
//                 let TypeKind::Struct { field_kinds, .. } = &self.types[*inner_type_kind] else {
//                     type_error!(
//                         self,
//                         "field access is not allowed on pointers to non-struct types"
//                     );
//                 };
//
//                 field_kinds
//             }
//             _ => type_error!(
//                 self,
//                 "field access is only allowed on structs or pointers to structs"
//             ),
//         };
//
//         for Field {
//             name: field_name,
//             type_kind: field_kind,
//         } in field_kinds.iter()
//         {
//             if *field_name == name {
//                 return Some(*field_kind);
//             }
//         }
//
//         type_error!(self, "field doesn't exist in struct");
//     }
//
//     fn function_call(&mut self, name: String, args: Arc<Vec<usize>>) -> Option<usize> {
//         for arg in args.iter() {
//             self.check_node(*arg);
//         }
//
//         let Some(declaration_index) = self.function_declaration_indices.get(&name) else {
//             type_error!(
//                 self,
//                 &format!("function with name \"{}\" does not exist", name)
//             );
//         };
//
//         let NodeKind::FunctionDeclaration {
//             return_type_name, ..
//         } = self.nodes[*declaration_index].kind
//         else {
//             return None;
//         };
//
//         self.check_node(return_type_name)
//     }

    fn int_literal(&mut self, _text: String) -> Option<usize> {
        Some(INT_INDEX)
    }

    fn float32_literal(&mut self, _text: String) -> Option<usize> {
        Some(FLOAT32_INDEX)
    }

    fn string_literal(&mut self, _text: String) -> Option<usize> {
        Some(STRING_INDEX)
    }

    fn bool_literal(&mut self, _value: bool) -> Option<usize> {
        Some(BOOL_INDEX)
    }

    fn array_literal(&mut self, elements: Arc<Vec<usize>>, repeat_count: usize) -> Option<usize> {
        for element in elements.iter() {
            self.check_node(*element);
        }

        let node_type = self.check_node(*elements.first()?)?;
        self.array_type_kinds
            .get(&ArrayLayout {
                element_type_kind: node_type,
                element_count: elements.len() * repeat_count,
            })
            .copied()
    }

    fn struct_literal(&mut self, name: String, fields: Arc<Vec<usize>>) -> Option<usize> {
        for field in fields.iter() {
            self.check_node(*field);
        }

        let Some(definition_index) = self.struct_definition_indices.get(&name) else {
            type_error!(
                self,
                &format!("struct with name \"{}\" does not exist", name)
            );
        };

        if let NodeKind::StructDefinition { type_kind, .. } = self.nodes[*definition_index].kind {
            Some(type_kind)
        } else {
            None
        }
    }

    fn field_literal(&mut self, _name: String, expression: usize) -> Option<usize> {
        self.check_node(expression)
    }

    fn type_size(&mut self, type_name: usize) -> Option<usize> {
        self.check_node(type_name);

        Some(INT_INDEX)
    }

    fn type_name(&mut self, type_kind: usize) -> Option<usize> {
        Some(type_kind)
    }

    fn is_type_numeric(type_kind: Option<usize>) -> bool {
        matches!(
            type_kind,
            Some(INT_INDEX)
                | Some(UINT_INDEX)
                | Some(INT8_INDEX)
                | Some(UINT8_INDEX)
                | Some(INT16_INDEX)
                | Some(UINT16_INDEX)
                | Some(INT32_INDEX)
                | Some(UINT32_INDEX)
                | Some(INT64_INDEX)
                | Some(UINT64_INDEX)
                | Some(FLOAT32_INDEX)
                | Some(FLOAT64_INDEX)
        )
    }
}
