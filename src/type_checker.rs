use std::{collections::HashMap, sync::Arc};

use crate::{
    environment::Environment,
    parser::{ArrayLayout, Field, NodeKind, Op, TypeKind, INT_INDEX, STRING_INDEX},
};

#[derive(Clone, Debug)]
pub struct TypedNode {
    pub node_kind: NodeKind,
    pub type_kind: Option<usize>,
}

pub struct TypeChecker {
    pub typed_nodes: Vec<Option<TypedNode>>,
    pub nodes: Vec<NodeKind>,
    pub types: Vec<TypeKind>,
    pub function_declaration_indices: HashMap<String, usize>,
    pub struct_definition_indices: HashMap<String, usize>,
    pub array_type_kinds: HashMap<ArrayLayout, usize>,
    environment: Environment,
    has_function_opened_block: bool,
}

impl TypeChecker {
    pub fn new(
        nodes: Vec<NodeKind>,
        types: Vec<TypeKind>,
        function_declaration_indices: HashMap<String, usize>,
        struct_declaration_indices: HashMap<String, usize>,
        array_type_indices: HashMap<ArrayLayout, usize>,
    ) -> Self {
        let node_count = nodes.len();

        let mut type_checker = Self {
            typed_nodes: Vec::new(),
            nodes,
            types,
            function_declaration_indices,
            struct_definition_indices: struct_declaration_indices,
            array_type_kinds: array_type_indices,
            environment: Environment::new(),
            has_function_opened_block: false,
        };

        type_checker.typed_nodes.resize(node_count, None);
        type_checker.environment.push();

        type_checker
    }

    pub fn check(&mut self, start_index: usize) {
        self.check_node(start_index);
    }

    fn check_node(&mut self, index: usize) -> Option<usize> {
        let type_kind = match self.nodes[index].clone() {
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
            NodeKind::VariableAssignment {
                variable,
                expression,
            } => self.variable_assignment(variable, expression),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression),
            NodeKind::Expression {
                term,
                trailing_terms,
            } => self.expression(term, trailing_terms),
            NodeKind::Term {
                unary,
                trailing_unaries,
            } => self.term(unary, trailing_unaries),
            NodeKind::Unary { op, primary } => self.unary(op, primary),
            NodeKind::Primary { inner } => self.primary(inner),
            NodeKind::Variable { inner } => self.variable(inner),
            NodeKind::VariableName { name } => self.variable_name(name),
            NodeKind::VariableIndex { parent, expression } => {
                self.variable_index(parent, expression)
            }
            NodeKind::VariableField { parent, name } => self.variable_field(parent, name),
            NodeKind::FunctionCall { name, args } => self.function_call(name, args),
            NodeKind::IntLiteral { text } => self.int_literal(text),
            NodeKind::StringLiteral { text } => self.string_literal(text),
            NodeKind::ArrayLiteral {
                elements,
                repeat_count,
            } => self.array_literal(elements, repeat_count),
            NodeKind::StructLiteral { name, fields } => self.struct_literal(name, fields),
            NodeKind::FieldLiteral { name, expression } => self.field_literal(name, expression),
            NodeKind::TypeName { type_kind } => self.type_name(type_kind),
        };

        self.typed_nodes[index] = Some(TypedNode {
            node_kind: self.nodes[index].clone(),
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
        type_name: usize,
        expression: usize,
    ) -> Option<usize> {
        let type_kind = self.check_node(type_name);
        self.check_node(expression);

        self.environment.insert(name, type_kind.unwrap());

        type_kind
    }

    fn variable_assignment(&mut self, variable: usize, expression: usize) -> Option<usize> {
        let type_kind = self.check_node(variable);
        self.check_node(expression);

        type_kind
    }

    fn return_statement(&mut self, expression: usize) -> Option<usize> {
        self.check_node(expression)
    }

    fn expression(
        &mut self,
        term: usize,
        trailing_terms: Arc<Vec<crate::parser::TrailingTerm>>,
    ) -> Option<usize> {
        let type_kind = self.check_node(term);

        for trailing_term in trailing_terms.iter() {
            self.check_node(trailing_term.term);
        }

        type_kind
    }

    fn term(
        &mut self,
        unary: usize,
        trailing_unaries: Arc<Vec<crate::parser::TrailingUnary>>,
    ) -> Option<usize> {
        let type_kind = self.check_node(unary);

        for trailing_unary in trailing_unaries.iter() {
            self.check_node(trailing_unary.unary);
        }

        type_kind
    }

    fn unary(&mut self, _op: Option<Op>, primary: usize) -> Option<usize> {
        self.check_node(primary)
    }

    fn primary(&mut self, inner: usize) -> Option<usize> {
        self.check_node(inner)
    }

    fn variable(&mut self, inner: usize) -> Option<usize> {
        self.check_node(inner)
    }

    fn variable_name(&mut self, name: String) -> Option<usize> {
        self.environment.get(&name)
    }

    fn variable_index(&mut self, parent: usize, expression: usize) -> Option<usize> {
        let parent_type = self.check_node(parent).unwrap();
        let element_type_kind = if let TypeKind::Array {
            element_type_kind, ..
        } = &self.types[parent_type]
        {
            *element_type_kind
        } else {
            panic!("Indexing is only allowed on arrays");
        };

        self.check_node(expression);

        Some(element_type_kind)
    }

    fn variable_field(&mut self, parent: usize, name: String) -> Option<usize> {
        let parent_type = self.check_node(parent).unwrap();
        let TypeKind::Struct { fields_kinds, .. } = &self.types[parent_type] else {
            panic!("Field access is only allowed on structs");
        };

        for Field {
            name: field_name,
            type_kind: field_kind,
        } in fields_kinds.iter()
        {
            if *field_name == name {
                return Some(*field_kind);
            }
        }

        panic!("Field doesn't exist in struct");
    }

    fn function_call(&mut self, name: String, args: Arc<Vec<usize>>) -> Option<usize> {
        for arg in args.iter() {
            self.check_node(*arg);
        }

        let NodeKind::FunctionDeclaration {
            return_type_name, ..
        } = self.nodes[self.function_declaration_indices[&name]]
        else {
            return None;
        };

        self.check_node(return_type_name)
    }

    fn int_literal(&mut self, _text: String) -> Option<usize> {
        Some(INT_INDEX)
    }

    fn string_literal(&mut self, _text: String) -> Option<usize> {
        Some(STRING_INDEX)
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

        if let NodeKind::StructDefinition { type_kind, .. } =
            self.nodes[self.struct_definition_indices[&name]]
        {
            Some(type_kind)
        } else {
            None
        }
    }

    fn field_literal(&mut self, _name: String, expression: usize) -> Option<usize> {
        self.check_node(expression)
    }

    fn type_name(&mut self, type_kind: usize) -> Option<usize> {
        Some(type_kind)
    }
}
