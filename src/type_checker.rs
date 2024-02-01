use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    environment::Environment,
    parser::{
        ArrayLayout, Field, Node, NodeKind, Op, TypeKind, BOOL_INDEX, CHAR_INDEX, FLOAT32_INDEX,
        FLOAT64_INDEX, INT16_INDEX, INT32_INDEX, INT64_INDEX, INT8_INDEX, INT_INDEX, STRING_INDEX,
        UINT16_INDEX, UINT32_INDEX, UINT64_INDEX, UINT8_INDEX, UINT_INDEX,
    },
    types::get_type_kind_as_pointer,
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
    pub function_declaration_indices: Vec<usize>,
    pub struct_definition_indices: HashMap<Arc<str>, usize>,
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
        function_declaration_indices: Vec<usize>,
        struct_definition_indices: HashMap<Arc<str>, usize>,
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
                type_kind,
            } => self.function_declaration(name, return_type_name, params, type_kind),
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
            NodeKind::ReturnStatement { expression } => self.return_statement(expression),
            NodeKind::DeferStatement { statement } => self.defer_statement(statement),
            NodeKind::IfStatement {
                expression,
                block,
                next,
            } => self.if_statement(expression, block, next),
            NodeKind::SwitchStatement {
                expression,
                case_block,
            } => self.switch_statement(expression, case_block),
            NodeKind::CaseBlock {
                expression,
                block,
                next,
            } => self.case_block(expression, block, next),
            NodeKind::WhileLoop { expression, block } => self.while_loop(expression, block),
            NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                block,
            } => self.for_loop(iterator, op, from, to, by, block),
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
        for function_declaration in &self.function_declaration_indices {
            let NodeKind::FunctionDeclaration { name, type_kind, .. } = &self.nodes[*function_declaration].kind else {
                type_error!(self, "invalid function declaration");
            };
            let NodeKind::Name { text: name_text } = self.nodes[*name].kind.clone() else {
                type_error!(self, "invalid function name");
            };

            self.environment.insert(name_text, *type_kind);
        }

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
        name: usize,
        fields: Arc<Vec<usize>>,
        type_kind: usize,
    ) -> Option<usize> {
        self.check_node(name);

        for field in fields.iter() {
            self.check_node(*field);
        }

        Some(type_kind)
    }

    fn field(&mut self, name: usize, type_name: usize) -> Option<usize> {
        self.check_node(name);
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
        name: usize,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
        type_kind: usize,
    ) -> Option<usize> {
        self.check_node(name);

        for param in params.iter() {
            self.check_node(*param);
        }

        self.check_node(return_type_name);

        Some(type_kind)
    }

    fn extern_function(&mut self, declaration: usize) -> Option<usize> {
        self.check_node(declaration)
    }

    fn param(&mut self, name: usize, type_name: usize) -> Option<usize> {
        self.check_node(name);

        let type_kind = self.check_node(type_name);

        let Node {
            kind: NodeKind::Name { text: name_text },
            ..
        } = self.nodes[name].clone()
        else {
            type_error!(self, "invalid parameter name");
        };

        self.environment.insert(name_text, type_kind.unwrap());

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
        name: usize,
        type_name: Option<usize>,
        expression: usize,
    ) -> Option<usize> {
        self.check_node(name);

        let expression_type_kind = self.check_node(expression);

        let type_kind = if let Some(type_name) = type_name {
            self.check_node(type_name)
        } else {
            expression_type_kind
        };

        if type_kind != expression_type_kind {
            type_error!(self, "mismatched types in variable declaration");
        }

        let Some(type_kind) = type_kind else {
            type_error!(self, "cannot declare untyped variable");
        };

        if let TypeKind::Function { .. } = self.types[type_kind] {
            type_error!(self, "variables can't have a function type, try a function pointer instead");
        }

        let Node {
            kind: NodeKind::Name { text: name_text },
            ..
        } = self.nodes[name].clone()
        else {
            type_error!(self, "invalid variable name");
        };

        self.environment.insert(name_text, type_kind);

        Some(type_kind)
    }

    fn return_statement(&mut self, expression: Option<usize>) -> Option<usize> {
        self.check_node(expression?)
    }

    fn defer_statement(&mut self, statement: usize) -> Option<usize> {
        self.check_node(statement)
    }

    fn if_statement(
        &mut self,
        expression: usize,
        block: usize,
        next: Option<usize>,
    ) -> Option<usize> {
        self.check_node(expression);
        self.check_node(block);

        if let Some(next) = next {
            self.check_node(next);
        }

        None
    }

    fn switch_statement(&mut self, expression: usize, case_block: usize) -> Option<usize> {
        self.check_node(expression);
        self.check_node(case_block);

        None
    }

    fn case_block(
        &mut self,
        expression: usize,
        block: usize,
        next: Option<usize>,
    ) -> Option<usize> {
        self.check_node(expression);
        self.check_node(block);

        if let Some(next) = next {
            self.check_node(next);
        }

        None
    }

    fn while_loop(&mut self, expression: usize, block: usize) -> Option<usize> {
        self.check_node(expression);
        self.check_node(block)
    }

    fn for_loop(
        &mut self,
        iterator: usize,
        _op: Op,
        from: usize,
        to: usize,
        by: Option<usize>,
        block: usize,
    ) -> Option<usize> {
        self.check_node(iterator);
        self.check_node(from);
        self.check_node(to);
        if let Some(by) = by {
            self.check_node(by);
        }
        self.check_node(block)
    }

    fn binary(&mut self, left: usize, op: Op, right: usize) -> Option<usize> {
        let type_kind = self.check_node(left);

        if type_kind != self.check_node(right) {
            type_error!(self, "type mismatch");
        }

        // TODO: Make sure assignments have a variable on the left side, rather than a literal.

        match op {
            Op::Plus
            | Op::Minus
            | Op::Multiply
            | Op::Divide
            | Op::PlusAssign
            | Op::MinusAssign
            | Op::MultiplyAssign
            | Op::DivideAssign => {
                if !TypeChecker::is_type_numeric(type_kind) {
                    type_error!(self, "expected arithmetic types");
                }
            }
            Op::Less | Op::Greater | Op::LessEqual | Op::GreaterEqual => {
                if !TypeChecker::is_type_numeric(type_kind) {
                    type_error!(self, "expected comparable types");
                }

                return Some(BOOL_INDEX);
            }
            Op::Equal | Op::NotEqual => {
                return Some(BOOL_INDEX);
            }
            Op::Not | Op::And | Op::Or => {
                if type_kind != Some(BOOL_INDEX) {
                    type_error!(self, "expected bool");
                }
            }
            _ => {}
        }

        type_kind
    }

    fn unary_prefix(&mut self, op: Op, right: usize) -> Option<usize> {
        let Some(type_kind) = self.check_node(right) else {
            type_error!(self, "cannot apply unary operator to untyped value");
        };

        match op {
            Op::Plus | Op::Minus => {
                if !TypeChecker::is_type_numeric(Some(type_kind)) {
                    type_error!(self, "expected numeric type");
                }
            }
            Op::Not => {
                if type_kind != BOOL_INDEX {
                    type_error!(self, "expected bool");
                }
            }
            // TODO: The reference operator should only be useable on variables, not literals.
            Op::Reference => {
                let pointer_type_kind = get_type_kind_as_pointer(
                    &mut self.types,
                    &mut self.pointer_type_kinds,
                    type_kind,
                );

                return Some(pointer_type_kind);
            }
            _ => {}
        }

        Some(type_kind)
    }

    fn unary_suffix(&mut self, left: usize, op: Op) -> Option<usize> {
        let Some(type_kind) = self.check_node(left) else {
            type_error!(self, "cannot apply unary operator to untyped value");
        };

        if let Op::Dereference = op {
            if !self.pointer_type_kind_set.contains(&type_kind) {
                type_error!(self, "only pointers can be dereferenced");
            }

            let TypeKind::Pointer { inner_type_kind } = &self.types[type_kind] else {
                type_error!(self, "only pointers can be dereferenced");
            };

            return Some(*inner_type_kind);
        }

        Some(type_kind)
    }

    fn call(&mut self, left: usize, args: Arc<Vec<usize>>) -> Option<usize> {
        let Some(type_kind) = self.check_node(left) else {
            type_error!(self, "cannot call untyped value");
        };

        for arg in args.iter() {
            self.check_node(*arg);
        }

        let TypeKind::Function { return_type_kind, .. } = self.types[type_kind] else {
            type_error!(self, "only functions can be called");
        };

        Some(return_type_kind)
    }

    fn index_access(&mut self, left: usize, expression: usize) -> Option<usize> {
        let Some(type_kind) = self.check_node(left) else {
            type_error!(self, "cannot index untyped value");
        };

        let element_type_kind = if let TypeKind::Array {
            element_type_kind, ..
        } = &self.types[type_kind]
        {
            *element_type_kind
        } else {
            type_error!(self, "indexing is only allowed on arrays");
        };

        self.check_node(expression);

        Some(element_type_kind)
    }

    fn field_access(&mut self, left: usize, name: usize) -> Option<usize> {
        let parent_type = self.check_node(left).unwrap();
        self.check_node(name);

        let field_kinds = match &self.types[parent_type] {
            TypeKind::Struct { field_kinds, .. } => field_kinds,
            TypeKind::Pointer { inner_type_kind } => {
                let TypeKind::Struct { field_kinds, .. } = &self.types[*inner_type_kind] else {
                    type_error!(
                        self,
                        "field access is not allowed on pointers to non-struct types"
                    );
                };

                field_kinds
            }
            _ => type_error!(
                self,
                "field access is only allowed on structs or pointers to structs"
            ),
        };

        let Node {
            kind: NodeKind::Name { text: name_text },
            ..
        } = &self.nodes[name]
        else {
            type_error!(self, "invalid field name");
        };

        for Field {
            name: field_name,
            type_kind: field_kind,
        } in field_kinds.iter()
        {
            let NodeKind::Name {
                text: field_name_text,
            } = &self.nodes[*field_name].kind
            else {
                type_error!(self, "invalid field name on struct");
            };

            if *field_name_text == *name_text {
                return Some(*field_kind);
            }
        }

        type_error!(self, "field doesn't exist in struct");
    }

    fn cast(&mut self, left: usize, type_name: usize) -> Option<usize> {
        self.check_node(left);
        self.check_node(type_name)
    }

    fn name(&mut self, _text: Arc<str>) -> Option<usize> {
        None
    }

    fn identifier(&mut self, name: usize) -> Option<usize> {
        self.check_node(name);

        let NodeKind::Name { text } = &self.nodes[name].kind else {
            type_error!(self, "invalid identifier name");
        };

        let Some(type_kind) = self.environment.get(text) else {
            type_error!(self, "undefined identifier");
        };

        Some(type_kind)
    }

    fn int_literal(&mut self, _text: Arc<str>) -> Option<usize> {
        Some(INT_INDEX)
    }

    fn float32_literal(&mut self, _text: Arc<str>) -> Option<usize> {
        Some(FLOAT32_INDEX)
    }

    fn string_literal(&mut self, _text: Arc<str>) -> Option<usize> {
        Some(STRING_INDEX)
    }

    fn bool_literal(&mut self, _value: bool) -> Option<usize> {
        Some(BOOL_INDEX)
    }

    fn char_literal(&mut self, _value: char) -> Option<usize> {
        Some(CHAR_INDEX)
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

    fn struct_literal(&mut self, name: usize, fields: Arc<Vec<usize>>) -> Option<usize> {
        self.check_node(name);

        for field in fields.iter() {
            self.check_node(*field);
        }

        let Node {
            kind: NodeKind::Name { text: name_text },
            ..
        } = &self.nodes[name]
        else {
            type_error!(self, "invalid struct name");
        };

        let Some(definition_index) = self.struct_definition_indices.get(name_text) else {
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

    fn field_literal(&mut self, name: usize, expression: usize) -> Option<usize> {
        self.check_node(name);
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
