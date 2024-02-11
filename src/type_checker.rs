use std::{
    collections::{HashMap, HashSet},
    mem,
    sync::Arc,
};

use crate::{
    const_value::ConstValue,
    environment::Environment,
    file_data::FileData,
    parser::{
        ArrayLayout, DeclarationKind, Field, FunctionLayout, Node, NodeKind, Op, StructLayout,
        TypeKind, BOOL_INDEX, CHAR_INDEX, FLOAT32_INDEX, FLOAT64_INDEX, INT16_INDEX, INT32_INDEX,
        INT64_INDEX, INT8_INDEX, INT_INDEX, STRING_INDEX, TAG_INDEX, UINT16_INDEX, UINT32_INDEX,
        UINT64_INDEX, UINT8_INDEX, UINT_INDEX,
    },
    types::{
        generic_function_to_concrete, generic_struct_to_concrete, get_function_type_kind,
        get_type_kind_as_array, get_type_kind_as_pointer, replace_generic_type_kinds,
    },
};

#[derive(Clone, Debug)]
pub struct TypedNode {
    pub node_kind: NodeKind,
    pub node_type: Option<Type>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InstanceKind {
    Variable,
    Literal,
    Name,
    Const(ConstValue),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Type {
    pub type_kind: usize,
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
    pub type_kinds: Vec<TypeKind>,
    pub array_type_kinds: HashMap<ArrayLayout, usize>,
    pub pointer_type_kinds: HashMap<usize, usize>,
    pub function_type_kinds: HashMap<FunctionLayout, usize>,
    pub struct_type_kinds: HashMap<StructLayout, usize>,
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
        type_kinds: Vec<TypeKind>,
        array_type_kinds: HashMap<ArrayLayout, usize>,
        pointer_type_kinds: HashMap<usize, usize>,
        function_type_kinds: HashMap<FunctionLayout, usize>,
        struct_type_kinds: HashMap<StructLayout, usize>,
        definition_indices: HashMap<Vec<Arc<str>>, usize>,
        files: Arc<Vec<FileData>>,
    ) -> Self {
        let node_count = nodes.len();

        let mut type_checker = Self {
            files,
            typed_nodes: Vec::new(),
            nodes,
            type_kinds,
            array_type_kinds,
            pointer_type_kinds,
            function_type_kinds,
            struct_type_kinds,
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
        self.handle_generic_usages();
    }

    fn handle_generic_usages(&mut self) {
        while let Some(pending_usage) = self.pending_generic_usages.pop() {
            let mut dealiased_usage = Vec::new();
            for mut type_kind in pending_usage.usage.iter().copied() {
                while let TypeKind::Alias { inner_type_kind } = &self.type_kinds[type_kind] {
                    type_kind = *inner_type_kind
                }

                dealiased_usage.push(type_kind);
            }

            self.generic_usages.entry(pending_usage.index).or_default();

            let dealiased_usage = Arc::new(dealiased_usage);
            if !self
                .generic_usages
                .get_mut(&pending_usage.index)
                .unwrap()
                .insert(dealiased_usage.clone())
            {
                continue;
            }

            let type_kind = match self.nodes[pending_usage.index].kind.clone() {
                NodeKind::Function {
                    declaration,
                    statement,
                } => self.function(declaration, statement, Some(dealiased_usage)),
                NodeKind::StructDefinition {
                    name,
                    fields,
                    generic_params,
                    functions,
                    type_kind,
                } => self.struct_definition(
                    name,
                    fields,
                    generic_params,
                    functions,
                    type_kind,
                    Some(dealiased_usage),
                ),
                _ => panic!("invalid generic usage"),
            };

            self.typed_nodes[pending_usage.index] = Some(TypedNode {
                node_kind: self.nodes[pending_usage.index].kind.clone(),
                node_type: type_kind,
            });
        }
    }

    fn check_node(&mut self, index: usize) -> Option<Type> {
        self.last_visited_index = index;

        let node_type = match self.nodes[index].kind.clone() {
            NodeKind::TopLevel {
                functions,
                structs,
                enums,
            } => self.top_level(functions, structs, enums),
            NodeKind::StructDefinition {
                name,
                fields,
                generic_params,
                functions,
                type_kind,
            } => self.struct_definition(name, fields, generic_params, functions, type_kind, None),
            NodeKind::EnumDefinition {
                name,
                variant_names,
                type_kind,
            } => self.enum_definition(name, variant_names, type_kind),
            NodeKind::Field { name, type_name } => self.field(name, type_name),
            NodeKind::Function {
                declaration,
                statement,
            } => self.function(declaration, statement, None),
            NodeKind::FunctionDeclaration {
                name,
                return_type_name,
                params,
                generic_params,
                type_kind,
            } => {
                self.function_declaration(name, return_type_name, params, generic_params, type_kind)
            }
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
            NodeKind::GenericSpecifier {
                left,
                generic_param_type_kinds,
            } => self.generic_specifier(left, generic_param_type_kinds),
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
            NodeKind::TypeName { type_kind } => self.type_name(type_kind),
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
        // TODO: Will this loop for functions do the right thing for functions defined in structs,
        // eg. if two structs have a function with the same name, will one overwrite the other?
        for function in functions.iter() {
            let declaration = match &self.nodes[*function].kind {
                NodeKind::Function { declaration, .. } => declaration,
                NodeKind::ExternFunction { declaration, .. } => declaration,
                _ => type_error!(self, "invalid function"),
            };
            let NodeKind::FunctionDeclaration {
                name, type_kind, ..
            } = &self.nodes[*declaration].kind
            else {
                type_error!(self, "invalid function declaration");
            };
            let NodeKind::Name { text: name_text } = self.nodes[*name].kind.clone() else {
                type_error!(self, "invalid function name");
            };

            let mut resolved_type_kind = 0;
            if let Some(error_type) =
                self.resolve_partial_types(*type_kind, &mut resolved_type_kind)
            {
                return Some(error_type);
            }

            self.environment.insert(
                name_text,
                Type {
                    type_kind: resolved_type_kind,
                    instance_kind: InstanceKind::Variable,
                },
            );
        }

        for struct_definition in structs.iter() {
            let NodeKind::StructDefinition {
                name, type_kind, ..
            } = &self.nodes[*struct_definition].kind
            else {
                type_error!(self, "invalid struct definition");
            };
            let NodeKind::Name { text: name_text } = self.nodes[*name].kind.clone() else {
                type_error!(self, "invalid struct name");
            };

            self.environment.insert(
                name_text,
                Type {
                    type_kind: *type_kind,
                    instance_kind: InstanceKind::Name,
                },
            );
        }

        for enum_definition in enums.iter() {
            let NodeKind::EnumDefinition {
                name, type_kind, ..
            } = &self.nodes[*enum_definition].kind
            else {
                type_error!(self, "invalid enum definition");
            };
            let NodeKind::Name { text: name_text } = self.nodes[*name].kind.clone() else {
                type_error!(self, "invalid enum name");
            };

            self.environment.insert(
                name_text,
                Type {
                    type_kind: *type_kind,
                    instance_kind: InstanceKind::Name,
                },
            );
        }

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

    fn struct_definition(
        &mut self,
        name: usize,
        fields: Arc<Vec<usize>>,
        generic_params: Arc<Vec<usize>>,
        functions: Arc<Vec<usize>>,
        type_kind: usize,
        generic_usage: Option<Arc<Vec<usize>>>,
    ) -> Option<Type> {
        let TypeKind::Struct {
            generic_type_kinds, ..
        } = self.type_kinds[type_kind].clone()
        else {
            type_error!(self, "invalid struct definition");
        };

        if !generic_type_kinds.is_empty() && generic_usage.is_none() {
            return None;
        }

        if let Some(generic_usage) = generic_usage {
            replace_generic_type_kinds(&mut self.type_kinds, &generic_type_kinds, &generic_usage);
        }

        self.check_node(name);

        let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
            type_error!(self, "invalid name in struct definition");
        };

        let struct_type = Type {
            type_kind,
            instance_kind: InstanceKind::Name,
        };
        self.environment.insert(name_text, struct_type.clone());

        self.environment.push();

        for generic_param in generic_params.iter() {
            self.check_node(*generic_param);
        }

        for field in fields.iter() {
            self.check_node(*field);
        }

        for function in functions.iter() {
            self.check_node(*function);
        }

        self.environment.pop();

        Some(struct_type)
    }

    fn enum_definition(
        &mut self,
        name: usize,
        variant_names: Arc<Vec<usize>>,
        type_kind: usize,
    ) -> Option<Type> {
        self.check_node(name);

        let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
            type_error!(self, "invalid name in enum definition");
        };

        let enum_type = Type {
            type_kind,
            instance_kind: InstanceKind::Name,
        };
        self.environment.insert(name_text, enum_type.clone());

        for variant_name in variant_names.iter() {
            self.check_node(*variant_name);
        }

        Some(enum_type)
    }

    fn field(&mut self, name: usize, type_name: usize) -> Option<Type> {
        self.check_node(name);
        self.check_node(type_name)
    }

    fn function(
        &mut self,
        declaration: usize,
        statement: usize,
        generic_usage: Option<Arc<Vec<usize>>>,
    ) -> Option<Type> {
        self.environment.push();
        let declaration_type = self.check_node(declaration)?;

        let TypeKind::Function {
            generic_type_kinds, ..
        } = self.type_kinds[declaration_type.type_kind].clone()
        else {
            type_error!(self, "invalid function declaration");
        };

        if !generic_type_kinds.is_empty() && generic_usage.is_none() {
            self.environment.pop();
            return None;
        }

        self.has_function_opened_block = true;

        if let Some(generic_usage) = generic_usage {
            replace_generic_type_kinds(&mut self.type_kinds, &generic_type_kinds, &generic_usage);
        }

        self.check_node(statement);

        Some(declaration_type)
    }

    fn function_declaration(
        &mut self,
        name: usize,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
        generic_params: Arc<Vec<usize>>,
        type_kind: usize,
    ) -> Option<Type> {
        self.check_node(name);

        for param in params.iter() {
            self.check_node(*param);
        }

        for generic_param in generic_params.iter() {
            self.check_node(*generic_param);
        }

        self.check_node(return_type_name);

        let mut resolved_type_kind = 0;
        if let Some(error_type) = self.resolve_partial_types(type_kind, &mut resolved_type_kind) {
            return Some(error_type);
        }

        Some(Type {
            type_kind: resolved_type_kind,
            instance_kind: InstanceKind::Literal,
        })
    }

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
            type_kind: type_name_type.type_kind,
            instance_kind: InstanceKind::Variable,
        };
        self.environment.insert(name_text, param_type.clone());

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
        _declaration_kind: DeclarationKind,
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
            if variable_type.type_kind != expression_type.type_kind {
                println!(
                    "{:?} != {:?}",
                    self.type_kinds[variable_type.type_kind],
                    self.type_kinds[expression_type.type_kind]
                );
                type_error!(self, "mismatched types in variable declaration");
            }

            variable_type
        } else {
            expression_type
        };

        if let TypeKind::Function { .. } = self.type_kinds[variable_type.type_kind] {
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

        if !matches!(variable_type.instance_kind, InstanceKind::Const(..)) {
            variable_type.instance_kind = InstanceKind::Variable;
        }

        self.environment.insert(name_text, variable_type.clone());

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

        if left_type.type_kind != right_type.type_kind {
            type_error!(self, "type mismatch");
        }

        if left_type.instance_kind == InstanceKind::Name
            || right_type.instance_kind == InstanceKind::Name
        {
            type_error!(self, "binary operators are only useable on instances");
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
                if !TypeChecker::is_type_numeric(left_type.type_kind) {
                    type_error!(self, "expected arithmetic types");
                }
            }
            Op::Less | Op::Greater | Op::LessEqual | Op::GreaterEqual => {
                if !TypeChecker::is_type_numeric(left_type.type_kind) {
                    type_error!(self, "expected comparable types");
                }

                return Some(Type {
                    type_kind: BOOL_INDEX,
                    instance_kind: InstanceKind::Literal,
                });
            }
            Op::Equal | Op::NotEqual => {
                return Some(Type {
                    type_kind: BOOL_INDEX,
                    instance_kind: InstanceKind::Literal,
                });
            }
            Op::And | Op::Or => {
                if left_type.type_kind != BOOL_INDEX {
                    type_error!(self, "expected bool");
                }
            }
            _ => {}
        }

        Some(Type {
            type_kind: left_type.type_kind,
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
            type_kind: binary_type.type_kind,
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
                if !TypeChecker::is_type_numeric(right_type.type_kind) {
                    type_error!(self, "expected numeric type");
                }

                Some(right_type)
            }
            Op::Not => {
                if right_type.type_kind != BOOL_INDEX {
                    type_error!(self, "expected bool");
                }

                Some(right_type)
            }
            Op::Reference => {
                if right_type.instance_kind != InstanceKind::Variable {
                    type_error!(self, "references must refer to a variable");
                }

                if let TypeKind::Function {
                    generic_type_kinds, ..
                } = &self.type_kinds[right_type.type_kind]
                {
                    if !generic_type_kinds.is_empty() {
                        type_error!(self, "cannot take a function pointer to a generic function, specify its types first");
                    }
                }

                let pointer_type_kind = get_type_kind_as_pointer(
                    &mut self.type_kinds,
                    &mut self.pointer_type_kinds,
                    right_type.type_kind,
                );

                Some(Type {
                    type_kind: pointer_type_kind,
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
            type_kind: unary_type.type_kind,
            instance_kind: InstanceKind::Const(result_value),
        })
    }

    fn unary_suffix(&mut self, left: usize, op: Op) -> Option<Type> {
        let left_type = self.check_node(left)?;

        if let Op::Dereference = op {
            let TypeKind::Pointer { inner_type_kind } = &self.type_kinds[left_type.type_kind]
            else {
                type_error!(self, "only pointers can be dereferenced");
            };

            if left_type.instance_kind == InstanceKind::Name {
                type_error!(self, "only pointer instances can be dereferenced");
            }

            Some(Type {
                type_kind: *inner_type_kind,
                instance_kind: InstanceKind::Variable,
            })
        } else {
            type_error!(self, "unknown unary suffix operator")
        }
    }

    fn call(&mut self, left: usize, args: Arc<Vec<usize>>) -> Option<Type> {
        let left_type = self.check_node(left)?;

        for arg in args.iter() {
            self.check_node(*arg);
        }

        let TypeKind::Function {
            return_type_kind,
            generic_type_kinds,
            ..
        } = &self.type_kinds[left_type.type_kind]
        else {
            type_error!(self, "only functions can be called");
        };

        if !generic_type_kinds.is_empty() {
            type_error!(
                self,
                "cannot call generic function without generic specifier"
            );
        }

        Some(Type {
            type_kind: *return_type_kind,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn index_access(&mut self, left: usize, expression: usize) -> Option<Type> {
        let left_type = self.check_node(left)?;

        let element_type_kind = if let TypeKind::Array {
            element_type_kind, ..
        } = &self.type_kinds[left_type.type_kind]
        {
            *element_type_kind
        } else {
            type_error!(self, "indexing is only allowed on arrays");
        };

        self.check_node(expression);

        Some(Type {
            type_kind: element_type_kind,
            instance_kind: InstanceKind::Variable,
        })
    }

    fn field_access(&mut self, left: usize, name: usize) -> Option<Type> {
        let parent_type = self.check_node(left)?;
        self.check_node(name);

        let NodeKind::Name { text: name_text } = &self.nodes[name].kind else {
            type_error!(self, "invalid field name");
        };

        let mut is_tag_access = false;
        let struct_type_kind = match &self.type_kinds[parent_type.type_kind] {
            TypeKind::Struct { is_union, .. } => {
                is_tag_access = parent_type.instance_kind == InstanceKind::Name && *is_union;
                parent_type.type_kind
            }
            TypeKind::Pointer { inner_type_kind } => *inner_type_kind,
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
                            type_kind: parent_type.type_kind,
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
                    type_kind: UINT_INDEX,
                    instance_kind: InstanceKind::Literal,
                });
            }
            _ => type_error!(
                self,
                "field access is only allowed on structs, enums, and pointers to structs"
            ),
        };

        let TypeKind::Struct { field_kinds, .. } = &self.type_kinds[struct_type_kind] else {
            type_error!(self, "field access is only allowed on struct types");
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

            if *field_name_text != *name_text {
                continue;
            }

            if is_tag_access {
                return Some(Type {
                    type_kind: TAG_INDEX,
                    instance_kind: InstanceKind::Literal,
                });
            }

            if let TypeKind::Function {
                param_type_kinds, ..
            } = &self.type_kinds[*field_kind]
            {
                if parent_type.instance_kind == InstanceKind::Literal {
                    type_error!(self, "method calls are not allowed on literals");
                }

                // A method is static if it's first parameter isn't a pointer to it's own struct's type.
                let mut is_method_static = true;
                if param_type_kinds.len() > 0 {
                    if let TypeKind::Pointer { inner_type_kind } =
                        self.type_kinds[param_type_kinds[0]]
                    {
                        is_method_static = inner_type_kind != struct_type_kind;
                    }
                }

                if is_method_static && parent_type.instance_kind != InstanceKind::Name {
                    type_error!(self, "static method calls are not allowed on instances");
                }
            } else if parent_type.instance_kind == InstanceKind::Name {
                type_error!(self, "struct field access is only allowed on instances");
            }

            return Some(Type {
                type_kind: *field_kind,
                instance_kind: InstanceKind::Variable,
            });
        }

        type_error!(self, "field doesn't exist in struct");
    }

    fn cast(&mut self, left: usize, type_name: usize) -> Option<Type> {
        self.check_node(left);
        let type_name_type = self.check_node(type_name)?;

        Some(Type {
            type_kind: type_name_type.type_kind,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_cast(&mut self, left: usize, _type_name: usize, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;
        let left_type = self.check_const_node(left)?;

        let InstanceKind::Const(left_const_value) = left_type.instance_kind else {
            type_error!(self, "cannot cast non const in const expression");
        };

        let result_value = match &self.type_kinds[const_type.type_kind] {
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
            type_kind: const_type.type_kind,
            instance_kind: InstanceKind::Const(result_value),
        })
    }

    fn get_namespaced_name(&mut self, node: usize) -> Option<Vec<Arc<str>>> {
        let mut namespaced_name = Vec::new();

        match self.nodes[node].kind.clone() {
            NodeKind::FieldAccess { left, name } => {
                let Some(Type { type_kind, .. }) =
                    &self.typed_nodes[left].as_ref().unwrap().node_type
                else {
                    return None;
                };

                let TypeKind::Struct {
                    name: left_name, ..
                } = &self.type_kinds[*type_kind]
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

    fn generic_specifier(
        &mut self,
        left: usize,
        generic_param_type_kinds: Arc<Vec<usize>>,
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

                    (function_index, concrete_type_kind, InstanceKind::Variable)
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
    }

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
            type_kind: INT_INDEX,
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
            type_kind: const_type.type_kind,
            instance_kind: InstanceKind::Const(ConstValue::Int { value }),
        })
    }

    fn float32_literal(&mut self, _text: Arc<str>) -> Option<Type> {
        Some(Type {
            type_kind: FLOAT32_INDEX,
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
            type_kind: const_type.type_kind,
            instance_kind: InstanceKind::Const(ConstValue::Float32 { value }),
        })
    }

    fn string_literal(&mut self, _text: Arc<str>) -> Option<Type> {
        Some(Type {
            type_kind: STRING_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_string_literal(&mut self, text: Arc<str>, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;

        Some(Type {
            type_kind: const_type.type_kind,
            instance_kind: InstanceKind::Const(ConstValue::String { value: text }),
        })
    }

    fn bool_literal(&mut self, _value: bool) -> Option<Type> {
        Some(Type {
            type_kind: BOOL_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_bool_literal(&mut self, value: bool, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;

        Some(Type {
            type_kind: const_type.type_kind,
            instance_kind: InstanceKind::Const(ConstValue::Bool { value }),
        })
    }

    fn char_literal(&mut self, _value: char) -> Option<Type> {
        Some(Type {
            type_kind: CHAR_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_char_literal(&mut self, value: char, index: usize) -> Option<Type> {
        let const_type = self.check_node(index)?;

        Some(Type {
            type_kind: const_type.type_kind,
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
        let type_kind = get_type_kind_as_array(
            &mut self.type_kinds,
            &mut self.array_type_kinds,
            node_type.type_kind,
            elements.len() * repeat_count,
        );

        Some(Type {
            type_kind,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn struct_literal(&mut self, left: usize, fields: Arc<Vec<usize>>) -> Option<Type> {
        let struct_type = self.check_node(left)?;

        for field in fields.iter() {
            self.check_node(*field);
        }

        if !matches!(
            self.type_kinds[struct_type.type_kind],
            TypeKind::Struct { .. }
        ) || struct_type.instance_kind != InstanceKind::Name
        {
            type_error!(
                self,
                "expected struct literal to start with the name of a struct type"
            );
        }

        Some(Type {
            type_kind: struct_type.type_kind,
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
            type_kind: UINT_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn const_type_size(&mut self, type_name: usize, index: usize) -> Option<Type> {
        let type_name_type = self.check_node(type_name)?;
        let const_type = self.check_node(index)?;

        let native_size = mem::size_of::<usize>() as u64;

        let value = match self.type_kinds[type_name_type.type_kind] {
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
            type_kind: const_type.type_kind,
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

    fn resolve_partial_types(
        &mut self,
        type_kind: usize,
        resolved_type_kind: &mut usize,
    ) -> Option<Type> {
        match self.type_kinds[type_kind].clone() {
            TypeKind::Array {
                mut element_type_kind,
                element_count,
            } => {
                if let Some(error_type) =
                    self.resolve_partial_types(element_type_kind, &mut element_type_kind)
                {
                    return Some(error_type);
                }
                *resolved_type_kind = get_type_kind_as_array(
                    &mut self.type_kinds,
                    &mut self.array_type_kinds,
                    element_type_kind,
                    element_count,
                );
            }
            TypeKind::Pointer {
                mut inner_type_kind,
            } => {
                if let Some(error_type) =
                    self.resolve_partial_types(inner_type_kind, &mut inner_type_kind)
                {
                    return Some(error_type);
                }
                *resolved_type_kind = get_type_kind_as_pointer(
                    &mut self.type_kinds,
                    &mut self.pointer_type_kinds,
                    inner_type_kind,
                );
            }
            TypeKind::Function {
                param_type_kinds,
                generic_type_kinds,
                return_type_kind,
            } => {
                let mut resolved_param_type_kinds = Vec::new();
                for param_type_kind in param_type_kinds.iter() {
                    let mut resolved_type_kind = 0;
                    if let Some(error_type) =
                        self.resolve_partial_types(*param_type_kind, &mut resolved_type_kind)
                    {
                        return Some(error_type);
                    }
                    resolved_param_type_kinds.push(resolved_type_kind);
                }
                let resolved_param_type_kinds = Arc::new(resolved_param_type_kinds);

                let mut resolved_return_type_kind = 0;
                if let Some(error_type) =
                    self.resolve_partial_types(return_type_kind, &mut resolved_return_type_kind)
                {
                    return Some(error_type);
                }

                let function_layout = FunctionLayout {
                    param_type_kinds: resolved_param_type_kinds,
                    generic_type_kinds,
                    return_type_kind: resolved_return_type_kind,
                };

                *resolved_type_kind = get_function_type_kind(
                    &mut self.type_kinds,
                    &mut self.function_type_kinds,
                    function_layout,
                );
            }
            TypeKind::PartialArray {
                mut element_type_kind,
                const_expression,
            } => {
                if let Some(error_type) =
                    self.resolve_partial_types(element_type_kind, &mut element_type_kind)
                {
                    return Some(error_type);
                }

                let mut element_count = 0;
                if let Some(error_type) =
                    self.const_expression_to_uint(const_expression, &mut element_count)
                {
                    return Some(error_type);
                }

                *resolved_type_kind = get_type_kind_as_array(
                    &mut self.type_kinds,
                    &mut self.array_type_kinds,
                    element_type_kind,
                    element_count,
                );
            }
            TypeKind::PartialGeneric {
                inner_type_kind,
                generic_param_type_kinds,
            } => {
                let TypeKind::Struct { name, .. } = self.type_kinds[inner_type_kind].clone() else {
                    return self
                        .type_error("cannot apply generic specifier to non-struct type name");
                };

                let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
                    return self.type_error("invalid name in generic struct");
                };

                let mut resolved_generic_param_type_kinds = Vec::new();
                for generic_param_type_kind in generic_param_type_kinds.iter() {
                    let mut resolved_type_kind = 0;
                    if let Some(error_type) = self
                        .resolve_partial_types(*generic_param_type_kind, &mut resolved_type_kind)
                    {
                        return Some(error_type);
                    }
                    resolved_generic_param_type_kinds.push(resolved_type_kind);
                }
                let resolved_generic_param_type_kinds = Arc::new(resolved_generic_param_type_kinds);

                let struct_layout = StructLayout {
                    name: name_text.clone(),
                    generic_param_type_kinds: resolved_generic_param_type_kinds.clone(),
                };

                let concrete_type_kind = generic_struct_to_concrete(
                    struct_layout,
                    &mut self.type_kinds,
                    inner_type_kind,
                    &mut self.struct_type_kinds,
                    &mut self.function_type_kinds,
                    &resolved_generic_param_type_kinds,
                );

                let Some(index) = self.definition_indices.get(&vec![name_text]) else {
                    panic!("invalid struct before generic specifier");
                };

                self.pending_generic_usages.push(PendingGenericUsage {
                    index: *index,
                    usage: resolved_generic_param_type_kinds.clone(),
                });

                *resolved_type_kind = concrete_type_kind;
            }
            _ => *resolved_type_kind = type_kind,
        };

        None
    }

    fn type_name(&mut self, mut type_kind: usize) -> Option<Type> {
        if let Some(error_type) = self.resolve_partial_types(type_kind, &mut type_kind) {
            return Some(error_type);
        }

        Some(Type {
            type_kind,
            instance_kind: InstanceKind::Name,
        })
    }

    fn is_type_numeric(type_kind: usize) -> bool {
        matches!(
            type_kind,
            INT_INDEX
                | UINT_INDEX
                | INT8_INDEX
                | UINT8_INDEX
                | INT16_INDEX
                | UINT16_INDEX
                | INT32_INDEX
                | UINT32_INDEX
                | INT64_INDEX
                | UINT64_INDEX
                | FLOAT32_INDEX
                | FLOAT64_INDEX
        )
    }
}
