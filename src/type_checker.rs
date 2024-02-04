use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    environment::Environment,
    file_data::FileData,
    parser::{
        ArrayLayout, Field, FunctionLayout, Node, NodeKind, Op, TypeKind, BOOL_INDEX, CHAR_INDEX, FLOAT32_INDEX, FLOAT64_INDEX, INT16_INDEX, INT32_INDEX, INT64_INDEX, INT8_INDEX, INT_INDEX, STRING_INDEX, UINT16_INDEX, UINT32_INDEX, UINT64_INDEX, UINT8_INDEX, UINT_INDEX
    },
    types::{add_type, generic_params_to_concrete, get_function_type_kind, get_type_kind_as_array, get_type_kind_as_pointer},
};

#[derive(Clone, Debug)]
pub struct TypedNode {
    pub node_kind: NodeKind,
    pub node_type: Option<Type>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InstanceKind {
    Variable,
    Literal,
    Name,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Type {
    pub type_kind: usize,
    pub instance_kind: InstanceKind,
}

macro_rules! type_error {
    ($self:ident, $message:expr) => {{
        $self.had_error = true;
        $self.nodes[$self.last_visited_index]
            .start
            .error("Type", $message, &$self.files);

        return None;
    }};
}

pub struct TypeChecker {
    pub typed_nodes: Vec<Option<TypedNode>>,
    pub nodes: Vec<Node>,
    pub types: Vec<TypeKind>,
    pub array_type_kinds: HashMap<ArrayLayout, usize>,
    pub pointer_type_kinds: HashMap<usize, usize>,
    pub function_type_kinds: HashMap<FunctionLayout, usize>,
    pub function_declaration_indices: HashMap<Vec<Arc<str>>, usize>,
    pub generic_function_usages: HashMap<usize, HashSet<Arc<Vec<usize>>>>,
    pub had_error: bool,
    files: Arc<Vec<FileData>>,
    environment: Environment<Type>,
    has_function_opened_block: bool,
    last_visited_index: usize,
}

impl TypeChecker {
    pub fn new(
        nodes: Vec<Node>,
        types: Vec<TypeKind>,
        array_type_kinds: HashMap<ArrayLayout, usize>,
        pointer_type_kinds: HashMap<usize, usize>,
        function_type_kinds: HashMap<FunctionLayout, usize>,
        function_declaration_indices: HashMap<Vec<Arc<str>>, usize>,
        files: Arc<Vec<FileData>>,
    ) -> Self {
        let node_count = nodes.len();

        let mut type_checker = Self {
            files,
            typed_nodes: Vec::new(),
            nodes,
            types,
            array_type_kinds,
            pointer_type_kinds,
            function_type_kinds,
            function_declaration_indices,
            generic_function_usages: HashMap::new(),
            had_error: false,
            environment: Environment::new(),
            has_function_opened_block: false,
            last_visited_index: 0,
        };

        type_checker.typed_nodes.resize(node_count, None);
        type_checker.environment.push();

        type_checker
    }

    fn add_type(&mut self, type_kind: TypeKind) -> usize {
        add_type(&mut self.types, type_kind)
    }

    pub fn check(&mut self, start_index: usize) {
        self.check_node(start_index);
    }

    fn check_node(&mut self, index: usize) -> Option<Type> {
        self.last_visited_index = index;

        let type_kind = match self.nodes[index].kind.clone() {
            NodeKind::TopLevel {
                functions,
                structs,
                enums,
            } => self.top_level(functions, structs, enums),
            NodeKind::StructDefinition {
                name,
                fields,
                functions,
                type_kind,
            } => self.struct_definition(name, fields, functions, type_kind),
            NodeKind::EnumDefinition {
                name,
                variant_names,
                type_kind,
            } => self.enum_definition(name, variant_names, type_kind),
            NodeKind::Field { name, type_name } => self.field(name, type_name),
            NodeKind::Function { declaration, block } => self.function(declaration, block),
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
            NodeKind::GenericSpecifier { left, type_names } => {
                self.generic_specifier(left, type_names)
            }
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
            node_type: type_kind,
        });

        type_kind
    }

    fn top_level(
        &mut self,
        functions: Arc<Vec<usize>>,
        structs: Arc<Vec<usize>>,
        enums: Arc<Vec<usize>>,
    ) -> Option<Type> {
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

            self.environment.insert(
                name_text,
                Type {
                    type_kind: *type_kind,
                    instance_kind: InstanceKind::Variable,
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
        functions: Arc<Vec<usize>>,
        type_kind: usize,
    ) -> Option<Type> {
        self.check_node(name);

        let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
            type_error!(self, "invalid name in struct definition");
        };

        let struct_type = Type {
            type_kind,
            instance_kind: InstanceKind::Name,
        };
        self.environment.insert(name_text, struct_type);

        for field in fields.iter() {
            self.check_node(*field);
        }

        for function in functions.iter() {
            self.check_node(*function);
        }

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
        self.environment.insert(name_text, enum_type);

        for variant_name in variant_names.iter() {
            self.check_node(*variant_name);
        }

        Some(enum_type)
    }

    fn field(&mut self, name: usize, type_name: usize) -> Option<Type> {
        self.check_node(name);
        self.check_node(type_name)
    }

    fn function(&mut self, declaration: usize, block: usize) -> Option<Type> {
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
        generic_params: Arc<Vec<usize>>,
        type_kind: usize,
    ) -> Option<Type> {
        self.check_node(name);

        for param in params.iter() {
            self.check_node(*param);
        }

        for generic_param in generic_params.iter() {
            self.check_node(*generic_param);

            let NodeKind::Name { text } = self.nodes[*generic_param].kind.clone() else {
                panic!("Invalid generic param");
            };

            let generic_type_kind = self.add_type(TypeKind::Partial);
            self.environment.insert(
                text,
                Type {
                    type_kind: generic_type_kind,
                    instance_kind: InstanceKind::Name,
                },
            )
        }

        self.check_node(return_type_name);

        Some(Type {
            type_kind,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn extern_function(&mut self, declaration: usize) -> Option<Type> {
        let NodeKind::FunctionDeclaration {
            generic_params,
            ..
        } = &self.nodes[declaration].kind
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
        self.environment.insert(name_text, param_type);

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

    fn statement(&mut self, inner: usize) -> Option<Type> {
        self.check_node(inner);

        None
    }

    fn variable_declaration(
        &mut self,
        _is_mutable: bool,
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
            self.check_node(type_name)?
        } else {
            expression_type
        };

        if variable_type.type_kind != expression_type.type_kind {
            type_error!(self, "mismatched types in variable declaration");
        }

        if let TypeKind::Function { .. } = self.types[variable_type.type_kind] {
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

        variable_type.instance_kind = InstanceKind::Variable;
        self.environment.insert(name_text, variable_type);

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
        block: usize,
        next: Option<usize>,
    ) -> Option<Type> {
        self.check_node(expression);
        self.check_node(block);

        if let Some(next) = next {
            self.check_node(next);
        }

        None
    }

    fn switch_statement(&mut self, expression: usize, case_block: usize) -> Option<Type> {
        self.check_node(expression);
        self.check_node(case_block);

        None
    }

    fn case_block(&mut self, expression: usize, block: usize, next: Option<usize>) -> Option<Type> {
        self.check_node(expression);
        self.check_node(block);

        if let Some(next) = next {
            self.check_node(next);
        }

        None
    }

    fn while_loop(&mut self, expression: usize, block: usize) -> Option<Type> {
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
    ) -> Option<Type> {
        self.check_node(iterator);
        self.check_node(from);
        self.check_node(to);
        if let Some(by) = by {
            self.check_node(by);
        }
        self.check_node(block)
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
            Op::Not | Op::And | Op::Or => {
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

                let pointer_type_kind = get_type_kind_as_pointer(
                    &mut self.types,
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

    fn unary_suffix(&mut self, left: usize, op: Op) -> Option<Type> {
        let left_type = self.check_node(left)?;

        if let Op::Dereference = op {
            let TypeKind::Pointer { inner_type_kind } = &self.types[left_type.type_kind] else {
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
            return_type_kind, ..
        } = self.types[left_type.type_kind]
        else {
            type_error!(self, "only functions can be called");
        };

        Some(Type {
            type_kind: return_type_kind,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn index_access(&mut self, left: usize, expression: usize) -> Option<Type> {
        let left_type = self.check_node(left)?;

        let element_type_kind = if let TypeKind::Array {
            element_type_kind, ..
        } = &self.types[left_type.type_kind]
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

        let struct_type_kind = match &self.types[parent_type.type_kind] {
            TypeKind::Struct { .. } => parent_type.type_kind,
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
            _ => type_error!(
                self,
                "field access is only allowed on structs, enums, and pointers to structs"
            ),
        };

        let TypeKind::Struct { field_kinds, .. } = &self.types[struct_type_kind] else {
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

            if let TypeKind::Function {
                param_type_kinds, ..
            } = &self.types[*field_kind]
            {
                if parent_type.instance_kind == InstanceKind::Literal {
                    type_error!(self, "method calls are not allowed on literals");
                }

                // A method is static if it's first parameter isn't a pointer to it's own struct's type.
                let mut is_method_static = true;
                if param_type_kinds.len() > 0 {
                    if let TypeKind::Pointer { inner_type_kind } = self.types[param_type_kinds[0]] {
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

    fn get_namespaced_name(&mut self, node: usize) -> Option<Vec<Arc<str>>> {
        let mut namespaced_name = Vec::new();

        match self.nodes[node].kind.clone() {
            NodeKind::FieldAccess { left, name } => {
                let NodeKind::Identifier { name: left_name } = &self.nodes[left].kind else {
                    return None;
                };

                let NodeKind::Name { text: left_name_text } = self.nodes[*left_name].kind.clone() else {
                    return None;
                };

                let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
                    return None;
                };

                namespaced_name.push(left_name_text);
                namespaced_name.push(name_text);
            },
            NodeKind::Identifier { name } => {
                let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
                    return None;
                };

                namespaced_name.push(name_text);
            },
            _ => return None,
        };

        Some(namespaced_name)
    }

    fn generic_specifier(&mut self, left: usize, type_names: Arc<Vec<usize>>) -> Option<Type> {
        let left_type = self.check_node(left)?;
        let TypeKind::Function {
            param_type_kinds,
            generic_type_kinds,
            return_type_kind,
        } = self.types[left_type.type_kind].clone()
        else {
            type_error!(self, "cannot apply generic specifier to non-function");
        };

        if generic_type_kinds.is_empty() {
            type_error!(self, "generic specifier can only be applied to generic functions");
        }

        let Some(namespaced_name) = self.get_namespaced_name(left) else {
            type_error!(self, "expected function name before generic specifier");
        };

        let Some(function_index) = self.function_declaration_indices.get(&namespaced_name) else {
            type_error!(self, "invalid function before generic specifier");
        };

        let Some(usages) = self.generic_function_usages.get_mut(function_index) else {
            type_error!(self, "invalid function before generic specifier");
        };

        usages.insert(type_names.clone());

        let concrete_param_type_kinds = generic_params_to_concrete(&param_type_kinds, &generic_type_kinds, &type_names);

        let concrete_function = FunctionLayout {
            param_type_kinds: Arc::new(concrete_param_type_kinds),
            generic_type_kinds: Arc::new(Vec::new()),
            return_type_kind,
        };

        let concrete_type_kind = get_function_type_kind(&mut self.types, &mut self.function_type_kinds, concrete_function);

        Some(Type {
            type_kind: concrete_type_kind,
            instance_kind: InstanceKind::Literal,
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

    fn int_literal(&mut self, _text: Arc<str>) -> Option<Type> {
        Some(Type {
            type_kind: INT_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn float32_literal(&mut self, _text: Arc<str>) -> Option<Type> {
        Some(Type {
            type_kind: FLOAT32_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn string_literal(&mut self, _text: Arc<str>) -> Option<Type> {
        Some(Type {
            type_kind: STRING_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn bool_literal(&mut self, _value: bool) -> Option<Type> {
        Some(Type {
            type_kind: BOOL_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn char_literal(&mut self, _value: char) -> Option<Type> {
        Some(Type {
            type_kind: CHAR_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn array_literal(&mut self, elements: Arc<Vec<usize>>, repeat_count: usize) -> Option<Type> {
        for element in elements.iter() {
            self.check_node(*element);
        }

        let node_type = self.check_node(*elements.first()?)?;
        let type_kind = get_type_kind_as_array(
            &mut self.types,
            &mut self.array_type_kinds,
            node_type.type_kind,
            elements.len() * repeat_count,
        );

        Some(Type {
            type_kind,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn struct_literal(&mut self, name: usize, fields: Arc<Vec<usize>>) -> Option<Type> {
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

        let Some(name_type) = self.environment.get(name_text) else {
            type_error!(
                self,
                &format!("struct with name \"{}\" does not exist", name)
            );
        };

        if !matches!(self.types[name_type.type_kind], TypeKind::Struct { .. })
            || name_type.instance_kind != InstanceKind::Name
        {
            type_error!(
                self,
                &format!("expected the name of a struct type, but got \"{}\"", name)
            );
        }

        Some(Type {
            type_kind: name_type.type_kind,
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
            type_kind: INT_INDEX,
            instance_kind: InstanceKind::Literal,
        })
    }

    fn type_name(&mut self, type_kind: usize) -> Option<Type> {
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
