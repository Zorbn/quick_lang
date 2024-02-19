use std::{collections::HashMap, ffi::OsString, hash::Hash, mem, sync::Arc};

use crate::{
    const_value::ConstValue,
    environment::Environment,
    file_data::FileData,
    parser::{DeclarationKind, Node, NodeIndex, NodeKind, Op},
    type_kinds::{get_field_index_by_name, Field, TypeKind, TypeKinds},
};

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

macro_rules! assert_typed {
    ($self:ident, $index:expr) => {{
        let Some(node_type) = $self.get_typer_node($index).node_type.clone() else {
            return $self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
            });
        };

        node_type
    }};
}

#[derive(Clone, Hash, Debug, PartialEq, Eq)]
pub struct GenericIdentifier {
    pub name: Arc<str>,
    pub extending_type_kind_id: Option<usize>,
    pub generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
}

impl GenericIdentifier {
    pub fn new(name: Arc<str>) -> Self {
        Self {
            name,
            extending_type_kind_id: None,
            generic_arg_type_kind_ids: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FunctionDefinition {
    type_kind_id: usize,
    file_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DefinitionLookupResult {
    Error,
    NotFound,
    Found(NodeIndex),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LookupKind {
    All,
    Functions,
    Types,
}

pub struct Typer {
    all_nodes: Arc<Vec<Vec<Node>>>,
    all_definition_indices: Arc<Vec<HashMap<Arc<str>, NodeIndex>>>,

    pub typed_nodes: Vec<TypedNode>,
    pub typed_definition_indices: Vec<NodeIndex>,
    pub type_kinds: TypeKinds,
    pub main_function_declaration: Option<NodeIndex>,
    pub error_count: usize,

    type_kind_environment: Environment<GenericIdentifier, usize>,
    function_definitions: HashMap<GenericIdentifier, FunctionDefinition>,
    files: Arc<Vec<FileData>>,
    file_paths_components: Arc<Vec<Vec<OsString>>>,
    environment: Environment<Arc<str>, Type>,
    was_block_already_opened: bool,
    node_index_stack: Vec<NodeIndex>,
    used_file_indices: Vec<usize>,
    loop_stack: usize,
    // We want to do as little work as possible when type checking types/functions that haven't been checked yet but
    // are needed as dependencies of other types/functions, since any non-generic type/function will eventually be fully checked.
    // So, we can increment the shallow check stack to request that type checks don't include function bodies when we don't need them.
    shallow_check_stack: usize,
}

impl Typer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        all_nodes: Arc<Vec<Vec<Node>>>,
        all_definition_indices: Arc<Vec<HashMap<Arc<str>, NodeIndex>>>,
        files: Arc<Vec<FileData>>,
        file_paths_components: Arc<Vec<Vec<OsString>>>,
        file_index: usize,
    ) -> Self {
        let mut type_checker = Self {
            all_nodes,
            all_definition_indices,
            files,
            file_paths_components,
            typed_nodes: Vec::new(),
            typed_definition_indices: Vec::new(),
            type_kinds: TypeKinds::new(),
            main_function_declaration: None,
            error_count: 0,
            environment: Environment::new(),
            type_kind_environment: Environment::new(),
            function_definitions: HashMap::new(),
            was_block_already_opened: false,
            node_index_stack: Vec::new(),
            used_file_indices: Vec::new(),
            loop_stack: 0,
            shallow_check_stack: 0,
        };

        type_checker.used_file_indices.push(file_index);

        // All of these primitives need to be defined by default.
        let int_id = type_checker.type_kinds.add_or_get(TypeKind::Int);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Int".into()),
            int_id,
            true,
        );
        let string_id = type_checker.type_kinds.add_or_get(TypeKind::String);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("String".into()),
            string_id,
            true,
        );
        let bool_id = type_checker.type_kinds.add_or_get(TypeKind::Bool);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Bool".into()),
            bool_id,
            true,
        );
        let char_id = type_checker.type_kinds.add_or_get(TypeKind::Char);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Char".into()),
            char_id,
            true,
        );
        let void_id = type_checker.type_kinds.add_or_get(TypeKind::Void);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Void".into()),
            void_id,
            true,
        );
        let uint_id = type_checker.type_kinds.add_or_get(TypeKind::UInt);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("UInt".into()),
            uint_id,
            true,
        );
        let int8_id = type_checker.type_kinds.add_or_get(TypeKind::Int8);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Int8".into()),
            int8_id,
            true,
        );
        let uint8_id = type_checker.type_kinds.add_or_get(TypeKind::UInt8);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("UInt8".into()),
            uint8_id,
            true,
        );
        let int16_id = type_checker.type_kinds.add_or_get(TypeKind::Int16);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Int16".into()),
            int16_id,
            true,
        );
        let uint16_id = type_checker.type_kinds.add_or_get(TypeKind::UInt16);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("UInt16".into()),
            uint16_id,
            true,
        );
        let int32_id = type_checker.type_kinds.add_or_get(TypeKind::Int32);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Int32".into()),
            int32_id,
            true,
        );
        let uint32_id = type_checker.type_kinds.add_or_get(TypeKind::UInt32);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("UInt32".into()),
            uint32_id,
            true,
        );
        let int64_id = type_checker.type_kinds.add_or_get(TypeKind::Int64);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Int64".into()),
            int64_id,
            true,
        );
        let uint64_id = type_checker.type_kinds.add_or_get(TypeKind::UInt64);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("UInt64".into()),
            uint64_id,
            true,
        );
        let float32_id = type_checker.type_kinds.add_or_get(TypeKind::Float32);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Float32".into()),
            float32_id,
            true,
        );
        let float64_id = type_checker.type_kinds.add_or_get(TypeKind::Float64);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Float64".into()),
            float64_id,
            true,
        );
        let tag_id = type_checker.type_kinds.add_or_get(TypeKind::Tag);
        type_checker.type_kind_environment.insert(
            GenericIdentifier::new("Tag".into()),
            tag_id,
            true,
        );

        type_checker
    }

    fn add_node(&mut self, typed_node: TypedNode) -> NodeIndex {
        let node_index = self.typed_nodes.len();
        self.typed_nodes.push(typed_node);
        let file_index = self.node_index_stack.last().copied().unwrap().file_index;

        NodeIndex {
            node_index,
            file_index,
        }
    }

    fn add_node_with_file_index(
        &mut self,
        typed_node: TypedNode,
        file_index: Option<usize>,
    ) -> NodeIndex {
        let node_index = self.typed_nodes.len();
        self.typed_nodes.push(typed_node);

        NodeIndex {
            node_index,
            file_index,
        }
    }

    fn get_parser_node(&self, index: NodeIndex) -> &Node {
        &self.all_nodes[index.file_index.unwrap()][index.node_index]
    }

    fn get_typer_node(&self, index: NodeIndex) -> &TypedNode {
        &self.typed_nodes[index.node_index]
    }

    fn lookup_definition_index(&mut self, name_text: &Arc<str>) -> DefinitionLookupResult {
        let mut found_index = None;

        for i in 0..self.used_file_indices.len() {
            let imported_file_index = self.used_file_indices[i];
            let definition_index = self.all_definition_indices[imported_file_index]
                .get(name_text)
                .copied();

            if definition_index.is_none() {
                continue;
            }

            if found_index.is_some() {
                self.error("ambiguous identifier, multiple possible definitions exist");
                return DefinitionLookupResult::Error;
            }

            found_index = definition_index;
        }

        if let Some(found_index) = found_index {
            DefinitionLookupResult::Found(found_index)
        } else {
            DefinitionLookupResult::NotFound
        }
    }

    fn lookup_identifier(
        &mut self,
        name: NodeIndex,
        extending_type_kind_id: Option<usize>,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
        kind: LookupKind,
    ) -> Option<(NodeIndex, Type)> {
        let mut typed_name = self.check_node(name);

        let NodeKind::Name { text: name_text } = self.get_parser_node(name).kind.clone() else {
            self.error("invalid identifier name");
            return None;
        };

        // Look up the definition first, to make sure there is exactly one definition available (aka: it isn't ambiguous).
        // We'll use it if the type information isn't already available in one of the caches.
        let definition_index = self.lookup_definition_index(&name_text);
        if definition_index == DefinitionLookupResult::Error {
            return None;
        }

        if let Some(identifier_type) = self.environment.get(&name_text) {
            return Some((typed_name, identifier_type));
        };

        let identifier = GenericIdentifier {
            name: name_text.clone(),
            extending_type_kind_id,
            generic_arg_type_kind_ids: generic_arg_type_kind_ids.clone(),
        };

        if kind != LookupKind::Types {
            if let Some(function_definition) = self.function_definitions.get(&identifier) {
                typed_name.file_index = function_definition.file_index;
                return Some((
                    typed_name,
                    Type {
                        type_kind_id: function_definition.type_kind_id,
                        instance_kind: InstanceKind::Val,
                    },
                ));
            }
        }

        if let Some(type_kind_id) = self.type_kind_environment.get(&identifier) {
            return Some((
                typed_name,
                Type {
                    type_kind_id,
                    instance_kind: InstanceKind::Name,
                },
            ));
        }

        let DefinitionLookupResult::Found(definition_index) = definition_index else {
            self.error("undefined identifier");
            return None;
        };

        let is_function = matches!(self.get_parser_node(definition_index).kind, NodeKind::Function { .. });

        if kind == LookupKind::Types && is_function {
            self.error("expected identifier to refer to a type");
            return None;
        }

        if kind == LookupKind::Functions && !is_function {
            self.error("expected identifier to refer to a function");
            return None;
        }

        self.shallow_check_stack += 1;
        let typed_definition =
            self.check_node_with_generic_args(definition_index, generic_arg_type_kind_ids);
        self.shallow_check_stack -= 1;

        let definition_type = self.get_typer_node(typed_definition).node_type.clone()?;

        typed_name.file_index = typed_definition.file_index;

        Some((typed_name, definition_type))
    }

    fn error(&mut self, message: &str) {
        self.error_count += 1;
        self.get_parser_node(self.node_index_stack.last().copied().unwrap())
            .start
            .error("Type", message, &self.files);
    }

    fn type_error(&mut self, message: &str) -> NodeIndex {
        self.error(message);

        self.add_node(TypedNode {
            node_kind: NodeKind::Error,
            node_type: None,
        })
    }

    pub fn check(&mut self, start_index: NodeIndex) {
        self.check_node(start_index);
    }

    fn check_optional_node(&mut self, index: Option<NodeIndex>, hint: Option<usize>) -> Option<NodeIndex> {
        index.map(|index| self.check_node_with_hint(index, hint))
    }

    fn check_node(&mut self, index: NodeIndex) -> NodeIndex {
        self.node_index_stack.push(index);

        let typed_index = match self.get_parser_node(index).kind.clone() {
            NodeKind::TopLevel {
                functions,
                structs,
                enums,
                usings,
                aliases,
            } => self.top_level(functions, structs, enums, usings, aliases),
            NodeKind::ExternFunction { declaration } => self.extern_function(declaration),
            NodeKind::Using {
                path_component_names,
            } => self.using(path_component_names),
            NodeKind::Alias {
                aliased_type_name,
                alias_name,
            } => self.alias(aliased_type_name, alias_name),
            NodeKind::Param { name, type_name } => self.param(name, type_name),
            NodeKind::Block { statements } => self.block(statements, None),
            NodeKind::Statement { inner } => self.statement(inner, None),
            NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                type_name,
                expression,
            } => self.variable_declaration(declaration_kind, name, type_name, expression),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression, None),
            NodeKind::BreakStatement => self.break_statement(),
            NodeKind::ContinueStatement => self.continue_statement(),
            NodeKind::DeferStatement { statement } => self.defer_statement(statement),
            NodeKind::IfStatement {
                expression,
                scoped_statement,
                next,
            } => self.if_statement(expression, scoped_statement, next),
            NodeKind::SwitchStatement {
                expression,
                case_statement,
            } => self.switch_statement(expression, case_statement),
            NodeKind::CaseStatement {
                expression,
                scoped_statement,
                next,
            } => self.case_statement(expression, scoped_statement, next),
            NodeKind::WhileLoop {
                expression,
                scoped_statement,
            } => self.while_loop(expression, scoped_statement),
            NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                scoped_statement,
            } => self.for_loop(iterator, op, from, to, by, scoped_statement),
            NodeKind::ConstExpression { inner } => self.const_expression(inner, None),
            NodeKind::Binary { left, op, right } => self.binary(left, op, right, None),
            NodeKind::UnaryPrefix { op, right } => self.unary_prefix(op, right, None),
            NodeKind::UnarySuffix { left, op } => self.unary_suffix(left, op),
            NodeKind::Call { left, args } => self.call(left, args),
            NodeKind::IndexAccess { left, expression } => self.index_access(left, expression),
            NodeKind::FieldAccess { left, name } => self.field_access(left, name),
            NodeKind::Cast { left, type_name } => self.cast(left, type_name),
            NodeKind::Name { text } => self.name(text),
            NodeKind::Identifier { name } => self.identifier(name),
            NodeKind::IntLiteral { text } => self.int_literal(text, None),
            NodeKind::FloatLiteral { text } => self.float_literal(text, None),
            NodeKind::StringLiteral { text } => self.string_literal(text),
            NodeKind::BoolLiteral { value } => self.bool_literal(value),
            NodeKind::CharLiteral { value } => self.char_literal(value),
            NodeKind::ArrayLiteral {
                elements,
                repeat_count_const_expression,
            } => self.array_literal(elements, repeat_count_const_expression),
            NodeKind::StructLiteral {
                left,
                field_literals,
            } => self.struct_literal(left, field_literals),
            NodeKind::FieldLiteral { name, expression } => self.field_literal(name, expression),
            NodeKind::TypeSize { type_name } => self.type_size(type_name, None),
            NodeKind::StructDefinition {
                name,
                fields,
                generic_params,
                is_union,
            } => self.struct_definition(name, fields, generic_params, is_union, None),
            NodeKind::EnumDefinition {
                name,
                variant_names,
            } => self.enum_definition(name, variant_names),
            NodeKind::Field { name, type_name } => self.field(name, type_name),
            NodeKind::FunctionDeclaration {
                name,
                params,
                extending_type_name,
                generic_params,
                return_type_name,
            } => self.function_declaration(name, params, extending_type_name, generic_params, return_type_name, false),
            NodeKind::Function {
                declaration,
                scoped_statement,
            } => self.function(declaration, scoped_statement, None),
            NodeKind::GenericSpecifier {
                name,
                generic_arg_type_names,
            } => self.generic_specifier(name, generic_arg_type_names),
            NodeKind::TypeName { name } => self.type_name(name),
            NodeKind::TypeNamePointer {
                inner,
                is_inner_mutable,
            } => self.type_name_pointer(inner, is_inner_mutable),
            NodeKind::TypeNameArray {
                inner,
                element_count_const_expression,
            } => self.type_name_array(inner, element_count_const_expression),
            NodeKind::TypeNameFunction {
                param_type_names,
                return_type_name,
            } => self.type_name_function(param_type_names, return_type_name),
            NodeKind::TypeNameGenericSpecifier {
                name,
                generic_arg_type_names,
            } => self.type_name_generic_specifier(name, generic_arg_type_names),
            NodeKind::Error => type_error!(self, "cannot generate error node"),
        };

        self.node_index_stack.pop();

        typed_index
    }

    fn check_node_with_generic_args(
        &mut self,
        index: NodeIndex,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
    ) -> NodeIndex {
        self.node_index_stack.push(index);

        let typed_index = match self.get_parser_node(index).kind.clone() {
            NodeKind::StructDefinition {
                name,
                fields,
                generic_params,
                is_union,
            } => self.struct_definition(
                name,
                fields,
                generic_params,
                is_union,
                generic_arg_type_kind_ids,
            ),
            NodeKind::Function {
                declaration,
                scoped_statement,
            } => self.function(declaration, scoped_statement, generic_arg_type_kind_ids),
            _ => self.check_node(index),
        };

        self.node_index_stack.pop();

        typed_index
    }

    fn check_node_with_hint(
        &mut self,
        index: NodeIndex,
        hint: Option<usize>,
    ) -> NodeIndex {
        self.node_index_stack.push(index);

        let typed_index = match self.get_parser_node(index).kind.clone() {
            NodeKind::Binary { left, op, right } => self.binary(left, op, right, hint),
            NodeKind::UnaryPrefix { op, right } => self.unary_prefix(op, right, hint),
            NodeKind::IntLiteral { text } => self.int_literal(text, hint),
            NodeKind::FloatLiteral { text } => self.float_literal(text, hint),
            NodeKind::TypeSize { type_name } => self.type_size(type_name, hint),
            NodeKind::Block { statements } => self.block(statements, hint),
            NodeKind::Statement { inner } => self.statement(inner, hint),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression, hint),
            NodeKind::ConstExpression { inner } => self.const_expression(inner, hint),
            _ => self.check_node(index),
        };

        self.node_index_stack.pop();

        typed_index
    }

    fn check_const_node(&mut self, index: NodeIndex, hint: Option<usize>) -> NodeIndex {
        self.node_index_stack.push(index);

        let typed_index = match self.get_parser_node(index).kind.clone() {
            NodeKind::Binary { left, op, right } => self.const_binary(left, op, right, index, hint),
            NodeKind::UnaryPrefix { op, right } => self.const_unary_prefix(op, right, index, hint),
            NodeKind::Cast { left, type_name } => self.const_cast(left, type_name, index),
            NodeKind::Identifier { name } => self.const_identifier(name, index),
            NodeKind::IntLiteral { text } => self.const_int_literal(text, index, hint),
            NodeKind::FloatLiteral { text } => self.const_float_literal(text, index, hint),
            NodeKind::StringLiteral { text } => self.const_string_literal(text, index),
            NodeKind::BoolLiteral { value } => self.const_bool_literal(value, index),
            NodeKind::CharLiteral { value } => self.const_char_literal(value, index),
            NodeKind::TypeSize { type_name } => self.const_type_size(type_name, index, hint),
            _ => self.type_error("non-constant in constant expression"),
        };

        self.node_index_stack.pop();

        typed_index
    }

    fn top_level(
        &mut self,
        functions: Arc<Vec<NodeIndex>>,
        structs: Arc<Vec<NodeIndex>>,
        enums: Arc<Vec<NodeIndex>>,
        usings: Arc<Vec<NodeIndex>>,
        aliases: Arc<Vec<NodeIndex>>,
    ) -> NodeIndex {
        let mut typed_usings = Vec::new();
        for using in usings.iter() {
            let typed_using = self.check_node(*using);
            typed_usings.push(typed_using);
        }

        let mut typed_aliases = Vec::new();
        for alias in aliases.iter() {
            let typed_alias = self.check_node(*alias);
            typed_aliases.push(typed_alias);
        }

        let mut typed_structs = Vec::new();
        for struct_definition in structs.iter() {
            let NodeKind::StructDefinition {
                name,
                generic_params,
                ..
            } = &self.get_parser_node(*struct_definition).kind
            else {
                panic!("invalid struct definition");
            };

            if !generic_params.is_empty() {
                continue;
            }

            let NodeKind::Name { text: name_text } = self.get_parser_node(*name).kind.clone()
            else {
                panic!("invalid struct name");
            };

            let identifier = GenericIdentifier {
                name: name_text,
                extending_type_kind_id: None,
                generic_arg_type_kind_ids: None,
            };

            if self.type_kind_environment.get(&identifier).is_some() {
                continue;
            }

            typed_structs.push(self.check_node(*struct_definition));
        }

        let mut typed_enums = Vec::new();
        for enum_definition in enums.iter() {
            let NodeKind::EnumDefinition { name, .. } =
                &self.get_parser_node(*enum_definition).kind
            else {
                panic!("invalid enum definition");
            };

            let NodeKind::Name { text: name_text } = self.get_parser_node(*name).kind.clone()
            else {
                panic!("invalid enum name");
            };

            let identifier = GenericIdentifier {
                name: name_text,
                extending_type_kind_id: None,
                generic_arg_type_kind_ids: None,
            };

            if self.type_kind_environment.get(&identifier).is_some() {
                continue;
            }

            typed_enums.push(self.check_node(*enum_definition));
        }

        let mut typed_functions = Vec::new();
        for function in functions.iter() {
            let declaration = if let NodeKind::Function { declaration, .. } =
                &self.get_parser_node(*function).kind
            {
                declaration
            } else if let NodeKind::ExternFunction { declaration } =
                &self.get_parser_node(*function).kind
            {
                declaration
            } else {
                panic!("invalid function");
            };

            let NodeKind::FunctionDeclaration { generic_params, .. } =
                &self.get_parser_node(*declaration).kind
            else {
                panic!("invalid function declaration");
            };

            if !generic_params.is_empty() {
                continue;
            }

            typed_functions.push(self.check_node(*function));
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::TopLevel {
                functions: Arc::new(typed_functions),
                structs: Arc::new(typed_structs),
                enums: Arc::new(typed_enums),
                usings: Arc::new(typed_usings),
                aliases: Arc::new(typed_aliases),
            },
            node_type: None,
        })
    }

    fn extern_function(&mut self, declaration: NodeIndex) -> NodeIndex {
        let NodeKind::FunctionDeclaration {
            name,
            params,
            extending_type_name,
            generic_params,
            return_type_name,
            ..
        } = self.get_parser_node(declaration).kind.clone()
        else {
            type_error!(self, "invalid function declaration");
        };

        let NodeKind::Name { text: name_text } = self.get_parser_node(name).kind.clone() else {
            type_error!(self, "invalid function name");
        };

        if !generic_params.is_empty() {
            type_error!(self, "extern function cannot be generic");
        }

        if extending_type_name.is_some() {
            type_error!(self, "extern function cannot be a method");
        }

        self.environment.push(false);

        let typed_declaration =
            self.function_declaration(name, params, extending_type_name, generic_params, return_type_name, true);

        self.environment.pop();

        let type_kind_id = assert_typed!(self, typed_declaration).type_kind_id;
        let identifier = GenericIdentifier {
            name: name_text.clone(),
            extending_type_kind_id: None,
            generic_arg_type_kind_ids: None,
        };
        self.function_definitions.insert(
            identifier,
            FunctionDefinition {
                type_kind_id,
                file_index: None,
            },
        );

        let node_type = Some(Type {
            type_kind_id,
            instance_kind: InstanceKind::Val,
        });

        let index = self.add_node_with_file_index(
            TypedNode {
                node_kind: NodeKind::ExternFunction {
                    declaration: typed_declaration,
                },
                node_type,
            },
            typed_declaration.file_index,
        );

        self.typed_definition_indices.push(index);

        index
    }

    fn using(&mut self, path_component_names: Arc<Vec<NodeIndex>>) -> NodeIndex {
        let mut did_find_file = false;
        for (i, file_path_components) in self.file_paths_components.iter().enumerate() {
            if file_path_components.len() != path_component_names.len() {
                continue;
            }

            let mut is_matching = true;
            for component_i in 0..file_path_components.len() {
                let NodeKind::Name { text: name_text } =
                    &self.get_parser_node(path_component_names[component_i]).kind
                else {
                    panic!("invalid component name");
                };

                if file_path_components[component_i] != name_text.as_ref() {
                    is_matching = false;
                    break;
                }
            }

            if !is_matching {
                continue;
            }

            self.used_file_indices.push(i);
            did_find_file = true;
            break;
        }

        if !did_find_file {
            type_error!(self, "couldn't file file referenced by using statement");
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::Using {
                path_component_names: Arc::new(Vec::new()),
            },
            node_type: None,
        })
    }

    fn alias(&mut self, aliased_type_name: NodeIndex, alias_name: NodeIndex) -> NodeIndex {
        let typed_aliased_type_name = self.check_node(aliased_type_name);
        let aliased_type_name_type = assert_typed!(self, typed_aliased_type_name);
        let typed_alias_name = self.check_node(alias_name);

        let NodeKind::Name {
            text: alias_name_text,
        } = self.get_typer_node(typed_alias_name).node_kind.clone()
        else {
            type_error!(self, "invalid enum name");
        };

        let identifier = GenericIdentifier {
            name: alias_name_text,
            extending_type_kind_id: None,
            generic_arg_type_kind_ids: None,
        };

        self.type_kind_environment
            .insert(identifier, aliased_type_name_type.type_kind_id, true);

        self.add_node(TypedNode {
            node_kind: NodeKind::Alias {
                aliased_type_name: typed_aliased_type_name,
                alias_name: typed_alias_name,
            },
            node_type: Some(aliased_type_name_type),
        })
    }

    fn param(&mut self, name: NodeIndex, type_name: NodeIndex) -> NodeIndex {
        let typed_name = self.check_node(name);
        let typed_type_name = self.check_node(type_name);
        let type_name_type = assert_typed!(self, typed_type_name);

        if let TypeKind::Void = self.type_kinds.get_by_id(type_name_type.type_kind_id) {
            type_error!(self, "parameter cannot be of type Void");
        }

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(type_name_type.type_kind_id) {
            type_error!(self, "field cannot be of type func");
        }

        let NodeKind::Name { text: name_text } = self.get_parser_node(name).kind.clone() else {
            type_error!(self, "invalid parameter name");
        };

        let node_type = Type {
            type_kind_id: type_name_type.type_kind_id,
            instance_kind: InstanceKind::Var,
        };
        self.environment.insert(name_text, node_type.clone(), false);

        self.add_node(TypedNode {
            node_kind: NodeKind::Param {
                name: typed_name,
                type_name: typed_type_name,
            },
            node_type: Some(node_type),
        })
    }

    fn block(&mut self, statements: Arc<Vec<NodeIndex>>, hint: Option<usize>) -> NodeIndex {
        if !self.was_block_already_opened {
            self.environment.push(true);
        } else {
            self.was_block_already_opened = false;
        }

        let mut typed_statements = Vec::new();
        for statement in statements.iter() {
            typed_statements.push(self.check_node_with_hint(*statement, hint));
        }

        self.environment.pop();

        self.add_node(TypedNode {
            node_kind: NodeKind::Block {
                statements: Arc::new(typed_statements),
            },
            node_type: None,
        })
    }

    fn statement(&mut self, inner: Option<NodeIndex>, hint: Option<usize>) -> NodeIndex {
        let typed_inner = self.check_optional_node(inner, hint);

        self.add_node(TypedNode {
            node_kind: NodeKind::Statement { inner: typed_inner },
            node_type: None,
        })
    }

    fn variable_declaration(
        &mut self,
        declaration_kind: DeclarationKind,
        name: NodeIndex,
        type_name: Option<NodeIndex>,
        expression: Option<NodeIndex>,
    ) -> NodeIndex {
        let typed_name = self.check_node(name);
        let typed_type_name = self.check_optional_node(type_name, None);

        let typed_expression = if let Some(typed_type_name) = typed_type_name {
            let type_name_type = assert_typed!(self, typed_type_name);
            self.check_optional_node(expression, Some(type_name_type.type_kind_id))
        } else {
            self.check_optional_node(expression, None)
        };

        if let Some(typed_expression) = typed_expression {
            let expression_type = assert_typed!(self, typed_expression);

            if expression_type.instance_kind == InstanceKind::Name {
                type_error!(self, "only instances of types can be stored in variables");
            }
        }

        let mut variable_type = if let Some(typed_type_name) = typed_type_name {
            let variable_type = assert_typed!(self, typed_type_name);

            if typed_expression.is_some() || declaration_kind == DeclarationKind::Const {
                let expression_type = assert_typed!(self, typed_expression.unwrap());

                if variable_type.type_kind_id != expression_type.type_kind_id {
                    type_error!(self, "mismatched types in variable declaration");
                }

                expression_type
            } else {
                variable_type
            }
        } else {
            assert_typed!(self, typed_expression.unwrap())
        };

        if declaration_kind == DeclarationKind::Const
            && !matches!(variable_type.instance_kind, InstanceKind::Const(..))
        {
            type_error!(self, "cannot declare a const with a non-const value");
        }

        match self.type_kinds.get_by_id(variable_type.type_kind_id) {
            TypeKind::Function { .. } => type_error!(
                self,
                "variables can't have a function type, try a function pointer instead"
            ),
            TypeKind::Void => type_error!(self, "variables cannot be of type Void"),
            _ => {}
        }

        let NodeKind::Name { text: name_text } = self.get_parser_node(name).kind.clone() else {
            type_error!(self, "invalid variable name");
        };

        variable_type.instance_kind = match declaration_kind {
            DeclarationKind::Var => InstanceKind::Var,
            DeclarationKind::Val => InstanceKind::Val,
            DeclarationKind::Const => variable_type.instance_kind,
        };

        self.environment
            .insert(name_text, variable_type.clone(), false);

        self.add_node(TypedNode {
            node_kind: NodeKind::VariableDeclaration {
                declaration_kind,
                name: typed_name,
                type_name: typed_type_name,
                expression: typed_expression,
            },
            node_type: Some(variable_type),
        })
    }

    fn return_statement(&mut self, expression: Option<NodeIndex>, hint: Option<usize>) -> NodeIndex {
        let typed_expression = self.check_optional_node(expression, hint);

        if let Some(typed_expression) = typed_expression {
            let expression_type = assert_typed!(self, typed_expression);

            if expression_type.instance_kind == InstanceKind::Name {
                type_error!(self, "cannot return type name");
            }
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::ReturnStatement {
                expression: typed_expression,
            },
            node_type: None,
        })
    }

    fn break_statement(&mut self) -> NodeIndex {
        if self.loop_stack == 0 {
            type_error!(self, "break statements can only appear in loops");
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::BreakStatement,
            node_type: None,
        })
    }

    fn continue_statement(&mut self) -> NodeIndex {
        if self.loop_stack == 0 {
            type_error!(self, "continue statements can only appear in loops");
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::ContinueStatement,
            node_type: None,
        })
    }

    fn defer_statement(&mut self, statement: NodeIndex) -> NodeIndex {
        let typed_statement = self.check_node(statement);

        self.add_node(TypedNode {
            node_kind: NodeKind::DeferStatement {
                statement: typed_statement,
            },
            node_type: None,
        })
    }

    fn if_statement(
        &mut self,
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        next: Option<NodeIndex>,
    ) -> NodeIndex {
        let typed_expression = self.check_node(expression);
        let typed_scoped_statement = self.check_node(scoped_statement);
        let typed_next = self.check_optional_node(next, None);

        self.add_node(TypedNode {
            node_kind: NodeKind::IfStatement {
                expression: typed_expression,
                scoped_statement: typed_scoped_statement,
                next: typed_next,
            },
            node_type: None,
        })
    }

    fn switch_statement(&mut self, expression: NodeIndex, case_statement: NodeIndex) -> NodeIndex {
        let typed_expression = self.check_node(expression);
        let typed_case_statement = self.check_node(case_statement);

        self.add_node(TypedNode {
            node_kind: NodeKind::SwitchStatement {
                expression: typed_expression,
                case_statement: typed_case_statement,
            },
            node_type: None,
        })
    }

    fn case_statement(
        &mut self,
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        next: Option<NodeIndex>,
    ) -> NodeIndex {
        let typed_expression = self.check_node(expression);
        let typed_scoped_statement = self.check_node(scoped_statement);
        let typed_next = self.check_optional_node(next, None);

        self.add_node(TypedNode {
            node_kind: NodeKind::CaseStatement {
                expression: typed_expression,
                scoped_statement: typed_scoped_statement,
                next: typed_next,
            },
            node_type: None,
        })
    }

    fn while_loop(&mut self, expression: NodeIndex, scoped_statement: NodeIndex) -> NodeIndex {
        let typed_expression = self.check_node(expression);

        self.loop_stack += 1;
        let typed_scoped_statement = self.check_node(scoped_statement);
        self.loop_stack -= 1;

        self.add_node(TypedNode {
            node_kind: NodeKind::WhileLoop {
                expression: typed_expression,
                scoped_statement: typed_scoped_statement,
            },
            node_type: None,
        })
    }

    fn for_loop(
        &mut self,
        iterator: NodeIndex,
        op: Op,
        from: NodeIndex,
        to: NodeIndex,
        by: Option<NodeIndex>,
        scoped_statement: NodeIndex,
    ) -> NodeIndex {
        let typed_iterator = self.check_node(iterator);

        let (typed_from, typed_to, typed_by) = if self.is_node_numeric_literal(from) && self.is_node_numeric_literal(to) && by.is_some() {
            // "from" and "to" are both literals, but we also have a "by" node, so start with that one.
            let typed_by = self.check_node(by.unwrap());
            let by_type = assert_typed!(self, typed_by);

            let typed_from = self.check_node_with_hint(from, Some(by_type.type_kind_id));
            let typed_to = self.check_node_with_hint(to, Some(by_type.type_kind_id));

            (typed_from, typed_to, Some(typed_by))
        } else if self.is_node_numeric_literal(from) {
            // "from" is a literal, so we will hint "to" in an attempt to get a hint to pass to "from"
            let typed_to = self.check_node(to);
            let to_type = assert_typed!(self, typed_to);

            let typed_from = self.check_node_with_hint(from, Some(to_type.type_kind_id));
            let typed_by = self.check_optional_node(by, Some(to_type.type_kind_id));

            (typed_from, typed_to, typed_by)
        } else {
            // Check "from" first, and use it's type as a hint for the others.
            let typed_from = self.check_node(from);
            let from_type = assert_typed!(self, typed_from);

            let typed_to = self.check_node_with_hint(to, Some(from_type.type_kind_id));
            let typed_by = self.check_optional_node(by, Some(from_type.type_kind_id));

            (typed_from, typed_to, typed_by)
        };

        let from_type = assert_typed!(self, typed_from);
        let to_type = assert_typed!(self, typed_to);

        if from_type.type_kind_id != to_type.type_kind_id {
            type_error!(self, "type mismatch between for loop bounds");
        }

        if let Some(typed_by) = typed_by {
            let by_type = assert_typed!(self, typed_by);

            if by_type.type_kind_id != from_type.type_kind_id {
                type_error!(self, "type mismatch between for loop increment and bounds");
            }
        }

        if !self
            .type_kinds
            .get_by_id(from_type.type_kind_id)
            .is_numeric()
        {
            type_error!(self, "for loop bounds must have numeric types");
        }

        self.environment.push(true);
        self.was_block_already_opened = true;

        let NodeKind::Name {
            text: iterator_text,
        } = self.get_typer_node(typed_iterator).node_kind.clone()
        else {
            type_error!(self, "invalid iterator name");
        };

        let node_type = Type {
            type_kind_id: from_type.type_kind_id,
            instance_kind: InstanceKind::Var,
        };
        self.environment
            .insert(iterator_text, node_type.clone(), false);

        self.loop_stack += 1;
        let typed_scoped_statement = self.check_node(scoped_statement);
        self.loop_stack -= 1;

        self.add_node(TypedNode {
            node_kind: NodeKind::ForLoop {
                iterator: typed_iterator,
                op,
                from: typed_from,
                to: typed_to,
                by: typed_by,
                scoped_statement: typed_scoped_statement,
            },
            node_type: None,
        })
    }

    fn const_expression(&mut self, inner: NodeIndex, hint: Option<usize>) -> NodeIndex {
        self.check_const_node(inner, hint)
    }

    fn binary(&mut self, left: NodeIndex, op: Op, right: NodeIndex, hint: Option<usize>) -> NodeIndex {
        // Maximize the accuracy of hinting by type checking the right side first if the left side is a literal, so that we might have a more useful hint to give the left side.
        let (typed_left, left_type, typed_right, right_type) = if self.is_node_numeric_literal(left) {
            let typed_right = self.check_node_with_hint(right, hint);
            let right_type = assert_typed!(self, typed_right);
            let typed_left = self.check_node_with_hint(left, Some(right_type.type_kind_id));
            let left_type = assert_typed!(self, typed_left);

            (typed_left, left_type, typed_right, right_type)
        } else {
            let typed_left = self.check_node_with_hint(left, hint);
            let left_type = assert_typed!(self, typed_left);
            let typed_right = self.check_node_with_hint(right, Some(left_type.type_kind_id));
            let right_type = assert_typed!(self, typed_right);

            (typed_left, left_type, typed_right, right_type)
        };

        let node_kind = NodeKind::Binary {
            left: typed_left,
            op,
            right: typed_right,
        };

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
            Op::Assign
                | Op::PlusAssign
                | Op::MinusAssign
                | Op::MultiplyAssign
                | Op::DivideAssign
                | Op::BitwiseAndAssign
                | Op::BitwiseOrAssign
                | Op::XorAssign
                | Op::LeftShiftAssign
                | Op::RightShiftAssign
                | Op::ModuloAssign
        ) && left_type.instance_kind != InstanceKind::Var
        {
            type_error!(self, "only vars can be assigned to");
        }

        match op {
            Op::Plus
            | Op::Minus
            | Op::Multiply
            | Op::Divide
            | Op::Modulo
            | Op::PlusAssign
            | Op::MinusAssign
            | Op::MultiplyAssign
            | Op::DivideAssign
            | Op::LeftShiftAssign
            | Op::RightShiftAssign
            | Op::ModuloAssign
            | Op::BitwiseAndAssign
            | Op::BitwiseOrAssign
            | Op::XorAssign
            | Op::LeftShift
            | Op::RightShift
            | Op::BitwiseAnd
            | Op::BitwiseOr
            | Op::BitwiseNot
            | Op::Xor => {
                if !self
                    .type_kinds
                    .get_by_id(left_type.type_kind_id)
                    .is_numeric()
                {
                    type_error!(self, "expected arithmetic types");
                }
            }
            Op::Less | Op::Greater | Op::LessEqual | Op::GreaterEqual => {
                if !self
                    .type_kinds
                    .get_by_id(left_type.type_kind_id)
                    .is_numeric()
                {
                    type_error!(self, "expected comparable types");
                }

                let type_kind_id = self.type_kinds.add_or_get(TypeKind::Bool);
                return self.add_node(TypedNode {
                    node_kind,
                    node_type: Some(Type {
                        type_kind_id,
                        instance_kind: InstanceKind::Literal,
                    }),
                });
            }
            Op::Equal | Op::NotEqual => {
                let type_kind_id = self.type_kinds.add_or_get(TypeKind::Bool);
                return self.add_node(TypedNode {
                    node_kind,
                    node_type: Some(Type {
                        type_kind_id,
                        instance_kind: InstanceKind::Literal,
                    }),
                });
            }
            Op::And | Op::Or => {
                if self.type_kinds.get_by_id(left_type.type_kind_id) != TypeKind::Bool {
                    type_error!(self, "expected Bool");
                }
            }
            _ => {}
        }

        self.add_node(TypedNode {
            node_kind,
            node_type: Some(Type {
                type_kind_id: left_type.type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn const_binary(
        &mut self,
        left: NodeIndex,
        op: Op,
        right: NodeIndex,
        index: NodeIndex,
        hint: Option<usize>,
    ) -> NodeIndex {
        let typed_binary = self.check_node_with_hint(index, hint);
        let binary_type = assert_typed!(self, typed_binary);

        let typed_left = self.check_const_node(left, Some(binary_type.type_kind_id));
        let InstanceKind::Const(left_const_value) = assert_typed!(self, typed_left).instance_kind
        else {
            type_error!(self, "expected const operand")
        };

        let typed_right = self.check_const_node(right, Some(binary_type.type_kind_id));
        let InstanceKind::Const(right_const_value) = assert_typed!(self, typed_right).instance_kind
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

        self.add_node(TypedNode {
            node_kind: NodeKind::Binary {
                left: typed_left,
                op,
                right: typed_right,
            },
            node_type: Some(Type {
                type_kind_id: binary_type.type_kind_id,
                instance_kind: InstanceKind::Const(result_value),
            }),
        })
    }

    fn unary_prefix(&mut self, op: Op, right: NodeIndex, hint: Option<usize>) -> NodeIndex {
        let typed_right = self.check_node_with_hint(right, hint);
        let right_type = assert_typed!(self, typed_right);

        if right_type.instance_kind == InstanceKind::Name {
            type_error!(
                self,
                "unary prefix operators can only be applied to instances"
            );
        }

        let node_kind = NodeKind::UnaryPrefix {
            op,
            right: typed_right,
        };

        match op {
            Op::Plus | Op::Minus => {
                if !self
                    .type_kinds
                    .get_by_id(right_type.type_kind_id)
                    .is_numeric()
                {
                    type_error!(self, "expected numeric type");
                }

                self.add_node(TypedNode {
                    node_kind,
                    node_type: Some(right_type),
                })
            }
            Op::Not => {
                if self.type_kinds.get_by_id(right_type.type_kind_id) != TypeKind::Bool {
                    type_error!(self, "expected Bool");
                }

                self.add_node(TypedNode {
                    node_kind,
                    node_type: Some(right_type),
                })
            }
            Op::Reference => {
                let is_mutable = match right_type.instance_kind {
                    InstanceKind::Val => false,
                    InstanceKind::Var => true,
                    _ => type_error!(self, "references must refer to a variable"),
                };

                let type_kind_id = self.type_kinds.add_or_get(TypeKind::Pointer {
                    inner_type_kind_id: right_type.type_kind_id,
                    is_inner_mutable: is_mutable,
                });

                self.add_node(TypedNode {
                    node_kind,
                    node_type: Some(Type {
                        type_kind_id,
                        instance_kind: InstanceKind::Literal,
                    }),
                })
            }
            _ => type_error!(self, "unknown unary prefix operator"),
        }
    }

    fn const_unary_prefix(&mut self, op: Op, right: NodeIndex, index: NodeIndex, hint: Option<usize>) -> NodeIndex {
        let typed_unary = self.check_node_with_hint(index, hint);
        let unary_type = assert_typed!(self, typed_unary);

        let typed_right = self.check_const_node(right, Some(unary_type.type_kind_id));
        let InstanceKind::Const(right_const_value) = assert_typed!(self, typed_right).instance_kind
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

        self.add_node(TypedNode {
            node_kind: NodeKind::UnaryPrefix {
                op,
                right: typed_right,
            },
            node_type: Some(Type {
                type_kind_id: unary_type.type_kind_id,
                instance_kind: InstanceKind::Const(result_value),
            }),
        })
    }

    fn unary_suffix(&mut self, left: NodeIndex, op: Op) -> NodeIndex {
        let typed_left = self.check_node(left);
        let left_type = assert_typed!(self, typed_left);

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

            self.add_node(TypedNode {
                node_kind: NodeKind::UnarySuffix {
                    left: typed_left,
                    op,
                },
                node_type: Some(Type {
                    type_kind_id: *inner_type_kind_id,
                    instance_kind,
                }),
            })
        } else {
            type_error!(self, "unknown unary suffix operator")
        }
    }

    fn call(&mut self, left: NodeIndex, args: Arc<Vec<NodeIndex>>) -> NodeIndex {
        let typed_left = self.check_node(left);
        let left_type = assert_typed!(self, typed_left);

        let TypeKind::Function {
            return_type_kind_id,
            param_type_kind_ids,
            ..
        } = self.type_kinds.get_by_id(left_type.type_kind_id).clone()
        else {
            type_error!(self, "only functions can be called");
        };

        // A call after a field access is a method call. A function pointer would have be to dereferenced first.
        let skip_count = if let NodeKind::FieldAccess { left, .. } = &self.get_typer_node(typed_left).node_kind {
            let left_type = assert_typed!(self, *left);

            if let TypeKind::Pointer { is_inner_mutable, .. } = self.type_kinds.get_by_id(left_type.type_kind_id) {
                let TypeKind::Pointer { is_inner_mutable: expected_is_inner_mutable, .. } = self.type_kinds.get_by_id(param_type_kind_ids[0]) else {
                    type_error!(self, "methods must accept a pointer as their first argument");
                };

                if !is_inner_mutable && expected_is_inner_mutable {
                    type_error!(self, "this method cannot accept a *val as a subject");
                }
            }

            1
        } else {
            0
        };

        if args.len() + skip_count != param_type_kind_ids.len() {
            type_error!(self, "wrong number of arguments");
        }

        let mut typed_args = Vec::new();
        for (arg, param_type_kind_id) in args.iter().zip(param_type_kind_ids.iter().skip(skip_count)) {
            let typed_arg = self.check_node_with_hint(*arg, Some(*param_type_kind_id));
            typed_args.push(typed_arg);

            let arg_type = assert_typed!(self, typed_arg);

            if arg_type.type_kind_id != *param_type_kind_id {
                type_error!(self, "incorrect argument type");
            }
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::Call {
                left: typed_left,
                args: Arc::new(typed_args),
            },
            node_type: Some(Type {
                type_kind_id: return_type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn index_access(&mut self, left: NodeIndex, expression: NodeIndex) -> NodeIndex {
        let typed_left = self.check_node(left);
        let left_type = assert_typed!(self, typed_left);

        if left_type.instance_kind == InstanceKind::Name {
            type_error!(self, "array type names cannot be indexed");
        }

        let element_type_kind_id = if let TypeKind::Array {
            element_type_kind_id,
            ..
        } = &self.type_kinds.get_by_id(left_type.type_kind_id)
        {
            *element_type_kind_id
        } else {
            type_error!(self, "indexing is only allowed on arrays");
        };

        let typed_expression = self.check_node(expression);
        let expression_type = assert_typed!(self, typed_expression);

        if !self.type_kinds.get_by_id(expression_type.type_kind_id).is_int() {
            type_error!(self, "expected index to have an integer type (Int, UInt, etc.)");
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::IndexAccess {
                left: typed_left,
                expression: typed_expression,
            },
            node_type: Some(Type {
                type_kind_id: element_type_kind_id,
                instance_kind: left_type.instance_kind,
            }),
        })
    }

    fn field_access(&mut self, left: NodeIndex, name: NodeIndex) -> NodeIndex {
        let typed_left = self.check_node(left);
        let left_type = assert_typed!(self, typed_left);
        let typed_name = self.check_node(name);

        let NodeKind::Name { text: name_text } = &self.get_parser_node(name).kind else {
            type_error!(self, "invalid field name");
        };

        let node_kind = NodeKind::FieldAccess {
            left: typed_left,
            name: typed_name,
        };

        let mut is_tag_access = false;
        let (struct_type_kind_id, field_instance_kind) =
            match &self.type_kinds.get_by_id(left_type.type_kind_id) {
                TypeKind::Struct { is_union, .. } => {
                    is_tag_access = left_type.instance_kind == InstanceKind::Name && *is_union;
                    (left_type.type_kind_id, left_type.instance_kind)
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
                        } = &self.get_typer_node(*variant_name).node_kind
                        else {
                            type_error!(self, "invalid enum variant name");
                        };

                        if *variant_name_text == *name_text {
                            return self.add_node(TypedNode {
                                node_kind,
                                node_type: Some(Type {
                                    type_kind_id: left_type.type_kind_id,
                                    instance_kind: InstanceKind::Literal,
                                }),
                            });
                        }
                    }

                    type_error!(self, "variant not found in enum");
                }
                TypeKind::Array { .. } => {
                    if name_text.as_ref() != "count" {
                        type_error!(self, "field not found on array");
                    }

                    let type_kind_id = self.type_kinds.add_or_get(TypeKind::UInt);
                    return self.add_node(TypedNode {
                        node_kind,
                        node_type: Some(Type {
                            type_kind_id,
                            instance_kind: InstanceKind::Literal,
                        }),
                    });
                }
                _ => type_error!(
                    self,
                    "field access is only allowed on structs, enums, and pointers to structs"
                ),
            };

        let TypeKind::Struct { fields, .. } = &self.type_kinds.get_by_id(struct_type_kind_id)
        else {
            type_error!(self, "field access is only allowed on struct types");
        };

        for Field {
            name: field_name,
            type_kind_id: field_kind_id,
        } in fields.iter()
        {
            let NodeKind::Name {
                text: field_name_text,
            } = &self.get_typer_node(*field_name).node_kind
            else {
                type_error!(self, "invalid field name on struct");
            };

            if *field_name_text != *name_text {
                continue;
            }

            if is_tag_access {
                let type_kind_id = self.type_kinds.add_or_get(TypeKind::Tag);
                return self.add_node(TypedNode {
                    node_kind,
                    node_type: Some(Type {
                        type_kind_id,
                        instance_kind: InstanceKind::Literal,
                    }),
                });
            }

            return self.add_node(TypedNode {
                node_kind,
                node_type: Some(Type {
                    type_kind_id: *field_kind_id,
                    instance_kind: field_instance_kind,
                }),
            });
        }

        // TODO: Also make it possible to have generic args here, probably requires making generic specifiers allowed as the rhs of field accesses.
        let Some((_, name_type)) = self.lookup_identifier(name, Some(left_type.type_kind_id), None, LookupKind::Functions) else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
            });
        };

        self.add_node(TypedNode {
            node_kind,
            node_type: Some(name_type),
        })
    }

    fn cast(&mut self, left: NodeIndex, type_name: NodeIndex) -> NodeIndex {
        let typed_left = self.check_node(left);
        let typed_type_name = self.check_node(type_name);
        let type_name_type = assert_typed!(self, typed_type_name);

        self.add_node(TypedNode {
            node_kind: NodeKind::Cast {
                left: typed_left,
                type_name: typed_type_name,
            },
            node_type: Some(Type {
                type_kind_id: type_name_type.type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn const_cast(&mut self, left: NodeIndex, type_name: NodeIndex, index: NodeIndex) -> NodeIndex {
        let typed_const = self.check_node(index);
        let const_type = assert_typed!(self, typed_const);

        let typed_left = self.check_const_node(left, None);
        let left_type = assert_typed!(self, typed_left);

        let typed_type_name = self.check_node(type_name);

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

        self.add_node(TypedNode {
            node_kind: NodeKind::Cast {
                left: typed_left,
                type_name: typed_type_name,
            },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(result_value),
            }),
        })
    }

    fn name(&mut self, text: Arc<str>) -> NodeIndex {
        self.add_node(TypedNode {
            node_kind: NodeKind::Name { text },
            node_type: None,
        })
    }

    fn identifier(&mut self, name: NodeIndex) -> NodeIndex {
        let Some((typed_name, name_type)) = self.lookup_identifier(name, None, None, LookupKind::All) else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::Identifier { name: typed_name },
            node_type: Some(name_type),
        })
    }

    fn const_identifier(&mut self, name: NodeIndex, index: NodeIndex) -> NodeIndex {
        let typed_name = self.check_node(name);

        let typed_const = self.check_node(index);
        let const_type = assert_typed!(self, typed_const);

        if !matches!(const_type.instance_kind, InstanceKind::Const(..)) {
            type_error!(self, "expected identifier to refer to a const value");
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::Identifier { name: typed_name },
            node_type: Some(const_type),
        })
    }

    fn int_literal(&mut self, text: Arc<str>, hint: Option<usize>) -> NodeIndex {
        let mut type_kind_id = self.type_kinds.add_or_get(TypeKind::Int);

        if let Some(hint) = hint {
            if self.type_kinds.get_by_id(hint).is_int() {
                type_kind_id = hint;
            }
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::IntLiteral { text },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn const_int_literal(&mut self, text: Arc<str>, index: NodeIndex, hint: Option<usize>) -> NodeIndex {
        let typed_const = self.check_node_with_hint(index, hint);
        let const_type = assert_typed!(self, typed_const);

        let value = if self.type_kinds.get_by_id(const_type.type_kind_id).is_unsigned() {
            let Ok(value) = text.parse::<u64>() else {
                type_error!(self, "invalid value of unsigned integer literal");
            };

            ConstValue::UInt { value }
        } else {
            let Ok(value) = text.parse::<i64>() else {
                type_error!(self, "invalid value of integer literal");
            };

            ConstValue::Int { value }
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::IntLiteral { text },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(value),
            }),
        })
    }

    fn float_literal(&mut self, text: Arc<str>, hint: Option<usize>) -> NodeIndex {
        let mut type_kind_id = self.type_kinds.add_or_get(TypeKind::Float32);

        if let Some(hint) = hint {
            if self.type_kinds.get_by_id(hint).is_float() {
                type_kind_id = hint;
            }
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::FloatLiteral { text },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn const_float_literal(&mut self, text: Arc<str>, index: NodeIndex, hint: Option<usize>) -> NodeIndex {
        let typed_const = self.check_node_with_hint(index, hint);
        let const_type = assert_typed!(self, typed_const);
        let Ok(value) = text.parse::<f64>() else {
            type_error!(self, "invalid value of float literal");
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::FloatLiteral { text },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(ConstValue::Float { value }),
            }),
        })
    }

    fn string_literal(&mut self, text: Arc<str>) -> NodeIndex {
        let type_kind_id = self.type_kinds.add_or_get(TypeKind::String);
        self.add_node(TypedNode {
            node_kind: NodeKind::StringLiteral { text },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn const_string_literal(&mut self, text: Arc<str>, index: NodeIndex) -> NodeIndex {
        let typed_const = self.check_node(index);
        let const_type = assert_typed!(self, typed_const);

        self.add_node(TypedNode {
            node_kind: NodeKind::StringLiteral { text: text.clone() },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(ConstValue::String { value: text }),
            }),
        })
    }

    fn bool_literal(&mut self, value: bool) -> NodeIndex {
        let type_kind_id = self.type_kinds.add_or_get(TypeKind::Bool);
        self.add_node(TypedNode {
            node_kind: NodeKind::BoolLiteral { value },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn const_bool_literal(&mut self, value: bool, index: NodeIndex) -> NodeIndex {
        let typed_const = self.check_node(index);
        let const_type = assert_typed!(self, typed_const);

        self.add_node(TypedNode {
            node_kind: NodeKind::BoolLiteral { value },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(ConstValue::Bool { value }),
            }),
        })
    }

    fn char_literal(&mut self, value: char) -> NodeIndex {
        let type_kind_id = self.type_kinds.add_or_get(TypeKind::Char);
        self.add_node(TypedNode {
            node_kind: NodeKind::CharLiteral { value },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn const_char_literal(&mut self, value: char, index: NodeIndex) -> NodeIndex {
        let typed_const = self.check_node(index);
        let const_type = assert_typed!(self, typed_const);

        self.add_node(TypedNode {
            node_kind: NodeKind::CharLiteral { value },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(ConstValue::Char { value }),
            }),
        })
    }

    fn array_literal(
        &mut self,
        elements: Arc<Vec<NodeIndex>>,
        repeat_count_const_expression: Option<NodeIndex>,
    ) -> NodeIndex {
        let mut typed_elements = Vec::new();
        for element in elements.iter() {
            typed_elements.push(self.check_node(*element));
        }

        let repeat_count = if let Some(const_expression) = repeat_count_const_expression {
            let mut repeat_count = 0;
            if let Some(error_node) =
                self.const_expression_to_uint(const_expression, &mut repeat_count)
            {
                return error_node;
            }

            repeat_count
        } else {
            1
        };

        let node_type = if let Some(first_element) = elements.first() {
            let typed_element = self.check_node(*first_element);
            let element_type = assert_typed!(self, typed_element).clone();
            let type_kind_id = self.type_kinds.add_or_get(TypeKind::Array {
                element_type_kind_id: element_type.type_kind_id,
                element_count: elements.len() * repeat_count,
            });

            Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Literal,
            })
        } else {
            type_error!(self, "array literal must contain at least one element");
        };

        let typed_repeat_count_const_expression =
            self.check_optional_node(repeat_count_const_expression, None);
        self.add_node(TypedNode {
            node_kind: NodeKind::ArrayLiteral {
                elements: Arc::new(typed_elements),
                repeat_count_const_expression: typed_repeat_count_const_expression,
            },
            node_type,
        })
    }

    fn struct_literal(
        &mut self,
        left: NodeIndex,
        field_literals: Arc<Vec<NodeIndex>>,
    ) -> NodeIndex {
        let typed_left = self.check_node(left);
        let struct_type = assert_typed!(self, typed_left);

        if struct_type.instance_kind != InstanceKind::Name {
            type_error!(
                self,
                "expected struct literal to start with the name of a struct type"
            );
        }

        let TypeKind::Struct {
            fields: expected_fields,
            is_union,
            ..
        } = self.type_kinds.get_by_id(struct_type.type_kind_id)
        else {
            type_error!(self, "expected struct type in struct literal");
        };

        let mut typed_field_literals = Vec::new();

        if is_union {
            if field_literals.len() != 1 && expected_fields.len() != 0 {
                type_error!(
                    self,
                    "incorrect number of fields, expected one field to initialize a union"
                );
            }

            if field_literals.len() > 0 {
                let NodeKind::FieldLiteral {
                    name: field_name, ..
                } = &self.get_parser_node(field_literals[0]).kind
                else {
                    type_error!(self, "invalid field literal");
                };

                let NodeKind::Name {
                    text: field_name_text,
                } = &self.get_parser_node(*field_name).kind
                else {
                    type_error!(self, "invalid field name");
                };

                let Some(expected_field_index) =
                    get_field_index_by_name(&self.typed_nodes, field_name_text, &expected_fields)
                else {
                    type_error!(self, "union doesn't contain a field with this name");
                };

                let expected_type_kind_id = expected_fields[expected_field_index].type_kind_id;

                let typed_field_literal = self.check_node_with_hint(field_literals[0], Some(expected_type_kind_id));
                typed_field_literals.push(typed_field_literal);

                let field_literal_type = assert_typed!(self, typed_field_literal);

                if field_literal_type.type_kind_id
                    != expected_type_kind_id
                {
                    type_error!(self, "incorrect field type");
                }
            }
        } else {
            if field_literals.len() != expected_fields.len() {
                type_error!(self, "incorrect number of fields");
            }

            for (field, expected_field) in field_literals.iter().zip(expected_fields.iter()) {
                let typed_field_literal = self.check_node(*field);
                typed_field_literals.push(typed_field_literal);

                let field_literal_type = assert_typed!(self, typed_field_literal);

                if field_literal_type.type_kind_id != expected_field.type_kind_id {
                    type_error!(self, "incorrect field type");
                }
            }
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::StructLiteral {
                left: typed_left,
                field_literals: Arc::new(typed_field_literals),
            },
            node_type: Some(Type {
                type_kind_id: struct_type.type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn field_literal(&mut self, name: NodeIndex, expression: NodeIndex) -> NodeIndex {
        let typed_name = self.check_node(name);
        let typed_expression = self.check_node(expression);
        let expression_type = assert_typed!(self, typed_expression);

        self.add_node(TypedNode {
            node_kind: NodeKind::FieldLiteral {
                name: typed_name,
                expression: typed_expression,
            },
            node_type: Some(expression_type),
        })
    }

    fn type_size(&mut self, type_name: NodeIndex, hint: Option<usize>) -> NodeIndex {
        let typed_type_name = self.check_node(type_name);
        let mut type_kind_id = self.type_kinds.add_or_get(TypeKind::UInt);

        if let Some(hint) = hint {
            if self.type_kinds.get_by_id(hint).is_int() {
                type_kind_id = hint;
            }
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeSize {
                type_name: typed_type_name,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
        })
    }

    fn const_type_size(&mut self, type_name: NodeIndex, index: NodeIndex, hint: Option<usize>) -> NodeIndex {
        let typed_type_name = self.check_node(type_name);
        let type_name_type = assert_typed!(self, typed_type_name);
        let typed_const = self.check_node_with_hint(index, hint);
        let const_type = assert_typed!(self, typed_const);

        let native_size = mem::size_of::<NodeIndex>() as u64;

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

        let const_value = if self.type_kinds.get_by_id(const_type.type_kind_id).is_unsigned() {
            ConstValue::UInt { value }
        } else {
            ConstValue::Int { value: value as i64 }
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeSize {
                type_name: typed_type_name,
            },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(const_value),
            }),
        })
    }

    fn struct_definition(
        &mut self,
        name: NodeIndex,
        fields: Arc<Vec<NodeIndex>>,
        generic_params: Arc<Vec<NodeIndex>>,
        is_union: bool,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
    ) -> NodeIndex {
        if !generic_params.is_empty() && generic_arg_type_kind_ids.is_none() {
            type_error!(self, "generic type requires generic arguments");
        }

        let NodeKind::Name { text: name_text } = self.get_parser_node(name).kind.clone() else {
            type_error!(self, "invalid name");
        };

        let typed_name = self.check_node(name);
        let identifier = GenericIdentifier {
            name: name_text,
            extending_type_kind_id: None,
            generic_arg_type_kind_ids: generic_arg_type_kind_ids.clone(),
        };

        let type_kind_id = self.type_kinds.add_placeholder();
        self.type_kind_environment
            .insert(identifier, type_kind_id, true);

        self.type_kind_environment.push(false);

        if let Some(generic_arg_type_kind_ids) = generic_arg_type_kind_ids.clone() {
            if generic_arg_type_kind_ids.len() != generic_params.len() {
                type_error!(self, "incorrect number of generic arguments");
            }

            for i in 0..generic_arg_type_kind_ids.len() {
                let NodeKind::Name { text: param_text } =
                    self.get_parser_node(generic_params[i]).kind.clone()
                else {
                    type_error!(self, "invalid parameter name");
                };

                self.type_kind_environment.insert(
                    GenericIdentifier {
                        name: param_text,
                        extending_type_kind_id: None,
                        generic_arg_type_kind_ids: None,
                    },
                    generic_arg_type_kind_ids[i],
                    false,
                );
            }
        }

        let mut typed_fields = Vec::new();
        let mut type_kind_fields = Vec::new();

        for field in fields.iter() {
            let typed_field = self.check_node(*field);
            typed_fields.push(typed_field);

            let field_type_kind_id = assert_typed!(self, typed_field).type_kind_id;

            let NodeKind::Field {
                name: field_name, ..
            } = self.get_parser_node(*field).kind
            else {
                type_error!(self, "invalid field");
            };

            let typed_field_name = self.check_node(field_name);

            type_kind_fields.push(Field {
                name: typed_field_name,
                type_kind_id: field_type_kind_id,
            })
        }

        self.type_kind_environment.pop();

        self.type_kinds.replace_placeholder(
            type_kind_id,
            TypeKind::Struct {
                name: typed_name,
                fields: Arc::new(type_kind_fields),
                is_union,
            },
        );

        let index = self.add_node(TypedNode {
            node_kind: NodeKind::StructDefinition {
                name: typed_name,
                fields: Arc::new(typed_fields),
                generic_params: Arc::new(Vec::new()),
                is_union,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Name,
            }),
        });

        self.typed_definition_indices.push(index);

        index
    }

    fn enum_definition(
        &mut self,
        name: NodeIndex,
        variant_names: Arc<Vec<NodeIndex>>,
    ) -> NodeIndex {
        let NodeKind::Name { text: name_text } = self.get_parser_node(name).kind.clone() else {
            type_error!(self, "invalid enum name");
        };

        let typed_name = self.check_node(name);

        let mut typed_variant_names = Vec::new();
        for variant_name in variant_names.iter() {
            typed_variant_names.push(self.check_node(*variant_name));
        }
        let typed_variant_names = Arc::new(typed_variant_names);

        let type_kind_id = self.type_kinds.add_or_get(TypeKind::Enum {
            name: typed_name,
            variant_names: typed_variant_names.clone(),
        });

        let identifier = GenericIdentifier {
            name: name_text,
            extending_type_kind_id: None,
            generic_arg_type_kind_ids: None,
        };
        self.type_kind_environment
            .insert(identifier, type_kind_id, true);

        let index = self.add_node(TypedNode {
            node_kind: NodeKind::EnumDefinition {
                name: typed_name,
                variant_names: typed_variant_names,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Name,
            }),
        });

        self.typed_definition_indices.push(index);

        index
    }

    fn field(&mut self, name: NodeIndex, type_name: NodeIndex) -> NodeIndex {
        let typed_name = self.check_node(name);
        let typed_type_name = self.check_node(type_name);
        let type_name_type = assert_typed!(self, typed_type_name);

        if let TypeKind::Void = self.type_kinds.get_by_id(type_name_type.type_kind_id) {
            type_error!(self, "field cannot be of type Void");
        }

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(type_name_type.type_kind_id) {
            type_error!(self, "field cannot be of type func");
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::Field {
                name: typed_name,
                type_name: typed_type_name,
            },
            node_type: Some(type_name_type),
        })
    }

    fn function_declaration(
        &mut self,
        name: NodeIndex,
        params: Arc<Vec<NodeIndex>>,
        extending_type_name: Option<NodeIndex>,
        generic_params: Arc<Vec<NodeIndex>>,
        return_type_name: NodeIndex,
        is_extern: bool,
    ) -> NodeIndex {
        let mut typed_name = self.check_node(name);
        if is_extern {
            typed_name.file_index = None;
        }

        let mut typed_params = Vec::new();
        let mut param_type_kind_ids = Vec::new();
        for param in params.iter() {
            let typed_param = self.check_node(*param);
            typed_params.push(typed_param);

            let param_type = assert_typed!(self, typed_param);
            param_type_kind_ids.push(param_type.type_kind_id);
        }
        let param_type_kind_ids = Arc::new(param_type_kind_ids);

        let typed_extending_type_name = self.check_optional_node(extending_type_name, None);
        if let Some(typed_extending_type_name) = typed_extending_type_name {
            let extending_type_name_type = assert_typed!(self, typed_extending_type_name);

            if let TypeKind::Pointer { .. } = self.type_kinds.get_by_id(extending_type_name_type.type_kind_id) {
                type_error!(self, "methods cannot be declared on pointers ");
            }

            if typed_params.is_empty() {
                type_error!(self, "methods must accept at least one argument");
            }

            let first_param_type = assert_typed!(self, typed_params[0]);

            let TypeKind::Pointer { inner_type_kind_id, .. } = &self.type_kinds.get_by_id(first_param_type.type_kind_id) else {
                type_error!(self, "methods must accept a pointer as their first argument");
            };

            if *inner_type_kind_id != extending_type_name_type.type_kind_id {
                type_error!(self, "methods must accept a pointer to their owning type as their first argument");
            }
        }

        let mut typed_generic_params = Vec::new();
        for generic_param in generic_params.iter() {
            let typed_generic_param = self.check_node(*generic_param);
            typed_generic_params.push(typed_generic_param);
        }

        let typed_return_type_name = self.check_node(return_type_name);
        let return_type = assert_typed!(self, typed_return_type_name);
        if return_type.instance_kind != InstanceKind::Name {
            type_error!(self, "expected type name");
        }

        let type_kind = TypeKind::Function {
            param_type_kind_ids: param_type_kind_ids.clone(),
            return_type_kind_id: return_type.type_kind_id,
        };
        let type_kind_id = self.type_kinds.add_or_get(type_kind);

        let index = self.add_node_with_file_index(
            TypedNode {
                node_kind: NodeKind::FunctionDeclaration {
                    name: typed_name,
                    params: Arc::new(typed_params),
                    extending_type_name: typed_extending_type_name,
                    generic_params: Arc::new(typed_generic_params),
                    return_type_name: typed_return_type_name,
                },
                node_type: Some(Type {
                    type_kind_id,
                    instance_kind: InstanceKind::Name,
                }),
            },
            typed_name.file_index,
        );

        let NodeKind::Name { text: name_text } = &self.get_typer_node(typed_name).node_kind else {
            type_error!(self, "invalid name in function declaration");
        };

        if name_text.as_ref() == "Main" && extending_type_name.is_none() {
            if is_extern {
                type_error!(self, "Main function cannot be defined as an extern");
            }

            if !generic_params.is_empty() {
                type_error!(self, "Main function cannot be generic");
            }

            if self.shallow_check_stack == 0 {
                self.main_function_declaration = Some(index);
            }

            if self.type_kinds.get_by_id(return_type.type_kind_id) != TypeKind::Int {
                type_error!(self, "expected Main to return an Int");
            }

            if param_type_kind_ids.len() > 0 {
                if param_type_kind_ids.len() < 2 {
                    type_error!(self, "invalid parameters for Main function");
                }

                let first_type_kind = self.type_kinds.get_by_id(param_type_kind_ids[0]);
                if first_type_kind != TypeKind::Int {
                    type_error!(self, "expected first argument of Main to be an Int");
                }

                let second_type_kind = self.type_kinds.get_by_id(param_type_kind_ids[1]);
                let TypeKind::Pointer {
                    inner_type_kind_id,
                    is_inner_mutable: false,
                } = second_type_kind
                else {
                    type_error!(self, "expected second argument of Main to be *val String");
                };

                let TypeKind::String = self.type_kinds.get_by_id(inner_type_kind_id) else {
                    type_error!(self, "expected second argument of Main to be *String");
                };
            }
        }

        index
    }

    fn function(
        &mut self,
        declaration: NodeIndex,
        scoped_statement: NodeIndex,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
    ) -> NodeIndex {
        let pre_error_count = self.error_count;

        let is_generic = generic_arg_type_kind_ids.is_some();

        let index = self.function_impl(declaration, scoped_statement, generic_arg_type_kind_ids);

        if self.error_count <= pre_error_count || !is_generic {
            return index;
        }

        // The place this node was used must have been the node before it in the stack.
        if let Some(usage_index) = self.node_index_stack.get(self.node_index_stack.len() - 2) {
            self.get_parser_node(*usage_index)
                .start
                .usage_error(&self.files);
        }

        index
    }

    fn function_impl(
        &mut self,
        declaration: NodeIndex,
        scoped_statement: NodeIndex,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
    ) -> NodeIndex {
        let NodeKind::FunctionDeclaration {
            name,
            extending_type_name,
            generic_params,
            ..
        } = self.get_parser_node(declaration).kind.clone()
        else {
            type_error!(self, "invalid function declaration");
        };

        let NodeKind::Name { text: name_text } = self.get_parser_node(name).kind.clone() else {
            type_error!(self, "invalid function name");
        };

        if !generic_params.is_empty() && generic_arg_type_kind_ids.is_none() {
            type_error!(self, "generic function requires generic arguments");
        }

        self.type_kind_environment.push(false);
        self.environment.push(false);

        if let Some(generic_arg_type_kind_ids) = generic_arg_type_kind_ids.clone() {
            if generic_arg_type_kind_ids.len() != generic_params.len() {
                type_error!(self, "incorrect number of generic arguments");
            }

            for i in 0..generic_arg_type_kind_ids.len() {
                let NodeKind::Name { text: param_text } =
                    self.get_parser_node(generic_params[i]).kind.clone()
                else {
                    type_error!(self, "invalid parameter name");
                };

                self.type_kind_environment.insert(
                    GenericIdentifier {
                        name: param_text,
                        extending_type_kind_id: None,
                        generic_arg_type_kind_ids: None,
                    },
                    generic_arg_type_kind_ids[i],
                    false,
                );
            }
        }

        let typed_extending_type_name = self.check_optional_node(extending_type_name, None);
        let extending_type_kind_id = if let Some(typed_extending_type_name) = typed_extending_type_name {
            Some(assert_typed!(self, typed_extending_type_name).type_kind_id)
        } else {
            None
        };

        let typed_declaration = self.check_node(declaration);
        let declaration_type = assert_typed!(self, typed_declaration);
        let identifier = GenericIdentifier {
            name: name_text.clone(),
            extending_type_kind_id,
            generic_arg_type_kind_ids: generic_arg_type_kind_ids.clone(),
        };
        self.function_definitions.insert(
            identifier,
            FunctionDefinition {
                type_kind_id: declaration_type.type_kind_id,
                file_index: typed_declaration.file_index,
            },
        );

        let is_deep_check = self.shallow_check_stack == 0 || generic_arg_type_kind_ids.is_some();

        let typed_scoped_statement = if is_deep_check {
            let TypeKind::Function {
                return_type_kind_id: expected_return_type_kind_id,
                ..
            } = self.type_kinds.get_by_id(declaration_type.type_kind_id)
            else {
                type_error!(self, "invalid function type");
            };

            let typed_scoped_statement = self.check_node_with_hint(scoped_statement, Some(expected_return_type_kind_id));

            if let Some(return_type) = self.ensure_typed_statement_returns(typed_scoped_statement) {
                if return_type.type_kind_id != expected_return_type_kind_id {
                    type_error!(self, "function does not return the right type");
                }
            } else if self.type_kinds.get_by_id(expected_return_type_kind_id) != TypeKind::Void {
                type_error!(
                    self,
                    "function does not return the correct type of value on all execution paths"
                );
            }

            typed_scoped_statement
        } else {
            self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
            })
        };

        self.environment.pop();
        self.type_kind_environment.pop();

        let index = self.add_node_with_file_index(
            TypedNode {
                node_kind: NodeKind::Function {
                    declaration: typed_declaration,
                    scoped_statement: typed_scoped_statement,
                },
                node_type: Some(Type {
                    type_kind_id: declaration_type.type_kind_id,
                    instance_kind: InstanceKind::Val,
                }),
            },
            typed_declaration.file_index,
        );

        self.typed_definition_indices.push(index);

        index
    }

    fn generic_specifier(
        &mut self,
        name: NodeIndex,
        generic_arg_type_names: Arc<Vec<NodeIndex>>,
    ) -> NodeIndex {
        let mut typed_generic_arg_type_names = Vec::new();
        for generic_arg_type_name in generic_arg_type_names.iter() {
            let typed_generic_arg_type_name = self.check_node(*generic_arg_type_name);
            typed_generic_arg_type_names.push(typed_generic_arg_type_name);
        }

        let mut generic_arg_type_kind_ids = Vec::new();
        for generic_arg_type_name in generic_arg_type_names.iter() {
            let typed_generic_arg = self.check_node(*generic_arg_type_name);
            let generic_arg_type = assert_typed!(self, typed_generic_arg);
            if generic_arg_type.instance_kind != InstanceKind::Name {
                type_error!(self, "expected type name");
            }

            generic_arg_type_kind_ids.push(generic_arg_type.type_kind_id);
        }
        let generic_arg_type_kind_ids = Arc::new(generic_arg_type_kind_ids);

        let Some((typed_name, name_type)) =
            self.lookup_identifier(name, None, Some(generic_arg_type_kind_ids), LookupKind::All)
        else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::GenericSpecifier {
                name: typed_name,
                generic_arg_type_names: Arc::new(typed_generic_arg_type_names),
            },
            node_type: Some(name_type),
        })
    }

    fn type_name(&mut self, name: NodeIndex) -> NodeIndex {
        let Some((typed_name, name_type)) = self.lookup_identifier(name, None, None, LookupKind::Types) else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeName { name: typed_name },
            node_type: Some(name_type),
        })
    }

    fn type_name_pointer(&mut self, inner: NodeIndex, is_inner_mutable: bool) -> NodeIndex {
        let typed_inner = self.check_node(inner);
        let inner_type = assert_typed!(self, typed_inner);
        if inner_type.instance_kind != InstanceKind::Name {
            type_error!(self, "expected type name");
        }

        let type_kind_id = self.type_kinds.add_or_get(TypeKind::Pointer {
            inner_type_kind_id: inner_type.type_kind_id,
            is_inner_mutable,
        });

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeNamePointer {
                inner: typed_inner,
                is_inner_mutable,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Name,
            }),
        })
    }

    fn type_name_array(
        &mut self,
        inner: NodeIndex,
        element_count_const_expression: NodeIndex,
    ) -> NodeIndex {
        let typed_inner = self.check_node(inner);
        let inner_type = assert_typed!(self, typed_inner);
        if inner_type.instance_kind != InstanceKind::Name {
            type_error!(self, "expected type name");
        }

        let expected_element_count_type_kind_id = self.type_kinds.add_or_get(TypeKind::Int);
        let typed_element_count_const_expression = self.check_node_with_hint(
            element_count_const_expression,
            Some(expected_element_count_type_kind_id),
        );
        let element_count_const_type = assert_typed!(self, typed_element_count_const_expression);

        let InstanceKind::Const(ConstValue::Int {
            value: element_count,
        }) = element_count_const_type.instance_kind
        else {
            type_error!(self, "expected Int for element count");
        };

        let type_kind_id = self.type_kinds.add_or_get(TypeKind::Array {
            element_type_kind_id: inner_type.type_kind_id,
            element_count: element_count as usize,
        });

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeNameArray {
                inner: typed_inner,
                element_count_const_expression: typed_element_count_const_expression,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Name,
            }),
        })
    }

    fn type_name_function(
        &mut self,
        param_type_names: Arc<Vec<NodeIndex>>,
        return_type_name: NodeIndex,
    ) -> NodeIndex {
        let mut param_type_kind_ids = Vec::new();
        for param_type_name in param_type_names.iter() {
            let typed_param = self.check_node(*param_type_name);
            let param_type = assert_typed!(self, typed_param);
            if param_type.instance_kind != InstanceKind::Name {
                type_error!(self, "expected type name");
            }

            param_type_kind_ids.push(param_type.type_kind_id);
        }

        let typed_return_type_name = self.check_node(return_type_name);
        let return_type = assert_typed!(self, typed_return_type_name);
        if return_type.instance_kind != InstanceKind::Name {
            type_error!(self, "expected type name");
        }

        let type_kind = TypeKind::Function {
            param_type_kind_ids: Arc::new(param_type_kind_ids),
            return_type_kind_id: return_type.type_kind_id,
        };
        let type_kind_id = self.type_kinds.add_or_get(type_kind);

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeNameFunction {
                param_type_names: Arc::new(Vec::new()),
                return_type_name: typed_return_type_name,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Name,
            }),
        })
    }

    fn type_name_generic_specifier(
        &mut self,
        name: NodeIndex,
        generic_arg_type_names: Arc<Vec<NodeIndex>>,
    ) -> NodeIndex {
        let mut generic_arg_type_kind_ids = Vec::new();
        for generic_arg_type_name in generic_arg_type_names.iter() {
            let typed_generic_arg = self.check_node(*generic_arg_type_name);
            let generic_arg_type = assert_typed!(self, typed_generic_arg);
            if generic_arg_type.instance_kind != InstanceKind::Name {
                type_error!(self, "expected type name");
            }

            generic_arg_type_kind_ids.push(generic_arg_type.type_kind_id);
        }
        let generic_arg_type_kind_ids = Arc::new(generic_arg_type_kind_ids);

        let Some((typed_name, name_type)) =
            self.lookup_identifier(name, None, Some(generic_arg_type_kind_ids), LookupKind::Types)
        else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeNameGenericSpecifier {
                name: typed_name,
                generic_arg_type_names: Arc::new(Vec::new()),
            },
            node_type: Some(name_type),
        })
    }

    fn const_expression_to_uint(
        &mut self,
        const_expression: NodeIndex,
        result: &mut usize,
    ) -> Option<NodeIndex> {
        let NodeKind::ConstExpression { inner } = self.get_parser_node(const_expression).kind
        else {
            return Some(self.type_error("expected const expression"));
        };

        let const_expression = self.const_expression(inner, None);

        let Some(Type {
            instance_kind: InstanceKind::Const(const_value),
            ..
        }) = &self.get_typer_node(const_expression).node_type
        else {
            return Some(self.type_error("expected const value from const expression"));
        };

        *result = match const_value {
            ConstValue::Int { value } => {
                if *value < 0 {
                    return Some(self.type_error("expected positive integer"));
                } else {
                    *value as usize
                }
            }
            ConstValue::UInt { value } => *value as usize,
            _ => return Some(self.type_error("expected integer")),
        };

        None
    }

    fn ensure_typed_statement_returns(&self, statement: NodeIndex) -> Option<Type> {
        match &self.get_typer_node(statement).node_kind {
            NodeKind::Block { statements } => {
                for statement in statements.iter() {
                    if let Some(return_type) = self.ensure_typed_statement_returns(*statement) {
                        return Some(return_type);
                    }
                }

                None
            }
            NodeKind::Statement { inner: Some(inner) } => {
                self.ensure_typed_statement_returns(*inner)
            }
            NodeKind::ReturnStatement {
                expression: Some(expression),
            } => self.get_typer_node(*expression).node_type.clone(),
            NodeKind::DeferStatement { statement } => {
                self.ensure_typed_statement_returns(*statement)
            }
            NodeKind::IfStatement {
                scoped_statement,
                next: Some(next),
                ..
            } => self.ensure_both_typed_statements_return(*scoped_statement, *next),
            NodeKind::SwitchStatement { case_statement, .. } => {
                self.ensure_typed_statement_returns(*case_statement)
            }
            NodeKind::CaseStatement {
                scoped_statement,
                next: Some(next),
                ..
            } => self.ensure_both_typed_statements_return(*scoped_statement, *next),
            _ => None,
        }
    }

    fn ensure_both_typed_statements_return(
        &self,
        statement: NodeIndex,
        next: NodeIndex,
    ) -> Option<Type> {
        let Some(statement_type) = self.ensure_typed_statement_returns(statement) else {
            return None;
        };

        let Some(next_type) = self.ensure_typed_statement_returns(next) else {
            return None;
        };

        if statement_type.type_kind_id == next_type.type_kind_id {
            return Some(statement_type);
        }

        None
    }

    fn is_node_numeric_literal(&self, index: NodeIndex) -> bool {
        matches!(self.get_parser_node(index).kind, NodeKind::IntLiteral { .. } | NodeKind::FloatLiteral { .. })
    }
}
