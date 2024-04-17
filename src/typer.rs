use std::{collections::HashSet, ffi::OsString, mem, sync::Arc};

use crate::{
    assert_matches,
    const_value::ConstValue,
    environment::Environment,
    file_data::FileData,
    namespace::{
        Definition, DefinitionIndexError, DefinitionIndices, Identifier, Namespace,
        NamespaceGenericArg, NamespaceLookupResult, DEFINITION_ERROR,
    },
    parser::{DeclarationKind, MethodKind, Node, NodeIndex, NodeKind, Op},
    position::Position,
    type_kinds::{get_field_index_by_name, Field, TypeKind, TypeKinds},
};

#[derive(Clone, Debug)]
pub struct TypedNode {
    pub node_kind: NodeKind,
    pub node_type: Option<Type>,
    pub namespace_id: Option<usize>,
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

macro_rules! type_error_at_parser_node {
    ($self:ident, $message:expr, $node:expr) => {{
        return $self.type_error_at_parser_node($message, $node);
    }};
}

macro_rules! assert_typed {
    ($self:ident, $index:expr) => {{
        let Some(node_type) = $self.get_typer_node($index).node_type.clone() else {
            return $self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
                namespace_id: None,
            });
        };

        node_type
    }};
}

pub enum LookupResult {
    DefinitionIndex(usize, NodeIndex),
    Definition(usize, Definition),
    Ambiguous,
    None,
}

impl LookupResult {
    fn from_namespace_lookup_result(namespace_id: usize, result: NamespaceLookupResult) -> Self {
        match result {
            NamespaceLookupResult::DefinitionIndex(definition_index) => {
                LookupResult::DefinitionIndex(namespace_id, definition_index)
            }
            NamespaceLookupResult::Definition(definition) => {
                LookupResult::Definition(namespace_id, definition)
            }
            NamespaceLookupResult::None => LookupResult::None,
        }
    }
}

enum IdentifierLookupResult {
    Some(usize, Type),
    Ambiguous,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LookupKind {
    All,
    Types,
}

#[derive(Clone, Copy, Debug)]
enum LookupLocation {
    Namespace(usize),
    File(usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReturnTypeComparison {
    Matches,
    DoesntMatch,
    DoesntReturn,
}

#[derive(Clone, Copy, Debug)]
pub struct TypedDefinition {
    pub index: NodeIndex,
    pub is_shallow: bool,
}

pub const GLOBAL_NAMESPACE_ID: usize = 0;

#[derive(Clone)]
pub struct Typer {
    all_nodes: Arc<Vec<Vec<Node>>>,

    pub typed_nodes: Vec<TypedNode>,
    pub typed_definitions: Vec<TypedDefinition>,
    pub type_kinds: TypeKinds,
    pub main_function_declaration: Option<NodeIndex>,
    pub error_count: usize,

    file_index: Option<usize>,
    files: Arc<Vec<FileData>>,

    pub namespaces: Vec<Namespace>,
    file_namespace_ids: Vec<usize>,
    file_used_namespace_ids: Vec<Vec<usize>>,
    pub string_view_type_kind_id: usize,

    scope_type_kind_environment: Environment<Identifier, usize>,
    scope_environment: Environment<Arc<str>, Type>,

    was_block_already_opened: bool,
    node_index_stack: Vec<NodeIndex>,
    loop_stack: usize,
}

impl Typer {
    // Create a blank typer representing no file in particular.
    pub fn new(all_nodes: Arc<Vec<Vec<Node>>>, files: Arc<Vec<FileData>>) -> Self {
        let file_count = files.len();

        Self {
            all_nodes,

            typed_nodes: Vec::new(),
            typed_definitions: Vec::new(),
            type_kinds: TypeKinds::new(),
            namespaces: Vec::with_capacity(file_count),
            string_view_type_kind_id: 0,
            main_function_declaration: None,
            error_count: 0,

            file_index: None,
            files,
            file_namespace_ids: Vec::with_capacity(file_count),
            file_used_namespace_ids: Vec::new(),

            scope_type_kind_environment: Environment::new(),
            scope_environment: Environment::new(),

            was_block_already_opened: false,
            node_index_stack: Vec::new(),
            loop_stack: 0,
        }
    }

    // Make a typer that inherits the data of an existing typer, but represents a specific file.
    pub fn new_for_file(base_typer: &Typer, file_index: usize) -> Self {
        let mut typer = base_typer.clone();
        typer.file_index = Some(file_index);
        typer.find_string_view();

        typer
    }

    pub fn check_namespaces(
        &mut self,
        all_start_indices: &[NodeIndex],
        file_path_components: &[Vec<OsString>],
    ) {
        let mut file_used_namespace_ids = Vec::new();

        self.namespaces
            .push(Namespace::new("".into(), None, Vec::new(), None));

        self.define_global_primitives();

        let mut definition_errors = Vec::new();

        for (i, start_index) in all_start_indices.iter().enumerate() {
            let mut current_namespace_id = GLOBAL_NAMESPACE_ID;

            for j in 0..(file_path_components[i].len() - 1) {
                let component_str = file_path_components[i][j].to_str().unwrap();

                if let Some(existing_namespace_id) = self.namespaces[current_namespace_id]
                    .child_ids
                    .get(component_str)
                {
                    current_namespace_id = *existing_namespace_id;
                    continue;
                }

                let new_namespace_id = self.namespaces.len();
                let new_namespace_name: Arc<str> = Arc::from(component_str);

                self.namespaces.push(Namespace::new(
                    new_namespace_name.clone(),
                    None,
                    Vec::new(),
                    Some(current_namespace_id),
                ));
                self.namespaces[current_namespace_id]
                    .child_ids
                    .insert(new_namespace_name.clone(), new_namespace_id);

                let namespace_type_kind_id = self.type_kinds.add_or_get(TypeKind::Namespace {
                    namespace_id: new_namespace_id,
                });

                if self.namespaces[current_namespace_id]
                    .insert(
                        new_namespace_name,
                        Definition::TypeKind {
                            type_kind_id: namespace_type_kind_id,
                        },
                    )
                    .is_err()
                {
                    self.error(DEFINITION_ERROR, *start_index);
                }

                current_namespace_id = new_namespace_id;
            }

            self.file_namespace_ids.push(current_namespace_id);

            assert_matches!(
                NodeKind::TopLevel {
                    definition_indices,
                    ..
                },
                self.get_parser_node(*start_index).kind.clone()
            );

            definition_errors.clear();
            self.namespaces[current_namespace_id]
                .extend_definition_indices(&definition_indices, &mut definition_errors);

            for DefinitionIndexError(index) in &definition_errors {
                self.error(DEFINITION_ERROR, *index);
            }

            self.file_used_namespace_ids.push(Vec::new());
            file_used_namespace_ids.push(HashSet::new());
            file_used_namespace_ids[i].insert(GLOBAL_NAMESPACE_ID);
        }

        for (i, start_index) in all_start_indices.iter().enumerate() {
            assert_matches!(
                NodeKind::TopLevel { usings, .. },
                self.get_parser_node(*start_index).kind.clone()
            );

            for using in usings.iter() {
                self.check_using(*using, &mut file_used_namespace_ids);
            }

            self.file_used_namespace_ids[i].extend(file_used_namespace_ids[i].iter());
        }
    }

    fn check_using(
        &mut self,
        using: NodeIndex,
        file_used_namespace_ids: &mut [HashSet<usize>],
    ) -> NodeIndex {
        self.node_index_stack.push(using);

        let Node {
            kind: NodeKind::Using {
                namespace_type_name,
            },
            ..
        } = self.get_parser_node(using).clone()
        else {
            panic!("invalid using node");
        };

        let typed_namespace_type_name = self.check_node(namespace_type_name);
        let namespace_type_name_type = assert_typed!(self, typed_namespace_type_name);

        let namespace_id = if let TypeKind::Namespace { namespace_id } = self
            .type_kinds
            .get_by_id(namespace_type_name_type.type_kind_id)
        {
            namespace_id
        } else {
            return self
                .type_error_at_parser_node("expected namespace after using", namespace_type_name);
        };

        file_used_namespace_ids[using.file_index].insert(namespace_id);

        let typed_using = self.add_node(TypedNode {
            node_kind: NodeKind::Using {
                namespace_type_name: typed_namespace_type_name,
            },
            node_type: None,
            namespace_id: None,
        });

        self.node_index_stack.pop();

        typed_using
    }

    fn define_global_primitives(&mut self) {
        let global_namespace = &mut self.namespaces[GLOBAL_NAMESPACE_ID];

        let int_id = self.type_kinds.add_or_get(TypeKind::Int);
        global_namespace.define(
            Identifier::new("Int"),
            Definition::TypeKind {
                type_kind_id: int_id,
            },
        );
        let bool_id = self.type_kinds.add_or_get(TypeKind::Bool);
        global_namespace.define(
            Identifier::new("Bool"),
            Definition::TypeKind {
                type_kind_id: bool_id,
            },
        );
        let char_id = self.type_kinds.add_or_get(TypeKind::Char);
        global_namespace.define(
            Identifier::new("Char"),
            Definition::TypeKind {
                type_kind_id: char_id,
            },
        );
        let void_id = self.type_kinds.add_or_get(TypeKind::Void);
        global_namespace.define(
            Identifier::new("Void"),
            Definition::TypeKind {
                type_kind_id: void_id,
            },
        );
        let uint_id = self.type_kinds.add_or_get(TypeKind::UInt);
        global_namespace.define(
            Identifier::new("UInt"),
            Definition::TypeKind {
                type_kind_id: uint_id,
            },
        );
        let int8_id = self.type_kinds.add_or_get(TypeKind::Int8);
        global_namespace.define(
            Identifier::new("Int8"),
            Definition::TypeKind {
                type_kind_id: int8_id,
            },
        );
        let uint8_id = self.type_kinds.add_or_get(TypeKind::UInt8);
        global_namespace.define(
            Identifier::new("UInt8"),
            Definition::TypeKind {
                type_kind_id: uint8_id,
            },
        );
        let int16_id = self.type_kinds.add_or_get(TypeKind::Int16);
        global_namespace.define(
            Identifier::new("Int16"),
            Definition::TypeKind {
                type_kind_id: int16_id,
            },
        );
        let uint16_id = self.type_kinds.add_or_get(TypeKind::UInt16);
        global_namespace.define(
            Identifier::new("UInt16"),
            Definition::TypeKind {
                type_kind_id: uint16_id,
            },
        );
        let int32_id = self.type_kinds.add_or_get(TypeKind::Int32);
        global_namespace.define(
            Identifier::new("Int32"),
            Definition::TypeKind {
                type_kind_id: int32_id,
            },
        );
        let uint32_id = self.type_kinds.add_or_get(TypeKind::UInt32);
        global_namespace.define(
            Identifier::new("UInt32"),
            Definition::TypeKind {
                type_kind_id: uint32_id,
            },
        );
        let int64_id = self.type_kinds.add_or_get(TypeKind::Int64);
        global_namespace.define(
            Identifier::new("Int64"),
            Definition::TypeKind {
                type_kind_id: int64_id,
            },
        );
        let uint64_id = self.type_kinds.add_or_get(TypeKind::UInt64);
        global_namespace.define(
            Identifier::new("UInt64"),
            Definition::TypeKind {
                type_kind_id: uint64_id,
            },
        );
        let float32_id = self.type_kinds.add_or_get(TypeKind::Float32);
        global_namespace.define(
            Identifier::new("Float32"),
            Definition::TypeKind {
                type_kind_id: float32_id,
            },
        );
        let float64_id = self.type_kinds.add_or_get(TypeKind::Float64);
        global_namespace.define(
            Identifier::new("Float64"),
            Definition::TypeKind {
                type_kind_id: float64_id,
            },
        );
        let tag_id = self.type_kinds.add_or_get(TypeKind::Tag);
        global_namespace.define(
            Identifier::new("Tag"),
            Definition::TypeKind {
                type_kind_id: tag_id,
            },
        );
    }

    fn find_string_view(&mut self) {
        self.string_view_type_kind_id = if let IdentifierLookupResult::Some(_, string_view_type) =
            self.lookup_identifier(
                Identifier::new("StringView"),
                LookupLocation::Namespace(GLOBAL_NAMESPACE_ID),
                LookupKind::Types,
                None,
            ) {
            string_view_type.type_kind_id
        } else {
            panic!("StringView type not found");
        };
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

    fn get_parser_node(&self, index: NodeIndex) -> &Node {
        &self.all_nodes[index.file_index][index.node_index]
    }

    fn get_typer_node(&self, index: NodeIndex) -> &TypedNode {
        &self.typed_nodes[index.node_index]
    }

    fn lookup_identifier_definition_or_index(
        &mut self,
        identifier: Identifier,
        location: LookupLocation,
    ) -> LookupResult {
        if let LookupLocation::Namespace(namespace_id) = location {
            let result = self.namespaces[namespace_id].lookup(&identifier);

            return LookupResult::from_namespace_lookup_result(namespace_id, result);
        }

        // The scope is not part of a namespace.
        if let Some(variable_type) = self.scope_environment.get(&identifier.name) {
            return LookupResult::Definition(
                GLOBAL_NAMESPACE_ID,
                Definition::Variable { variable_type },
            );
        };

        if let Some(type_kind_id) = self.scope_type_kind_environment.get(&identifier) {
            return LookupResult::Definition(
                GLOBAL_NAMESPACE_ID,
                Definition::TypeKind { type_kind_id },
            );
        };

        let LookupLocation::File(file_index) = location else {
            unreachable!()
        };

        let mut namespace_id = self.file_namespace_ids[file_index];
        loop {
            let result = self.namespaces[namespace_id].lookup(&identifier);

            if let NamespaceLookupResult::None = result {
                if let Some(parent_namespace_id) = self.namespaces[namespace_id].parent_id {
                    namespace_id = parent_namespace_id;
                    continue;
                } else {
                    break;
                }
            }

            return LookupResult::from_namespace_lookup_result(namespace_id, result);
        }

        let mut found_in_namespace_id = GLOBAL_NAMESPACE_ID;
        let mut result = NamespaceLookupResult::None;

        // Look in all namespaces this file is using to find the identifier.
        // If this symbol is in multiple of the used namespaces it is considered ambiguous.
        // We have no logical way to prioritize one used namespace over another, unlike with
        // the file's namespace and it's parent namespaces.
        for used_namespace_id in &self.file_used_namespace_ids[file_index] {
            let used_namespace_result = self.namespaces[*used_namespace_id].lookup(&identifier);

            if let NamespaceLookupResult::None = used_namespace_result {
                continue;
            }

            if let NamespaceLookupResult::None = result {
                found_in_namespace_id = *used_namespace_id;
                result = used_namespace_result;

                continue;
            }

            return LookupResult::Ambiguous;
        }

        LookupResult::from_namespace_lookup_result(found_in_namespace_id, result)
    }

    fn lookup_identifier(
        &mut self,
        identifier: Identifier,
        location: LookupLocation,
        kind: LookupKind,
        usage_index: Option<NodeIndex>,
    ) -> IdentifierLookupResult {
        let mut result = self.lookup_identifier_definition_or_index(identifier.clone(), location);

        if let LookupResult::DefinitionIndex(namespace_id, definition_index) = result {
            self.check_node_with_generic_args(
                definition_index,
                usage_index,
                identifier.generic_arg_type_kind_ids.clone(),
                namespace_id,
            );

            result = self.lookup_identifier_definition_or_index(identifier.clone(), location);
        }

        let LookupResult::Definition(namespace_id, definition) = result else {
            if let LookupResult::Ambiguous = result {
                return IdentifierLookupResult::Ambiguous;
            }

            return IdentifierLookupResult::None;
        };

        match definition {
            Definition::Function {
                type_kind_id,
                is_extern,
            } if kind == LookupKind::All => {
                let function_namespace_id = if is_extern {
                    GLOBAL_NAMESPACE_ID
                } else {
                    namespace_id
                };

                IdentifierLookupResult::Some(
                    function_namespace_id,
                    Type {
                        type_kind_id,
                        instance_kind: InstanceKind::Val,
                    },
                )
            }
            Definition::Variable { variable_type } if kind == LookupKind::All => {
                if identifier.generic_arg_type_kind_ids.is_some() {
                    return IdentifierLookupResult::None;
                }

                IdentifierLookupResult::Some(namespace_id, variable_type)
            }
            Definition::TypeKind { type_kind_id } => IdentifierLookupResult::Some(
                namespace_id,
                Type {
                    type_kind_id,
                    instance_kind: InstanceKind::Name,
                },
            ),
            _ => IdentifierLookupResult::None,
        }
    }

    fn lookup_identifier_name(
        &mut self,
        name: NodeIndex,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
        namespace_id: Option<usize>,
        kind: LookupKind,
    ) -> Option<(NodeIndex, Type)> {
        assert_matches!(
            NodeKind::Name { text: name_text },
            self.get_parser_node(name).kind.clone()
        );

        let identifier = Identifier {
            name: name_text,
            generic_arg_type_kind_ids,
        };

        let location = if let Some(namespace_id) = namespace_id {
            LookupLocation::Namespace(namespace_id)
        } else {
            LookupLocation::File(name.file_index)
        };

        let result = self.lookup_identifier(identifier, location, kind, Some(name));

        let IdentifierLookupResult::Some(namespace_id, name_type) = result else {
            if let IdentifierLookupResult::Ambiguous = result {
                self.error(
                    "abmiguous identifier, multiple definitions are in scope",
                    name,
                );
            } else if kind == LookupKind::Types {
                self.error("undefined type", name);
            } else {
                self.error("undefined identifier", name);
            }

            return None;
        };

        let typed_name = self.check_node_with_namespace(name, namespace_id);

        Some((typed_name, name_type))
    }

    fn get_file_namespace(&mut self, file_index: usize) -> &mut Namespace {
        &mut self.namespaces[self.file_namespace_ids[file_index]]
    }

    fn error(&mut self, message: &str, node_index: NodeIndex) {
        self.error_count += 1;
        self.get_parser_node(node_index)
            .start
            .error("Type", message, &self.files);
    }

    fn type_error(&mut self, message: &str) -> NodeIndex {
        self.type_error_at_parser_node(message, self.node_index_stack.last().copied().unwrap())
    }

    fn type_error_at_parser_node(&mut self, message: &str, node: NodeIndex) -> NodeIndex {
        self.error(message, node);

        self.add_node(TypedNode {
            node_kind: NodeKind::Error,
            node_type: None,
            namespace_id: None,
        })
    }

    pub fn check(&mut self, start_index: NodeIndex) {
        self.check_node(start_index);
    }

    fn check_optional_node(
        &mut self,
        index: Option<NodeIndex>,
        hint: Option<usize>,
    ) -> Option<NodeIndex> {
        index.map(|index| self.check_node_with_hint(index, hint))
    }

    fn check_node(&mut self, index: NodeIndex) -> NodeIndex {
        self.node_index_stack.push(index);
        let file_index = index.file_index;
        let file_namespace_id = self.file_namespace_ids[file_index];

        let typed_index = match self.get_parser_node(index).kind.clone() {
            NodeKind::TopLevel {
                functions,
                structs,
                enums,
                aliases,
                variable_declarations,
                definition_indices,
                ..
            } => self.top_level(
                functions,
                structs,
                enums,
                aliases,
                variable_declarations,
                definition_indices,
                file_index,
            ),
            NodeKind::ExternFunction { declaration } => {
                self.extern_function(declaration, file_index)
            }
            NodeKind::Alias {
                aliased_type_name,
                alias_name,
            } => self.alias(aliased_type_name, alias_name, file_index),
            NodeKind::Param { name, type_name } => self.param(name, type_name),
            NodeKind::Block { statements } => self.block(statements, None),
            NodeKind::Statement { inner } => self.statement(inner, None),
            NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                type_name,
                expression,
            } => self.variable_declaration(
                declaration_kind,
                name,
                type_name,
                expression,
                None,
                file_index,
            ),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression, None),
            NodeKind::BreakStatement => self.break_statement(),
            NodeKind::ContinueStatement => self.continue_statement(),
            NodeKind::DeferStatement { statement } => self.defer_statement(statement),
            NodeKind::CrashStatement { position } => self.crash_statement(position),
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
            NodeKind::Call { left, args, .. } => self.call(left, args),
            NodeKind::IndexAccess {
                left,
                expression,
                position,
            } => self.index_access(left, expression, position),
            NodeKind::FieldAccess {
                left,
                name,
                position,
            } => self.field_access(left, name, position),
            NodeKind::Cast { left, type_name } => self.cast(left, type_name),
            NodeKind::Name { text } => self.name(text, None),
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
                functions,
                generic_params,
                definition_indices,
                is_union,
            } => self.struct_definition(
                name,
                fields,
                functions,
                generic_params,
                definition_indices,
                is_union,
                None,
                None,
                file_index,
            ),
            NodeKind::EnumDefinition {
                name,
                variant_names,
            } => self.enum_definition(name, variant_names, file_index),
            NodeKind::Field { name, type_name } => self.field(name, type_name),
            NodeKind::FunctionDeclaration {
                name,
                params,
                generic_params,
                return_type_name,
            } => self.function_declaration(
                name,
                params,
                generic_params,
                return_type_name,
                false,
                file_namespace_id,
                file_index,
            ),
            NodeKind::Function {
                declaration,
                scoped_statement,
            } => self.function(
                declaration,
                scoped_statement,
                None,
                None,
                file_namespace_id,
                file_index,
            ),
            NodeKind::GenericSpecifier {
                left,
                generic_arg_type_names,
            } => self.generic_specifier(left, generic_arg_type_names),
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
            NodeKind::TypeNameFieldAccess { left, name } => self.type_name_field_access(left, name),
            NodeKind::TypeNameGenericSpecifier {
                left,
                generic_arg_type_names,
            } => self.type_name_generic_specifier(left, generic_arg_type_names),
            NodeKind::Using { .. } => type_error!(self, "cannot generate using node"),
            NodeKind::Error => type_error!(self, "cannot generate error node"),
        };

        self.node_index_stack.pop();

        typed_index
    }

    fn check_node_with_generic_args(
        &mut self,
        index: NodeIndex,
        usage_index: Option<NodeIndex>,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
        namespace_id: usize,
    ) -> Option<NodeIndex> {
        self.node_index_stack.push(index);
        let file_index = index.file_index;

        let typed_index = match self.get_parser_node(index).kind.clone() {
            NodeKind::StructDefinition {
                name,
                fields,
                functions,
                generic_params,
                definition_indices,
                is_union,
            } => Some(self.struct_definition(
                name,
                fields,
                functions,
                generic_params,
                definition_indices,
                is_union,
                generic_arg_type_kind_ids,
                usage_index,
                file_index,
            )),
            NodeKind::Function {
                declaration,
                scoped_statement,
            } => Some(self.function(
                declaration,
                scoped_statement,
                generic_arg_type_kind_ids,
                usage_index,
                namespace_id,
                file_index,
            )),
            _ if generic_arg_type_kind_ids.is_none() => {
                Some(self.check_node_with_namespace(index, namespace_id))
            }
            _ => None,
        };

        self.node_index_stack.pop();

        typed_index
    }

    fn check_node_with_hint(&mut self, index: NodeIndex, hint: Option<usize>) -> NodeIndex {
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

    fn check_node_with_namespace(&mut self, index: NodeIndex, namespace_id: usize) -> NodeIndex {
        self.node_index_stack.push(index);
        let file_index = index.file_index;

        let typed_index = match self.get_parser_node(index).kind.clone() {
            NodeKind::Name { text } => self.name(text, Some(namespace_id)),
            NodeKind::FunctionDeclaration {
                name,
                params,
                generic_params,
                return_type_name,
            } => self.function_declaration(
                name,
                params,
                generic_params,
                return_type_name,
                false,
                namespace_id,
                file_index,
            ),
            NodeKind::Function {
                declaration,
                scoped_statement,
            } => self.function(
                declaration,
                scoped_statement,
                None,
                None,
                namespace_id,
                file_index,
            ),
            NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                type_name,
                expression,
            } => self.variable_declaration(
                declaration_kind,
                name,
                type_name,
                expression,
                Some(namespace_id),
                file_index,
            ),
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

    fn check_functions(
        &mut self,
        functions: Arc<Vec<NodeIndex>>,
        namespace_id: usize,
        typed_functions: &mut Vec<NodeIndex>,
    ) {
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

            assert_matches!(
                NodeKind::FunctionDeclaration {
                    name,
                    generic_params,
                    ..
                },
                &self.get_parser_node(*declaration).kind
            );

            if !generic_params.is_empty() {
                continue;
            }

            assert_matches!(
                NodeKind::Name { text: name_text },
                self.get_parser_node(*name).kind.clone()
            );

            if self.namespaces[namespace_id].is_name_defined(name_text) {
                continue;
            }

            typed_functions.push(self.check_node_with_namespace(*function, namespace_id));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn top_level(
        &mut self,
        functions: Arc<Vec<NodeIndex>>,
        structs: Arc<Vec<NodeIndex>>,
        enums: Arc<Vec<NodeIndex>>,
        aliases: Arc<Vec<NodeIndex>>,
        variable_declarations: Arc<Vec<NodeIndex>>,
        definition_indices: Arc<DefinitionIndices>,
        file_index: usize,
    ) -> NodeIndex {
        let mut typed_aliases = Vec::new();
        for alias in aliases.iter() {
            let typed_alias = self.check_node(*alias);
            typed_aliases.push(typed_alias);
        }

        let mut typed_structs = Vec::new();
        for struct_definition in structs.iter() {
            assert_matches!(
                NodeKind::StructDefinition {
                    name,
                    generic_params,
                    ..
                },
                &self.get_parser_node(*struct_definition).kind
            );

            if !generic_params.is_empty() {
                continue;
            }

            assert_matches!(
                NodeKind::Name { text: name_text },
                self.get_parser_node(*name).kind.clone()
            );

            if self
                .get_file_namespace(file_index)
                .is_name_defined(name_text)
            {
                continue;
            }

            typed_structs.push(self.check_node(*struct_definition));
        }

        let mut typed_enums = Vec::new();
        for enum_definition in enums.iter() {
            assert_matches!(
                NodeKind::EnumDefinition { name, .. },
                &self.get_parser_node(*enum_definition).kind
            );
            assert_matches!(
                NodeKind::Name { text: name_text },
                self.get_parser_node(*name).kind.clone()
            );

            if self
                .get_file_namespace(file_index)
                .is_name_defined(name_text)
            {
                continue;
            }

            typed_enums.push(self.check_node(*enum_definition));
        }

        let mut typed_functions = Vec::new();
        self.check_functions(
            functions,
            self.file_namespace_ids[file_index],
            &mut typed_functions,
        );

        let mut typed_variable_declarations = Vec::new();
        for variable_declaration in variable_declarations.iter() {
            assert_matches!(
                NodeKind::VariableDeclaration { name, .. },
                &self.get_parser_node(*variable_declaration).kind
            );
            assert_matches!(
                NodeKind::Name { text: name_text },
                self.get_parser_node(*name).kind.clone()
            );

            if self
                .get_file_namespace(file_index)
                .is_name_defined(name_text)
            {
                continue;
            }

            let namespace_id = self.file_namespace_ids[file_index];
            typed_variable_declarations
                .push(self.check_node_with_namespace(*variable_declaration, namespace_id));
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::TopLevel {
                functions: Arc::new(typed_functions),
                structs: Arc::new(typed_structs),
                enums: Arc::new(typed_enums),
                aliases: Arc::new(typed_aliases),
                variable_declarations: Arc::new(typed_variable_declarations),
                usings: Arc::new(Vec::new()),
                definition_indices,
            },
            node_type: None,
            namespace_id: None,
        })
    }

    fn extern_function(&mut self, declaration: NodeIndex, file_index: usize) -> NodeIndex {
        assert_matches!(
            NodeKind::FunctionDeclaration {
                name,
                params,
                generic_params,
                return_type_name,
                ..
            },
            self.get_parser_node(declaration).kind.clone()
        );

        assert_matches!(
            NodeKind::Name { text: name_text },
            self.get_parser_node(name).kind.clone()
        );

        if !generic_params.is_empty() {
            type_error!(self, "extern function cannot be generic");
        }

        self.scope_environment.push(false);

        let namespace_id = self.file_namespace_ids[file_index];
        let typed_declaration = self.function_declaration(
            name,
            params,
            generic_params,
            return_type_name,
            true,
            GLOBAL_NAMESPACE_ID,
            file_index,
        );

        self.scope_environment.pop();

        let type_kind_id = assert_typed!(self, typed_declaration).type_kind_id;
        let identifier = Identifier::new(name_text.clone());

        self.namespaces[namespace_id].define(
            identifier,
            Definition::Function {
                type_kind_id,
                is_extern: true,
            },
        );

        let node_type = Some(Type {
            type_kind_id,
            instance_kind: InstanceKind::Val,
        });

        let index = self.add_node(TypedNode {
            node_kind: NodeKind::ExternFunction {
                declaration: typed_declaration,
            },
            node_type,
            namespace_id: Some(namespace_id),
        });

        self.typed_definitions.push(TypedDefinition {
            index,
            is_shallow: false,
        });

        index
    }

    fn alias(
        &mut self,
        aliased_type_name: NodeIndex,
        alias_name: NodeIndex,
        file_index: usize,
    ) -> NodeIndex {
        let typed_aliased_type_name = self.check_node(aliased_type_name);
        let aliased_type_name_type = assert_typed!(self, typed_aliased_type_name);
        let typed_alias_name = self.check_node(alias_name);

        assert_matches!(
            NodeKind::Name {
                text: alias_name_text,
            },
            self.get_typer_node(typed_alias_name).node_kind.clone()
        );

        let identifier = Identifier::new(alias_name_text);

        self.get_file_namespace(file_index).define(
            identifier,
            Definition::TypeKind {
                type_kind_id: aliased_type_name_type.type_kind_id,
            },
        );

        self.add_node(TypedNode {
            node_kind: NodeKind::Alias {
                aliased_type_name: typed_aliased_type_name,
                alias_name: typed_alias_name,
            },
            node_type: Some(aliased_type_name_type),
            namespace_id: Some(self.file_namespace_ids[file_index]),
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

        assert_matches!(
            NodeKind::Name { text: name_text },
            self.get_parser_node(name).kind.clone()
        );

        let node_type = Type {
            type_kind_id: type_name_type.type_kind_id,
            instance_kind: InstanceKind::Var,
        };
        self.scope_environment.insert(name_text, node_type.clone());

        self.add_node(TypedNode {
            node_kind: NodeKind::Param {
                name: typed_name,
                type_name: typed_type_name,
            },
            node_type: Some(node_type),
            namespace_id: None,
        })
    }

    fn block(&mut self, statements: Arc<Vec<NodeIndex>>, hint: Option<usize>) -> NodeIndex {
        if !self.was_block_already_opened {
            self.scope_environment.push(true);
        } else {
            self.was_block_already_opened = false;
        }

        let mut typed_statements = Vec::new();
        for statement in statements.iter() {
            typed_statements.push(self.check_node_with_hint(*statement, hint));
        }

        self.scope_environment.pop();

        self.add_node(TypedNode {
            node_kind: NodeKind::Block {
                statements: Arc::new(typed_statements),
            },
            node_type: None,
            namespace_id: None,
        })
    }

    fn statement(&mut self, inner: Option<NodeIndex>, hint: Option<usize>) -> NodeIndex {
        let typed_inner = self.check_optional_node(inner, hint);

        self.add_node(TypedNode {
            node_kind: NodeKind::Statement { inner: typed_inner },
            node_type: None,
            namespace_id: None,
        })
    }

    fn variable_declaration(
        &mut self,
        declaration_kind: DeclarationKind,
        name: NodeIndex,
        type_name: Option<NodeIndex>,
        expression: Option<NodeIndex>,
        namespace_id: Option<usize>,
        file_index: usize,
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

                if !self
                    .is_assignment_valid(variable_type.type_kind_id, expression_type.type_kind_id)
                {
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

        assert_matches!(
            NodeKind::Name { text: name_text },
            self.get_parser_node(name).kind.clone()
        );

        variable_type.instance_kind = match declaration_kind {
            DeclarationKind::Var => InstanceKind::Var,
            DeclarationKind::Val => InstanceKind::Val,
            DeclarationKind::Const => variable_type.instance_kind,
        };

        let index = self.add_node(TypedNode {
            node_kind: NodeKind::VariableDeclaration {
                declaration_kind,
                name: typed_name,
                type_name: typed_type_name,
                expression: typed_expression,
            },
            node_type: Some(variable_type.clone()),
            namespace_id,
        });

        if let Some(namespace_id) = namespace_id {
            let identifier = Identifier::new(name_text);

            self.namespaces[namespace_id]
                .define(identifier, Definition::Variable { variable_type });
            self.typed_definitions.push(TypedDefinition {
                index,
                is_shallow: Some(file_index) != self.file_index,
            });
        } else {
            self.scope_environment.insert(name_text, variable_type);
        }

        index
    }

    fn return_statement(
        &mut self,
        expression: Option<NodeIndex>,
        hint: Option<usize>,
    ) -> NodeIndex {
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
            namespace_id: None,
        })
    }

    fn break_statement(&mut self) -> NodeIndex {
        if self.loop_stack == 0 {
            type_error!(self, "break statements can only appear in loops");
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::BreakStatement,
            node_type: None,
            namespace_id: None,
        })
    }

    fn continue_statement(&mut self) -> NodeIndex {
        if self.loop_stack == 0 {
            type_error!(self, "continue statements can only appear in loops");
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::ContinueStatement,
            node_type: None,
            namespace_id: None,
        })
    }

    fn defer_statement(&mut self, statement: NodeIndex) -> NodeIndex {
        let typed_statement = self.check_node(statement);

        self.add_node(TypedNode {
            node_kind: NodeKind::DeferStatement {
                statement: typed_statement,
            },
            node_type: None,
            namespace_id: None,
        })
    }

    fn crash_statement(&mut self, position: Position) -> NodeIndex {
        self.add_node(TypedNode {
            node_kind: NodeKind::CrashStatement { position },
            node_type: None,
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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

        let (typed_from, typed_to, typed_by) = if self.is_node_numeric_literal(from)
            && self.is_node_numeric_literal(to)
            && by.is_some()
        {
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

        self.scope_environment.push(true);
        self.was_block_already_opened = true;

        assert_matches!(
            NodeKind::Name {
                text: iterator_text,
            },
            self.get_typer_node(typed_iterator).node_kind.clone()
        );

        let node_type = Type {
            type_kind_id: from_type.type_kind_id,
            instance_kind: InstanceKind::Var,
        };
        self.scope_environment
            .insert(iterator_text, node_type.clone());

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
            namespace_id: None,
        })
    }

    fn const_expression(&mut self, inner: NodeIndex, hint: Option<usize>) -> NodeIndex {
        self.check_const_node(inner, hint)
    }

    fn binary(
        &mut self,
        left: NodeIndex,
        op: Op,
        right: NodeIndex,
        hint: Option<usize>,
    ) -> NodeIndex {
        // Maximize the accuracy of hinting by type checking the right side first if the left side is a literal, so that we might have a more useful hint to give the left side.
        let (typed_left, left_type, typed_right, right_type) = if self.is_node_numeric_literal(left)
        {
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
        ) {
            if left_type.instance_kind != InstanceKind::Var {
                type_error!(self, "only vars can be assigned to");
            }

            if !self.is_assignment_valid(left_type.type_kind_id, right_type.type_kind_id) {
                type_error!(self, "type mismatch");
            }
        } else if left_type.type_kind_id != right_type.type_kind_id {
            type_error!(self, "type mismatch");
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
                    namespace_id: None,
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
                    namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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
                    namespace_id: None,
                })
            }
            Op::Not => {
                if self.type_kinds.get_by_id(right_type.type_kind_id) != TypeKind::Bool {
                    type_error!(self, "expected Bool");
                }

                self.add_node(TypedNode {
                    node_kind,
                    node_type: Some(right_type),
                    namespace_id: None,
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
                    namespace_id: None,
                })
            }
            _ => type_error!(self, "unknown unary prefix operator"),
        }
    }

    fn const_unary_prefix(
        &mut self,
        op: Op,
        right: NodeIndex,
        index: NodeIndex,
        hint: Option<usize>,
    ) -> NodeIndex {
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
            namespace_id: None,
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
                namespace_id: None,
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

        // A field access of an instance on the left means this is a method call, calling a function pointer
        // stored in an instance of a struct would require dereferencing the function pointer first.
        let mut is_method_call = false;
        let mut method_kind = MethodKind::None;

        let mut typed_caller = typed_left;

        if let NodeKind::GenericSpecifier { left, .. } = self.get_typer_node(typed_caller).node_kind
        {
            typed_caller = left;
        }

        if let NodeKind::FieldAccess { left, .. } = self.get_typer_node(typed_caller).node_kind {
            let left_type = assert_typed!(self, left);
            if left_type.instance_kind != InstanceKind::Name {
                is_method_call = true;

                if param_type_kind_ids.is_empty() {
                    type_error!(self, "static methods cannot be called on instances");
                }

                let Some(validated_method_kind) =
                    self.is_method_call_valid(param_type_kind_ids[0], left_type)
                else {
                    type_error!(
                        self,
                        "type mismatch, instance cannot be used as the first parameter of this method"
                    );
                };

                method_kind = validated_method_kind;
            }
        }

        let implicit_arg_count = if is_method_call { 1 } else { 0 };

        if args.len() + implicit_arg_count != param_type_kind_ids.len() {
            type_error!(self, "wrong number of arguments");
        }

        let mut typed_args = Vec::new();
        for (arg, param_type_kind_id) in args
            .iter()
            .zip(param_type_kind_ids.iter().skip(implicit_arg_count))
        {
            let typed_arg = self.check_node_with_hint(*arg, Some(*param_type_kind_id));
            typed_args.push(typed_arg);

            let arg_type = assert_typed!(self, typed_arg);

            if !self.is_assignment_valid(*param_type_kind_id, arg_type.type_kind_id) {
                type_error_at_parser_node!(self, "incorrect argument type", *arg);
            }
        }

        self.add_node(TypedNode {
            node_kind: NodeKind::Call {
                left: typed_left,
                args: Arc::new(typed_args),
                method_kind,
            },
            node_type: Some(Type {
                type_kind_id: return_type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
            namespace_id: None,
        })
    }

    fn index_access(
        &mut self,
        left: NodeIndex,
        expression: NodeIndex,
        position: Position,
    ) -> NodeIndex {
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

        if !self
            .type_kinds
            .get_by_id(expression_type.type_kind_id)
            .is_int()
        {
            type_error!(
                self,
                "expected index to have an integer type (Int, UInt, etc.)"
            );
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::IndexAccess {
                left: typed_left,
                expression: typed_expression,
                position,
            },
            node_type: Some(Type {
                type_kind_id: element_type_kind_id,
                instance_kind: left_type.instance_kind,
            }),
            namespace_id: None,
        })
    }

    fn field_access(&mut self, left: NodeIndex, name: NodeIndex, position: Position) -> NodeIndex {
        let typed_left = self.check_node(left);
        let left_type = assert_typed!(self, typed_left);
        let typed_name = self.check_node(name);

        assert_matches!(
            NodeKind::Name { text: name_text },
            &self.get_parser_node(name).kind
        );

        let node_kind = NodeKind::FieldAccess {
            left: typed_left,
            name: typed_name,
            position,
        };

        let mut is_tag_access = false;
        let (struct_type_kind_id, field_instance_kind) = match &self
            .type_kinds
            .get_by_id(left_type.type_kind_id)
        {
            TypeKind::Struct { is_union, .. } => {
                is_tag_access = left_type.instance_kind == InstanceKind::Name && *is_union;
                (left_type.type_kind_id, left_type.instance_kind.clone())
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
                    assert_matches!(
                        NodeKind::Name {
                            text: variant_name_text,
                        },
                        &self.get_typer_node(*variant_name).node_kind
                    );

                    if *variant_name_text == *name_text {
                        return self.add_node(TypedNode {
                            node_kind,
                            node_type: Some(Type {
                                type_kind_id: left_type.type_kind_id,
                                instance_kind: InstanceKind::Literal,
                            }),
                            namespace_id: None,
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
                    namespace_id: None,
                });
            }
            TypeKind::Namespace { namespace_id } => {
                let Some((typed_name, name_type)) =
                    self.lookup_identifier_name(name, None, Some(*namespace_id), LookupKind::All)
                else {
                    return self.add_node(TypedNode {
                        node_kind: NodeKind::Error,
                        node_type: None,
                        namespace_id: None,
                    });
                };

                return self.add_node(TypedNode {
                    node_kind: NodeKind::FieldAccess {
                        left: typed_left,
                        name: typed_name,
                        position,
                    },
                    node_type: Some(name_type),
                    namespace_id: None,
                });
            }
            _ => type_error!(
                self,
                "field access is only allowed on structs, enums, and pointers to structs"
            ),
        };

        let TypeKind::Struct {
            fields,
            namespace_id,
            ..
        } = &self.type_kinds.get_by_id(struct_type_kind_id)
        else {
            type_error!(self, "field access is only allowed on struct types");
        };

        if is_tag_access || field_instance_kind != InstanceKind::Name {
            for Field {
                name: field_name,
                type_kind_id: field_kind_id,
            } in fields.iter()
            {
                assert_matches!(
                    NodeKind::Name {
                        text: field_name_text,
                    },
                    &self.get_typer_node(*field_name).node_kind
                );

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
                        namespace_id: None,
                    });
                }

                return self.add_node(TypedNode {
                    node_kind,
                    node_type: Some(Type {
                        type_kind_id: *field_kind_id,
                        instance_kind: field_instance_kind,
                    }),
                    namespace_id: None,
                });
            }
        }

        let Some((typed_name, name_type)) =
            self.lookup_identifier_name(name, None, Some(*namespace_id), LookupKind::All)
        else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
                namespace_id: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::FieldAccess {
                left: typed_left,
                name: typed_name,
                position,
            },
            node_type: Some(name_type),
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
        })
    }

    fn name(&mut self, text: Arc<str>, namespace_id: Option<usize>) -> NodeIndex {
        self.add_node(TypedNode {
            node_kind: NodeKind::Name { text },
            node_type: None,
            namespace_id,
        })
    }

    fn identifier(&mut self, name: NodeIndex) -> NodeIndex {
        let Some((typed_name, name_type)) =
            self.lookup_identifier_name(name, None, None, LookupKind::All)
        else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
                namespace_id: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::Identifier { name: typed_name },
            node_type: Some(name_type),
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
        })
    }

    fn const_int_literal(
        &mut self,
        text: Arc<str>,
        index: NodeIndex,
        hint: Option<usize>,
    ) -> NodeIndex {
        let typed_const = self.check_node_with_hint(index, hint);
        let const_type = assert_typed!(self, typed_const);

        let value = if self
            .type_kinds
            .get_by_id(const_type.type_kind_id)
            .is_unsigned()
        {
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
            namespace_id: None,
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
            namespace_id: None,
        })
    }

    fn const_float_literal(
        &mut self,
        text: Arc<str>,
        index: NodeIndex,
        hint: Option<usize>,
    ) -> NodeIndex {
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
            namespace_id: None,
        })
    }

    fn string_literal(&mut self, text: Arc<String>) -> NodeIndex {
        self.add_node(TypedNode {
            node_kind: NodeKind::StringLiteral { text },
            node_type: Some(Type {
                type_kind_id: self.string_view_type_kind_id,
                instance_kind: InstanceKind::Literal,
            }),
            namespace_id: None,
        })
    }

    fn const_string_literal(&mut self, text: Arc<String>, index: NodeIndex) -> NodeIndex {
        let typed_const = self.check_node(index);
        let const_type = assert_typed!(self, typed_const);

        self.add_node(TypedNode {
            node_kind: NodeKind::StringLiteral { text: text.clone() },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(ConstValue::String { value: text }),
            }),
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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
                assert_matches!(
                    NodeKind::FieldLiteral {
                        name: field_name,
                        ..
                    },
                    &self.get_parser_node(field_literals[0]).kind
                );

                assert_matches!(
                    NodeKind::Name {
                        text: field_name_text,
                    },
                    &self.get_parser_node(*field_name).kind
                );

                let Some(expected_field_index) =
                    get_field_index_by_name(&self.typed_nodes, field_name_text, &expected_fields)
                else {
                    type_error_at_parser_node!(
                        self,
                        "union doesn't contain a field with this name",
                        field_literals[0]
                    );
                };

                let expected_type_kind_id = expected_fields[expected_field_index].type_kind_id;

                let typed_field_literal =
                    self.check_node_with_hint(field_literals[0], Some(expected_type_kind_id));
                typed_field_literals.push(typed_field_literal);

                let field_literal_type = assert_typed!(self, typed_field_literal);

                if !self.is_assignment_valid(field_literal_type.type_kind_id, expected_type_kind_id)
                {
                    type_error_at_parser_node!(self, "incorrect field type", field_literals[0]);
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

                assert_matches!(
                    NodeKind::FieldLiteral {
                        name: field_name,
                        ..
                    },
                    &self.get_typer_node(typed_field_literal).node_kind
                );
                assert_matches!(
                    NodeKind::Name {
                        text: field_name_text
                    },
                    &self.get_typer_node(*field_name).node_kind
                );
                assert_matches!(
                    NodeKind::Name {
                        text: expected_field_name_text
                    },
                    &self.get_typer_node(expected_field.name).node_kind
                );

                if field_name_text != expected_field_name_text {
                    type_error_at_parser_node!(self, "incorrect field name", *field);
                }

                if !self.is_assignment_valid(
                    field_literal_type.type_kind_id,
                    expected_field.type_kind_id,
                ) {
                    type_error_at_parser_node!(self, "incorrect field type", *field);
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
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
        })
    }

    fn const_type_size(
        &mut self,
        type_name: NodeIndex,
        index: NodeIndex,
        hint: Option<usize>,
    ) -> NodeIndex {
        let typed_type_name = self.check_node(type_name);
        let type_name_type = assert_typed!(self, typed_type_name);
        let typed_const = self.check_node_with_hint(index, hint);
        let const_type = assert_typed!(self, typed_const);

        let native_size = mem::size_of::<NodeIndex>() as u64;

        let value = match self.type_kinds.get_by_id(type_name_type.type_kind_id) {
            TypeKind::Int => native_size,
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

        let const_value = if self
            .type_kinds
            .get_by_id(const_type.type_kind_id)
            .is_unsigned()
        {
            ConstValue::UInt { value }
        } else {
            ConstValue::Int {
                value: value as i64,
            }
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeSize {
                type_name: typed_type_name,
            },
            node_type: Some(Type {
                type_kind_id: const_type.type_kind_id,
                instance_kind: InstanceKind::Const(const_value),
            }),
            namespace_id: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn struct_definition(
        &mut self,
        name: NodeIndex,
        fields: Arc<Vec<NodeIndex>>,
        functions: Arc<Vec<NodeIndex>>,
        generic_params: Arc<Vec<NodeIndex>>,
        definition_indices: Arc<DefinitionIndices>,
        is_union: bool,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
        usage_index: Option<NodeIndex>,
        file_index: usize,
    ) -> NodeIndex {
        if !generic_params.is_empty() && generic_arg_type_kind_ids.is_none() {
            type_error_at_parser_node!(
                self,
                "generic type requires generic arguments",
                usage_index.unwrap()
            );
        }

        assert_matches!(
            NodeKind::Name { text: name_text },
            self.get_parser_node(name).kind.clone()
        );

        let typed_name = self.check_node(name);
        let identifier = Identifier {
            name: name_text.clone(),
            generic_arg_type_kind_ids: generic_arg_type_kind_ids.clone(),
        };

        let type_kind_id = self.type_kinds.add_placeholder();
        self.get_file_namespace(file_index)
            .define(identifier, Definition::TypeKind { type_kind_id });

        self.scope_type_kind_environment.push(false);
        let mut generic_args = Vec::new();

        if let Some(generic_arg_type_kind_ids) = generic_arg_type_kind_ids.clone() {
            if generic_arg_type_kind_ids.len() != generic_params.len() {
                type_error_at_parser_node!(
                    self,
                    "incorrect number of generic arguments",
                    usage_index.unwrap()
                );
            }

            for i in 0..generic_arg_type_kind_ids.len() {
                assert_matches!(
                    NodeKind::Name { text: param_text },
                    self.get_parser_node(generic_params[i]).kind.clone()
                );

                self.scope_type_kind_environment.insert(
                    Identifier::new(param_text.clone()),
                    generic_arg_type_kind_ids[i],
                );

                generic_args.push(NamespaceGenericArg {
                    param_name: param_text,
                    type_kind_id: generic_arg_type_kind_ids[i],
                })
            }
        }

        let mut typed_fields = Vec::new();
        let mut type_kind_fields = Vec::new();

        for field in fields.iter() {
            let typed_field = self.check_node(*field);
            typed_fields.push(typed_field);

            let field_type_kind_id = assert_typed!(self, typed_field).type_kind_id;

            assert_matches!(
                NodeKind::Field {
                    name: field_name,
                    ..
                },
                self.get_parser_node(*field).kind
            );

            let typed_field_name = self.check_node(field_name);

            type_kind_fields.push(Field {
                name: typed_field_name,
                type_kind_id: field_type_kind_id,
            })
        }

        self.scope_type_kind_environment.pop();

        let namespace_id = self.namespaces.len();
        let mut namespace = Namespace::new(
            name_text.clone(),
            Some(type_kind_id),
            generic_args,
            Some(self.file_namespace_ids[file_index]),
        );
        namespace.extend_definition_indices_unchecked(&definition_indices);
        self.namespaces.push(namespace);

        self.type_kinds.replace_placeholder(
            type_kind_id,
            TypeKind::Struct {
                name: typed_name,
                fields: Arc::new(type_kind_fields),
                is_union,
                namespace_id,
            },
        );

        let mut typed_functions = Vec::new();
        self.check_functions(functions, namespace_id, &mut typed_functions);

        let index = self.add_node(TypedNode {
            node_kind: NodeKind::StructDefinition {
                name: typed_name,
                fields: Arc::new(typed_fields),
                functions: Arc::new(typed_functions),
                generic_params: Arc::new(Vec::new()),
                definition_indices,
                is_union,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Name,
            }),
            namespace_id: Some(self.file_namespace_ids[file_index]),
        });

        self.typed_definitions.push(TypedDefinition {
            index,
            is_shallow: Some(file_index) != self.file_index,
        });

        index
    }

    fn enum_definition(
        &mut self,
        name: NodeIndex,
        variant_names: Arc<Vec<NodeIndex>>,
        file_index: usize,
    ) -> NodeIndex {
        assert_matches!(
            NodeKind::Name { text: name_text },
            self.get_parser_node(name).kind.clone()
        );

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

        let identifier = Identifier::new(name_text);
        self.get_file_namespace(file_index)
            .define(identifier, Definition::TypeKind { type_kind_id });

        let index = self.add_node(TypedNode {
            node_kind: NodeKind::EnumDefinition {
                name: typed_name,
                variant_names: typed_variant_names,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Name,
            }),
            namespace_id: Some(self.file_namespace_ids[file_index]),
        });

        self.typed_definitions.push(TypedDefinition {
            index,
            is_shallow: false,
        });

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
            namespace_id: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn function_declaration(
        &mut self,
        name: NodeIndex,
        params: Arc<Vec<NodeIndex>>,
        generic_params: Arc<Vec<NodeIndex>>,
        return_type_name: NodeIndex,
        is_extern: bool,
        namespace_id: usize,
        file_index: usize,
    ) -> NodeIndex {
        let typed_name = self.check_node_with_namespace(name, namespace_id);

        let mut typed_params = Vec::new();
        let mut param_type_kind_ids = Vec::new();
        for param in params.iter() {
            let typed_param = self.check_node(*param);
            typed_params.push(typed_param);

            let param_type = assert_typed!(self, typed_param);
            param_type_kind_ids.push(param_type.type_kind_id);
        }
        let typed_params = Arc::new(typed_params);
        let param_type_kind_ids = Arc::new(param_type_kind_ids);

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

        let index = self.add_node(TypedNode {
            node_kind: NodeKind::FunctionDeclaration {
                name: typed_name,
                params: typed_params,
                generic_params: Arc::new(typed_generic_params),
                return_type_name: typed_return_type_name,
            },
            node_type: Some(Type {
                type_kind_id,
                instance_kind: InstanceKind::Name,
            }),
            namespace_id: Some(namespace_id),
        });

        assert_matches!(
            NodeKind::Name { text: name_text },
            &self.get_typer_node(typed_name).node_kind
        );

        if name_text.as_ref() == "Main" {
            if is_extern {
                type_error!(self, "Main function cannot be defined as an extern");
            }

            if !generic_params.is_empty() {
                type_error!(self, "Main function cannot be generic");
            }

            if Some(file_index) == self.file_index {
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

                let TypeKind::Pointer {
                    inner_type_kind_id,
                    is_inner_mutable: false,
                } = self.type_kinds.get_by_id(param_type_kind_ids[1])
                else {
                    type_error!(
                        self,
                        "expected second argument of Main to be a *val *val Char"
                    );
                };

                let TypeKind::Pointer {
                    inner_type_kind_id,
                    is_inner_mutable: false,
                } = self.type_kinds.get_by_id(inner_type_kind_id)
                else {
                    type_error!(
                        self,
                        "expected second argument of Main to be a *val *val Char"
                    );
                };

                if self.type_kinds.get_by_id(inner_type_kind_id) != TypeKind::Char {
                    type_error!(
                        self,
                        "expected second argument of Main to be *val *val Char"
                    );
                }
            }
        }

        index
    }

    fn function(
        &mut self,
        declaration: NodeIndex,
        scoped_statement: NodeIndex,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
        usage_index: Option<NodeIndex>,
        namespace_id: usize,
        file_index: usize,
    ) -> NodeIndex {
        let pre_error_count = self.error_count;

        let is_generic = generic_arg_type_kind_ids.is_some();

        let index = self.function_impl(
            declaration,
            scoped_statement,
            generic_arg_type_kind_ids,
            usage_index,
            namespace_id,
            file_index,
        );

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
        usage_index: Option<NodeIndex>,
        namespace_id: usize,
        file_index: usize,
    ) -> NodeIndex {
        assert_matches!(
            NodeKind::FunctionDeclaration {
                name,
                generic_params,
                ..
            },
            self.get_parser_node(declaration).kind.clone()
        );

        assert_matches!(
            NodeKind::Name { text: name_text },
            self.get_parser_node(name).kind.clone()
        );

        if !generic_params.is_empty() && generic_arg_type_kind_ids.is_none() {
            type_error_at_parser_node!(
                self,
                "generic function requires generic arguments",
                usage_index.unwrap()
            );
        }

        self.scope_type_kind_environment.push(false);
        self.scope_environment.push(false);

        for generic_arg in &self.namespaces[namespace_id].generic_args {
            self.scope_type_kind_environment.insert(
                Identifier::new(generic_arg.param_name.clone()),
                generic_arg.type_kind_id,
            );
        }

        if let Some(generic_arg_type_kind_ids) = generic_arg_type_kind_ids.clone() {
            if generic_arg_type_kind_ids.len() != generic_params.len() {
                type_error_at_parser_node!(
                    self,
                    "incorrect number of generic arguments",
                    usage_index.unwrap()
                );
            }

            for i in 0..generic_arg_type_kind_ids.len() {
                assert_matches!(
                    NodeKind::Name { text: param_text },
                    self.get_parser_node(generic_params[i]).kind.clone()
                );

                self.scope_type_kind_environment
                    .insert(Identifier::new(param_text), generic_arg_type_kind_ids[i]);
            }
        }

        let typed_declaration = self.check_node_with_namespace(declaration, namespace_id);
        let declaration_type = assert_typed!(self, typed_declaration);
        let identifier = Identifier {
            name: name_text.clone(),
            generic_arg_type_kind_ids: generic_arg_type_kind_ids.clone(),
        };
        self.namespaces[namespace_id].define(
            identifier,
            Definition::Function {
                type_kind_id: declaration_type.type_kind_id,
                is_extern: false,
            },
        );

        let is_deep_check = Some(file_index) == self.file_index
            || generic_arg_type_kind_ids.is_some()
            || !self.namespaces[namespace_id].generic_args.is_empty();

        let typed_scoped_statement = if is_deep_check {
            let TypeKind::Function {
                return_type_kind_id: expected_return_type_kind_id,
                ..
            } = self.type_kinds.get_by_id(declaration_type.type_kind_id)
            else {
                type_error!(self, "invalid function type");
            };

            let typed_scoped_statement =
                self.check_node_with_hint(scoped_statement, Some(expected_return_type_kind_id));

            let is_return_type_valid = match self.compare_typed_statement_return_type(
                typed_scoped_statement,
                expected_return_type_kind_id,
            ) {
                ReturnTypeComparison::Matches => true,
                ReturnTypeComparison::DoesntReturn => {
                    self.type_kinds.get_by_id(expected_return_type_kind_id) == TypeKind::Void
                }
                ReturnTypeComparison::DoesntMatch => false,
            };

            if !is_return_type_valid {
                type_error!(
                    self,
                    "function does not return the correct type on all execution paths"
                );
            }

            typed_scoped_statement
        } else {
            self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
                namespace_id: None,
            })
        };

        self.scope_environment.pop();
        self.scope_type_kind_environment.pop();

        let index = self.add_node(TypedNode {
            node_kind: NodeKind::Function {
                declaration: typed_declaration,
                scoped_statement: typed_scoped_statement,
            },
            node_type: Some(Type {
                type_kind_id: declaration_type.type_kind_id,
                instance_kind: InstanceKind::Val,
            }),
            namespace_id: Some(namespace_id),
        });

        self.typed_definitions.push(TypedDefinition {
            index,
            is_shallow: !is_deep_check,
        });

        index
    }

    fn generic_specifier(
        &mut self,
        left: NodeIndex,
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

        let (typed_left, name_type) = if let NodeKind::FieldAccess {
            left,
            name,
            position,
        } = self.get_parser_node(left).kind
        {
            let typed_left = self.check_node(left);
            let left_type = assert_typed!(self, typed_left);

            let (dereferenced_left_type_kind_id, _) = self
                .type_kinds
                .dereference_type_kind_id(left_type.type_kind_id);

            let (TypeKind::Namespace { namespace_id } | TypeKind::Struct { namespace_id, .. }) =
                self.type_kinds.get_by_id(dereferenced_left_type_kind_id)
            else {
                type_error!(self, "expected namespace before field access in type name");
            };

            let Some((typed_name, name_type)) = self.lookup_identifier_name(
                name,
                Some(generic_arg_type_kind_ids),
                Some(namespace_id),
                LookupKind::All,
            ) else {
                return self.add_node(TypedNode {
                    node_kind: NodeKind::Error,
                    node_type: None,
                    namespace_id: None,
                });
            };

            (
                self.add_node(TypedNode {
                    node_kind: NodeKind::FieldAccess {
                        left: typed_left,
                        name: typed_name,
                        position,
                    },
                    node_type: None,
                    namespace_id: None,
                }),
                name_type,
            )
        } else {
            let NodeKind::Identifier { name } = &self.get_parser_node(left).kind else {
                type_error!(self, "expected identifier before generic specifier");
            };

            let Some((typed_name, name_type)) = self.lookup_identifier_name(
                *name,
                Some(generic_arg_type_kind_ids),
                None,
                LookupKind::All,
            ) else {
                return self.add_node(TypedNode {
                    node_kind: NodeKind::Error,
                    node_type: None,
                    namespace_id: None,
                });
            };

            (
                self.add_node(TypedNode {
                    node_kind: NodeKind::Identifier { name: typed_name },
                    node_type: None,
                    namespace_id: None,
                }),
                name_type,
            )
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::GenericSpecifier {
                left: typed_left,
                generic_arg_type_names: Arc::new(typed_generic_arg_type_names),
            },
            node_type: Some(name_type),
            namespace_id: None,
        })
    }

    fn type_name(&mut self, name: NodeIndex) -> NodeIndex {
        let Some((typed_name, name_type)) =
            self.lookup_identifier_name(name, None, None, LookupKind::Types)
        else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
                namespace_id: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeName { name: typed_name },
            node_type: Some(name_type),
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
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
            namespace_id: None,
        })
    }

    fn type_name_field_access(&mut self, left: NodeIndex, name: NodeIndex) -> NodeIndex {
        let typed_left = self.check_node(left);
        let left_type = assert_typed!(self, typed_left);

        let TypeKind::Namespace { namespace_id } =
            &self.type_kinds.get_by_id(left_type.type_kind_id)
        else {
            type_error!(self, "expected namespace before field access in type name");
        };

        let Some((typed_name, name_type)) =
            self.lookup_identifier_name(name, None, Some(*namespace_id), LookupKind::Types)
        else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
                namespace_id: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeNameFieldAccess {
                left: typed_left,
                name: typed_name,
            },
            node_type: Some(name_type),
            namespace_id: None,
        })
    }

    fn type_name_generic_specifier(
        &mut self,
        left: NodeIndex,
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

        let (name, namespace_id) =
            if let NodeKind::TypeNameFieldAccess { left, name } = self.get_parser_node(left).kind {
                let typed_left = self.check_node(left);
                let left_type = assert_typed!(self, typed_left);

                let TypeKind::Namespace { namespace_id } =
                    self.type_kinds.get_by_id(left_type.type_kind_id)
                else {
                    type_error!(self, "expected namespace before field access in type name");
                };

                (name, Some(namespace_id))
            } else {
                let NodeKind::TypeName { name } = &self.get_parser_node(left).kind else {
                    type_error!(self, "expected type name before generic specifier");
                };

                (*name, None)
            };

        let Some((typed_name, name_type)) = self.lookup_identifier_name(
            name,
            Some(generic_arg_type_kind_ids),
            namespace_id,
            LookupKind::Types,
        ) else {
            return self.add_node(TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
                namespace_id: None,
            });
        };

        self.add_node(TypedNode {
            node_kind: NodeKind::TypeNameGenericSpecifier {
                left: typed_name,
                generic_arg_type_names: Arc::new(Vec::new()),
            },
            node_type: Some(name_type),
            namespace_id: None,
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

    fn is_return_type_valid(
        &self,
        return_type: &Option<Type>,
        expected_type_kind_id: usize,
    ) -> bool {
        if let Some(return_type) = return_type {
            return_type.type_kind_id == expected_type_kind_id
        } else {
            self.type_kinds.get_by_id(expected_type_kind_id) == TypeKind::Void
        }
    }

    // Returns None if the statement doesn't return any type, otherwise returns whether or not
    // the statement's return type is valid based on the desired type.
    fn compare_typed_statement_return_type(
        &self,
        statement: NodeIndex,
        expected_type_kind_id: usize,
    ) -> ReturnTypeComparison {
        match &self.get_typer_node(statement).node_kind {
            NodeKind::Block { statements } => {
                let mut result = ReturnTypeComparison::DoesntReturn;

                for statement in statements.iter() {
                    match self
                        .compare_typed_statement_return_type(*statement, expected_type_kind_id)
                    {
                        ReturnTypeComparison::Matches => result = ReturnTypeComparison::Matches,
                        ReturnTypeComparison::DoesntMatch => {
                            return ReturnTypeComparison::DoesntMatch
                        }
                        ReturnTypeComparison::DoesntReturn => {}
                    }
                }

                result
            }
            NodeKind::Statement { inner: Some(inner) } => {
                self.compare_typed_statement_return_type(*inner, expected_type_kind_id)
            }
            NodeKind::ReturnStatement { expression } => {
                let return_type = if let Some(expression) = expression {
                    &self.get_typer_node(*expression).node_type
                } else {
                    &None
                };

                if self.is_return_type_valid(return_type, expected_type_kind_id) {
                    ReturnTypeComparison::Matches
                } else {
                    ReturnTypeComparison::DoesntMatch
                }
            }
            NodeKind::DeferStatement { statement } => {
                self.compare_typed_statement_return_type(*statement, expected_type_kind_id)
            }
            NodeKind::IfStatement {
                scoped_statement,
                next: Some(next),
                ..
            } => self.compare_both_typed_statement_return_types(
                *scoped_statement,
                *next,
                expected_type_kind_id,
            ),
            NodeKind::SwitchStatement { case_statement, .. } => {
                self.compare_typed_statement_return_type(*case_statement, expected_type_kind_id)
            }
            NodeKind::CaseStatement {
                scoped_statement,
                next: Some(next),
                ..
            } => self.compare_both_typed_statement_return_types(
                *scoped_statement,
                *next,
                expected_type_kind_id,
            ),
            NodeKind::IfStatement {
                scoped_statement,
                next: None,
                ..
            } => self.compare_typed_statement_return_type(*scoped_statement, expected_type_kind_id),
            NodeKind::WhileLoop {
                scoped_statement, ..
            } => self.compare_typed_statement_return_type(*scoped_statement, expected_type_kind_id),
            NodeKind::ForLoop {
                scoped_statement, ..
            } => self.compare_typed_statement_return_type(*scoped_statement, expected_type_kind_id),
            _ => ReturnTypeComparison::DoesntReturn,
        }
    }

    fn compare_both_typed_statement_return_types(
        &self,
        statement: NodeIndex,
        next: NodeIndex,
        expected_type_kind_id: usize,
    ) -> ReturnTypeComparison {
        let statement_result =
            self.compare_typed_statement_return_type(statement, expected_type_kind_id);
        let next_result = self.compare_typed_statement_return_type(next, expected_type_kind_id);

        if statement_result == ReturnTypeComparison::DoesntMatch
            || next_result == ReturnTypeComparison::DoesntMatch
        {
            return ReturnTypeComparison::DoesntMatch;
        }

        if statement_result == ReturnTypeComparison::Matches
            || next_result == ReturnTypeComparison::Matches
        {
            return ReturnTypeComparison::Matches;
        }

        ReturnTypeComparison::DoesntReturn
    }

    fn is_node_numeric_literal(&self, index: NodeIndex) -> bool {
        matches!(
            self.get_parser_node(index).kind,
            NodeKind::IntLiteral { .. } | NodeKind::FloatLiteral { .. }
        )
    }

    fn is_assignment_valid(&self, to_type_kind_id: usize, from_type_kind_id: usize) -> bool {
        if to_type_kind_id == from_type_kind_id {
            return true;
        }

        match self.type_kinds.get_by_id(to_type_kind_id) {
            TypeKind::Array {
                element_type_kind_id: to_element_type_kind_id,
                element_count: to_element_count,
            } => {
                let TypeKind::Array {
                    element_type_kind_id: from_element_type_kind_id,
                    element_count: from_element_count,
                } = self.type_kinds.get_by_id(from_type_kind_id)
                else {
                    return false;
                };

                from_element_count == to_element_count
                    && self.is_assignment_valid(to_element_type_kind_id, from_element_type_kind_id)
            }
            // It's possible to assign either a val or var pointer to a val pointer, because assigning a var pointer to a val pointer
            // just reduces the number of things you can do with the type, it doesn't let you modify immutable values like going the other direction would.
            TypeKind::Pointer {
                inner_type_kind_id: to_inner_type_kind_id,
                is_inner_mutable: false,
            } => {
                let TypeKind::Pointer {
                    inner_type_kind_id: from_inner_type_kind_id,
                    ..
                } = self.type_kinds.get_by_id(from_type_kind_id)
                else {
                    return false;
                };

                from_inner_type_kind_id == to_inner_type_kind_id
            }
            _ => false,
        }
    }

    fn is_method_call_valid(
        &self,
        param_type_kind_id: usize,
        instance_type: Type,
    ) -> Option<MethodKind> {
        if self.is_assignment_valid(param_type_kind_id, instance_type.type_kind_id) {
            return Some(MethodKind::ByValue);
        }

        if let TypeKind::Pointer {
            inner_type_kind_id,
            is_inner_mutable,
        } = self.type_kinds.get_by_id(param_type_kind_id)
        {
            if is_inner_mutable && instance_type.instance_kind != InstanceKind::Var {
                return None;
            }

            if self.is_assignment_valid(inner_type_kind_id, instance_type.type_kind_id) {
                return Some(MethodKind::ByReference);
            }
        }

        None
    }
}
