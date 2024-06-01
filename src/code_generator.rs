use core::panic;
use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};

use crate::{
    assert_matches,
    const_value::ConstValue,
    emitter::Emitter,
    emitter_stack::EmitterStack,
    file_data::FileData,
    name_generator::NameGenerator,
    namespace::Namespace,
    parser::{DeclarationKind, MethodKind, NodeIndex, NodeKind, Op},
    type_kinds::{
        get_field_index_by_name, PrimitiveType, TypeKind, TypeKinds, BOOL_TYPE_KIND_ID,
        CHAR_TYPE_KIND_ID, FLOAT32_TYPE_KIND_ID, FLOAT64_TYPE_KIND_ID, INT16_TYPE_KIND_ID,
        INT32_TYPE_KIND_ID, INT64_TYPE_KIND_ID, INT8_TYPE_KIND_ID, INT_TYPE_KIND_ID,
        UINT16_TYPE_KIND_ID, UINT32_TYPE_KIND_ID, UINT64_TYPE_KIND_ID, UINT8_TYPE_KIND_ID,
        UINT_TYPE_KIND_ID,
    },
    typer::{InstanceKind, Type, TypedNode, GLOBAL_NAMESPACE_ID},
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum EmitterKind {
    Header,
    TypePrototype,
    TypeDefinition,
    FunctionPrototype,
    GlobalVariable,
    Top,
    Body,
}

fn reserved_names() -> &'static HashSet<Arc<str>> {
    static NAMES: OnceLock<HashSet<Arc<str>>> = OnceLock::new();
    NAMES.get_or_init(|| {
        [
            "alignas",
            "alignof",
            "auto",
            "bool",
            "break",
            "case",
            "char",
            "const",
            "constexpr",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "false",
            "float",
            "for",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "main",
            "nullptr",
            "register",
            "restrict",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "static_assert",
            "struct",
            "switch",
            "thread_local",
            "true",
            "typedef",
            "typeof",
            "typeof_unqual",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
            "_Alignas",
            "_Alignof",
            "_Atomic",
            "_BitInt",
            "_Bool",
            "_Complex",
            "_Decimal128",
            "_Decimal32",
            "_Decimal64",
            "_Generic",
            "_Imaginary",
            "_Noreturn",
            "_Static_assert",
            "_Thread_local",
        ]
        .iter()
        .map(|s| Arc::from(*s))
        .collect()
    })
}

#[derive(Clone, Copy, Debug)]
struct LoopDepth {
    index: NodeIndex,
    depth: usize,
    was_label_used: bool,
}

pub struct CodeGenerator {
    typed_nodes: Vec<TypedNode>,
    type_kinds: TypeKinds,
    namespaces: Vec<Namespace>,
    span_char_type_kind_id: usize,
    main_function_declaration: Option<NodeIndex>,
    typed_definitions: Vec<NodeIndex>,
    name_generator: NameGenerator,
    files: Arc<Vec<FileData>>,

    pub header_emitter: Emitter,
    pub type_prototype_emitter: Emitter,
    pub type_definition_emitter: Emitter,
    pub function_prototype_emitter: Emitter,
    pub global_variable_emitter: Emitter,
    pub body_emitters: EmitterStack,

    function_declaration_needing_init: Option<NodeIndex>,
    loop_depth_stack: Vec<LoopDepth>,
    switch_depth_stack: Vec<usize>,
    node_index_stack: Vec<NodeIndex>,
    is_debug_mode: bool,
    is_unsafe_mode: bool,
}

impl CodeGenerator {
    pub fn new(
        typed_nodes: Vec<TypedNode>,
        type_kinds: TypeKinds,
        namespaces: Vec<Namespace>,
        span_char_type_kind_id: usize,
        main_function_declaration: Option<NodeIndex>,
        typed_definitions: Vec<NodeIndex>,
        name_generator: NameGenerator,
        files: Arc<Vec<FileData>>,
        is_debug_mode: bool,
        is_unsafe_mode: bool,
    ) -> Self {
        let mut code_generator = Self {
            typed_nodes,
            type_kinds,
            namespaces,
            span_char_type_kind_id,
            main_function_declaration,
            typed_definitions,
            name_generator,
            files,
            header_emitter: Emitter::new(0),
            type_prototype_emitter: Emitter::new(0),
            type_definition_emitter: Emitter::new(0),
            function_prototype_emitter: Emitter::new(0),
            global_variable_emitter: Emitter::new(0),
            body_emitters: EmitterStack::new(),
            function_declaration_needing_init: None,
            loop_depth_stack: Vec::new(),
            switch_depth_stack: Vec::new(),
            node_index_stack: Vec::new(),
            is_debug_mode,
            is_unsafe_mode,
        };

        code_generator.emitln("#include <stdint.h>", EmitterKind::Header);
        code_generator.emitln("#include <stdbool.h>", EmitterKind::Header);
        code_generator.newline(EmitterKind::Header);
        code_generator.emitln(
            "void Internal__ErrorTrace(char const *message, intptr_t skipCount);",
            EmitterKind::Header,
        );
        code_generator.emitln(
            "void *memmove(void *dst, const void *src, size_t size);",
            EmitterKind::Header,
        );
        code_generator.emitln(
            "int memcmp(const void *ptr1, const void *ptr2, size_t num);",
            EmitterKind::Header,
        );
        code_generator.newline(EmitterKind::Header);
        code_generator.body_emitters.push(1);

        if !is_unsafe_mode {
            code_generator.emit_bounds_check();
        }

        code_generator.emit_main_function();

        code_generator
    }

    fn emit(&mut self, str: &str, emitter_kind: EmitterKind) {
        self.emitter(emitter_kind).emit(str);
    }

    fn newline(&mut self, emitter_kind: EmitterKind) {
        self.emitter(emitter_kind).newline();
    }

    fn emit_char(&mut self, c: char, emitter_kind: EmitterKind) {
        self.emitter(emitter_kind).emit_char(c);
    }

    fn emitln(&mut self, str: &str, emitter_kind: EmitterKind) {
        if self.is_debug_mode {
            if let Some(current_node_index) = self.node_index_stack.last() {
                let current_node = self.get_typer_node(*current_node_index);

                let line = current_node.start.line;
                let file = &self.files[current_node.start.file_index].path;

                let line_pragma = format!("#line {} {:?}\n", line, file);

                self.emitter(emitter_kind).emitln_before(&line_pragma);
            }
        }

        self.emitter(emitter_kind).emitln(str);
    }

    fn indent(&mut self, emitter_kind: EmitterKind) {
        self.emitter(emitter_kind).indent();
    }

    fn unindent(&mut self, emitter_kind: EmitterKind) {
        self.emitter(emitter_kind).unindent();
    }

    fn get_typer_node(&self, index: NodeIndex) -> &TypedNode {
        &self.typed_nodes[index.node_index]
    }

    pub fn gen(&mut self) {
        for i in 0..self.typed_definitions.len() {
            let index = self.typed_definitions[i];

            self.node_index_stack.push(index);

            let TypedNode {
                node_kind,
                node_type,
                namespace_id,
                ..
            } = self.get_typer_node(index).clone();

            match node_kind {
                NodeKind::VariableDeclaration {
                    declaration_kind,
                    name,
                    expression,
                    is_shallow,
                    ..
                } => {
                    self.variable_declaration(
                        declaration_kind,
                        name,
                        expression,
                        is_shallow,
                        node_type,
                        namespace_id,
                        EmitterKind::GlobalVariable,
                    );
                    self.emitln(";", EmitterKind::GlobalVariable);
                    self.newline(EmitterKind::GlobalVariable)
                }
                _ => self.gen_node(index),
            }

            self.node_index_stack.pop();
        }
    }

    fn gen_node(&mut self, index: NodeIndex) {
        self.node_index_stack.push(index);

        let TypedNode {
            node_kind,
            node_type,
            namespace_id,
            ..
        } = self.get_typer_node(index).clone();

        match node_kind {
            NodeKind::TopLevel { .. } => self.top_level(node_type, namespace_id),
            NodeKind::StructDefinition {
                fields, is_union, ..
            } => self.struct_definition(fields, is_union, node_type),
            NodeKind::EnumDefinition {
                name,
                variant_names,
                ..
            } => self.enum_definition(name, variant_names),
            NodeKind::Field { name, .. } => self.field(name, node_type),
            NodeKind::Function {
                declaration,
                scoped_statement,
                is_shallow,
            } => self.function(declaration, scoped_statement, is_shallow, node_type),
            NodeKind::FunctionDeclaration {
                name,
                params,
                generic_params,
                ..
            } => self.function_declaration(name, params, generic_params, node_type),
            NodeKind::ExternFunction { declaration } => {
                self.extern_function(declaration, node_type)
            }
            NodeKind::Param { name, .. } => self.param(name, node_type),
            NodeKind::Block { statements } => self.block(statements),
            NodeKind::Statement { inner } => self.statement(inner),
            NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                expression,
                is_shallow,
                ..
            } => self.variable_declaration(
                declaration_kind,
                name,
                expression,
                is_shallow,
                node_type,
                namespace_id,
                EmitterKind::Body,
            ),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression),
            NodeKind::BreakStatement => self.break_statement(),
            NodeKind::ContinueStatement => self.continue_statement(),
            NodeKind::DeferStatement { statement } => self.defer_statement(statement),
            NodeKind::DeleteStatement { expression } => self.delete_statement(expression),
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
            } => self.while_loop(expression, scoped_statement, index),
            NodeKind::ForOfLoop {
                iterator,
                op,
                from,
                to,
                by,
                scoped_statement,
                ..
            } => self.for_loop(iterator, op, from, to, by, scoped_statement, index),
            NodeKind::ForInLoop { .. } => panic!("cannot generate for-in loop"),
            NodeKind::ConstExpression { .. } => self.const_expression(node_type, EmitterKind::Body),
            NodeKind::Binary { left, op, right } => {
                self.binary(left, op, right, node_type, EmitterKind::Body)
            }
            NodeKind::UnaryPrefix { op, right } => {
                self.unary_prefix(op, right, node_type, EmitterKind::Body)
            }
            NodeKind::UnarySuffix { left, op } => self.unary_suffix(left, op, EmitterKind::Body),
            NodeKind::Call {
                left,
                args,
                method_kind,
            } => self.call(left, args, method_kind, node_type, EmitterKind::Body),
            NodeKind::IndexAccess { left, expression } => {
                self.index_access(left, expression, EmitterKind::Body)
            }
            NodeKind::FieldAccess { left, name } => {
                self.field_access(left, name, node_type, EmitterKind::Body)
            }
            NodeKind::Cast { left, .. } => self.cast(left, node_type, EmitterKind::Body),
            NodeKind::GenericSpecifier {
                left,
                generic_arg_type_names,
            } => self.generic_specifier(left, generic_arg_type_names, node_type, EmitterKind::Body),
            NodeKind::Name { text } => self.name(text, namespace_id, EmitterKind::Body),
            NodeKind::Identifier { name } => self.identifier(name, node_type, EmitterKind::Body),
            NodeKind::IntLiteral { text } => self.int_literal(text, node_type, EmitterKind::Body),
            NodeKind::FloatLiteral { text } => {
                self.float_literal(text, node_type, EmitterKind::Body)
            }
            NodeKind::CharLiteral { value } => self.char_literal(value, EmitterKind::Body),
            NodeKind::StringLiteral { text } => self.string_literal(text, EmitterKind::Body),
            NodeKind::StringInterpolation { .. } => panic!("cannot generate string interpolation"),
            NodeKind::BoolLiteral { value } => self.bool_literal(value, EmitterKind::Body),
            NodeKind::ArrayLiteral { elements, .. } => {
                self.array_literal(elements, node_type, EmitterKind::Body)
            }
            NodeKind::StructLiteral { field_literals, .. } => {
                self.struct_literal(field_literals, node_type, EmitterKind::Body)
            }
            NodeKind::FieldLiteral { name, expression } => {
                self.field_literal(name, expression, EmitterKind::Body)
            }
            NodeKind::TypeSize { type_name } => {
                self.type_size(type_name, node_type, namespace_id, EmitterKind::Body)
            }
            NodeKind::Using { .. } => panic!("cannot generate using statement"),
            NodeKind::Alias { .. } => panic!("cannot generate alias statement"),
            NodeKind::Error => panic!("cannot generate error node"),
            NodeKind::TypeName { .. }
            | NodeKind::TypeNameArray { .. }
            | NodeKind::TypeNamePointer { .. }
            | NodeKind::TypeNameFunction { .. }
            | NodeKind::TypeNameFieldAccess { .. }
            | NodeKind::TypeNameGenericSpecifier { .. } => {
                panic!("cannot generate type name, generate the corresponding type kind instead")
            }
        }

        self.node_index_stack.pop();
    }

    fn gen_node_with_emitter(&mut self, index: NodeIndex, emitter_kind: EmitterKind) {
        self.node_index_stack.push(index);

        let TypedNode {
            node_kind,
            node_type,
            namespace_id,
            ..
        } = self.get_typer_node(index).clone();

        match node_kind {
            NodeKind::ConstExpression { .. } => self.const_expression(node_type, emitter_kind),
            NodeKind::Binary { left, op, right } => {
                self.binary(left, op, right, node_type, emitter_kind)
            }
            NodeKind::UnaryPrefix { op, right } => {
                self.unary_prefix(op, right, node_type, emitter_kind)
            }
            NodeKind::UnarySuffix { left, op } => self.unary_suffix(left, op, emitter_kind),
            NodeKind::Call {
                left,
                args,
                method_kind,
            } => self.call(left, args, method_kind, node_type, emitter_kind),
            NodeKind::IndexAccess { left, expression } => {
                self.index_access(left, expression, emitter_kind)
            }
            NodeKind::FieldAccess { left, name } => {
                self.field_access(left, name, node_type, emitter_kind)
            }
            NodeKind::Cast { left, .. } => self.cast(left, node_type, emitter_kind),
            NodeKind::GenericSpecifier {
                left,
                generic_arg_type_names,
            } => self.generic_specifier(left, generic_arg_type_names, node_type, emitter_kind),
            NodeKind::Name { text } => self.name(text, namespace_id, emitter_kind),
            NodeKind::Identifier { name } => self.identifier(name, node_type, emitter_kind),
            NodeKind::IntLiteral { text } => self.int_literal(text, node_type, emitter_kind),
            NodeKind::FloatLiteral { text } => self.float_literal(text, node_type, emitter_kind),
            NodeKind::CharLiteral { value } => self.char_literal(value, emitter_kind),
            NodeKind::StringLiteral { text } => self.string_literal(text, emitter_kind),
            NodeKind::BoolLiteral { value } => self.bool_literal(value, emitter_kind),
            NodeKind::ArrayLiteral { elements, .. } => {
                self.array_literal(elements, node_type, emitter_kind)
            }
            NodeKind::StructLiteral { field_literals, .. } => {
                self.struct_literal(field_literals, node_type, emitter_kind)
            }
            NodeKind::FieldLiteral { name, expression } => {
                self.field_literal(name, expression, emitter_kind)
            }
            NodeKind::TypeSize { type_name } => {
                self.type_size(type_name, node_type, namespace_id, emitter_kind)
            }
            _ => self.gen_node(index),
        }

        self.node_index_stack.pop();
    }

    fn top_level(&mut self, _node_type: Option<Type>, _namespace_id: Option<usize>) {
        // We ignore the top level node because we want to generate all typed definitions, some of which aren't in the top level.
        // eg. generic definitions, definitions pulled from another file.
        // TODO: Should these definitions be added to the typed top level rather than stored in a special typed definitions list?
    }

    fn struct_definition(
        &mut self,
        fields: Arc<Vec<NodeIndex>>,
        is_union: bool,
        node_type: Option<Type>,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        if !matches!(
            self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Struct {
                primitive_type: PrimitiveType::None,
                ..
            }
        ) {
            return;
        };

        self.emit_struct_equals(type_kind_id);

        if is_union {
            self.emit_union_check_tag(type_kind_id);
            self.emit_union_with_tag(type_kind_id);
        }

        self.emit("struct ", EmitterKind::TypePrototype);
        self.emit_struct_name(type_kind_id, EmitterKind::TypePrototype);
        self.emitln(";", EmitterKind::TypePrototype);
        self.newline(EmitterKind::TypePrototype);

        self.emit("struct ", EmitterKind::TypeDefinition);
        self.emit_struct_name(type_kind_id, EmitterKind::TypeDefinition);

        self.emit(" ", EmitterKind::TypeDefinition);

        if is_union {
            self.emitln("{", EmitterKind::TypeDefinition);
            self.indent(EmitterKind::TypeDefinition);
            self.emitln("intptr_t tag;", EmitterKind::TypeDefinition);
            self.emit("union ", EmitterKind::TypeDefinition);
        }

        self.emitln("{", EmitterKind::TypeDefinition);
        self.indent(EmitterKind::TypeDefinition);

        if fields.is_empty() {
            // C doesn't allow empty structs.
            self.emitln("bool placeholder;", EmitterKind::TypeDefinition);
        }

        for field in fields.iter() {
            self.gen_node(*field);
        }

        self.unindent(EmitterKind::TypeDefinition);
        self.emit("}", EmitterKind::TypeDefinition);

        if is_union {
            self.emitln(" variant;", EmitterKind::TypeDefinition);
            self.unindent(EmitterKind::TypeDefinition);
            self.emit("}", EmitterKind::TypeDefinition);
        }

        self.emitln(";", EmitterKind::TypeDefinition);
        self.newline(EmitterKind::TypeDefinition);
    }

    fn enum_definition(&mut self, name: NodeIndex, variant_names: Arc<Vec<NodeIndex>>) {
        self.emit("enum ", EmitterKind::TypePrototype);
        self.gen_node_with_emitter(name, EmitterKind::TypePrototype);
        self.emitln(";", EmitterKind::TypePrototype);
        self.newline(EmitterKind::TypePrototype);

        self.emit("enum ", EmitterKind::TypeDefinition);
        self.gen_node_with_emitter(name, EmitterKind::TypeDefinition);
        self.emitln(" {", EmitterKind::TypeDefinition);
        self.indent(EmitterKind::TypeDefinition);

        for variant_name in variant_names.iter() {
            self.emit("__", EmitterKind::TypeDefinition);
            self.gen_node_with_emitter(name, EmitterKind::TypeDefinition);
            self.gen_node_with_emitter(*variant_name, EmitterKind::TypeDefinition);
            self.emitln(",", EmitterKind::TypeDefinition);
        }

        self.unindent(EmitterKind::TypeDefinition);
        self.emitln("};", EmitterKind::TypeDefinition);
        self.newline(EmitterKind::TypeDefinition);
    }

    fn field(&mut self, name: NodeIndex, node_type: Option<Type>) {
        self.emit_type_kind_left(
            node_type.clone().unwrap().type_kind_id,
            EmitterKind::TypeDefinition,
            false,
            true,
        );
        self.gen_node_with_emitter(name, EmitterKind::TypeDefinition);
        self.emit_type_kind_right(
            node_type.unwrap().type_kind_id,
            EmitterKind::TypeDefinition,
            false,
        );
        self.emitln(";", EmitterKind::TypeDefinition);
    }

    fn function(
        &mut self,
        declaration: NodeIndex,
        scoped_statement: NodeIndex,
        is_shallow: bool,
        node_type: Option<Type>,
    ) {
        if is_shallow {
            assert_matches!(
                NodeKind::FunctionDeclaration {
                    name,
                    params,
                    generic_params,
                    ..
                },
                self.get_typer_node(declaration).node_kind.clone()
            );

            let type_kind_id = node_type.unwrap().type_kind_id;
            let has_generic_params = !generic_params.is_empty();

            self.emit_function_declaration(
                EmitterKind::FunctionPrototype,
                name,
                &params,
                has_generic_params,
                type_kind_id,
            );

            self.emitln(";", EmitterKind::FunctionPrototype);
            self.newline(EmitterKind::FunctionPrototype);
        } else {
            self.gen_node(declaration);
            self.function_declaration_needing_init = Some(declaration);
            self.gen_node(scoped_statement);
            self.newline(EmitterKind::Body);
            self.newline(EmitterKind::Body);
        }
    }

    fn function_declaration(
        &mut self,
        name: NodeIndex,
        params: Arc<Vec<NodeIndex>>,
        generic_params: Arc<Vec<NodeIndex>>,
        node_type: Option<Type>,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;
        let has_generic_params = !generic_params.is_empty();

        self.emit_function_declaration(
            EmitterKind::FunctionPrototype,
            name,
            &params,
            has_generic_params,
            type_kind_id,
        );
        self.emitln(";", EmitterKind::FunctionPrototype);
        self.newline(EmitterKind::FunctionPrototype);

        self.emit_function_declaration(
            EmitterKind::Body,
            name,
            &params,
            has_generic_params,
            type_kind_id,
        );
        self.emit(" ", EmitterKind::Body);
    }

    fn extern_function(&mut self, declaration: NodeIndex, node_type: Option<Type>) {
        self.emit("extern ", EmitterKind::FunctionPrototype);

        assert_matches!(
            NodeKind::FunctionDeclaration { name, params, .. },
            self.get_typer_node(declaration).node_kind.clone()
        );

        let type_kind_id = node_type.unwrap().type_kind_id;

        self.emit_function_declaration(
            EmitterKind::FunctionPrototype,
            name,
            &params,
            false,
            type_kind_id,
        );

        self.emitln(";", EmitterKind::FunctionPrototype);
        self.newline(EmitterKind::FunctionPrototype);
    }

    fn param(&mut self, name: NodeIndex, node_type: Option<Type>) {
        self.emit_param(name, node_type.unwrap().type_kind_id, EmitterKind::Body);
    }

    fn copy_array_params(&mut self, function_declaration: NodeIndex) {
        assert_matches!(
            NodeKind::FunctionDeclaration { params, .. },
            self.get_typer_node(function_declaration).node_kind.clone()
        );

        for param in params.iter() {
            assert_matches!(
                TypedNode {
                    node_kind: NodeKind::Param { name, .. },
                    node_type,
                    ..
                },
                self.get_typer_node(*param).clone()
            );

            let type_kind_id = node_type.unwrap().type_kind_id;

            if !matches!(
                &self.type_kinds.get_by_id(type_kind_id),
                TypeKind::Array { .. }
            ) {
                continue;
            }

            assert_matches!(
                NodeKind::Name { text: name_text },
                self.get_typer_node(name).node_kind.clone()
            );

            let copy_name = format!("__{}", &name_text);

            self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, true);
            self.emit(&copy_name, EmitterKind::Body);
            self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
            self.emitln(";", EmitterKind::Body);

            self.emit_memmove_name_to_name(&copy_name, &name_text, type_kind_id, EmitterKind::Body);
            self.emitln(";", EmitterKind::Body);

            self.gen_node(name);
            self.emit(" = ", EmitterKind::Body);
            self.emit(&copy_name, EmitterKind::Body);
            self.emitln(";", EmitterKind::Body);
        }
    }

    fn block(&mut self, statements: Arc<Vec<NodeIndex>>) {
        self.emitln("{", EmitterKind::Body);
        self.body_emitters.push(1);

        // Make copies of any parameters that are arrays, because arrays are supposed to be passed by value.
        if let Some(function_declaration) = self.function_declaration_needing_init {
            self.copy_array_params(function_declaration);
            self.function_declaration_needing_init = None;
        }

        for statement in statements.iter() {
            self.gen_node(*statement);
        }

        let mut was_last_statement_early_exit = false;
        for statement in statements.iter().rev() {
            assert_matches!(
                TypedNode {
                    node_kind: NodeKind::Statement { inner },
                    ..
                },
                self.get_typer_node(*statement)
            );

            let Some(inner) = inner else {
                continue;
            };

            was_last_statement_early_exit = matches!(
                self.get_typer_node(*inner).node_kind,
                NodeKind::ReturnStatement { .. }
                    | NodeKind::BreakStatement
                    | NodeKind::ContinueStatement,
            );

            break;
        }

        self.body_emitters.pop(!was_last_statement_early_exit);
        self.emit("}", EmitterKind::Body);
    }

    fn statement(&mut self, inner: Option<NodeIndex>) {
        let Some(inner) = inner else {
            self.emitln(";", EmitterKind::Body);
            return;
        };

        let needs_semicolon = !matches!(
            self.get_typer_node(inner),
            TypedNode {
                node_kind: NodeKind::DeferStatement { .. },
                ..
            } | TypedNode {
                node_kind: NodeKind::IfStatement { .. },
                ..
            } | TypedNode {
                node_kind: NodeKind::SwitchStatement { .. },
                ..
            } | TypedNode {
                node_kind: NodeKind::WhileLoop { .. },
                ..
            } | TypedNode {
                node_kind: NodeKind::ForOfLoop { .. },
                ..
            } | TypedNode {
                node_kind: NodeKind::Block { .. },
                ..
            }
        );

        let needs_newline = matches!(
            self.get_typer_node(inner),
            TypedNode {
                node_kind: NodeKind::Block { .. },
                ..
            }
        );

        self.gen_node(inner);

        if needs_semicolon {
            self.emitln(";", EmitterKind::Body);
        }

        if needs_newline {
            self.newline(EmitterKind::Body);
        }
    }

    fn emit_variable_declaration_initializer(
        &mut self,
        name: NodeIndex,
        expression: NodeIndex,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        let is_array = matches!(
            &self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Array { .. }
        );

        if is_array && !self.is_typed_expression_array_literal(expression) {
            self.emitln(";", emitter_kind);

            assert_matches!(
                NodeKind::Name { text: name_text },
                self.get_typer_node(name).node_kind.clone()
            );

            self.emit_memmove_expression_to_name(
                &name_text,
                expression,
                type_kind_id,
                emitter_kind,
            );
        } else {
            self.emit(" = ", emitter_kind);
            self.gen_node_with_emitter(expression, emitter_kind);
        }
    }

    fn variable_declaration(
        &mut self,
        declaration_kind: DeclarationKind,
        name: NodeIndex,
        expression: Option<NodeIndex>,
        is_shallow: bool,
        node_type: Option<Type>,
        namespace_id: Option<usize>,
        emitter_kind: EmitterKind,
    ) {
        if is_shallow {
            self.emit("extern ", emitter_kind);
        }

        let type_kind_id = node_type.as_ref().unwrap().type_kind_id;

        let is_array = matches!(
            &self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Array { .. }
        );
        let needs_const = declaration_kind != DeclarationKind::Var
            && !is_array
            && emitter_kind != EmitterKind::GlobalVariable;

        self.emit_type_kind_left(type_kind_id, emitter_kind, false, true);

        if needs_const {
            self.emit("const ", emitter_kind);
        }

        if let Some(namespace_id) = namespace_id {
            self.emit_namespace(namespace_id, emitter_kind);
        }

        self.gen_node_with_emitter(name, emitter_kind);

        self.emit_type_kind_right(type_kind_id, emitter_kind, false);

        if is_shallow || emitter_kind == EmitterKind::GlobalVariable {
            return;
        }

        let Some(expression) = expression else {
            return;
        };

        self.emit_variable_declaration_initializer(name, expression, node_type, emitter_kind);
    }

    fn return_statement(&mut self, expression: Option<NodeIndex>) {
        self.body_emitters.early_exiting_scopes(None);

        let expression = if let Some(expression) = expression {
            expression
        } else {
            self.emit("return", EmitterKind::Body);
            return;
        };

        let type_kind_id = self
            .get_typer_node(expression)
            .node_type
            .as_ref()
            .unwrap()
            .type_kind_id;

        if matches!(
            &self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Array { .. }
        ) {
            if self.is_typed_expression_array_literal(expression) {
                let temp_name = self.name_generator.temp_name("temp");

                self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, true);
                self.emit(&temp_name, EmitterKind::Body);
                self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
                self.emit(" = ", EmitterKind::Body);
                self.gen_node(expression);
                self.emitln(";", EmitterKind::Body);

                self.emit_memmove_name_to_name(
                    "__return",
                    &temp_name,
                    type_kind_id,
                    EmitterKind::Body,
                );
                self.emitln(";", EmitterKind::Body);
            } else {
                self.emit_memmove_expression_to_name(
                    "__return",
                    expression,
                    type_kind_id,
                    EmitterKind::Body,
                );
                self.emitln(";", EmitterKind::Body);
            }

            self.emit("return __return", EmitterKind::Body);
        } else {
            self.emit("return ", EmitterKind::Body);
            self.gen_node(expression);
        }
    }

    fn break_statement(&mut self) {
        let last_loop = self.loop_depth_stack.last_mut().unwrap();
        let scope_count = self.body_emitters.len() - last_loop.depth;

        self.body_emitters.early_exiting_scopes(Some(scope_count));

        if let Some(last_switch_depth) = self.switch_depth_stack.last() {
            if *last_switch_depth > last_loop.depth {
                last_loop.was_label_used = true;

                let node_index = last_loop.index.node_index;
                self.emit("goto __break", EmitterKind::Body);
                self.emit_number_backwards(node_index, EmitterKind::Body);

                return;
            }
        }

        self.emit("break", EmitterKind::Body);
    }

    fn continue_statement(&mut self) {
        let scope_count = self.body_emitters.len() - self.loop_depth_stack.last().unwrap().depth;

        self.body_emitters.early_exiting_scopes(Some(scope_count));
        self.emit("continue", EmitterKind::Body);
    }

    fn defer_statement(&mut self, statement: NodeIndex) {
        self.body_emitters.push(0);
        self.gen_node(statement);
        self.body_emitters.pop_to_bottom();
    }

    fn delete_statement(&mut self, expression: NodeIndex) {
        let expression_type = self.get_typer_node(expression).node_type.clone().unwrap();
        let expression_type_kind_id = expression_type.type_kind_id;

        let void_id = self.type_kinds.add_or_get(TypeKind::Void);

        let free_type_kind_id = self.type_kinds.add_or_get(TypeKind::Function {
            param_type_kind_ids: vec![expression_type_kind_id].into(),
            return_type_kind_id: void_id,
        });

        // The expression will always be assigned to a var before the destructor is called on it.
        let expression_type_for_destructor = Type {
            type_kind_id: expression_type_kind_id,
            instance_kind: InstanceKind::Var,
        };

        if let Some(method_kind) = self
            .type_kinds
            .is_destructor_call_valid(&expression_type_for_destructor, &self.namespaces)
        {
            // We don't want to re-evalutate the expression when we use it
            // multiple times (when calling the destructor, and when freeing).
            let free_subject = self.name_generator.temp_name("freeSubject");

            self.emit_type_kind_left(expression_type_kind_id, EmitterKind::Top, false, true);
            self.emit(&free_subject, EmitterKind::Top);
            self.emit_type_kind_right(expression_type_kind_id, EmitterKind::Top, false);
            self.emitln(" = ", EmitterKind::Top);
            self.gen_node_with_emitter(expression, EmitterKind::Top);
            self.emitln(";", EmitterKind::Top);

            self.emit_destructor(
                &free_subject,
                expression_type_kind_id,
                EmitterKind::Body,
                method_kind,
            );

            self.emit_function_name_string("Free", free_type_kind_id, true, EmitterKind::Body);
            self.emit("(", EmitterKind::Body);
            self.emit(&free_subject, EmitterKind::Body);
            self.emit(")", EmitterKind::Body);
        } else {
            self.emit_function_name_string("Free", free_type_kind_id, true, EmitterKind::Body);
            self.emit("(", EmitterKind::Body);
            self.gen_node(expression);
            self.emit(")", EmitterKind::Body);
        }
    }

    fn if_statement(
        &mut self,
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        next: Option<NodeIndex>,
    ) {
        self.emit("if (", EmitterKind::Body);
        self.gen_node(expression);
        self.emit(") ", EmitterKind::Body);
        self.gen_node(scoped_statement);

        if let Some(next) = next {
            self.emit(" else ", EmitterKind::Body);
            self.gen_node(next);

            if !matches!(
                self.get_typer_node(next).node_kind,
                NodeKind::IfStatement { .. }
            ) {
                self.newline(EmitterKind::Body);
            }
        } else {
            self.newline(EmitterKind::Body);
        }
    }

    fn switch_statement(&mut self, expression: NodeIndex, case_statement: NodeIndex) {
        self.switch_depth_stack.push(self.body_emitters.len());

        self.emit("switch (", EmitterKind::Body);
        self.gen_node(expression);
        self.emitln(") {", EmitterKind::Body);
        self.gen_node(case_statement);
        self.emitln("}", EmitterKind::Body);

        self.switch_depth_stack.pop();
    }

    fn case_statement(
        &mut self,
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        next: Option<NodeIndex>,
    ) {
        self.emit("case ", EmitterKind::Body);
        self.gen_node(expression);
        self.emit(": ", EmitterKind::Body);
        self.gen_node(scoped_statement);
        self.emitln(" break;", EmitterKind::Body);

        if let Some(next) = next {
            if !matches!(
                self.get_typer_node(next).node_kind,
                NodeKind::CaseStatement { .. }
            ) {
                self.emit("default: ", EmitterKind::Body);
                self.gen_node(next);
                self.emitln(" break;", EmitterKind::Body);
            } else {
                self.gen_node(next);
            }
        }
    }

    fn emit_break_label_if_used(&mut self, index: NodeIndex, loop_depth: Option<LoopDepth>) {
        let Some(loop_depth) = loop_depth else {
            return;
        };

        if !loop_depth.was_label_used {
            return;
        }

        self.emit("__break", EmitterKind::Body);
        self.emit_number_backwards(index.node_index, EmitterKind::Body);
        self.emitln(":;", EmitterKind::Body);
    }

    fn while_loop(&mut self, expression: NodeIndex, scoped_statement: NodeIndex, index: NodeIndex) {
        self.emit("while (", EmitterKind::Body);
        self.gen_node(expression);
        self.emit(") ", EmitterKind::Body);

        self.loop_depth_stack.push(LoopDepth {
            index,
            depth: self.body_emitters.len(),
            was_label_used: false,
        });

        self.gen_node(scoped_statement);
        self.newline(EmitterKind::Body);

        let loop_depth = self.loop_depth_stack.pop();
        self.emit_break_label_if_used(index, loop_depth)
    }

    fn for_loop(
        &mut self,
        iterator: NodeIndex,
        op: Op,
        from: NodeIndex,
        to: NodeIndex,
        by: Option<NodeIndex>,
        scoped_statement: NodeIndex,
        index: NodeIndex,
    ) {
        let type_kind_id = self
            .get_typer_node(from)
            .node_type
            .as_ref()
            .unwrap()
            .type_kind_id;

        self.emit("for (", EmitterKind::Body);
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, true);
        self.gen_node(iterator);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.emit(" = ", EmitterKind::Body);
        self.gen_node(from);
        self.emit("; ", EmitterKind::Body);

        self.gen_node(iterator);
        self.emit_binary_op(op, EmitterKind::Body);
        self.gen_node(to);
        self.emit("; ", EmitterKind::Body);

        self.gen_node(iterator);
        self.emit(" += ", EmitterKind::Body);
        if let Some(by) = by {
            self.gen_node(by);
        } else {
            self.emit("1", EmitterKind::Body);
        }
        self.emit(") ", EmitterKind::Body);

        self.loop_depth_stack.push(LoopDepth {
            index,
            depth: self.body_emitters.len(),
            was_label_used: false,
        });

        self.gen_node(scoped_statement);
        self.newline(EmitterKind::Body);

        let loop_depth = self.loop_depth_stack.pop();
        self.emit_break_label_if_used(index, loop_depth)
    }

    fn const_expression(&mut self, node_type: Option<Type>, emitter_kind: EmitterKind) {
        assert_matches!(
            Some(Type {
                instance_kind: InstanceKind::Const(const_value),
                ..
            }),
            node_type
        );

        match const_value {
            ConstValue::Int { value } => self.emit(&value.to_string(), emitter_kind),
            ConstValue::UInt { value } => self.emit(&value.to_string(), emitter_kind),
            ConstValue::Float { value } => self.emit(&value.to_string(), emitter_kind),
            ConstValue::String { value } => {
                self.emit_char('"', emitter_kind);
                self.emit(&value, emitter_kind);
                self.emit_char('"', emitter_kind);
            }
            ConstValue::Char { value } => self.emit_char(value, emitter_kind),
            ConstValue::Bool { value } => {
                if value {
                    self.emit("true", emitter_kind);
                } else {
                    self.emit("false", emitter_kind);
                }
            }
        }
    }

    fn binary(
        &mut self,
        left: NodeIndex,
        op: Op,
        right: NodeIndex,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        let left_type_kind_id = self
            .get_typer_node(left)
            .node_type
            .as_ref()
            .unwrap()
            .type_kind_id;

        if op == Op::Assign {
            let type_kind_id = node_type.unwrap().type_kind_id;
            let is_array = matches!(
                &self.type_kinds.get_by_id(type_kind_id),
                TypeKind::Array { .. }
            );

            if is_array && !self.is_typed_expression_array_literal(left) {
                self.emit_memmove_expression_to_variable(left, right, type_kind_id, emitter_kind);
                return;
            }

            if let NodeKind::FieldAccess { left, name, .. } = self.get_typer_node(left).node_kind {
                let left_type = self.get_typer_node(left).node_type.as_ref().unwrap();

                let (dereferenced_left_type_kind_id, is_left_pointer) = self
                    .type_kinds
                    .dereference_type_kind_id(left_type.type_kind_id);

                if let TypeKind::Struct {
                    fields, is_union, ..
                } = &self.type_kinds.get_by_id(dereferenced_left_type_kind_id)
                {
                    if *is_union {
                        assert_matches!(
                            NodeKind::Name { text: name_text },
                            self.get_typer_node(name).node_kind.clone()
                        );

                        let tag =
                            get_field_index_by_name(&self.typed_nodes, &name_text, fields).unwrap();

                        self.emit_union_with_tag_usage(
                            dereferenced_left_type_kind_id,
                            is_left_pointer,
                            left,
                            name,
                            tag,
                            emitter_kind,
                        );
                        self.emit_binary_op(op, emitter_kind);
                        self.gen_node_with_emitter(right, emitter_kind);

                        return;
                    }
                }
            }
        }

        let needs_increased_precedence = matches!(
            op,
            Op::BitwiseOr | Op::Xor | Op::BitwiseAnd | Op::LeftShift | Op::RightShift
        );

        if needs_increased_precedence {
            self.emit("(", emitter_kind)
        }

        if matches!(op, Op::Equal | Op::NotEqual) {
            self.emit_equality(
                left_type_kind_id,
                None,
                left,
                None,
                right,
                op == Op::Equal,
                emitter_kind,
            );
        } else {
            self.gen_node_with_emitter(left, emitter_kind);
            self.emit_binary_op(op, emitter_kind);
            self.gen_node_with_emitter(right, emitter_kind);
        }

        if needs_increased_precedence {
            self.emit(")", emitter_kind)
        }
    }

    fn unary_prefix(
        &mut self,
        op: Op,
        right: NodeIndex,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        let node_type = node_type.unwrap();
        let mut needs_closing_paren = false;

        match op {
            Op::New => {
                let type_kind_id = node_type.type_kind_id;
                let (dereferenced_type_kind_id, _) =
                    self.type_kinds.dereference_type_kind_id(type_kind_id);

                let function_type_kind_id = self.type_kinds.add_or_get(TypeKind::Function {
                    param_type_kind_ids: vec![dereferenced_type_kind_id].into(),
                    return_type_kind_id: type_kind_id,
                });

                self.emit_function_name_string("Alloc", function_type_kind_id, true, emitter_kind);
                self.emit("(", emitter_kind);

                needs_closing_paren = true;
            }
            Op::Scope => {
                let type_kind_id = node_type.type_kind_id;

                let (dereferenced_type_kind_id, _) =
                    self.type_kinds.dereference_type_kind_id(type_kind_id);

                let dereferenced_node_type = Type {
                    type_kind_id: dereferenced_type_kind_id,
                    // The expression will always be assigned to a temporary var.
                    instance_kind: InstanceKind::Var,
                };

                let scope_result = self.emit_stack_allocation(&dereferenced_node_type, emitter_kind);

                self.body_emitters.push(0);

                if let Some(method_kind) = self
                    .type_kinds
                    .is_destructor_call_valid(&dereferenced_node_type, &self.namespaces)
                {
                    self.emit_destructor(
                        &scope_result,
                        dereferenced_type_kind_id,
                        EmitterKind::Body,
                        method_kind,
                    );
                }

                self.body_emitters.pop_to_bottom();

                needs_closing_paren = true;
            }
            Op::Reference => {
                let right_type = self.get_typer_node(right).node_type.as_ref().unwrap();

                if right_type.instance_kind == InstanceKind::Literal {
                    self.emit_stack_allocation(&node_type, emitter_kind);
                    needs_closing_paren = true;
                } else {
                    self.emit("&", emitter_kind)
                }
            }
            _ => {
                self.emit(
                    match op {
                        Op::Plus => "+",
                        Op::Minus => "-",
                        Op::Not => "!",
                        _ => panic!("expected unary prefix operator"),
                    },
                    emitter_kind,
                );
            }
        }

        self.gen_node_with_emitter(right, emitter_kind);

        if needs_closing_paren {
            self.emit(")", emitter_kind);
        }
    }

    fn unary_suffix(&mut self, left: NodeIndex, op: Op, emitter_kind: EmitterKind) {
        self.emit("(", emitter_kind);
        self.emit(
            match op {
                Op::Dereference => "*",
                _ => panic!("expected unary suffix operator"),
            },
            emitter_kind,
        );

        self.gen_node_with_emitter(left, emitter_kind);
        self.emit(")", emitter_kind);
    }

    fn call(
        &mut self,
        left: NodeIndex,
        args: Arc<Vec<NodeIndex>>,
        method_kind: MethodKind,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        self.gen_node_with_emitter(left, emitter_kind);

        self.emit("(", emitter_kind);

        if method_kind != MethodKind::None {
            let mut caller = left;
            if let NodeKind::GenericSpecifier { left, .. } = self.get_typer_node(caller).node_kind {
                caller = left;
            }

            assert_matches!(
                NodeKind::FieldAccess { left, .. },
                self.get_typer_node(caller).node_kind
            );

            let left_type = self.get_typer_node(left).node_type.clone().unwrap();

            let mut needs_closing_paren = false;

            match method_kind {
                MethodKind::ByReference => {
                    if left_type.instance_kind == InstanceKind::Literal {
                        self.emit_stack_allocation(&left_type, emitter_kind);
                        needs_closing_paren = true;
                    } else {
                        self.emit("&", emitter_kind)
                    }
                },
                MethodKind::ByDereference => self.emit("*", emitter_kind),
                _ => {}
            }

            self.gen_node_with_emitter(left, emitter_kind);

            if needs_closing_paren {
                self.emit(")", emitter_kind);
            }

            if !args.is_empty() {
                self.emit(", ", emitter_kind);
            }
        }

        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                self.emit(", ", emitter_kind);
            }

            self.gen_node_with_emitter(*arg, emitter_kind);
        }

        let type_kind_id = node_type.unwrap().type_kind_id;

        if matches!(
            &self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Array { .. }
        ) {
            if args.len() > 0 {
                self.emit(", ", emitter_kind);
            }

            let return_array_name = self.name_generator.temp_name("returnArray");

            self.emit_type_kind_left(type_kind_id, EmitterKind::Top, false, true);
            self.emit(&return_array_name, EmitterKind::Top);
            self.emit_type_kind_right(type_kind_id, EmitterKind::Top, false);
            self.emitln(";", EmitterKind::Top);

            self.emit(&return_array_name, emitter_kind);
        }
        self.emit(")", emitter_kind);
    }

    fn index_access(&mut self, left: NodeIndex, expression: NodeIndex, emitter_kind: EmitterKind) {
        self.gen_node_with_emitter(left, emitter_kind);
        self.emit("[", emitter_kind);

        if self.is_unsafe_mode {
            self.gen_node_with_emitter(expression, emitter_kind);
        } else {
            let left_type = self.get_typer_node(left).node_type.as_ref().unwrap();

            assert_matches!(
                TypeKind::Array { element_count, .. },
                &self.type_kinds.get_by_id(left_type.type_kind_id)
            );

            let element_count = *element_count;

            self.emit("__BoundsCheck(", emitter_kind);
            self.gen_node_with_emitter(expression, emitter_kind);
            self.emit(", ", emitter_kind);
            self.emit(&element_count.to_string(), EmitterKind::Body);
            self.emit(")", emitter_kind);
        }

        self.emit("]", emitter_kind);
    }

    fn field_access(
        &mut self,
        left: NodeIndex,
        name: NodeIndex,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        let left_type = self.get_typer_node(left).node_type.as_ref().unwrap();

        let field_type_kind_id = node_type.unwrap().type_kind_id;
        let mut is_method_access = false;
        match self.type_kinds.get_by_id(field_type_kind_id) {
            TypeKind::Tag => {
                assert_matches!(
                    TypeKind::Struct {
                        fields,
                        is_union,
                        ..
                    },
                    &self.type_kinds.get_by_id(left_type.type_kind_id)
                );

                if left_type.instance_kind == InstanceKind::Name && *is_union {
                    assert_matches!(
                        NodeKind::Name { text: name_text },
                        self.get_typer_node(name).node_kind.clone()
                    );

                    let tag =
                        get_field_index_by_name(&self.typed_nodes, &name_text, fields).unwrap();

                    self.emit(&tag.to_string(), emitter_kind);

                    return;
                }
            }
            TypeKind::Function { .. } => {
                is_method_access = true;
            }
            _ => {}
        }

        let (dereferenced_left_type_kind_id, is_left_pointer) = self
            .type_kinds
            .dereference_type_kind_id(left_type.type_kind_id);

        if let TypeKind::Struct {
            fields, is_union, ..
        } = &self.type_kinds.get_by_id(dereferenced_left_type_kind_id)
        {
            if *is_union {
                assert_matches!(
                    NodeKind::Name { text: name_text },
                    self.get_typer_node(name).node_kind.clone()
                );

                let tag = get_field_index_by_name(&self.typed_nodes, &name_text, fields).unwrap();

                self.emit_union_check_tag_usage(
                    dereferenced_left_type_kind_id,
                    is_left_pointer,
                    left,
                    name,
                    tag,
                    emitter_kind,
                );

                return;
            }
        }

        match self.type_kinds.get_by_id(dereferenced_left_type_kind_id) {
            TypeKind::Struct { .. }
                if left_type.instance_kind != InstanceKind::Name && !is_method_access =>
            {
                self.gen_node_with_emitter(left, emitter_kind);

                if is_left_pointer {
                    self.emit("->", emitter_kind)
                } else {
                    self.emit(".", emitter_kind)
                }
            }
            TypeKind::Enum {
                name: enum_name, ..
            } => {
                self.emit("__", emitter_kind);
                self.gen_node_with_emitter(enum_name, emitter_kind);
            }
            TypeKind::Array { element_count, .. } => {
                // On arrays, only the "count" field is allowed.
                self.emit(&element_count.to_string(), EmitterKind::Body);
                return;
            }
            TypeKind::Namespace { .. } | TypeKind::Struct { .. } => {}
            _ => panic!("tried to access type that cannot be accessed"),
        }

        self.gen_node_with_emitter(name, emitter_kind);
    }

    fn cast(&mut self, left: NodeIndex, node_type: Option<Type>, emitter_kind: EmitterKind) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        if let TypeKind::Tag { .. } = &self.type_kinds.get_by_id(type_kind_id) {
            let left_type_kind_id = self
                .get_typer_node(left)
                .node_type
                .as_ref()
                .unwrap()
                .type_kind_id;

            assert_matches!(
                TypeKind::Struct { is_union, .. },
                &self.type_kinds.get_by_id(left_type_kind_id)
            );

            if !is_union {
                panic!("casting to a tag is not allowed for this value");
            }

            self.gen_node_with_emitter(left, emitter_kind);
            self.emit(".tag", emitter_kind);

            return;
        }

        self.emit("((", emitter_kind);
        self.emit_type_kind_left(type_kind_id, emitter_kind, false, false);
        self.emit_type_kind_right(type_kind_id, emitter_kind, false);
        self.emit(")", emitter_kind);
        self.gen_node_with_emitter(left, emitter_kind);
        self.emit(")", emitter_kind);
    }

    fn generic_specifier(
        &mut self,
        left: NodeIndex,
        generic_arg_type_names: Arc<Vec<NodeIndex>>,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        let name = match self.get_typer_node(left).node_kind {
            NodeKind::FieldAccess { name, .. } => name,
            NodeKind::Identifier { name } => name,
            _ => panic!("expected field access or identifier before generic specifier"),
        };

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(type_kind_id) {
            let is_generic = !generic_arg_type_names.is_empty();
            self.emit_function_name(name, type_kind_id, is_generic, emitter_kind)
        } else {
            self.emit_type_kind_left(type_kind_id, emitter_kind, false, false);
            self.emit_type_kind_right(type_kind_id, emitter_kind, false);
        }
    }

    fn name(&mut self, text: Arc<str>, namespace_id: Option<usize>, emitter_kind: EmitterKind) {
        if let Some(namespace_id) = namespace_id {
            self.emit_namespace(namespace_id, emitter_kind);
        }

        if reserved_names().contains(&text) {
            self.emit("__", emitter_kind);
        }

        self.emit(&text, emitter_kind);
    }

    fn identifier(&mut self, name: NodeIndex, node_type: Option<Type>, emitter_kind: EmitterKind) {
        let node_type = node_type.unwrap();

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(node_type.type_kind_id) {
            self.emit_function_name(name, node_type.type_kind_id, false, emitter_kind)
        } else {
            self.gen_node_with_emitter(name, emitter_kind);
        }
    }

    fn int_literal(&mut self, text: Arc<str>, node_type: Option<Type>, emitter_kind: EmitterKind) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        self.emit("((", emitter_kind);
        self.emit_type_kind_left(type_kind_id, emitter_kind, false, false);
        self.emit_type_kind_right(type_kind_id, emitter_kind, false);
        self.emit(")", emitter_kind);

        self.emit(&text, emitter_kind);

        if self.type_kinds.get_by_id(type_kind_id).is_unsigned() {
            self.emit("u", emitter_kind);
        }

        self.emit(")", emitter_kind);
    }

    fn float_literal(
        &mut self,
        text: Arc<str>,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        self.emit(&text, emitter_kind);

        let type_kind_id = node_type.unwrap().type_kind_id;
        if type_kind_id == FLOAT32_TYPE_KIND_ID {
            self.emit("f", emitter_kind);
        }
    }

    fn emit_char_with_escaping(&mut self, c: char, emitter_kind: EmitterKind) {
        match c {
            '\'' => self.emit("\\\'", emitter_kind),
            '\"' => self.emit("\\\"", emitter_kind),
            '\\' => self.emit("\\\\", emitter_kind),
            '\0' => self.emit("\\0", emitter_kind),
            '\n' => self.emit("\\n", emitter_kind),
            '\r' => self.emit("\\r", emitter_kind),
            '\t' => self.emit("\\t", emitter_kind),
            _ => self.emit_char(c, emitter_kind),
        }
    }

    fn char_literal(&mut self, value: char, emitter_kind: EmitterKind) {
        self.emit("'", emitter_kind);
        self.emit_char_with_escaping(value, emitter_kind);
        self.emit("'", emitter_kind);
    }

    fn string_literal(&mut self, text: Arc<String>, emitter_kind: EmitterKind) {
        self.emit("(struct ", emitter_kind);
        self.emit_struct_name(self.span_char_type_kind_id, emitter_kind);
        self.emitln(") {", emitter_kind);
        self.indent(emitter_kind);
        self.emit(".count = ", emitter_kind);
        self.emit(&text.len().to_string(), emitter_kind);
        self.emitln(",", emitter_kind);
        self.emit(".data = ", emitter_kind);

        self.emit("\"", emitter_kind);
        for c in text.chars() {
            self.emit_char_with_escaping(c, emitter_kind);
        }
        self.emitln("\",", emitter_kind);

        self.unindent(emitter_kind);
        self.emit("}", emitter_kind);
    }

    fn bool_literal(&mut self, value: bool, emitter_kind: EmitterKind) {
        if value {
            self.emit("true", emitter_kind);
        } else {
            self.emit("false", emitter_kind);
        }
    }

    fn array_literal(
        &mut self,
        elements: Arc<Vec<NodeIndex>>,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        assert_matches!(
            TypeKind::Array { element_count, .. },
            self.type_kinds.get_by_id(node_type.unwrap().type_kind_id)
        );

        let repeat_count = element_count / elements.len();

        self.emit("{", emitter_kind);
        let mut i = 0;
        for _ in 0..repeat_count {
            for element in elements.iter() {
                if i > 0 {
                    self.emit(", ", emitter_kind);
                }

                self.gen_node_with_emitter(*element, emitter_kind);

                i += 1;
            }
        }
        self.emit("}", emitter_kind);
    }

    fn struct_literal(
        &mut self,
        field_literals: Arc<Vec<NodeIndex>>,
        node_type: Option<Type>,
        emitter_kind: EmitterKind,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        self.emit("(", emitter_kind);
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.emit(") ", emitter_kind);

        assert_matches!(
            TypeKind::Struct {
                fields: type_kind_fields,
                is_union,
                ..
            },
            &self.type_kinds.get_by_id(type_kind_id)
        );

        let is_union = *is_union;

        if is_union {
            self.emitln("{", emitter_kind);
            self.indent(emitter_kind);

            if field_literals.len() != 1 {
                panic!("expected union literal to contain a single field");
            }

            assert_matches!(
                NodeKind::FieldLiteral { name, .. },
                &self.get_typer_node(field_literals[0]).node_kind
            );

            assert_matches!(
                NodeKind::Name { text: name_text },
                &self.get_typer_node(*name).node_kind
            );

            let tag =
                get_field_index_by_name(&self.typed_nodes, name_text, type_kind_fields).unwrap();

            self.emit(".tag = ", emitter_kind);
            self.emit(&tag.to_string(), emitter_kind);
            self.emitln(",", emitter_kind);

            self.emit(".variant = ", emitter_kind);
        }

        self.emitln("{", emitter_kind);
        self.indent(emitter_kind);

        if field_literals.is_empty() {
            // Since empty structs aren't allowed in C, we generate a placeholder field
            // in structs that would be empty otherwise. We also have to initialize it here.
            self.emitln("0,", emitter_kind);
        }

        for field_literal in field_literals.iter() {
            self.gen_node_with_emitter(*field_literal, emitter_kind);
            self.emitln(",", emitter_kind);
        }

        self.unindent(emitter_kind);
        self.emit("}", emitter_kind);

        if is_union {
            self.emitln(",", emitter_kind);
            self.unindent(emitter_kind);
            self.emit("}", emitter_kind);
        }
    }

    fn field_literal(&mut self, name: NodeIndex, expression: NodeIndex, emitter_kind: EmitterKind) {
        self.emit(".", emitter_kind);
        self.gen_node_with_emitter(name, emitter_kind);
        self.emit(" = ", emitter_kind);
        self.gen_node_with_emitter(expression, emitter_kind);
    }

    fn type_size(
        &mut self,
        type_name: NodeIndex,
        _node_type: Option<Type>,
        _namespace_id: Option<usize>,
        emitter_kind: EmitterKind,
    ) {
        let type_name_type_kind = &self.get_typer_node(type_name).node_type;
        self.emit_type_size(
            type_name_type_kind.as_ref().unwrap().type_kind_id,
            emitter_kind,
        );
    }

    fn emit_stack_allocation(&mut self, node_type: &Type, emitter_kind: EmitterKind) -> String {
        let type_kind_id = node_type.type_kind_id;

        let pointer_type_kind_id = self.type_kinds.add_or_get(TypeKind::Pointer {
            inner_type_kind_id: type_kind_id,
            is_inner_mutable: true,
        });

        let function_type_kind_id = self.type_kinds.add_or_get(TypeKind::Function {
            param_type_kind_ids: vec![pointer_type_kind_id, type_kind_id].into(),
            return_type_kind_id: pointer_type_kind_id,
        });

        let stack_var = self.name_generator.temp_name("stackVar");

        self.emit_type_kind_left(type_kind_id, EmitterKind::Top, false, true);
        self.emit(&stack_var, EmitterKind::Top);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Top, false);
        self.emitln(";", EmitterKind::Top);

        self.emit_function_name_string(
            "AllocInto",
            function_type_kind_id,
            true,
            emitter_kind,
        );
        self.emit("(&", emitter_kind);
        self.emit(&stack_var, emitter_kind);
        self.emit(", ", emitter_kind);

        stack_var
    }

    fn emit_destructor(
        &mut self,
        subject_name: &str,
        type_kind_id: usize,
        emitter_kind: EmitterKind,
        method_kind: MethodKind,
    ) {
        let (dereferenced_type_kind_id, _) = self.type_kinds.dereference_type_kind_id(type_kind_id);

        assert_matches!(
            TypeKind::Struct { namespace_id, .. },
            self.type_kinds.get_by_id(dereferenced_type_kind_id)
        );
        self.emit_namespace(namespace_id, emitter_kind);
        self.emit("Destroy(", emitter_kind);

        match method_kind {
            MethodKind::ByReference => self.emit("&", emitter_kind),
            MethodKind::ByDereference => self.emit("*", emitter_kind),
            _ => {}
        }

        self.emit(subject_name, emitter_kind);
        self.emitln(");", emitter_kind);
    }

    fn emit_memmove_expression_to_variable(
        &mut self,
        destination: NodeIndex,
        source: NodeIndex,
        type_kind_id: usize,
        emitter_kind: EmitterKind,
    ) {
        self.emit("memmove(", emitter_kind);
        self.gen_node_with_emitter(destination, emitter_kind);
        self.emit(", ", emitter_kind);
        self.gen_node_with_emitter(source, emitter_kind);
        self.emit(", ", emitter_kind);
        self.emit_type_size(type_kind_id, emitter_kind);
        self.emit(")", emitter_kind);
    }

    fn emit_memmove_expression_to_name(
        &mut self,
        destination: &str,
        source: NodeIndex,
        type_kind_id: usize,
        emitter_kind: EmitterKind,
    ) {
        self.emit("memmove(", emitter_kind);
        self.emit(destination, emitter_kind);
        self.emit(", ", emitter_kind);
        self.gen_node_with_emitter(source, emitter_kind);
        self.emit(", ", emitter_kind);
        self.emit_type_size(type_kind_id, emitter_kind);
        self.emit(")", emitter_kind);
    }

    fn emit_memmove_name_to_name(
        &mut self,
        destination: &str,
        source: &str,
        type_kind_id: usize,
        emitter_kind: EmitterKind,
    ) {
        self.emit("memmove(", emitter_kind);
        self.emit(destination, emitter_kind);
        self.emit(", ", emitter_kind);
        self.emit(source, emitter_kind);
        self.emit(", ", emitter_kind);
        self.emit_type_size(type_kind_id, emitter_kind);
        self.emit(")", emitter_kind);
    }

    fn emit_type_size(&mut self, type_kind_id: usize, emitter_kind: EmitterKind) {
        match self.type_kinds.get_by_id(type_kind_id) {
            TypeKind::Array {
                element_type_kind_id,
                element_count,
            } => {
                self.emit_type_size(element_type_kind_id, emitter_kind);
                self.emit(" * ", emitter_kind);
                self.emit(&element_count.to_string(), emitter_kind);
            }
            _ => {
                self.emit("sizeof(", emitter_kind);
                self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
                self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
                self.emit(")", emitter_kind);
            }
        };
    }

    fn emit_type_kind_left(
        &mut self,
        type_kind_id: usize,
        emitter_kind: EmitterKind,
        do_arrays_as_pointers: bool,
        is_prefix: bool,
    ) {
        let type_kind = &self.type_kinds.get_by_id(type_kind_id);
        let needs_trailing_space = is_prefix
            && !matches!(
                type_kind,
                TypeKind::Array { .. } | TypeKind::Pointer { .. } | TypeKind::Function { .. }
            );

        match type_kind_id {
            INT_TYPE_KIND_ID => self.emit("intptr_t", emitter_kind),
            UINT_TYPE_KIND_ID => self.emit("uintptr_t", emitter_kind),
            INT8_TYPE_KIND_ID => self.emit("int8_t", emitter_kind),
            UINT8_TYPE_KIND_ID => self.emit("uint8_t", emitter_kind),
            INT16_TYPE_KIND_ID => self.emit("int16_t", emitter_kind),
            UINT16_TYPE_KIND_ID => self.emit("uint16_t", emitter_kind),
            INT32_TYPE_KIND_ID => self.emit("int32_t", emitter_kind),
            UINT32_TYPE_KIND_ID => self.emit("uint32_t", emitter_kind),
            INT64_TYPE_KIND_ID => self.emit("int64_t", emitter_kind),
            UINT64_TYPE_KIND_ID => self.emit("uint64_t", emitter_kind),
            FLOAT32_TYPE_KIND_ID => self.emit("float", emitter_kind),
            FLOAT64_TYPE_KIND_ID => self.emit("double", emitter_kind),
            CHAR_TYPE_KIND_ID => self.emit("char", emitter_kind),
            BOOL_TYPE_KIND_ID => self.emit("bool", emitter_kind),
            _ => match type_kind.clone() {
                TypeKind::Tag { .. } => self.emit("intptr_t", emitter_kind),
                TypeKind::Void => self.emit("void", emitter_kind),
                TypeKind::Struct { .. } => {
                    self.emit("struct ", emitter_kind);
                    self.emit_struct_name(type_kind_id, emitter_kind);
                }
                TypeKind::Enum { name, .. } => {
                    self.emit("enum ", emitter_kind);
                    self.gen_node_with_emitter(name, emitter_kind);
                }
                TypeKind::Array {
                    element_type_kind_id,
                    ..
                } => {
                    self.emit_type_kind_left(
                        element_type_kind_id,
                        emitter_kind,
                        do_arrays_as_pointers,
                        true,
                    );
                    if do_arrays_as_pointers {
                        self.emit("*", emitter_kind);
                    }
                }
                TypeKind::Pointer {
                    inner_type_kind_id,
                    is_inner_mutable,
                } => {
                    self.emit_type_kind_left(
                        inner_type_kind_id,
                        emitter_kind,
                        do_arrays_as_pointers,
                        true,
                    );

                    // If the pointer points to an immutable value, then add a const to the generated code.
                    // Except for functions, because a const function has no meaning in C.
                    if !is_inner_mutable
                        && !matches!(
                            self.type_kinds.get_by_id(inner_type_kind_id),
                            TypeKind::Function { .. }
                        )
                    {
                        self.emit("const ", emitter_kind);
                    }

                    self.emit("*", emitter_kind);
                }
                TypeKind::Placeholder { .. } => {
                    panic!("can't emit placeholder type: {:?}", type_kind)
                }
                TypeKind::Function {
                    return_type_kind_id,
                    ..
                } => {
                    self.emit_type_kind_left(return_type_kind_id, emitter_kind, true, true);
                    self.emit_type_kind_right(return_type_kind_id, emitter_kind, true);
                    self.emit("(", emitter_kind);
                }
                TypeKind::Namespace { .. } => panic!("cannot emit namespace types"),
            },
        }

        if needs_trailing_space {
            self.emit(" ", emitter_kind);
        }
    }

    fn emit_type_kind_right(
        &mut self,
        type_kind_id: usize,
        emitter_kind: EmitterKind,
        do_arrays_as_pointers: bool,
    ) {
        let type_kind_id = self.type_kinds.get_by_id(type_kind_id).clone();

        match type_kind_id {
            TypeKind::Array {
                element_type_kind_id,
                element_count,
            } => {
                if !do_arrays_as_pointers {
                    self.emit("[", emitter_kind);
                    self.emit(&element_count.to_string(), emitter_kind);
                    self.emit("]", emitter_kind);
                }
                self.emit_type_kind_right(
                    element_type_kind_id,
                    emitter_kind,
                    do_arrays_as_pointers,
                );
            }
            TypeKind::Pointer {
                inner_type_kind_id, ..
            } => {
                self.emit_type_kind_right(inner_type_kind_id, emitter_kind, do_arrays_as_pointers);
            }
            TypeKind::Function {
                param_type_kind_ids,
                ..
            } => {
                self.emit(")(", emitter_kind);
                for (i, param_kind_id) in param_type_kind_ids.iter().enumerate() {
                    if i > 0 {
                        self.emit(", ", emitter_kind);
                    }

                    self.emit_type_kind_left(*param_kind_id, emitter_kind, false, false);
                    self.emit_type_kind_right(*param_kind_id, emitter_kind, false);
                }
                self.emit(")", emitter_kind);
            }
            _ => {}
        }
    }

    fn emit_binary_op(&mut self, op: Op, emitter_kind: EmitterKind) {
        self.emit(
            match op {
                Op::Less => " < ",
                Op::Greater => " > ",
                Op::LessEqual => " <= ",
                Op::GreaterEqual => " >= ",
                Op::Plus => " + ",
                Op::Minus => " - ",
                Op::Multiply => " * ",
                Op::Divide => " / ",
                Op::Modulo => " % ",
                Op::Assign => " = ",
                Op::And => " && ",
                Op::Or => " || ",
                Op::BitwiseOr => " | ",
                Op::Xor => " ^ ",
                Op::BitwiseAnd => " & ",
                Op::LeftShift => " << ",
                Op::RightShift => " >> ",
                Op::PlusAssign => " += ",
                Op::MinusAssign => " -= ",
                Op::MultiplyAssign => " *= ",
                Op::DivideAssign => " /= ",
                Op::ModuloAssign => " %= ",
                Op::LeftShiftAssign => " <<= ",
                Op::RightShiftAssign => " >>= ",
                Op::BitwiseAndAssign => " &= ",
                Op::BitwiseOrAssign => " |= ",
                Op::XorAssign => " ^= ",
                _ => panic!("this operator cannot be emitted using emit_binary_op"),
            },
            emitter_kind,
        );
    }

    fn emit_function_declaration(
        &mut self,
        emitter_kind: EmitterKind,
        name: NodeIndex,
        params: &Arc<Vec<NodeIndex>>,
        has_generic_params: bool,
        type_kind_id: usize,
    ) {
        assert_matches!(
            TypeKind::Function {
                return_type_kind_id,
                ..
            },
            self.type_kinds.get_by_id(type_kind_id)
        );

        let mut is_generic = has_generic_params;

        if let TypedNode {
            namespace_id: Some(namespace_id),
            ..
        } = self.get_typer_node(name)
        {
            is_generic = is_generic || !self.namespaces[*namespace_id].generic_args.is_empty();
        }

        // TODO: This creates excess code for generic functions! Is that ok?
        // Generic functions may be generated in multiple files, since they are generated in all files that they are used.
        if is_generic {
            self.emit("static ", emitter_kind)
        }

        self.emit_type_kind_left(return_type_kind_id, emitter_kind, true, true);

        self.emit_function_name(name, type_kind_id, has_generic_params, emitter_kind);

        let mut param_count = 0;

        self.emit("(", emitter_kind);
        for param in params.iter() {
            if param_count > 0 {
                self.emit(", ", emitter_kind);
            }

            param_count += 1;

            self.emit_param_node(*param, emitter_kind);
        }

        if matches!(
            &self.type_kinds.get_by_id(return_type_kind_id),
            TypeKind::Array { .. }
        ) {
            if param_count > 0 {
                self.emit(", ", emitter_kind);
            }

            param_count += 1;

            self.emit_param_string("__return", return_type_kind_id, emitter_kind);
        }

        if param_count == 0 {
            self.emit("void", emitter_kind);
        }

        self.emit(")", emitter_kind);

        self.emit_type_kind_right(return_type_kind_id, emitter_kind, true);
    }

    fn emit_param_node(&mut self, param: NodeIndex, emitter_kind: EmitterKind) {
        let type_kind_id = self
            .get_typer_node(param)
            .node_type
            .as_ref()
            .unwrap()
            .type_kind_id;

        assert_matches!(
            NodeKind::Param { name, .. },
            self.get_typer_node(param).node_kind
        );

        self.emit_param(name, type_kind_id, emitter_kind);
    }

    fn emit_param(&mut self, name: NodeIndex, type_kind_id: usize, emitter_kind: EmitterKind) {
        self.emit_type_kind_left(type_kind_id, emitter_kind, false, true);
        self.gen_node_with_emitter(name, emitter_kind);
        self.emit_type_kind_right(type_kind_id, emitter_kind, false);
    }

    fn emit_param_string(&mut self, name: &str, type_kind_id: usize, emitter_kind: EmitterKind) {
        self.emit_type_kind_left(type_kind_id, emitter_kind, false, true);
        self.emit(name, emitter_kind);
        self.emit_type_kind_right(type_kind_id, emitter_kind, false);
    }

    fn emit_namespace(&mut self, namespace_id: usize, emitter_kind: EmitterKind) {
        if namespace_id == GLOBAL_NAMESPACE_ID {
            return;
        }

        if let Some(parent_id) = self.namespaces[namespace_id].parent_id {
            self.emit_namespace(parent_id, emitter_kind);
        }

        let namespace_name = self.namespaces[namespace_id].name.clone();
        self.emit(&namespace_name, emitter_kind);
        self.emit("__", emitter_kind);

        if let Some(associated_type_kind_id) = self.namespaces[namespace_id].associated_type_kind_id
        {
            if !self.namespaces[namespace_id].generic_args.is_empty() {
                self.emit_number_backwards(associated_type_kind_id, emitter_kind);
                self.emit("__", emitter_kind);
            }
        }
    }

    fn emit_number_backwards(&mut self, mut number: usize, emitter_kind: EmitterKind) {
        let mut digit = 0;
        while number > 0 || digit == 0 {
            self.emit_char(((number % 10) as u8 + b'0') as char, emitter_kind);
            number /= 10;
            digit += 1;
        }
    }

    fn emit_bounds_check(&mut self) {
        self.emitln(
            "static inline intptr_t __BoundsCheck(intptr_t index, intptr_t count);",
            EmitterKind::FunctionPrototype,
        );
        self.newline(EmitterKind::FunctionPrototype);

        self.emitln(
            "intptr_t __BoundsCheck(intptr_t index, intptr_t count) {",
            EmitterKind::Body,
        );
        self.indent(EmitterKind::Body);

        self.emitln("if (index < 0 || index >= count) {", EmitterKind::Body);
        self.indent(EmitterKind::Body);

        self.emitln(
            "Internal__ErrorTrace(\"Array access out of bounds!\", 3);",
            EmitterKind::Body,
        );

        self.unindent(EmitterKind::Body);
        self.emitln("}", EmitterKind::Body);

        self.emitln("return index;", EmitterKind::Body);
        self.unindent(EmitterKind::Body);
        self.emitln("}", EmitterKind::Body);
        self.newline(EmitterKind::Body);
    }

    fn emit_struct_name(&mut self, type_kind_id: usize, emitter_kind: EmitterKind) {
        assert_matches!(
            TypeKind::Struct { name, .. },
            self.type_kinds.get_by_id(type_kind_id)
        );

        self.gen_node_with_emitter(name, emitter_kind);
        self.emit("__", emitter_kind);
        self.emit_number_backwards(type_kind_id, emitter_kind);
    }

    fn emit_function_name(
        &mut self,
        name: NodeIndex,
        type_kind_id: usize,
        is_generic: bool,
        emitter_kind: EmitterKind,
    ) {
        self.gen_node_with_emitter(name, emitter_kind);

        if is_generic {
            self.emit("__", emitter_kind);
            self.emit_number_backwards(type_kind_id, emitter_kind);
        }
    }

    fn emit_function_name_string(
        &mut self,
        name: &str,
        type_kind_id: usize,
        is_generic: bool,
        emitter_kind: EmitterKind,
    ) {
        self.emit(name, emitter_kind);

        if is_generic {
            self.emit("__", emitter_kind);
            self.emit_number_backwards(type_kind_id, emitter_kind);
        }
    }

    fn emit_top_level_variable_initializers(&mut self) {
        for i in 0..self.typed_definitions.len() {
            let index = self.typed_definitions[i];

            let TypedNode {
                node_kind,
                node_type,
                namespace_id,
                ..
            } = self.get_typer_node(index).clone();

            if let NodeKind::VariableDeclaration {
                name, expression, ..
            } = node_kind
            {
                if let Some(namespace_id) = namespace_id {
                    self.emit_namespace(namespace_id, EmitterKind::Body);
                }

                self.gen_node_with_emitter(name, EmitterKind::Body);

                self.emit_variable_declaration_initializer(
                    name,
                    expression.unwrap(),
                    node_type,
                    EmitterKind::Body,
                );
                self.emitln(";", EmitterKind::Body);
                self.newline(EmitterKind::Body);
            }
        }
    }

    fn emit_main_function(&mut self) {
        let Some(main_function_declaration) = self.main_function_declaration else {
            return;
        };

        assert_matches!(
            TypedNode {
                node_kind: NodeKind::FunctionDeclaration { name, .. },
                node_type: Some(function_type),
                ..
            },
            self.get_typer_node(main_function_declaration).clone()
        );

        assert_matches!(
            TypeKind::Function {
                param_type_kind_ids,
                ..
            },
            &self.type_kinds.get_by_id(function_type.type_kind_id)
        );

        if param_type_kind_ids.len() > 0 {
            self.emitln("int main(int argc, char** argv) {", EmitterKind::Body);
            self.indent(EmitterKind::Body);
            self.emit_top_level_variable_initializers();
            self.emit("return (int)", EmitterKind::Body);
            self.emit_function_name(name, function_type.type_kind_id, false, EmitterKind::Body);
            self.emitln(
                "((intptr_t)argv, (const char *const *)argv);",
                EmitterKind::Body,
            );
            self.unindent(EmitterKind::Body);
            self.emitln("}", EmitterKind::Body);
            self.newline(EmitterKind::Body);

            return;
        }

        self.emitln("int main(void) {", EmitterKind::Body);
        self.indent(EmitterKind::Body);
        self.emit_top_level_variable_initializers();
        self.emit("return (int)", EmitterKind::Body);
        self.emit_function_name(name, function_type.type_kind_id, false, EmitterKind::Body);
        self.emitln("();", EmitterKind::Body);
        self.unindent(EmitterKind::Body);
        self.emitln("}", EmitterKind::Body);
        self.newline(EmitterKind::Body);
    }

    fn emitter(&mut self, emitter_kind: EmitterKind) -> &mut Emitter {
        match emitter_kind {
            EmitterKind::Header => &mut self.header_emitter,
            EmitterKind::TypePrototype => &mut self.type_prototype_emitter,
            EmitterKind::TypeDefinition => &mut self.type_definition_emitter,
            EmitterKind::FunctionPrototype => &mut self.function_prototype_emitter,
            EmitterKind::GlobalVariable => &mut self.global_variable_emitter,
            EmitterKind::Top => &mut self.body_emitters.top().top,
            EmitterKind::Body => &mut self.body_emitters.top().body,
        }
    }

    fn emit_union_check_tag(&mut self, type_kind_id: usize) {
        if self.is_unsafe_mode {
            return;
        }

        self.emit("static inline ", EmitterKind::FunctionPrototype);
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.emit("* __", EmitterKind::FunctionPrototype);
        self.emit_struct_name(type_kind_id, EmitterKind::FunctionPrototype);
        self.emit("__CheckTag(", EmitterKind::FunctionPrototype);
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.emit(" *self", EmitterKind::FunctionPrototype);
        self.emit_type_kind_right(type_kind_id, EmitterKind::FunctionPrototype, false);
        self.emitln(", intptr_t tag);", EmitterKind::FunctionPrototype);
        self.newline(EmitterKind::FunctionPrototype);

        self.emit("static inline ", EmitterKind::Body);
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, true, false);
        self.emit("* __", EmitterKind::Body);
        self.emit_struct_name(type_kind_id, EmitterKind::Body);
        self.emit("__CheckTag(", EmitterKind::Body);
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emit(" *self", EmitterKind::Body);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.emitln(", intptr_t tag) {", EmitterKind::Body);
        self.indent(EmitterKind::Body);

        self.emitln("if (self->tag != tag) {", EmitterKind::Body);
        self.indent(EmitterKind::Body);

        self.emitln(
            "Internal__ErrorTrace(\"Accessed wrong union variant!\", 3);",
            EmitterKind::Body,
        );

        self.unindent(EmitterKind::Body);
        self.emitln("}", EmitterKind::Body);

        self.emitln("return self;", EmitterKind::Body);
        self.unindent(EmitterKind::Body);
        self.emitln("}", EmitterKind::Body);
        self.newline(EmitterKind::Body);
    }

    fn emit_union_check_tag_usage(
        &mut self,
        dereferenced_left_type_kind_id: usize,
        is_left_pointer: bool,
        left: NodeIndex,
        name: NodeIndex,
        tag: usize,
        emitter_kind: EmitterKind,
    ) {
        if self.is_unsafe_mode {
            self.gen_node_with_emitter(left, emitter_kind);

            if is_left_pointer {
                self.emit("->variant.", emitter_kind);
            } else {
                self.emit(".variant.", emitter_kind);
            }

            self.gen_node_with_emitter(name, emitter_kind);

            return;
        }

        self.emit("__", emitter_kind);
        self.emit_struct_name(dereferenced_left_type_kind_id, EmitterKind::Body);
        self.emit("__CheckTag((", emitter_kind);
        self.emit_type_kind_left(
            dereferenced_left_type_kind_id,
            EmitterKind::Body,
            false,
            false,
        );
        self.emit_type_kind_right(dereferenced_left_type_kind_id, emitter_kind, false);
        self.emit("*)", emitter_kind);

        if !is_left_pointer {
            self.emit("&", emitter_kind);
        }

        self.gen_node_with_emitter(left, emitter_kind);

        self.emit(", ", emitter_kind);
        self.emit(&tag.to_string(), emitter_kind);
        self.emit(")->variant.", emitter_kind);

        self.gen_node_with_emitter(name, emitter_kind);
    }

    fn emit_union_with_tag(&mut self, type_kind_id: usize) {
        self.emit("static inline ", EmitterKind::FunctionPrototype);
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::FunctionPrototype, true);
        self.emit("* __", EmitterKind::FunctionPrototype);
        self.emit_struct_name(type_kind_id, EmitterKind::FunctionPrototype);
        self.emit("__WithTag(", EmitterKind::FunctionPrototype);
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.emit(" *self", EmitterKind::FunctionPrototype);
        self.emit_type_kind_right(type_kind_id, EmitterKind::FunctionPrototype, false);
        self.emitln(", intptr_t tag);", EmitterKind::FunctionPrototype);
        self.newline(EmitterKind::FunctionPrototype);

        self.emit("static inline ", EmitterKind::Body);
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, true, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, true);
        self.emit("* __", EmitterKind::Body);
        self.emit_struct_name(type_kind_id, EmitterKind::Body);
        self.emit("__WithTag(", EmitterKind::Body);
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emit(" *self", EmitterKind::Body);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.emitln(", intptr_t tag) {", EmitterKind::Body);
        self.indent(EmitterKind::Body);
        self.emitln("self->tag = tag;", EmitterKind::Body);
        self.emitln("return self;", EmitterKind::Body);
        self.unindent(EmitterKind::Body);
        self.emitln("}", EmitterKind::Body);
        self.newline(EmitterKind::Body);
    }

    fn emit_union_with_tag_usage(
        &mut self,
        dereferenced_left_type_kind_id: usize,
        is_left_pointer: bool,
        left: NodeIndex,
        name: NodeIndex,
        tag: usize,
        emitter_kind: EmitterKind,
    ) {
        self.emit("__", emitter_kind);
        self.emit_struct_name(dereferenced_left_type_kind_id, emitter_kind);
        self.emit("__WithTag((", emitter_kind);
        self.emit_type_kind_left(dereferenced_left_type_kind_id, emitter_kind, false, false);
        self.emit_type_kind_right(dereferenced_left_type_kind_id, emitter_kind, false);
        self.emit("*)", emitter_kind);

        if !is_left_pointer {
            self.emit("&", emitter_kind);
        }

        self.gen_node_with_emitter(left, emitter_kind);

        self.emit(", ", emitter_kind);
        self.emit(&tag.to_string(), emitter_kind);

        self.emit(")->variant.", emitter_kind);
        self.gen_node_with_emitter(name, emitter_kind);
    }

    fn emit_equality(
        &mut self,
        type_kind_id: usize,
        left_prefix: Option<&str>,
        left: NodeIndex,
        right_prefix: Option<&str>,
        right: NodeIndex,
        is_equal: bool,
        emitter_kind: EmitterKind,
    ) {
        match self.type_kinds.get_by_id(type_kind_id) {
            TypeKind::Struct {
                primitive_type: PrimitiveType::None,
                ..
            } => {
                if !is_equal {
                    self.emit("!", emitter_kind);
                }

                self.emit_struct_equals_usage(
                    type_kind_id,
                    left_prefix,
                    left,
                    right_prefix,
                    right,
                    emitter_kind,
                )
            }
            TypeKind::Array { .. } => {
                if !is_equal {
                    self.emit("!", emitter_kind);
                }

                self.emit("memcmp(", emitter_kind);

                if let Some(left_prefix) = left_prefix {
                    self.emit(left_prefix, emitter_kind);
                }

                self.gen_node_with_emitter(left, emitter_kind);

                self.emit(", ", emitter_kind);

                if let Some(right_prefix) = right_prefix {
                    self.emit(right_prefix, emitter_kind);
                }

                self.gen_node_with_emitter(right, emitter_kind);

                self.emit(", ", emitter_kind);
                self.emit_type_size(type_kind_id, emitter_kind);
                self.emit(")", emitter_kind);
            }
            _ => {
                if let Some(left_prefix) = left_prefix {
                    self.emit(left_prefix, emitter_kind);
                }

                self.gen_node_with_emitter(left, emitter_kind);

                if is_equal {
                    self.emit(" == ", emitter_kind);
                } else {
                    self.emit(" != ", emitter_kind);
                }

                if let Some(right_prefix) = right_prefix {
                    self.emit(right_prefix, emitter_kind);
                }

                self.gen_node_with_emitter(right, emitter_kind);
            }
        }
    }

    fn emit_struct_equals(&mut self, type_kind_id: usize) {
        self.emit("static inline bool __", EmitterKind::FunctionPrototype);
        self.emit_struct_name(type_kind_id, EmitterKind::FunctionPrototype);
        self.emit("__Equals(", EmitterKind::FunctionPrototype);
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.emit(" *left, ", EmitterKind::FunctionPrototype);
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.emitln(" *right);", EmitterKind::FunctionPrototype);
        self.newline(EmitterKind::FunctionPrototype);

        self.emit("static inline bool __", EmitterKind::Body);
        self.emit_struct_name(type_kind_id, EmitterKind::Body);
        self.emit("__Equals(", EmitterKind::Body);
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emit(" *left, ", EmitterKind::Body);
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emitln(" *right) {", EmitterKind::Body);
        self.indent(EmitterKind::Body);

        assert_matches!(
            TypeKind::Struct {
                fields,
                is_union,
                ..
            },
            self.type_kinds.get_by_id(type_kind_id)
        );

        if is_union {
            self.emitln("if (left->tag != right->tag) {", EmitterKind::Body);
            self.indent(EmitterKind::Body);
            self.emitln("return false;", EmitterKind::Body);
            self.unindent(EmitterKind::Body);
            self.emitln("}", EmitterKind::Body);

            self.emitln("switch (left->tag) {", EmitterKind::Body);
            for (i, field) in fields.iter().enumerate() {
                self.emit("case ", EmitterKind::Body);
                self.emit(&i.to_string(), EmitterKind::Body);
                self.emit(": return ", EmitterKind::Body);
                self.emit_equality(
                    field.type_kind_id,
                    Some("left->variant."),
                    field.name,
                    Some("right->variant."),
                    field.name,
                    true,
                    EmitterKind::Body,
                );
                self.emitln(";", EmitterKind::Body);
            }
            self.emitln("}", EmitterKind::Body);
            self.emitln("return true;", EmitterKind::Body);
        } else {
            self.emit("return ", EmitterKind::Body);

            if fields.is_empty() {
                self.emit("true", EmitterKind::Body);
            }

            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    self.emit(" && ", EmitterKind::Body);
                }

                self.emit_equality(
                    field.type_kind_id,
                    Some("left->"),
                    field.name,
                    Some("right->"),
                    field.name,
                    true,
                    EmitterKind::Body,
                );
            }

            self.emitln(";", EmitterKind::Body);
        }

        self.unindent(EmitterKind::Body);
        self.emitln("}", EmitterKind::Body);
        self.newline(EmitterKind::Body);
    }

    fn emit_struct_equals_usage(
        &mut self,
        type_kind_id: usize,
        left_prefix: Option<&str>,
        left: NodeIndex,
        right_prefix: Option<&str>,
        right: NodeIndex,
        emitter_kind: EmitterKind,
    ) {
        self.emit("__", emitter_kind);
        self.emit_struct_name(type_kind_id, emitter_kind);
        self.emit("__Equals(&", emitter_kind);

        if let Some(left_prefix) = left_prefix {
            self.emit(left_prefix, emitter_kind);
        }

        self.gen_node_with_emitter(left, emitter_kind);

        self.emit(", &", emitter_kind);

        if let Some(right_prefix) = right_prefix {
            self.emit(right_prefix, emitter_kind);
        }

        self.gen_node_with_emitter(right, emitter_kind);

        self.emit(")", emitter_kind);
    }

    fn is_typed_expression_array_literal(&self, expression: NodeIndex) -> bool {
        matches!(
            self.get_typer_node(expression).node_kind,
            NodeKind::ArrayLiteral { .. }
        )
    }
}
