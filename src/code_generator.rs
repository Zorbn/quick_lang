use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};

use crate::{
    const_value::ConstValue,
    emitter::Emitter,
    emitter_stack::EmitterStack,
    parser::{DeclarationKind, MethodKind, NodeIndex, NodeKind, Op},
    type_kinds::{get_field_index_by_name, TypeKind, TypeKinds},
    typer::{InstanceKind, Namespace, Type, TypedDefinition, TypedNode},
    utils::is_typed_expression_array_literal,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum EmitterKind {
    TypePrototype,
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
    string_view_type_kind_id: usize,
    main_function_declaration: Option<NodeIndex>,
    typed_definitions: Vec<TypedDefinition>,

    pub header_emitter: Emitter,
    pub type_prototype_emitter: Emitter,
    pub function_prototype_emitter: Emitter,
    pub global_variable_emitter: Emitter,
    pub body_emitters: EmitterStack,

    function_declaration_needing_init: Option<NodeIndex>,
    temp_variable_count: usize,
    loop_depth_stack: Vec<LoopDepth>,
    switch_depth_stack: Vec<usize>,
    is_debug_mode: bool,
}

impl CodeGenerator {
    pub fn new(
        typed_nodes: Vec<TypedNode>,
        type_kinds: TypeKinds,
        namespaces: Vec<Namespace>,
        string_view_type_kind_id: usize,
        main_function_declaration: Option<NodeIndex>,
        typed_definitions: Vec<TypedDefinition>,
        is_debug_mode: bool,
    ) -> Self {
        let mut code_generator = Self {
            typed_nodes,
            type_kinds,
            namespaces,
            string_view_type_kind_id,
            main_function_declaration,
            typed_definitions,
            header_emitter: Emitter::new(0),
            type_prototype_emitter: Emitter::new(0),
            function_prototype_emitter: Emitter::new(0),
            global_variable_emitter: Emitter::new(0),
            body_emitters: EmitterStack::new(),
            function_declaration_needing_init: None,
            temp_variable_count: 0,
            loop_depth_stack: Vec::new(),
            switch_depth_stack: Vec::new(),
            is_debug_mode,
        };

        code_generator.header_emitter.emitln("#include <stdint.h>");
        code_generator.header_emitter.emitln("#include <stdbool.h>");
        code_generator.header_emitter.emitln("#include <assert.h>");
        code_generator
            .header_emitter
            .emitln("void *memmove(void *dst, const void *src, size_t size);");
        code_generator
            .header_emitter
            .emitln("int memcmp(const void *ptr1, const void *ptr2, size_t num);");
        code_generator.header_emitter.newline();
        code_generator.body_emitters.push(1);

        if is_debug_mode {
            code_generator.emit_bounds_check();
        }

        code_generator.emit_main_function();

        code_generator
    }

    fn get_typer_node(&self, index: NodeIndex) -> &TypedNode {
        &self.typed_nodes[index.node_index]
    }

    pub fn gen(&mut self) {
        for i in 0..self.typed_definitions.len() {
            let TypedDefinition { index, is_shallow } = self.typed_definitions[i];

            let TypedNode {
                node_kind,
                node_type,
                namespace_id,
            } = self
                .get_typer_node(index)
                .clone();

            match node_kind {
                NodeKind::StructDefinition {
                    name,
                    fields,
                    is_union,
                    ..
                } => self.struct_definition(name, fields, is_union, node_type, namespace_id),
                NodeKind::EnumDefinition {
                    name,
                    variant_names,
                } => self.enum_definition(name, variant_names, node_type, namespace_id),
                NodeKind::Function {
                    declaration,
                    scoped_statement,
                } => self.function(declaration, scoped_statement, is_shallow, node_type, namespace_id),
                NodeKind::ExternFunction { declaration } => self.extern_function(declaration, node_type, namespace_id),
                NodeKind::VariableDeclaration {
                    declaration_kind,
                    name,
                    expression,
                    ..
                } => {
                    self.variable_declaration(declaration_kind, name, expression, is_shallow, node_type, namespace_id, EmitterKind::GlobalVariable);
                    self.global_variable_emitter.emitln(";");
                    self.global_variable_emitter.newline()
                },
                _ => panic!("unexpected definition kind: {:?}", node_kind),
            }
        }
    }

    fn gen_node(&mut self, index: NodeIndex) {
        let TypedNode {
            node_kind,
            node_type,
            namespace_id,
        } = self.get_typer_node(index).clone();

        match node_kind {
            NodeKind::TopLevel {
                functions,
                structs,
                enums,
                ..
            } => self.top_level(functions, structs, enums, node_type, namespace_id),
            NodeKind::StructDefinition {
                name,
                fields,
                is_union,
                ..
            } => self.struct_definition(name, fields, is_union, node_type, namespace_id),
            NodeKind::EnumDefinition {
                name,
                variant_names,
                ..
            } => self.enum_definition(name, variant_names, node_type, namespace_id),
            NodeKind::Field { name, type_name } => self.field(name, type_name, node_type, namespace_id),
            NodeKind::Function {
                declaration,
                scoped_statement,
            } => self.function(declaration, scoped_statement, false, node_type, namespace_id),
            NodeKind::FunctionDeclaration {
                name,
                params,
                generic_params,
                return_type_name,
                ..
            } => {
                self.function_declaration(name, params, generic_params, return_type_name, node_type, namespace_id)
            }
            NodeKind::ExternFunction { declaration } => {
                self.extern_function(declaration, node_type, namespace_id)
            }
            NodeKind::Param { name, type_name } => self.param(name, type_name, node_type, namespace_id),
            NodeKind::Block { statements } => self.block(statements, node_type, namespace_id),
            NodeKind::Statement { inner } => self.statement(inner, node_type, namespace_id),
            NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                expression,
                ..
            } => {
                self.variable_declaration(declaration_kind, name, expression, false, node_type, namespace_id, EmitterKind::Body)
            }
            NodeKind::ReturnStatement { expression } => {
                self.return_statement(expression, node_type, namespace_id)
            }
            NodeKind::BreakStatement => self.break_statement(node_type, namespace_id),
            NodeKind::ContinueStatement => self.continue_statement(node_type, namespace_id),
            NodeKind::DeferStatement { statement } => self.defer_statement(statement, node_type, namespace_id),
            NodeKind::IfStatement {
                expression,
                scoped_statement,
                next,
            } => self.if_statement(expression, scoped_statement, next, node_type, namespace_id),
            NodeKind::SwitchStatement {
                expression,
                case_statement,
            } => self.switch_statement(expression, case_statement, node_type, namespace_id),
            NodeKind::CaseStatement {
                expression,
                scoped_statement,
                next,
            } => self.case_statement(expression, scoped_statement, next, node_type, namespace_id),
            NodeKind::WhileLoop {
                expression,
                scoped_statement,
            } => self.while_loop(expression, scoped_statement, index, node_type, namespace_id),
            NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                scoped_statement,
            } => self.for_loop(iterator, op, from, to, by, scoped_statement, index, node_type, namespace_id),
            NodeKind::ConstExpression { inner } => self.const_expression(inner, node_type, namespace_id, EmitterKind::Body),
            NodeKind::Binary { left, op, right } => self.binary(left, op, right, node_type, namespace_id, EmitterKind::Body),
            NodeKind::UnaryPrefix { op, right } => self.unary_prefix(op, right, node_type, namespace_id, EmitterKind::Body),
            NodeKind::UnarySuffix { left, op } => self.unary_suffix(left, op, node_type, namespace_id, EmitterKind::Body),
            NodeKind::Call { left, args, method_kind } => self.call(left, args, method_kind, node_type, namespace_id, EmitterKind::Body),
            NodeKind::IndexAccess { left, expression } => {
                self.index_access(left, expression, node_type, namespace_id, EmitterKind::Body)
            }
            NodeKind::FieldAccess { left, name } => self.field_access(left, name, node_type, namespace_id, EmitterKind::Body),
            NodeKind::Cast { left, type_name } => self.cast(left, type_name, node_type, namespace_id, EmitterKind::Body),
            NodeKind::GenericSpecifier {
                left,
                generic_arg_type_names,
            } => self.generic_specifier(left, generic_arg_type_names, node_type, namespace_id, EmitterKind::Body),
            NodeKind::Name { text } => self.name(text, node_type, namespace_id, EmitterKind::Body),
            NodeKind::Identifier { name } => self.identifier(name, node_type, namespace_id, EmitterKind::Body),
            NodeKind::IntLiteral { text } => self.int_literal(text, node_type, namespace_id, EmitterKind::Body),
            NodeKind::FloatLiteral { text } => self.float_literal(text, node_type, namespace_id, EmitterKind::Body),
            NodeKind::CharLiteral { value } => self.char_literal(value, node_type, namespace_id, EmitterKind::Body),
            NodeKind::StringLiteral { text } => self.string_literal(text, node_type, namespace_id, EmitterKind::Body),
            NodeKind::BoolLiteral { value } => self.bool_literal(value, node_type, namespace_id, EmitterKind::Body),
            NodeKind::ArrayLiteral {
                elements,
                repeat_count_const_expression,
            } => self.array_literal(elements, repeat_count_const_expression, node_type, namespace_id, EmitterKind::Body),
            NodeKind::StructLiteral {
                left,
                field_literals,
            } => self.struct_literal(left, field_literals, node_type, namespace_id, EmitterKind::Body),
            NodeKind::FieldLiteral { name, expression } => {
                self.field_literal(name, expression, node_type, namespace_id, EmitterKind::Body)
            }
            NodeKind::TypeSize { type_name } => self.type_size(type_name, node_type, namespace_id, EmitterKind::Body),
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
    }

    fn gen_node_with_emitter(&mut self, index: NodeIndex, kind: EmitterKind) {
        let TypedNode {
            node_kind,
            node_type,
            namespace_id,
        } = self.get_typer_node(index).clone();

        match node_kind {
            NodeKind::ConstExpression { inner } => self.const_expression(inner, node_type, namespace_id, kind),
            NodeKind::Binary { left, op, right } => self.binary(left, op, right, node_type, namespace_id, kind),
            NodeKind::UnaryPrefix { op, right } => self.unary_prefix(op, right, node_type, namespace_id, kind),
            NodeKind::UnarySuffix { left, op } => self.unary_suffix(left, op, node_type, namespace_id, kind),
            NodeKind::Call { left, args, method_kind } => self.call(left, args, method_kind, node_type, namespace_id, kind),
            NodeKind::IndexAccess { left, expression } => {
                self.index_access(left, expression, node_type, namespace_id, kind)
            }
            NodeKind::FieldAccess { left, name } => self.field_access(left, name, node_type, namespace_id, kind),
            NodeKind::Cast { left, type_name } => self.cast(left, type_name, node_type, namespace_id, kind),
            NodeKind::GenericSpecifier {
                left,
                generic_arg_type_names,
            } => self.generic_specifier(left, generic_arg_type_names, node_type, namespace_id, kind),
            NodeKind::Name { text } => self.name(text, node_type, namespace_id, kind),
            NodeKind::Identifier { name } => self.identifier(name, node_type, namespace_id, kind),
            NodeKind::IntLiteral { text } => self.int_literal(text, node_type, namespace_id, kind),
            NodeKind::FloatLiteral { text } => self.float_literal(text, node_type, namespace_id, kind),
            NodeKind::CharLiteral { value } => self.char_literal(value, node_type, namespace_id, kind),
            NodeKind::StringLiteral { text } => self.string_literal(text, node_type, namespace_id, kind),
            NodeKind::BoolLiteral { value } => self.bool_literal(value, node_type, namespace_id, kind),
            NodeKind::ArrayLiteral {
                elements,
                repeat_count_const_expression,
            } => self.array_literal(elements, repeat_count_const_expression, node_type, namespace_id, kind),
            NodeKind::StructLiteral {
                left,
                field_literals,
            } => self.struct_literal(left, field_literals, node_type, namespace_id, kind),
            NodeKind::FieldLiteral { name, expression } => {
                self.field_literal(name, expression, node_type, namespace_id, kind)
            }
            NodeKind::TypeSize { type_name } => self.type_size(type_name, node_type, namespace_id, kind),
            _ => self.gen_node(index)
        }
    }

    fn top_level(
        &mut self,
        _functions: Arc<Vec<NodeIndex>>,
        _structs: Arc<Vec<NodeIndex>>,
        _enums: Arc<Vec<NodeIndex>>,
        _node_type: Option<Type>,
        _namespace_id: Option<usize>,
    ) {
    }

    fn struct_definition(
        &mut self,
        _name: NodeIndex,
        fields: Arc<Vec<NodeIndex>>,
        is_union: bool,
        node_type: Option<Type>,
        _namespace_id: Option<usize>,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        self.emit_struct_equals(type_kind_id);

        if is_union {
            self.emit_union_check_tag(type_kind_id);
            self.emit_union_with_tag(type_kind_id);
        }

        self.type_prototype_emitter.emit("struct ");
        self.emit_struct_name(type_kind_id, EmitterKind::TypePrototype);

        self.type_prototype_emitter.emit(" ");

        if is_union {
            self.type_prototype_emitter.emitln("{");
            self.type_prototype_emitter.indent();
            self.type_prototype_emitter.emitln("intptr_t tag;");
            self.type_prototype_emitter.emit("union ");
        }

        self.type_prototype_emitter.emitln("{");
        self.type_prototype_emitter.indent();

        if fields.is_empty() {
            // C doesn't allow empty structs.
            self.type_prototype_emitter.emitln("bool placeholder;");
        }

        for field in fields.iter() {
            self.gen_node(*field);
        }

        self.type_prototype_emitter.unindent();
        self.type_prototype_emitter.emit("}");

        if is_union {
            self.type_prototype_emitter.emitln(" variant;");
            self.type_prototype_emitter.unindent();
            self.type_prototype_emitter.emit("}");
        }

        self.type_prototype_emitter.emitln(";");
        self.type_prototype_emitter.newline();
    }

    fn enum_definition(
        &mut self,
        name: NodeIndex,
        variant_names: Arc<Vec<NodeIndex>>,
        _node_type: Option<Type>,
        _namespace_id: Option<usize>,
    ) {
        self.type_prototype_emitter.emit("enum ");
        self.gen_node_with_emitter(name, EmitterKind::TypePrototype);
        self.type_prototype_emitter.emitln(" {");
        self.type_prototype_emitter.indent();

        for variant_name in variant_names.iter() {
            self.type_prototype_emitter.emit("__");
            self.gen_node_with_emitter(name, EmitterKind::TypePrototype);
            self.gen_node_with_emitter(*variant_name, EmitterKind::TypePrototype);
            self.type_prototype_emitter.emitln(",");
        }

        self.type_prototype_emitter.unindent();
        self.type_prototype_emitter.emitln("};");
        self.type_prototype_emitter.newline();
    }

    fn field(&mut self, name: NodeIndex, _type_name: NodeIndex, node_type: Option<Type>, _namespace_id: Option<usize>) {
        self.emit_type_kind_left(
            node_type.clone().unwrap().type_kind_id,
            EmitterKind::TypePrototype,
            false,
            true,
        );
        self.gen_node_with_emitter(name, EmitterKind::TypePrototype);
        self.emit_type_kind_right(
            node_type.unwrap().type_kind_id,
            EmitterKind::TypePrototype,
            false,
        );
        self.type_prototype_emitter.emitln(";");
    }

    fn function(
        &mut self,
        declaration: NodeIndex,
        scoped_statement: NodeIndex,
        is_shallow: bool,
        node_type: Option<Type>,
        _namespace_id: Option<usize>,
    ) {
        if is_shallow {
            let NodeKind::FunctionDeclaration {
                name,
                params,
                generic_params,
                ..
            } = self.get_typer_node(declaration).node_kind.clone()
            else {
                panic!("invalid function declaration");
            };

            let type_kind_id = node_type.unwrap().type_kind_id;
            let has_generic_params = !generic_params.is_empty();

            self.emit_function_declaration(
                EmitterKind::FunctionPrototype,
                name,
                &params,
                has_generic_params,
                type_kind_id,
            );

            self.function_prototype_emitter.emitln(";");
            self.function_prototype_emitter.newline();
        } else {
            self.gen_node(declaration);
            self.function_declaration_needing_init = Some(declaration);
            self.gen_node(scoped_statement);
            self.body_emitters.top().body.newline();
            self.body_emitters.top().body.newline();
        }
    }

    fn function_declaration(
        &mut self,
        name: NodeIndex,
        params: Arc<Vec<NodeIndex>>,
        generic_params: Arc<Vec<NodeIndex>>,
        _return_type_name: NodeIndex,
        node_type: Option<Type>,
        _namespace_id: Option<usize>,
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
        self.function_prototype_emitter.emitln(";");
        self.function_prototype_emitter.newline();

        self.emit_function_declaration(
            EmitterKind::Body,
            name,
            &params,
            has_generic_params,
            type_kind_id,
        );
        self.body_emitters.top().body.emit(" ");
    }

    fn extern_function(&mut self, declaration: NodeIndex, node_type: Option<Type>, _namespace_id: Option<usize>) {
        self.function_prototype_emitter.emit("extern ");

        let NodeKind::FunctionDeclaration { name, params, .. } =
            self.get_typer_node(declaration).node_kind.clone()
        else {
            panic!("invalid function declaration");
        };

        let type_kind_id = node_type.unwrap().type_kind_id;

        self.emit_function_declaration(
            EmitterKind::FunctionPrototype,
            name,
            &params,
            false,
            type_kind_id,
        );

        self.function_prototype_emitter.emitln(";");
        self.function_prototype_emitter.newline();
    }

    fn param(&mut self, name: NodeIndex, _type_name: NodeIndex, node_type: Option<Type>, _namespace_id: Option<usize>) {
        self.emit_param(name, node_type.unwrap().type_kind_id, EmitterKind::Body);
    }

    fn copy_array_params(&mut self, function_declaration: NodeIndex) {
        let NodeKind::FunctionDeclaration { params, .. } =
            self.get_typer_node(function_declaration).node_kind.clone()
        else {
            panic!("invalid function declaration needing init");
        };

        for param in params.iter() {
            let TypedNode {
                node_kind: NodeKind::Param { name, .. },
                node_type,
                ..
            } = self.get_typer_node(*param).clone()
            else {
                panic!("invalid param in function declaration needing init");
            };

            let type_kind_id = node_type.unwrap().type_kind_id;

            if !matches!(
                &self.type_kinds.get_by_id(type_kind_id),
                TypeKind::Array { .. }
            ) {
                continue;
            }

            let NodeKind::Name { text: name_text } = self.get_typer_node(name).node_kind.clone()
            else {
                panic!("invalid parameter name");
            };
            let copy_name = format!("__{}", &name_text);

            self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, true);
            self.body_emitters.top().body.emit(&copy_name);
            self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
            self.body_emitters.top().body.emitln(";");

            self.emit_memmove_name_to_name(&copy_name, &name_text, type_kind_id, EmitterKind::Body);
            self.body_emitters.top().body.emitln(";");

            self.gen_node(name);
            self.body_emitters.top().body.emit(" = ");
            self.body_emitters.top().body.emit(&copy_name);
            self.body_emitters.top().body.emitln(";");
        }
    }

    fn block(&mut self, statements: Arc<Vec<NodeIndex>>, _node_type: Option<Type>, _namespace_id: Option<usize>) {
        self.body_emitters.top().body.emitln("{");
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
            let TypedNode {
                node_kind: NodeKind::Statement { inner },
                ..
            } = self.get_typer_node(*statement)
            else {
                panic!("last statement is not a statement");
            };

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
        self.body_emitters.top().body.emit("}");
    }

    fn statement(&mut self, inner: Option<NodeIndex>, _node_type: Option<Type>, _namespace_id: Option<usize>) {
        let Some(inner) = inner else {
            self.body_emitters.top().body.emitln(";");
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
                node_kind: NodeKind::ForLoop { .. },
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
            self.body_emitters.top().body.emitln(";");
        }

        if needs_newline {
            self.body_emitters.top().body.newline();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn variable_declaration(
        &mut self,
        declaration_kind: DeclarationKind,
        name: NodeIndex,
        expression: Option<NodeIndex>,
        is_shallow: bool,
        node_type: Option<Type>,
        namespace_id: Option<usize>,
        kind: EmitterKind,
    ) {
        if is_shallow {
            self.emitter(kind).emit("extern ");
        }

        let type_kind_id = node_type.unwrap().type_kind_id;

        let is_array = matches!(
            &self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Array { .. }
        );
        let needs_const = declaration_kind != DeclarationKind::Var && !is_array && kind != EmitterKind::GlobalVariable;

        self.emit_type_kind_left(type_kind_id, kind, false, true);

        if needs_const {
            self.emitter(kind).emit("const ");
        }

        if let Some(namespace_id) = namespace_id {
            self.emit_namespace(namespace_id, kind);
        }

        self.gen_node_with_emitter(name, kind);

        self.emit_type_kind_right(type_kind_id, kind, false);

        if is_shallow {
            return;
        }

        let Some(expression) = expression else {
            return;
        };

        if is_array && !is_typed_expression_array_literal(&self.typed_nodes, expression) {
            self.emitter(kind).emitln(";");

            let NodeKind::Name { text: name_text } = self.get_typer_node(name).node_kind.clone()
            else {
                panic!("invalid variable name");
            };
            self.emit_memmove_expression_to_name(&name_text, expression, type_kind_id, kind);
        } else {
            self.emitter(kind).emit(" = ");
            self.gen_node_with_emitter(expression, kind);
        }
    }

    fn return_statement(&mut self, expression: Option<NodeIndex>, _node_type: Option<Type>, _namespace_id: Option<usize>) {
        self.body_emitters.early_exiting_scopes(None);

        let expression = if let Some(expression) = expression {
            expression
        } else {
            self.body_emitters.top().body.emit("return");
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
            if is_typed_expression_array_literal(&self.typed_nodes, expression) {
                let temp_name = self.temp_variable_name("temp");

                self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, true);
                self.body_emitters.top().body.emit(&temp_name);
                self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
                self.body_emitters.top().body.emit(" = ");
                self.gen_node(expression);
                self.body_emitters.top().body.emitln(";");

                self.emit_memmove_name_to_name("__return", &temp_name, type_kind_id, EmitterKind::Body);
                self.body_emitters.top().body.emitln(";");
            } else {
                self.emit_memmove_expression_to_name("__return", expression, type_kind_id, EmitterKind::Body);
                self.body_emitters.top().body.emitln(";");
            }

            self.body_emitters.top().body.emit("return __return");
        } else {
            self.body_emitters.top().body.emit("return ");
            self.gen_node(expression);
        }
    }

    fn break_statement(&mut self, _node_type: Option<Type>, _namespace_id: Option<usize>) {
        let last_loop = self.loop_depth_stack.last_mut().unwrap();
        let scope_count = self.body_emitters.len() - last_loop.depth;

        self.body_emitters.early_exiting_scopes(Some(scope_count));

        if let Some(last_switch_depth) = self.switch_depth_stack.last() {
            if *last_switch_depth > last_loop.depth {
                last_loop.was_label_used = true;

                let node_index = last_loop.index.node_index;
                self.body_emitters.top().body.emit("goto __break");
                self.emit_number_backwards(node_index, EmitterKind::Body);

                return;
            }
        }

        self.body_emitters.top().body.emit("break");
    }

    fn continue_statement(&mut self, _node_type: Option<Type>, _namespace_id: Option<usize>) {
        let scope_count = self.body_emitters.len() - self.loop_depth_stack.last().unwrap().depth;

        self.body_emitters.early_exiting_scopes(Some(scope_count));
        self.body_emitters.top().body.emit("continue");
    }

    fn defer_statement(&mut self, statement: NodeIndex, _node_type: Option<Type>, _namespace_id: Option<usize>) {
        self.body_emitters.push(0);
        self.gen_node(statement);
        self.body_emitters.pop_to_bottom();
    }

    fn if_statement(
        &mut self,
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        next: Option<NodeIndex>,
        _node_type: Option<Type>,
        _namespace_id: Option<usize>
    ) {
        self.body_emitters.top().body.emit("if (");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(") ");
        self.gen_node(scoped_statement);

        if let Some(next) = next {
            self.body_emitters.top().body.emit(" else ");
            self.gen_node(next);

            if !matches!(
                self.get_typer_node(next).node_kind,
                NodeKind::IfStatement { .. }
            ) {
                self.body_emitters.top().body.newline();
            }
        } else {
            self.body_emitters.top().body.newline();
        }
    }

    fn switch_statement(
        &mut self,
        expression: NodeIndex,
        case_statement: NodeIndex,
        _node_type: Option<Type>,
        _namespace_id: Option<usize>
    ) {
        self.switch_depth_stack.push(self.body_emitters.len());

        self.body_emitters.top().body.emit("switch (");
        self.gen_node(expression);
        self.body_emitters.top().body.emitln(") {");
        self.gen_node(case_statement);
        self.body_emitters.top().body.emitln("}");

        self.switch_depth_stack.pop();
    }

    fn case_statement(
        &mut self,
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        next: Option<NodeIndex>,
        _node_type: Option<Type>,
        _namespace_id: Option<usize>
    ) {
        self.body_emitters.top().body.emit("case ");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(": ");
        self.gen_node(scoped_statement);
        self.body_emitters.top().body.emitln(" break;");

        if let Some(next) = next {
            if !matches!(
                self.get_typer_node(next).node_kind,
                NodeKind::CaseStatement { .. }
            ) {
                self.body_emitters.top().body.emit("default: ");
                self.gen_node(next);
                self.body_emitters.top().body.emitln(" break;");
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

        self.body_emitters.top().body.emit("__break");
        self.emit_number_backwards(index.node_index, EmitterKind::Body);
        self.body_emitters.top().body.emitln(":;");
    }

    fn while_loop(
        &mut self,
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        index: NodeIndex,
        _node_type: Option<Type>,
        _namespace_id: Option<usize>
    ) {
        self.body_emitters.top().body.emit("while (");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(") ");

        self.loop_depth_stack.push(LoopDepth {
            index,
            depth: self.body_emitters.len(),
            was_label_used: false
        });

        self.gen_node(scoped_statement);
        self.body_emitters.top().body.newline();

        let loop_depth = self.loop_depth_stack.pop();
        self.emit_break_label_if_used(index, loop_depth)
    }

    #[allow(clippy::too_many_arguments)]
    fn for_loop(
        &mut self,
        iterator: NodeIndex,
        op: Op,
        from: NodeIndex,
        to: NodeIndex,
        by: Option<NodeIndex>,
        scoped_statement: NodeIndex,
        index: NodeIndex,
        _node_type: Option<Type>,
        _namespace_id: Option<usize>
    ) {
        self.body_emitters.top().body.emit("for (intptr_t ");
        self.gen_node(iterator);
        self.body_emitters.top().body.emit(" = ");
        self.gen_node(from);
        self.body_emitters.top().body.emit("; ");

        self.gen_node(iterator);
        self.emit_binary_op(op, EmitterKind::Body);
        self.gen_node(to);
        self.body_emitters.top().body.emit("; ");

        self.gen_node(iterator);
        self.body_emitters.top().body.emit(" += ");
        if let Some(by) = by {
            self.gen_node(by);
        } else {
            self.body_emitters.top().body.emit("1");
        }
        self.body_emitters.top().body.emit(") ");

        self.loop_depth_stack.push(LoopDepth {
            index,
            depth: self.body_emitters.len(),
            was_label_used: false
        });

        self.gen_node(scoped_statement);
        self.body_emitters.top().body.newline();

        let loop_depth = self.loop_depth_stack.pop();
        self.emit_break_label_if_used(index, loop_depth)
    }

    fn const_expression(&mut self, _inner: NodeIndex, node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        let Some(Type {
            instance_kind: InstanceKind::Const(const_value),
            ..
        }) = node_type
        else {
            panic!("invalid node type of const expression");
        };

        match const_value {
            ConstValue::Int { value } => self.emitter(kind).emit(&value.to_string()),
            ConstValue::UInt { value } => self.emitter(kind).emit(&value.to_string()),
            ConstValue::Float { value } => self.emitter(kind).emit(&value.to_string()),
            ConstValue::String { value } => {
                self.emitter(kind).emit_char('"');
                self.emitter(kind).emit(&value);
                self.emitter(kind).emit_char('"');
            }
            ConstValue::Char { value } => self.emitter(kind).emit_char(value),
            ConstValue::Bool { value } => {
                if value {
                    self.emitter(kind).emit("true");
                } else {
                    self.emitter(kind).emit("false");
                }
            }
        }
    }

    fn binary(&mut self, left: NodeIndex, op: Op, right: NodeIndex, node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        let left_type_kind_id = self.get_typer_node(left).node_type.as_ref().unwrap().type_kind_id;

        if op == Op::Assign {
            let type_kind_id = node_type.unwrap().type_kind_id;
            let is_array = matches!(
                &self.type_kinds.get_by_id(type_kind_id),
                TypeKind::Array { .. }
            );

            if is_array && !is_typed_expression_array_literal(&self.typed_nodes, left) {
                self.emit_memmove_expression_to_variable(left, right, type_kind_id, kind);
                return;
            }

            if let NodeKind::FieldAccess { left, name } = self.get_typer_node(left).node_kind {
                let left_type = self.get_typer_node(left).node_type.as_ref().unwrap();

                let (dereferenced_left_type_kind_id, is_left_pointer) =
                    if let TypeKind::Pointer {
                        inner_type_kind_id, ..
                    } = self.type_kinds.get_by_id(left_type.type_kind_id)
                    {
                        (inner_type_kind_id, true)
                    } else {
                        (left_type.type_kind_id, false)
                    };

                if let TypeKind::Struct {
                    fields, is_union, ..
                } = &self.type_kinds.get_by_id(dereferenced_left_type_kind_id)
                {
                    if *is_union {
                        let NodeKind::Name { text: name_text } =
                            self.get_typer_node(name).node_kind.clone()
                        else {
                            panic!("invalid field name in field access");
                        };

                        let Some(tag) =
                            get_field_index_by_name(&self.typed_nodes, &name_text, fields)
                        else {
                            panic!("tag not found in union assignment");
                        };

                        self.emit_union_with_tag_usage(dereferenced_left_type_kind_id, is_left_pointer, left, name, tag, kind);
                        self.emit_binary_op(op, kind);
                        self.gen_node_with_emitter(right, kind);

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
            self.emitter(kind).emit("(")
        }

        if matches!(op, Op::Equal | Op::NotEqual) {
            self.emit_equality(left_type_kind_id, None, left, None, right, op == Op::Equal, kind);
        } else {
            self.gen_node_with_emitter(left, kind);
            self.emit_binary_op(op, kind);
            self.gen_node_with_emitter(right, kind);
        }

        if needs_increased_precedence {
            self.emitter(kind).emit(")")
        }
    }

    fn unary_prefix(&mut self, op: Op, right: NodeIndex, _node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        self.emitter(kind).emit(match op {
            Op::Plus => "+",
            Op::Minus => "-",
            Op::Not => "!",
            Op::Reference => "&",
            _ => panic!("expected unary prefix operator"),
        });

        self.gen_node_with_emitter(right, kind);
    }

    fn unary_suffix(&mut self, left: NodeIndex, op: Op, _node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        self.emitter(kind).emit("(");
        self.emitter(kind).emit(match op {
            Op::Dereference => "*",
            _ => panic!("expected unary suffix operator"),
        });

        self.gen_node_with_emitter(left, kind);
        self.emitter(kind).emit(")");
    }

    fn call(&mut self, left: NodeIndex, args: Arc<Vec<NodeIndex>>, method_kind: MethodKind, node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        self.gen_node_with_emitter(left, kind);

        self.emitter(kind).emit("(");

        if method_kind != MethodKind::None {
            let mut caller = left;
            if let NodeKind::GenericSpecifier { left, .. } = self.get_typer_node(caller).node_kind {
                caller = left;
            }

            let NodeKind::FieldAccess { left, .. } = self.get_typer_node(caller).node_kind else {
                panic!("expected field access before method call");
            };

            if method_kind == MethodKind::ByReference {
                self.emitter(kind).emit("&");
            }

            self.gen_node_with_emitter(left, kind);

            if !args.is_empty() {
                self.emitter(kind).emit(", ");
            }
        }

        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                self.emitter(kind).emit(", ");
            }

            self.gen_node_with_emitter(*arg, kind);
        }

        let type_kind_id = node_type.unwrap().type_kind_id;
        if matches!(
            &self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Array { .. }
        ) {
            if args.len() > 0 {
                self.emitter(kind).emit(", ");
            }

            let return_array_name = self.temp_variable_name("returnArray");

            self.emit_type_kind_left(type_kind_id, EmitterKind::Top, false, true);
            self.body_emitters.top().top.emit(&return_array_name);
            self.emit_type_kind_right(type_kind_id, EmitterKind::Top, false);
            self.body_emitters.top().top.emitln(";");

            self.emitter(kind).emit(&return_array_name);
        }
        self.emitter(kind).emit(")");
    }

    fn index_access(&mut self, left: NodeIndex, expression: NodeIndex, _node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        self.gen_node_with_emitter(left, kind);
        self.emitter(kind).emit("[");

        if self.is_debug_mode {
            let left_type = self.get_typer_node(left).node_type.as_ref().unwrap();
            let TypeKind::Array { element_count, .. } =
                &self.type_kinds.get_by_id(left_type.type_kind_id)
            else {
                panic!("tried to perform an index access on a non-array type");
            };
            let element_count = *element_count;

            self.emitter(kind).emit("__BoundsCheck(");
            self.gen_node_with_emitter(expression, kind);
            self.emitter(kind).emit(", ");
            self.body_emitters
                .top()
                .body
                .emit(&element_count.to_string());
            self.emitter(kind).emit(")");
        } else {
            self.gen_node_with_emitter(expression, kind);
        }

        self.emitter(kind).emit("]");
    }

    fn field_access(&mut self, left: NodeIndex, name: NodeIndex, node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        let left_type = self.get_typer_node(left).node_type.as_ref().unwrap();

        let field_type_kind_id = node_type.unwrap().type_kind_id;
        let mut is_method_access = false;
        match self.type_kinds.get_by_id(field_type_kind_id) {
            TypeKind::Tag => {
                let TypeKind::Struct {
                    fields, is_union, ..
                } = &self.type_kinds.get_by_id(left_type.type_kind_id)
                else {
                    panic!("expected tag field to be part of a struct");
                };

                if left_type.instance_kind == InstanceKind::Name && *is_union {
                    let NodeKind::Name { text: name_text } =
                        self.get_typer_node(name).node_kind.clone()
                    else {
                        panic!("invalid tag name in tag access");
                    };

                    let Some(tag) = get_field_index_by_name(&self.typed_nodes, &name_text, fields)
                    else {
                        panic!("tag not found in field access");
                    };

                    self.emitter(kind).emit(&tag.to_string());

                    return;
                }
            }
            TypeKind::Function { .. } => {
                is_method_access = true;
            }
            _ => {}
        }

        let (dereferenced_left_type_kind_id, is_left_pointer) =
            if let TypeKind::Pointer {
                inner_type_kind_id, ..
            } = self.type_kinds.get_by_id(left_type.type_kind_id)
            {
                (inner_type_kind_id, true)
            } else {
                (left_type.type_kind_id, false)
            };

        if let TypeKind::Struct {
            fields, is_union, ..
        } = &self.type_kinds.get_by_id(dereferenced_left_type_kind_id)
        {
            if *is_union {
                let NodeKind::Name { text: name_text } =
                    self.get_typer_node(name).node_kind.clone()
                else {
                    panic!("invalid field name in field access");
                };

                let Some(tag) = get_field_index_by_name(&self.typed_nodes, &name_text, fields)
                else {
                    panic!("tag not found in field access");
                };

                self.emit_union_check_tag_usage(dereferenced_left_type_kind_id, is_left_pointer, left, name, tag, kind);

                return;
            }
        }

        match self.type_kinds.get_by_id(dereferenced_left_type_kind_id) {
            TypeKind::Struct { .. } if left_type.instance_kind != InstanceKind::Name && !is_method_access => {
                self.gen_node_with_emitter(left, kind);

                if is_left_pointer {
                    self.emitter(kind).emit("->")
                } else {
                    self.emitter(kind).emit(".")
                }
            }
            TypeKind::Enum {
                name: enum_name, ..
            } => {
                self.emitter(kind).emit("__");
                self.gen_node_with_emitter(enum_name, kind);
            }
            TypeKind::Array { element_count, .. } => {
                // On arrays, only the "count" field is allowed.
                self.body_emitters
                    .top()
                    .body
                    .emit(&element_count.to_string());
                return;
            }
            TypeKind::Namespace { .. } | TypeKind::Struct { .. } => {}
            _ => panic!("tried to access type that cannot be accessed"),
        }

        self.gen_node_with_emitter(name, kind);
    }

    fn cast(&mut self, left: NodeIndex, _type_name: NodeIndex, node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        if let TypeKind::Tag { .. } = &self.type_kinds.get_by_id(type_kind_id) {
            let left_type_kind_id = self
                .get_typer_node(left)
                .node_type
                .as_ref()
                .unwrap()
                .type_kind_id;

            let TypeKind::Struct { is_union, .. } = &self.type_kinds.get_by_id(left_type_kind_id)
            else {
                panic!("casting to a tag is not allowed for this value");
            };

            if !is_union {
                panic!("casting to a tag is not allowed for this value");
            }

            self.gen_node_with_emitter(left, kind);
            self.emitter(kind).emit(".tag");

            return;
        }

        self.emitter(kind).emit("((");
        self.emit_type_kind_left(type_kind_id, kind, false, false);
        self.emit_type_kind_right(type_kind_id, kind, false);
        self.emitter(kind).emit(")");
        self.gen_node_with_emitter(left, kind);
        self.emitter(kind).emit(")");
    }

    fn generic_specifier(
        &mut self,
        left: NodeIndex,
        generic_arg_type_names: Arc<Vec<NodeIndex>>,
        node_type: Option<Type>,
        _namespace_id: Option<usize>,
        kind: EmitterKind,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        let name = match self.get_typer_node(left).node_kind {
            NodeKind::FieldAccess { name, .. } => name,
            NodeKind::Identifier { name } => name,
            _ => panic!("expected field access or identifier before generic specifier")
        };

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(type_kind_id) {
            let is_generic = !generic_arg_type_names.is_empty();
            self.emit_function_name(name, type_kind_id, is_generic, kind)
        } else {
            self.emit_type_kind_left(type_kind_id, kind, false, false);
            self.emit_type_kind_right(type_kind_id, kind, false);
        }
    }

    fn name(&mut self, text: Arc<str>, _node_type: Option<Type>, namespace_id: Option<usize>, kind: EmitterKind) {
        if let Some(namespace_id) = namespace_id {
            self.emit_namespace(namespace_id, kind);
        }

        if reserved_names().contains(&text) {
            self.emitter(kind).emit("__");
        }

        self.emitter(kind).emit(&text);
    }

    fn identifier(&mut self, name: NodeIndex, node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        let node_type = node_type.unwrap();

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(node_type.type_kind_id) {
            self.emit_function_name(name, node_type.type_kind_id, false, kind)
        } else {
            self.gen_node_with_emitter(name, kind);
        }
    }

    fn int_literal(&mut self, text: Arc<str>, node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        self.emitter(kind).emit(&text);

        let type_kind_id = node_type.unwrap().type_kind_id;
        if self.type_kinds.get_by_id(type_kind_id).is_unsigned() {
            self.emitter(kind).emit("u");
        }
    }

    fn float_literal(&mut self, text: Arc<str>, node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        self.emitter(kind).emit(&text);

        let type_kind_id = node_type.unwrap().type_kind_id;
        if self.type_kinds.get_by_id(type_kind_id) == TypeKind::Float32 {
            self.emitter(kind).emit("f");
        }
    }

    fn char_literal(&mut self, value: char, _node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        let mut char_buffer = [0u8];

        self.emitter(kind).emit("'");
        self.emitter(kind).emit(value.encode_utf8(&mut char_buffer));
        self.emitter(kind).emit("'");
    }

    fn string_literal(&mut self, text: Arc<str>, _node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        self.emitter(kind).emit("(struct ");
        self.emit_struct_name(self.string_view_type_kind_id, kind);
        self.emitter(kind).emitln(") {");
        self.emitter(kind).indent();
        self.emitter(kind).emit(".count = ");
        self.emitter(kind).emit(&text.len().to_string());
        self.emitter(kind).emitln(",");
        self.emitter(kind).emit(".data = ");

        self.emitter(kind).emit("\"");
        for (i, line) in text.lines().enumerate() {
            if i > 0 {
                self.emitter(kind).emit("\\n");
            }

            self.emitter(kind).emit(line);
        }
        self.emitter(kind).emitln("\",");

        self.emitter(kind).unindent();
        self.emitter(kind).emit("}");
    }

    fn bool_literal(&mut self, value: bool, _node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        if value {
            self.emitter(kind).emit("true");
        } else {
            self.emitter(kind).emit("false");
        }
    }

    fn array_literal(
        &mut self,
        elements: Arc<Vec<NodeIndex>>,
        _repeat_count_const_expression: Option<NodeIndex>,
        node_type: Option<Type>,
        _namespace_id: Option<usize>,
        kind: EmitterKind,
    ) {
        let TypeKind::Array { element_count, .. } =
            self.type_kinds.get_by_id(node_type.unwrap().type_kind_id)
        else {
            panic!("invalid type for array literal");
        };

        let repeat_count = element_count / elements.len();

        self.emitter(kind).emit("{");
        let mut i = 0;
        for _ in 0..repeat_count {
            for element in elements.iter() {
                if i > 0 {
                    self.emitter(kind).emit(", ");
                }

                self.gen_node_with_emitter(*element, kind);

                i += 1;
            }
        }
        self.emitter(kind).emit("}");
    }

    fn struct_literal(
        &mut self,
        _left: NodeIndex,
        field_literals: Arc<Vec<NodeIndex>>,
        node_type: Option<Type>,
        _namespace_id: Option<usize>,
        kind: EmitterKind,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        self.emitter(kind).emit("(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.emitter(kind).emit(") ");

        let TypeKind::Struct {
            fields: type_kind_fields,
            is_union,
            ..
        } = &self.type_kinds.get_by_id(type_kind_id)
        else {
            panic!("struct literal does not have a struct type");
        };
        let is_union = *is_union;

        if is_union {
            self.emitter(kind).emitln("{");
            self.emitter(kind).indent();

            if field_literals.len() != 1 {
                panic!("expected union literal to contain a single field");
            }

            let NodeKind::FieldLiteral { name, .. } =
                &self.get_typer_node(field_literals[0]).node_kind
            else {
                panic!("invalid field in union literal");
            };

            let NodeKind::Name { text: name_text } = &self.get_typer_node(*name).node_kind else {
                panic!("invalid field name text in union literal");
            };

            let Some(tag) = get_field_index_by_name(&self.typed_nodes, name_text, type_kind_fields)
            else {
                panic!("tag not found in union literal");
            };

            self.emitter(kind).emit(".tag = ");
            self.emitter(kind).emit(&tag.to_string());
            self.emitter(kind).emitln(",");

            self.emitter(kind).emit(".variant = ");
        }

        self.emitter(kind).emitln("{");
        self.emitter(kind).indent();

        if field_literals.is_empty() {
            // Since empty structs aren't allowed in C, we generate a placeholder field
            // in structs that would be empty otherwise. We also have to initialize it here.
            self.emitter(kind).emitln("0,");
        }

        for field_literal in field_literals.iter() {
            self.gen_node_with_emitter(*field_literal, kind);
            self.emitter(kind).emitln(",");
        }

        self.emitter(kind).unindent();
        self.emitter(kind).emit("}");

        if is_union {
            self.emitter(kind).emitln(",");
            self.emitter(kind).unindent();
            self.emitter(kind).emit("}");
        }
    }

    fn field_literal(&mut self, name: NodeIndex, expression: NodeIndex, _node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        self.emitter(kind).emit(".");
        self.gen_node_with_emitter(name, kind);
        self.emitter(kind).emit(" = ");
        self.gen_node_with_emitter(expression, kind);
    }

    fn type_size(&mut self, type_name: NodeIndex, _node_type: Option<Type>, _namespace_id: Option<usize>, kind: EmitterKind) {
        let type_name_type_kind = &self.get_typer_node(type_name).node_type;
        self.emit_type_size(type_name_type_kind.as_ref().unwrap().type_kind_id, kind);
    }

    fn emit_memmove_expression_to_variable(
        &mut self,
        destination: NodeIndex,
        source: NodeIndex,
        type_kind_id: usize,
        kind: EmitterKind,
    ) {
        self.emitter(kind).emit("memmove(");
        self.gen_node_with_emitter(destination, kind);
        self.emitter(kind).emit(", ");
        self.gen_node_with_emitter(source, kind);
        self.emitter(kind).emit(", ");
        self.emit_type_size(type_kind_id, kind);
        self.emitter(kind).emit(")");
    }

    fn emit_memmove_expression_to_name(
        &mut self,
        destination: &str,
        source: NodeIndex,
        type_kind_id: usize,
        kind: EmitterKind,
    ) {
        self.emitter(kind).emit("memmove(");
        self.emitter(kind).emit(destination);
        self.emitter(kind).emit(", ");
        self.gen_node_with_emitter(source, kind);
        self.emitter(kind).emit(", ");
        self.emit_type_size(type_kind_id, kind);
        self.emitter(kind).emit(")");
    }

    fn emit_memmove_name_to_name(&mut self, destination: &str, source: &str, type_kind_id: usize, kind: EmitterKind) {
        self.emitter(kind).emit("memmove(");
        self.emitter(kind).emit(destination);
        self.emitter(kind).emit(", ");
        self.emitter(kind).emit(source);
        self.emitter(kind).emit(", ");
        self.emit_type_size(type_kind_id, kind);
        self.emitter(kind).emit(")");
    }

    fn emit_type_size(&mut self, type_kind_id: usize, kind: EmitterKind) {
        match self.type_kinds.get_by_id(type_kind_id) {
            TypeKind::Array {
                element_type_kind_id,
                element_count,
            } => {
                self.emit_type_size(element_type_kind_id, kind);
                self.emitter(kind).emit(" * ");
                self.emitter(kind).emit(&element_count.to_string());
            }
            _ => {
                self.emitter(kind).emit("sizeof(");
                self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
                self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
                self.emitter(kind).emit(")");
            }
        };
    }

    fn emit_type_kind_left(
        &mut self,
        type_kind_id: usize,
        kind: EmitterKind,
        do_arrays_as_pointers: bool,
        is_prefix: bool,
    ) {
        let type_kind = &self.type_kinds.get_by_id(type_kind_id);
        let needs_trailing_space = is_prefix
            && !matches!(
                type_kind,
                TypeKind::Array { .. } | TypeKind::Pointer { .. } | TypeKind::Function { .. }
            );

        match type_kind.clone() {
            TypeKind::Int | TypeKind::Tag { .. } => self.emitter(kind).emit("intptr_t"),
            TypeKind::Bool => self.emitter(kind).emit("bool"),
            TypeKind::Char => self.emitter(kind).emit("char"),
            TypeKind::Void => self.emitter(kind).emit("void"),
            TypeKind::UInt => self.emitter(kind).emit("uintptr_t"),
            TypeKind::Int8 => self.emitter(kind).emit("int8_t"),
            TypeKind::UInt8 => self.emitter(kind).emit("uint8_t"),
            TypeKind::Int16 => self.emitter(kind).emit("int16_t"),
            TypeKind::UInt16 => self.emitter(kind).emit("uint16_t"),
            TypeKind::Int32 => self.emitter(kind).emit("int32_t"),
            TypeKind::UInt32 => self.emitter(kind).emit("uint32_t"),
            TypeKind::Int64 => self.emitter(kind).emit("int64_t"),
            TypeKind::UInt64 => self.emitter(kind).emit("uint64_t"),
            TypeKind::Float32 => self.emitter(kind).emit("float"),
            TypeKind::Float64 => self.emitter(kind).emit("double"),
            TypeKind::Struct { .. } => {
                self.emitter(kind).emit("struct ");
                self.emit_struct_name(type_kind_id, kind);
            }
            TypeKind::Enum { name, .. } => {
                self.emitter(kind).emit("enum ");
                self.gen_node_with_emitter(name, kind);
            }
            TypeKind::Array {
                element_type_kind_id,
                ..
            } => {
                self.emit_type_kind_left(element_type_kind_id, kind, do_arrays_as_pointers, true);
                if do_arrays_as_pointers {
                    self.emitter(kind).emit("*");
                }
            }
            TypeKind::Pointer {
                inner_type_kind_id,
                is_inner_mutable,
            } => {
                self.emit_type_kind_left(inner_type_kind_id, kind, do_arrays_as_pointers, true);

                // If the pointer points to an immutable value, then add a const to the generated code.
                // Except for functions, because a const function has no meaning in C.
                if !is_inner_mutable
                    && !matches!(
                        self.type_kinds.get_by_id(inner_type_kind_id),
                        TypeKind::Function { .. }
                    )
                {
                    self.emitter(kind).emit("const ");
                }

                self.emitter(kind).emit("*");
            }
            TypeKind::Placeholder { .. } => {
                panic!("can't emit placeholder type: {:?}", type_kind)
            }
            TypeKind::Function {
                return_type_kind_id,
                ..
            } => {
                self.emit_type_kind_left(return_type_kind_id, kind, true, true);
                self.emit_type_kind_right(return_type_kind_id, kind, true);
                self.emitter(kind).emit("(");
            }
            TypeKind::Namespace { .. } => panic!("cannot emit namespace types"),
        };

        if needs_trailing_space {
            self.emitter(kind).emit(" ");
        }
    }

    fn emit_type_kind_right(
        &mut self,
        type_kind_id: usize,
        kind: EmitterKind,
        do_arrays_as_pointers: bool,
    ) {
        let type_kind_id = self.type_kinds.get_by_id(type_kind_id).clone();

        match type_kind_id {
            TypeKind::Array {
                element_type_kind_id,
                element_count,
            } => {
                if !do_arrays_as_pointers {
                    self.emitter(kind).emit("[");
                    self.emitter(kind).emit(&element_count.to_string());
                    self.emitter(kind).emit("]");
                }
                self.emit_type_kind_right(element_type_kind_id, kind, do_arrays_as_pointers);
            }
            TypeKind::Pointer {
                inner_type_kind_id, ..
            } => {
                self.emit_type_kind_right(inner_type_kind_id, kind, do_arrays_as_pointers);
            }
            TypeKind::Function {
                param_type_kind_ids,
                ..
            } => {
                self.emitter(kind).emit(")(");
                for (i, param_kind_id) in param_type_kind_ids.iter().enumerate() {
                    if i > 0 {
                        self.emitter(kind).emit(", ");
                    }

                    self.emit_type_kind_left(*param_kind_id, kind, false, false);
                    self.emit_type_kind_right(*param_kind_id, kind, false);
                }
                self.emitter(kind).emit(")");
            }
            _ => {}
        }
    }

    fn emit_binary_op(&mut self, op: Op, kind: EmitterKind) {
        self.emitter(kind).emit(match op {
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
        });
    }

    fn emit_function_declaration(
        &mut self,
        kind: EmitterKind,
        name: NodeIndex,
        params: &Arc<Vec<NodeIndex>>,
        has_generic_params: bool,
        type_kind_id: usize,
    ) {
        let TypeKind::Function {
            return_type_kind_id,
            ..
        } = self.type_kinds.get_by_id(type_kind_id)
        else {
            panic!("tried to emit function declaration for non-function type");
        };

        self.emit_type_kind_left(return_type_kind_id, kind, true, true);

        self.emit_function_name(name, type_kind_id, has_generic_params, kind);

        let mut param_count = 0;

        self.emitter(kind).emit("(");
        for param in params.iter() {
            if param_count > 0 {
                self.emitter(kind).emit(", ");
            }

            param_count += 1;

            self.emit_param_node(*param, kind);
        }

        if matches!(
            &self.type_kinds.get_by_id(return_type_kind_id),
            TypeKind::Array { .. }
        ) {
            if param_count > 0 {
                self.emitter(kind).emit(", ");
            }

            param_count += 1;

            self.emit_param_string("__return", return_type_kind_id, kind);
        }

        if param_count == 0 {
            self.emitter(kind).emit("void");
        }

        self.emitter(kind).emit(")");

        self.emit_type_kind_right(return_type_kind_id, kind, true);
    }

    fn emit_param_node(&mut self, param: NodeIndex, kind: EmitterKind) {
        let type_kind_id = self
            .get_typer_node(param)
            .node_type
            .as_ref()
            .unwrap()
            .type_kind_id;
        let NodeKind::Param { name, .. } = self.get_typer_node(param).node_kind else {
            panic!("invalid param");
        };

        self.emit_param(name, type_kind_id, kind);
    }

    fn emit_param(&mut self, name: NodeIndex, type_kind_id: usize, kind: EmitterKind) {
        self.emit_type_kind_left(type_kind_id, kind, false, true);
        self.gen_node_with_emitter(name, kind);
        self.emit_type_kind_right(type_kind_id, kind, false);
    }

    fn emit_param_string(&mut self, name: &str, type_kind_id: usize, kind: EmitterKind) {
        self.emit_type_kind_left(type_kind_id, kind, false, true);
        self.emitter(kind).emit(name);
        self.emit_type_kind_right(type_kind_id, kind, false);
    }

    fn emit_namespace(&mut self, namespace_id: usize, kind: EmitterKind) {
        if let Some(parent_id) = self.namespaces[namespace_id].parent_id {
            self.emit_namespace(parent_id, kind);
        }

        let namespace_name = self.namespaces[namespace_id].name.clone();
        self.emitter(kind).emit(&namespace_name);
        self.emitter(kind).emit("__");

        if let Some(associated_type_kind_id) = self.namespaces[namespace_id].associated_type_kind_id {
            if !self.namespaces[namespace_id].generic_args.is_empty() {
                self.emit_number_backwards(associated_type_kind_id, kind);
                self.emitter(kind).emit("__");
            }
        }
    }

    fn emit_number_backwards(&mut self, mut number: usize, kind: EmitterKind) {
        let mut digit = 0;
        while number > 0 || digit == 0 {
            self.emitter(kind)
                .emit_char(((number % 10) as u8 + b'0') as char);
            number /= 10;
            digit += 1;
        }
    }

    fn emit_bounds_check(&mut self) {
        self.function_prototype_emitter
            .emitln("static inline intptr_t __BoundsCheck(intptr_t index, intptr_t count);");
        self.function_prototype_emitter.newline();

        self.body_emitters
            .top()
            .body
            .emitln("intptr_t __BoundsCheck(intptr_t index, intptr_t count) {");
        self.body_emitters.top().body.indent();
        self.body_emitters
            .top()
            .body
            .emitln("assert(index >= 0 && index < count);");
        self.body_emitters.top().body.emitln("return index;");
        self.body_emitters.top().body.unindent();
        self.body_emitters.top().body.emitln("}");
        self.body_emitters.top().body.newline();
    }

    fn emit_struct_name(&mut self, type_kind_id: usize, kind: EmitterKind) {
        let TypeKind::Struct { name, .. } = self.type_kinds.get_by_id(type_kind_id) else {
            panic!("invalid struct");
        };

        self.gen_node_with_emitter(name, kind);
        self.emitter(kind).emit("__");
        self.emit_number_backwards(type_kind_id, kind);
    }

    fn emit_function_name(
        &mut self,
        name: NodeIndex,
        type_kind_id: usize,
        is_generic: bool,
        kind: EmitterKind,
    ) {
        self.gen_node_with_emitter(name, kind);

        if is_generic {
            self.emitter(kind).emit("__");
            self.emit_number_backwards(type_kind_id, kind);
        }
    }

    fn emit_main_function(&mut self) {
        let Some(main_function_declaration) = self.main_function_declaration else {
            return;
        };

        let TypedNode {
            node_kind: NodeKind::FunctionDeclaration { name, .. },
            node_type: Some(function_type),
            ..
        } = self.get_typer_node(main_function_declaration).clone()
        else {
            panic!("invalid main function declaration");
        };

        let TypeKind::Function {
            param_type_kind_ids,
            ..
        } = &self.type_kinds.get_by_id(function_type.type_kind_id)
        else {
            panic!("invalid main function");
        };

        if param_type_kind_ids.len() > 0 {
            self.body_emitters
                .top()
                .body
                .emitln("int main(int argc, char** argv) {");
            self.body_emitters.top().body.indent();
            self.body_emitters.top().body.emit("return (int)");
            self.emit_function_name(name, function_type.type_kind_id, false, EmitterKind::Body);
            self.body_emitters
                .top()
                .body
                .emitln("((intptr_t)argv, (const char *const *)argv);");
            self.body_emitters.top().body.unindent();
            self.body_emitters.top().body.emitln("}");
            self.body_emitters.top().body.newline();

            return;
        }

        self.body_emitters.top().body.emitln("int main(void) {");
        self.body_emitters.top().body.indent();
        self.body_emitters.top().body.emit("return (int)");
        self.emit_function_name(name, function_type.type_kind_id, false, EmitterKind::Body);
        self.body_emitters.top().body.emitln("();");
        self.body_emitters.top().body.unindent();
        self.body_emitters.top().body.emitln("}");
        self.body_emitters.top().body.newline();
    }

    fn emitter(&mut self, kind: EmitterKind) -> &mut Emitter {
        match kind {
            EmitterKind::TypePrototype => &mut self.type_prototype_emitter,
            EmitterKind::FunctionPrototype => &mut self.function_prototype_emitter,
            EmitterKind::GlobalVariable => &mut self.global_variable_emitter,
            EmitterKind::Top => &mut self.body_emitters.top().top,
            EmitterKind::Body => &mut self.body_emitters.top().body,
        }
    }

    fn temp_variable_name(&mut self, prefix: &str) -> String {
        let temp_variable_index = self.temp_variable_count;
        self.temp_variable_count += 1;

        format!("__{}{}", prefix, temp_variable_index)
    }

    fn emit_union_check_tag(&mut self, type_kind_id: usize) {
        self.function_prototype_emitter.emit("static inline ");
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.function_prototype_emitter.emit("* ");
        self.emit_struct_name(type_kind_id, EmitterKind::FunctionPrototype);
        self.function_prototype_emitter.emit("__CheckTag(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.function_prototype_emitter.emit(" *self");
        self.emit_type_kind_right(type_kind_id, EmitterKind::FunctionPrototype, false);
        self.function_prototype_emitter.emitln(", intptr_t tag);");
        self.function_prototype_emitter.newline();

        self.body_emitters.top().body.emit("static inline ");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, true, false);
        self.body_emitters.top().body.emit("* ");
        self.emit_struct_name(type_kind_id, EmitterKind::Body);
        self.body_emitters.top().body.emit("__CheckTag(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.body_emitters.top().body.emit(" *self");
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.body_emitters.top().body.emitln(", intptr_t tag) {");
        self.body_emitters.top().body.indent();
        self.body_emitters
            .top()
            .body
            .emitln("assert(self->tag == tag);");
        self.body_emitters.top().body.emitln("return self;");
        self.body_emitters.top().body.unindent();
        self.body_emitters.top().body.emitln("}");
        self.body_emitters.top().body.newline();
    }

    fn emit_union_check_tag_usage(&mut self, dereferenced_left_type_kind_id: usize, is_left_pointer: bool, left: NodeIndex, name: NodeIndex, tag: usize, kind: EmitterKind) {
        self.emit_struct_name(dereferenced_left_type_kind_id, EmitterKind::Body);
        self.emitter(kind).emit("__CheckTag((");
        self.emit_type_kind_left(
            dereferenced_left_type_kind_id,
            EmitterKind::Body,
            false,
            false,
        );
        self.emit_type_kind_right(dereferenced_left_type_kind_id, kind, false);
        self.emitter(kind).emit("*)");

        if !is_left_pointer {
            self.emitter(kind).emit("&");
        }

        self.gen_node_with_emitter(left, kind);

        self.emitter(kind).emit(", ");
        self.emitter(kind).emit(&tag.to_string());

        self.emitter(kind).emit(")->variant.");
        self.gen_node_with_emitter(name, kind);
    }

    fn emit_union_with_tag(&mut self, type_kind_id: usize) {
        self.function_prototype_emitter.emit("static inline ");
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::FunctionPrototype, true);
        self.function_prototype_emitter.emit("* ");
        self.emit_struct_name(type_kind_id, EmitterKind::FunctionPrototype);
        self.function_prototype_emitter.emit("__WithTag(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.function_prototype_emitter.emit(" *self");
        self.emit_type_kind_right(type_kind_id, EmitterKind::FunctionPrototype, false);
        self.function_prototype_emitter.emitln(", intptr_t tag);");
        self.function_prototype_emitter.newline();

        self.body_emitters.top().body.emit("static inline ");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, true, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, true);
        self.body_emitters.top().body.emit("* ");
        self.emit_struct_name(type_kind_id, EmitterKind::Body);
        self.body_emitters.top().body.emit("__WithTag(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.body_emitters.top().body.emit(" *self");
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.body_emitters.top().body.emitln(", intptr_t tag) {");
        self.body_emitters.top().body.indent();
        self.body_emitters.top().body.emitln("self->tag = tag;");
        self.body_emitters.top().body.emitln("return self;");
        self.body_emitters.top().body.unindent();
        self.body_emitters.top().body.emitln("}");
        self.body_emitters.top().body.newline();
    }

    fn emit_union_with_tag_usage(&mut self, dereferenced_left_type_kind_id: usize, is_left_pointer: bool, left: NodeIndex, name: NodeIndex, tag: usize, kind: EmitterKind) {
        self.emit_struct_name(dereferenced_left_type_kind_id, kind);
        self.emitter(kind).emit("__WithTag((");
        self.emit_type_kind_left(
            dereferenced_left_type_kind_id,
            kind,
            false,
            false,
        );
        self.emit_type_kind_right(
            dereferenced_left_type_kind_id,
            kind,
            false,
        );
        self.emitter(kind).emit("*)");

        if !is_left_pointer {
            self.emitter(kind).emit("&");
        }

        self.gen_node_with_emitter(left, kind);

        self.emitter(kind).emit(", ");
        self.emitter(kind).emit(&tag.to_string());

        self.emitter(kind).emit(")->variant.");
        self.gen_node_with_emitter(name, kind);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_equality(
        &mut self,
        type_kind_id: usize,
        left_prefix: Option<&str>,
        left: NodeIndex,
        right_prefix: Option<&str>,
        right: NodeIndex,
        is_equal: bool,
        kind: EmitterKind
    ) {
        match self.type_kinds.get_by_id(type_kind_id) {
            TypeKind::Struct { .. } => {
                if !is_equal {
                    self.emitter(kind).emit("!");
                }

                self.emit_struct_equals_usage(type_kind_id, left_prefix, left, right_prefix, right, kind)
            },
            TypeKind::Array { .. } => {
                if !is_equal {
                    self.emitter(kind).emit("!");
                }

                self.emitter(kind).emit("memcmp(");

                if let Some(left_prefix) = left_prefix {
                    self.emitter(kind).emit(left_prefix);
                }

                self.gen_node_with_emitter(left, kind);

                self.emitter(kind).emit(", ");

                if let Some(right_prefix) = right_prefix {
                    self.emitter(kind).emit(right_prefix);
                }

                self.gen_node_with_emitter(right, kind);

                self.emitter(kind).emit(", ");
                self.emit_type_size(type_kind_id, kind);
                self.emitter(kind).emit(")");
            },
            _ => {
                if let Some(left_prefix) = left_prefix {
                    self.emitter(kind).emit(left_prefix);
                }

                self.gen_node_with_emitter(left, kind);

                if is_equal {
                    self.emitter(kind).emit(" == ");
                } else {
                    self.emitter(kind).emit(" != ");
                }

                if let Some(right_prefix) = right_prefix {
                    self.emitter(kind).emit(right_prefix);
                }

                self.gen_node_with_emitter(right, kind);
            }
        }
    }

    fn emit_struct_equals(&mut self, type_kind_id: usize) {
        self.function_prototype_emitter.emit("static inline bool ");
        self.emit_struct_name(type_kind_id, EmitterKind::FunctionPrototype);
        self.function_prototype_emitter.emit("__Equals(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.function_prototype_emitter.emit(" *left, ");
        self.emit_type_kind_left(type_kind_id, EmitterKind::FunctionPrototype, false, false);
        self.function_prototype_emitter.emitln(" *right);");
        self.function_prototype_emitter.newline();

        self.body_emitters.top().body.emit("static inline bool ");
        self.emit_struct_name(type_kind_id, EmitterKind::Body);
        self.body_emitters.top().body.emit("__Equals(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.body_emitters.top().body.emit(" *left, ");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.body_emitters.top().body.emitln(" *right) {");
        self.body_emitters.top().body.indent();

        let TypeKind::Struct { fields, is_union, .. } = self.type_kinds.get_by_id(type_kind_id) else {
            panic!("cannot emit struct comparison for non-struct type");
        };

        if is_union {
            self.body_emitters.top().body.emitln("if (left->tag != right->tag) {");
            self.body_emitters.top().body.indent();
            self.body_emitters.top().body.emitln("return false;");
            self.body_emitters.top().body.unindent();
            self.body_emitters.top().body.emitln("}");

            self.body_emitters.top().body.emitln("switch (left->tag) {");
            for (i, field) in fields.iter().enumerate() {
                self.body_emitters.top().body.emit("case ");
                self.body_emitters.top().body.emit(&i.to_string());
                self.body_emitters.top().body.emit(": return ");
                self.emit_equality(field.type_kind_id, Some("left->variant."), field.name, Some("right->variant."), field.name, true, EmitterKind::Body);
                self.body_emitters.top().body.emitln(";");
            }
            self.body_emitters.top().body.emitln("}");
            self.body_emitters.top().body.emitln("return true;");
        } else {
            self.body_emitters.top().body.emit("return ");

            if fields.is_empty() {
                self.body_emitters.top().body.emit("true");
            }

            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    self.body_emitters.top().body.emit(" && ");
                }

                self.emit_equality(field.type_kind_id, Some("left->"), field.name, Some("right->"), field.name, true, EmitterKind::Body);
            }

            self.body_emitters.top().body.emitln(";");
        }

        self.body_emitters.top().body.unindent();
        self.body_emitters.top().body.emitln("}");
        self.body_emitters.top().body.newline();
    }

    fn emit_struct_equals_usage(&mut self, type_kind_id: usize, left_prefix: Option<&str>, left: NodeIndex, right_prefix: Option<&str>, right: NodeIndex, kind: EmitterKind) {
        self.emit_struct_name(type_kind_id, kind);
        self.emitter(kind).emit("__Equals(&");

        if let Some(left_prefix) = left_prefix {
            self.emitter(kind).emit(left_prefix);
        }

        self.gen_node_with_emitter(left, kind);

        self.emitter(kind).emit(", &");

        if let Some(right_prefix) = right_prefix {
            self.emitter(kind).emit(right_prefix);
        }

        self.gen_node_with_emitter(right, kind);

        self.emitter(kind).emit(")");
    }
}
