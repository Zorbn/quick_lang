use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};

use crate::{
    const_value::ConstValue, emitter::Emitter, emitter_stack::EmitterStack, parser::{DeclarationKind, NodeKind, Op}, type_kinds::{TypeKind, TypeKinds}, typer::{InstanceKind, Type, TypedDefinition, TypedNode}, utils::{get_field_index_by_name, is_typed_expression_array_literal}
};

#[derive(Clone, Copy, Debug)]
enum EmitterKind {
    TypePrototype,
    FunctionPrototype,
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

/*
TODOs FOR WIP REFACTOR

Need some alternative to the namespace system for enum field names and auto generated union With/Check functions.
Switch to a new system of name mangling that doesn't require generic args to be known to generate a struct/function name. Maybe just print the type_kind_id after the name.

*/

pub struct CodeGenerator {
    typed_nodes: Vec<TypedNode>,
    type_kinds: TypeKinds,
    typed_definitions: Vec<TypedDefinition>,

    pub header_emitter: Emitter,
    pub type_prototype_emitter: Emitter,
    pub function_prototype_emitter: Emitter,
    pub body_emitters: EmitterStack,

    function_declaration_needing_init: Option<usize>,
    temp_variable_count: usize,
    is_debug_mode: bool,
}

impl CodeGenerator {
    pub fn new(
        typed_nodes: Vec<TypedNode>,
        type_kinds: TypeKinds,
        typed_definitions: Vec<TypedDefinition>,
        is_debug_mode: bool,
    ) -> Self {
        let mut code_generator = Self {
            typed_nodes,
            type_kinds,
            typed_definitions,
            header_emitter: Emitter::new(0),
            type_prototype_emitter: Emitter::new(0),
            function_prototype_emitter: Emitter::new(0),
            body_emitters: EmitterStack::new(),
            function_declaration_needing_init: None,
            temp_variable_count: 0,
            is_debug_mode,
        };

        code_generator.header_emitter.emitln("#include <string.h>");
        code_generator
            .header_emitter
            .emitln("#include <inttypes.h>");
        code_generator.header_emitter.emitln("#include <stdbool.h>");
        code_generator.header_emitter.emitln("#include <assert.h>");
        code_generator.header_emitter.newline();
        code_generator.body_emitters.push(1);

        if is_debug_mode {
            code_generator.emit_bounds_check();
        }

        code_generator
    }

    pub fn gen(&mut self, start_index: usize) {
        self.gen_node(start_index);
    }

    fn gen_node(&mut self, index: usize) {
        let TypedNode {
            node_kind,
            node_type,
        } = self.typed_nodes[index].clone();

        match node_kind {
            NodeKind::TopLevel {
                functions,
                structs,
                enums,
            } => self.top_level(functions, structs, enums, node_type),
            NodeKind::StructDefinition {
                name,
                fields,
                functions,
                is_union,
                ..
            } => self.struct_definition(name, fields, functions, is_union, node_type, None),
            NodeKind::EnumDefinition {
                name,
                variant_names,
                ..
            } => self.enum_definition(name, variant_names, node_type),
            NodeKind::Field { name, type_name } => self.field(name, type_name, node_type),
            NodeKind::Function {
                declaration,
                statement,
            } => self.function(declaration, statement, node_type, None),
            NodeKind::FunctionDeclaration {
                name,
                params,
                return_type_name,
                ..
            } => self.function_declaration(name, params, return_type_name, node_type),
            NodeKind::ExternFunction { declaration } => {
                self.extern_function(declaration, node_type)
            }
            NodeKind::Param { name, type_name } => self.param(name, type_name, node_type),
            NodeKind::Block { statements } => self.block(statements, node_type),
            NodeKind::Statement { inner } => self.statement(inner, node_type),
            NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                type_name,
                expression,
            } => {
                self.variable_declaration(declaration_kind, name, type_name, expression, node_type)
            }
            NodeKind::ReturnStatement { expression } => {
                self.return_statement(expression, node_type)
            }
            NodeKind::DeferStatement { statement } => self.defer_statement(statement, node_type),
            NodeKind::IfStatement {
                expression,
                statement,
                next,
            } => self.if_statement(expression, statement, next, node_type),
            NodeKind::SwitchStatement {
                expression,
                case_statement,
            } => self.switch_statement(expression, case_statement, node_type),
            NodeKind::CaseStatement {
                expression,
                statement,
                next,
            } => self.case_statement(expression, statement, next, node_type),
            NodeKind::WhileLoop {
                expression,
                statement,
            } => self.while_loop(expression, statement, node_type),
            NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                statement,
            } => self.for_loop(iterator, op, from, to, by, statement, node_type),
            NodeKind::ConstExpression { inner } => self.const_expression(inner, node_type),
            NodeKind::Binary { left, op, right } => self.binary(left, op, right, node_type),
            NodeKind::UnaryPrefix { op, right } => self.unary_prefix(op, right, node_type),
            NodeKind::UnarySuffix { left, op } => self.unary_suffix(left, op, node_type),
            NodeKind::Call { left, args } => self.call(left, args, node_type),
            NodeKind::IndexAccess { left, expression } => {
                self.index_access(left, expression, node_type)
            }
            NodeKind::FieldAccess { left, name } => self.field_access(left, name, node_type),
            NodeKind::Cast { left, type_name } => self.cast(left, type_name, node_type),
            NodeKind::GenericSpecifier {
                name_text,
                generic_arg_type_names,
            } => self.generic_specifier(name_text, generic_arg_type_names, node_type),
            NodeKind::Name { text } => self.name(text, node_type),
            NodeKind::Identifier { name } => self.identifier(name, node_type),
            NodeKind::IntLiteral { text } => self.int_literal(text, node_type),
            NodeKind::Float32Literal { text } => self.float32_literal(text, node_type),
            NodeKind::CharLiteral { value } => self.char_literal(value, node_type),
            NodeKind::StringLiteral { text } => self.string_literal(text, node_type),
            NodeKind::BoolLiteral { value } => self.bool_literal(value, node_type),
            NodeKind::ArrayLiteral {
                elements,
                repeat_count_const_expression,
            } => self.array_literal(elements, repeat_count_const_expression, node_type),
            NodeKind::StructLiteral { left, fields } => {
                self.struct_literal(left, fields, node_type)
            }
            NodeKind::FieldLiteral { name, expression } => {
                self.field_literal(name, expression, node_type)
            }
            NodeKind::TypeSize { type_name } => self.type_size(type_name, node_type),
            NodeKind::Error => panic!("cannot generate error node"),
            NodeKind::TypeName { .. }
            | NodeKind::TypeNameArray { .. }
            | NodeKind::TypeNamePointer { .. }
            | NodeKind::TypeNameFunction { .. }
            | NodeKind::TypeNameGenericSpecifier { .. } => {
                panic!("cannot generate type name, generate the corresponding type kind instead")
            }
        }
    }

    fn top_level(
        &mut self,
        _functions: Arc<Vec<usize>>,
        _structs: Arc<Vec<usize>>,
        _enums: Arc<Vec<usize>>,
        _node_type: Option<Type>,
    ) {
        for i in 0..self.typed_definitions.len() {
            let TypedNode { node_kind, node_type } = self.typed_nodes[self.typed_definitions[i].index].clone();

            if let NodeKind::Function { declaration, statement } = node_kind.clone() {
                if let NodeKind::FunctionDeclaration { name, params, generic_params, return_type_name } = self.typed_nodes[declaration].node_kind.clone() {
                    if let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() {
                        println!("function: {:?}", text);
                    }
                }
            }

            match node_kind {
                NodeKind::StructDefinition { name, fields, functions, is_union, .. } => self.struct_definition(name, fields, functions, is_union, node_type, self.typed_definitions[i].generic_arg_type_kind_ids.clone()),
                NodeKind::EnumDefinition { name, variant_names } => self.enum_definition(name, variant_names, node_type),
                NodeKind::Function { declaration, statement } => self.function(declaration, statement, node_type, self.typed_definitions[i].generic_arg_type_kind_ids.clone()),
                NodeKind::ExternFunction { declaration } => self.extern_function(declaration, node_type),
                _ => panic!("unexpected definition kind: {:?}", node_kind)
            }
        }
    }

    fn struct_definition(
        &mut self,
        name: usize,
        fields: Arc<Vec<usize>>,
        functions: Arc<Vec<usize>>,
        is_union: bool,
        node_type: Option<Type>,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        for function in functions.iter() {
            self.gen_node(*function);
        }

        if is_union {
            // TODO: Refactor, emit_union_check_tag
            self.emit_type_kind_left(
                type_kind_id,
                EmitterKind::FunctionPrototype,
                false,
                false,
            );
            self.function_prototype_emitter.emit("* ");
            self.function_prototype_emitter.emit("__CheckTag(");
            self.emit_type_kind_left(
                type_kind_id,
                EmitterKind::FunctionPrototype,
                false,
                false,
            );
            self.function_prototype_emitter.emit(" *self");
            self.emit_type_kind_right(
                type_kind_id,
                EmitterKind::FunctionPrototype,
                false,
            );
            self.function_prototype_emitter.emitln(", intptr_t tag);");
            self.function_prototype_emitter.newline();

            self.emit_type_kind_left(type_kind_id, EmitterKind::Body, true, false);
            self.body_emitters.top().body.emit("* ");
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

            // TODO: Refactor, emit_union_set_tag
            self.emit_type_kind_left(
                type_kind_id,
                EmitterKind::FunctionPrototype,
                false,
                false,
            );
            self.emit_type_kind_right(
                type_kind_id,
                EmitterKind::FunctionPrototype,
                true,
            );
            self.function_prototype_emitter.emit("* ");
            self.function_prototype_emitter.emit("__WithTag(");
            self.emit_type_kind_left(
                type_kind_id,
                EmitterKind::FunctionPrototype,
                false,
                false,
            );
            self.function_prototype_emitter.emit(" *self");
            self.emit_type_kind_right(
                type_kind_id,
                EmitterKind::FunctionPrototype,
                false,
            );
            self.function_prototype_emitter.emitln(", intptr_t tag);");
            self.function_prototype_emitter.newline();

            self.emit_type_kind_left(type_kind_id, EmitterKind::Body, true, false);
            self.emit_type_kind_right(type_kind_id, EmitterKind::Body, true);
            self.body_emitters.top().body.emit("* ");
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

        self.emit_type_kind_left(
            type_kind_id,
            EmitterKind::TypePrototype,
            false,
            false,
        );

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
        name: usize,
        variant_names: Arc<Vec<usize>>,
        _node_type: Option<Type>,
    ) {
        self.type_prototype_emitter.emit("enum ");
        self.emit_name_node(name, EmitterKind::TypePrototype);
        self.type_prototype_emitter.emitln(" {");
        self.type_prototype_emitter.indent();

        for variant_name in variant_names.iter() {
            self.type_prototype_emitter.emit("__");
            self.emit_name_node(name, EmitterKind::TypePrototype);
            self.emit_name_node(*variant_name, EmitterKind::TypePrototype);
            self.type_prototype_emitter.emitln(",");
        }

        self.type_prototype_emitter.unindent();
        self.type_prototype_emitter.emitln("};");
        self.type_prototype_emitter.newline();
    }

    fn field(&mut self, name: usize, _type_name: usize, node_type: Option<Type>) {
        self.emit_type_kind_left(
            node_type.clone().unwrap().type_kind_id,
            EmitterKind::TypePrototype,
            false,
            true,
        );
        self.emit_name_node(name, EmitterKind::TypePrototype);
        self.emit_type_kind_right(
            node_type.unwrap().type_kind_id,
            EmitterKind::TypePrototype,
            false,
        );
        self.type_prototype_emitter.emitln(";");
    }

    fn function(
        &mut self,
        declaration: usize,
        statement: usize,
        node_type: Option<Type>,
        generic_arg_type_kind_ids: Option<Arc<Vec<usize>>>,
    ) {
        self.gen_node(declaration);
        self.function_declaration_needing_init = Some(declaration);
        self.emit_scoped_statement(statement);
        self.body_emitters.top().body.newline();
    }

    fn function_declaration(
        &mut self,
        name: usize,
        params: Arc<Vec<usize>>,
        _return_type_name: usize,
        node_type: Option<Type>,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        self.emit_function_declaration(
            EmitterKind::FunctionPrototype,
            name,
            &params,
            type_kind_id,
        );
        self.function_prototype_emitter.emitln(";");
        self.function_prototype_emitter.newline();

        self.emit_function_declaration(
            EmitterKind::Body,
            name,
            &params,
            type_kind_id,
        );
        self.body_emitters.top().body.emit(" ");
    }

    fn extern_function(&mut self, declaration: usize, node_type: Option<Type>) {
        self.function_prototype_emitter.emit("extern ");

        // TODO: Can this stuff for emit_function_decl... just be a call to gen_node?
        let NodeKind::FunctionDeclaration {
            name,
            params,
            ..
        } = self.typed_nodes[declaration].node_kind.clone()
        else {
            panic!("invalid function declaration");
        };

        let type_kind_id = node_type.unwrap().type_kind_id;

        self.emit_function_declaration(
            EmitterKind::FunctionPrototype,
            name,
            &params,
            type_kind_id,
        );

        self.function_prototype_emitter.emitln(";");
        self.function_prototype_emitter.newline();
    }

    fn param(&mut self, name: usize, _type_name: usize, node_type: Option<Type>) {
        self.emit_param(name, node_type.unwrap().type_kind_id, EmitterKind::Body);
    }

    fn copy_array_params(&mut self, function_declaration: usize) {
        let NodeKind::FunctionDeclaration { params, .. } = self.typed_nodes[function_declaration].node_kind.clone()
        else {
            panic!("invalid function declaration needing init");
        };

        for param in params.iter() {
            let TypedNode {
                node_kind: NodeKind::Param { name, .. },
                node_type,
            } = self.typed_nodes[*param].clone()
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

            let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
            else {
                panic!("invalid parameter name");
            };
            let copy_name = format!("__{}", &name_text);

            self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, true);
            self.body_emitters.top().body.emit(&copy_name);
            self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
            self.body_emitters.top().body.emitln(";");

            self.emit_memmove_name_to_name(&copy_name, &name_text, type_kind_id);
            self.body_emitters.top().body.emitln(";");

            self.gen_node(name);
            self.body_emitters.top().body.emit(" = ");
            self.body_emitters.top().body.emit(&copy_name);
            self.body_emitters.top().body.emitln(";");
        }
    }

    fn block(&mut self, statements: Arc<Vec<usize>>, _node_type: Option<Type>) {
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

        let mut was_last_statement_return = false;
        for statement in statements.iter().rev() {
            let TypedNode {
                node_kind: NodeKind::Statement { inner },
                ..
            } = self.typed_nodes[*statement]
            else {
                panic!("last statement is not a statement");
            };

            let Some(inner) = inner else {
                continue;
            };

            was_last_statement_return = matches!(
                self.typed_nodes[inner],
                TypedNode {
                    node_kind: NodeKind::ReturnStatement { .. },
                    ..
                }
            );

            break;
        }

        self.body_emitters.pop(!was_last_statement_return);
        self.body_emitters.top().body.emit("}");
    }

    fn statement(&mut self, inner: Option<usize>, _node_type: Option<Type>) {
        let Some(inner) = inner else {
            self.body_emitters.top().body.emitln(";");
            return;
        };

        let needs_semicolon = !matches!(
            self.typed_nodes[inner],
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
            self.typed_nodes[inner],
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

    fn variable_declaration(
        &mut self,
        declaration_kind: DeclarationKind,
        name: usize,
        _type_name: Option<usize>,
        expression: usize,
        node_type: Option<Type>,
    ) {
        let type_kind_id = self.typed_nodes[expression].node_type.as_ref().unwrap().type_kind_id;

        let is_array = matches!(
            &self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Array { .. }
        );
        let needs_const = declaration_kind != DeclarationKind::Var && !is_array;

        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, true);
        if needs_const {
            self.body_emitters.top().body.emit("const ");
        }
        self.gen_node(name);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);

        if is_array && !is_typed_expression_array_literal(&self.typed_nodes, expression) {
            self.body_emitters.top().body.emitln(";");

            let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
            else {
                panic!("invalid variable name");
            };
            self.emit_memmove_expression_to_name(&name_text, expression, type_kind_id);
        } else {
            self.body_emitters.top().body.emit(" = ");
            self.gen_node(expression);
        }
    }

    fn return_statement(&mut self, expression: Option<usize>, node_type: Option<Type>) {
        self.body_emitters.exiting_all_scopes();

        let expression = if let Some(expression) = expression {
            expression
        } else {
            self.body_emitters.top().body.emit("return");
            return;
        };

        let type_kind_id = self.typed_nodes[expression].node_type.as_ref().unwrap().type_kind_id;

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

                self.emit_memmove_name_to_name("__return", &temp_name, type_kind_id);
                self.body_emitters.top().body.emitln(";");
            } else {
                self.emit_memmove_expression_to_name("__return", expression, type_kind_id);
                self.body_emitters.top().body.emitln(";");
            }

            self.body_emitters.top().body.emit("return __return");
        } else {
            self.body_emitters.top().body.emit("return ");
            self.gen_node(expression);
        }
    }

    fn defer_statement(&mut self, statement: usize, _node_type: Option<Type>) {
        self.body_emitters.push(0);
        self.gen_node(statement);
        self.body_emitters.pop_to_bottom();
    }

    fn if_statement(
        &mut self,
        expression: usize,
        statement: usize,
        next: Option<usize>,
        _node_type: Option<Type>,
    ) {
        self.body_emitters.top().body.emit("if (");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(") ");
        self.gen_node(statement);

        if let Some(next) = next {
            self.body_emitters.top().body.emit("else ");
            self.gen_node(next);
        }
    }

    fn switch_statement(
        &mut self,
        expression: usize,
        case_statement: usize,
        _node_type: Option<Type>,
    ) {
        self.body_emitters.top().body.emit("switch (");
        self.gen_node(expression);
        self.body_emitters.top().body.emitln(") {");
        self.gen_node(case_statement);
        self.body_emitters.top().body.emitln("}");
    }

    fn case_statement(
        &mut self,
        expression: usize,
        statement: usize,
        next: Option<usize>,
        _node_type: Option<Type>,
    ) {
        self.body_emitters.top().body.emit("case ");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(": ");
        self.emit_scoped_statement(statement);
        self.body_emitters.top().body.emitln("break;");

        if let Some(next) = next {
            if !matches!(
                self.typed_nodes[next].node_kind,
                NodeKind::CaseStatement { .. }
            ) {
                self.body_emitters.top().body.emit("default: ");
                self.emit_scoped_statement(next);
                self.body_emitters.top().body.emitln("break;");
            } else {
                self.gen_node(next);
            }
        }
    }

    fn while_loop(&mut self, expression: usize, statement: usize, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit("while (");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(") ");
        self.gen_node(statement);
    }

    #[allow(clippy::too_many_arguments)]
    fn for_loop(
        &mut self,
        iterator: usize,
        op: Op,
        from: usize,
        to: usize,
        by: Option<usize>,
        statement: usize,
        _node_type: Option<Type>,
    ) {
        self.body_emitters.top().body.emit("for (intptr_t ");
        self.gen_node(iterator);
        self.body_emitters.top().body.emit(" = ");
        self.gen_node(from);
        self.body_emitters.top().body.emit("; ");

        self.gen_node(iterator);
        self.emit_binary_op(op);
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

        self.gen_node(statement);
    }

    fn const_expression(&mut self, _inner: usize, node_type: Option<Type>) {
        let Some(Type {
            instance_kind: InstanceKind::Const(const_value),
            ..
        }) = node_type
        else {
            panic!("invalid node type of const expression");
        };

        match const_value {
            ConstValue::Int { value } => self.body_emitters.top().body.emit(&value.to_string()),
            ConstValue::UInt { value } => self.body_emitters.top().body.emit(&value.to_string()),
            ConstValue::Float32 { value } => self.body_emitters.top().body.emit(&value.to_string()),
            ConstValue::String { value } => {
                self.body_emitters.top().body.emit_char('"');
                self.body_emitters.top().body.emit(&value);
                self.body_emitters.top().body.emit_char('"');
            }
            ConstValue::Char { value } => self.body_emitters.top().body.emit_char(value),
            ConstValue::Bool { value } => {
                if value {
                    self.body_emitters.top().body.emit("true");
                } else {
                    self.body_emitters.top().body.emit("false");
                }
            }
        }
    }

    fn binary(&mut self, left: usize, op: Op, right: usize, node_type: Option<Type>) {
        if op == Op::Assign {
            let type_kind_id = node_type.unwrap().type_kind_id;
            let is_array = matches!(
                &self.type_kinds.get_by_id(type_kind_id),
                TypeKind::Array { .. }
            );

            if is_array && !is_typed_expression_array_literal(&self.typed_nodes, left) {
                self.emit_memmove_expression_to_variable(left, right, type_kind_id);
                return;
            }

            // TODO: Needs refactor into it's own fn, also consider the similar code that exists in field access.
            if let NodeKind::FieldAccess { left, name } = self.typed_nodes[left].node_kind {
                let left_type = self.typed_nodes[left].node_type.as_ref().unwrap();

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
                    name: struct_name,
                    field_kinds,
                    is_union,
                    ..
                } = &self.type_kinds.get_by_id(dereferenced_left_type_kind_id)
                {
                    if *is_union {
                        let NodeKind::Name { text: name_text } =
                            self.typed_nodes[name].node_kind.clone()
                        else {
                            panic!("invalid field name in field access");
                        };

                        let Some(tag) =
                            get_field_index_by_name(&name_text, &self.typed_nodes, field_kinds)
                        else {
                            panic!("tag not found in union assignment");
                        };

                        let NodeKind::Name {
                            text: struct_name_text,
                        } = self.typed_nodes[*struct_name].node_kind.clone()
                        else {
                            panic!("invalid name in struct field access");
                        };

                        self.body_emitters.top().body.emit("__WithTag((");
                        self.emit_type_kind_left(
                            dereferenced_left_type_kind_id,
                            EmitterKind::Body,
                            false,
                            false,
                        );
                        self.emit_type_kind_right(
                            dereferenced_left_type_kind_id,
                            EmitterKind::Body,
                            false,
                        );
                        self.body_emitters.top().body.emit("*)");

                        if !is_left_pointer {
                            self.body_emitters.top().body.emit("&");
                        }

                        self.gen_node(left);

                        self.body_emitters.top().body.emit(", ");
                        self.body_emitters.top().body.emit(&tag.to_string());

                        self.body_emitters.top().body.emit(")->variant.");
                        self.gen_node(name);

                        self.emit_binary_op(op);
                        self.gen_node(right);

                        return;
                    }
                }
            }
        }

        self.gen_node(left);
        self.emit_binary_op(op);
        self.gen_node(right);
    }

    fn unary_prefix(&mut self, op: Op, right: usize, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit(match op {
            Op::Plus => "+",
            Op::Minus => "-",
            Op::Not => "!",
            Op::Reference => "&",
            _ => panic!("expected unary prefix operator"),
        });

        self.gen_node(right);
    }

    fn unary_suffix(&mut self, left: usize, op: Op, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit("(");
        self.body_emitters.top().body.emit(match op {
            Op::Dereference => "*",
            _ => panic!("expected unary suffix operator"),
        });

        self.gen_node(left);
        self.body_emitters.top().body.emit(")");
    }

    fn call(&mut self, left: usize, args: Arc<Vec<usize>>, node_type: Option<Type>) {
        self.gen_node(left);

        self.body_emitters.top().body.emit("(");
        let mut i = 0;

        for arg in args.iter() {
            if i > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.gen_node(*arg);

            i += 1;
        }

        let type_kind_id = node_type.unwrap().type_kind_id;
        if matches!(
            &self.type_kinds.get_by_id(type_kind_id),
            TypeKind::Array { .. }
        ) {
            if args.len() > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            let return_array_name = self.temp_variable_name("returnArray");

            self.emit_type_kind_left(type_kind_id, EmitterKind::Top, false, true);
            self.body_emitters.top().top.emit(&return_array_name);
            self.emit_type_kind_right(type_kind_id, EmitterKind::Top, false);
            self.body_emitters.top().top.emitln(";");

            self.body_emitters.top().body.emit(&return_array_name);
        }
        self.body_emitters.top().body.emit(")");
    }

    fn index_access(&mut self, left: usize, expression: usize, _node_type: Option<Type>) {
        self.gen_node(left);
        self.body_emitters.top().body.emit("[");

        if self.is_debug_mode {
            let left_type = self.typed_nodes[left].node_type.as_ref().unwrap();
            let TypeKind::Array { element_count, .. } =
                &self.type_kinds.get_by_id(left_type.type_kind_id)
            else {
                panic!("tried to perform an index access on a non-array type");
            };
            let element_count = *element_count;

            self.body_emitters.top().body.emit("__BoundsCheck(");
            self.gen_node(expression);
            self.body_emitters.top().body.emit(", ");
            self.body_emitters
                .top()
                .body
                .emit(&element_count.to_string());
            self.body_emitters.top().body.emit(")");
        } else {
            self.gen_node(expression);
        }

        self.body_emitters.top().body.emit("]");
    }

    fn field_access(&mut self, left: usize, name: usize, node_type: Option<Type>) {
        let left_type = self.typed_nodes[left].node_type.as_ref().unwrap();

        match self.type_kinds.get_by_id(node_type.unwrap().type_kind_id) {
            TypeKind::Function { .. } => {
                let TypeKind::Struct {
                    name: struct_name, ..
                } = self.type_kinds.get_by_id(left_type.type_kind_id)
                else {
                    panic!("expected function field to be part of a struct");
                };

                let NodeKind::Name {
                    text: struct_name_text,
                } = self.typed_nodes[struct_name].node_kind.clone()
                else {
                    panic!("invalid name in struct field access");
                };

                self.gen_node(name);

                return;
            }
            TypeKind::Tag => {
                let TypeKind::Struct {
                    field_kinds,
                    is_union,
                    ..
                } = &self.type_kinds.get_by_id(left_type.type_kind_id)
                else {
                    panic!("expected tag field to be part of a struct");
                };

                if left_type.instance_kind == InstanceKind::Name && *is_union {
                    let NodeKind::Name { text: name_text } =
                        self.typed_nodes[name].node_kind.clone()
                    else {
                        panic!("invalid tag name in tag access");
                    };

                    let Some(tag) =
                        get_field_index_by_name(&name_text, &self.typed_nodes, field_kinds)
                    else {
                        panic!("tag not found in field access");
                    };

                    self.body_emitters.top().body.emit(&tag.to_string());

                    return;
                }
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
            name: struct_name,
            field_kinds,
            is_union,
            ..
        } = &self.type_kinds.get_by_id(dereferenced_left_type_kind_id)
        {
            if *is_union {
                let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
                else {
                    panic!("invalid field name in field access");
                };

                let Some(tag) = get_field_index_by_name(&name_text, &self.typed_nodes, field_kinds)
                else {
                    panic!("tag not found in field access");
                };

                let NodeKind::Name {
                    text: struct_name_text,
                } = self.typed_nodes[*struct_name].node_kind.clone()
                else {
                    panic!("invalid name in struct field access");
                };

                self.body_emitters.top().body.emit("__CheckTag((");
                self.emit_type_kind_left(
                    dereferenced_left_type_kind_id,
                    EmitterKind::Body,
                    false,
                    false,
                );
                self.emit_type_kind_right(dereferenced_left_type_kind_id, EmitterKind::Body, false);
                self.body_emitters.top().body.emit("*)");

                if !is_left_pointer {
                    self.body_emitters.top().body.emit("&");
                }

                self.gen_node(left);

                self.body_emitters.top().body.emit(", ");
                self.body_emitters.top().body.emit(&tag.to_string());

                self.body_emitters.top().body.emit(")->variant.");
                self.gen_node(name);

                return;
            }
        }

        match self.type_kinds.get_by_id(left_type.type_kind_id) {
            TypeKind::Pointer { .. } => {
                self.gen_node(left);
                self.body_emitters.top().body.emit("->")
            }
            TypeKind::Struct { .. } => {
                self.gen_node(left);
                self.body_emitters.top().body.emit(".")
            }
            TypeKind::Enum {
                name: enum_name, ..
            } => {
                self.body_emitters.top().body.emit("__");
                self.gen_node(enum_name);
            }
            TypeKind::Array { element_count, .. } => {
                // On arrays, only the "count" field is allowed.
                self.body_emitters
                    .top()
                    .body
                    .emit(&element_count.to_string());
                return;
            }
            _ => panic!("tried to access type that cannot be accessed"),
        }

        self.gen_node(name);
    }

    fn cast(&mut self, left: usize, _type_name: usize, node_type: Option<Type>) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        if let TypeKind::Tag { .. } = &self.type_kinds.get_by_id(type_kind_id) {
            let left_type_kind_id = self.typed_nodes[left]
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

            self.gen_node(left);
            self.body_emitters.top().body.emit(".tag");

            return;
        }

        self.body_emitters.top().body.emit("((");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.body_emitters.top().body.emit(")");
        self.gen_node(left);
        self.body_emitters.top().body.emit(")");
    }

    fn generic_specifier(
        &mut self,
        name_text: Arc<str>,
        _generic_arg_type_names: Arc<Vec<usize>>,
        node_type: Option<Type>,
    ) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(type_kind_id) {
            self.body_emitters.top().body.emit(&name_text);
            self.emit_number_backwards(type_kind_id, EmitterKind::Body);
        } else {
            self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
            self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        }
    }

    fn name(&mut self, text: Arc<str>, _node_type: Option<Type>) {
        self.emit_name(text, EmitterKind::Body);
    }

    fn identifier(&mut self, name: usize, node_type: Option<Type>) {
        self.gen_node(name);

        let type_kind_id = node_type.unwrap().type_kind_id;

        if let TypeKind::Function { .. } = self.type_kinds.get_by_id(type_kind_id) {
            self.emit_number_backwards(type_kind_id, EmitterKind::Body);
        }
    }

    fn int_literal(&mut self, text: Arc<str>, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit(&text);
    }

    fn float32_literal(&mut self, text: Arc<str>, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit(&text);
        self.body_emitters.top().body.emit("f");
    }

    fn char_literal(&mut self, value: char, _node_type: Option<Type>) {
        let mut char_buffer = [0u8];

        self.body_emitters.top().body.emit("'");
        self.body_emitters
            .top()
            .body
            .emit(value.encode_utf8(&mut char_buffer));
        self.body_emitters.top().body.emit("'");
    }

    fn string_literal(&mut self, text: Arc<str>, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit("\"");
        for (i, line) in text.lines().enumerate() {
            if i > 0 {
                self.body_emitters.top().body.emit("\\n");
            }

            self.body_emitters.top().body.emit(line);
        }
        self.body_emitters.top().body.emit("\"");
    }

    fn bool_literal(&mut self, value: bool, _node_type: Option<Type>) {
        if value {
            self.body_emitters.top().body.emit("true");
        } else {
            self.body_emitters.top().body.emit("false");
        }
    }

    fn array_literal(
        &mut self,
        elements: Arc<Vec<usize>>,
        _repeat_count_const_expression: Option<usize>,
        node_type: Option<Type>,
    ) {
        let TypeKind::Array { element_count, .. } =
            self.type_kinds.get_by_id(node_type.unwrap().type_kind_id)
        else {
            panic!("invalid type for array literal");
        };

        let repeat_count = element_count / elements.len();

        self.body_emitters.top().body.emit("{");
        let mut i = 0;
        for _ in 0..repeat_count {
            for element in elements.iter() {
                if i > 0 {
                    self.body_emitters.top().body.emit(", ");
                }

                self.gen_node(*element);

                i += 1;
            }
        }
        self.body_emitters.top().body.emit("}");
    }

    fn struct_literal(&mut self, _left: usize, fields: Arc<Vec<usize>>, node_type: Option<Type>) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        self.body_emitters.top().body.emit("(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.body_emitters.top().body.emit(") ");

        let TypeKind::Struct {
            field_kinds,
            is_union,
            ..
        } = &self.type_kinds.get_by_id(type_kind_id)
        else {
            panic!("struct literal does not have a struct type");
        };
        let is_union = *is_union;

        if is_union {
            self.body_emitters.top().body.emitln("{");
            self.body_emitters.top().body.indent();

            if fields.len() != 1 {
                panic!("expected union literal to contain a single field");
            }

            let NodeKind::FieldLiteral { name, .. } = &self.typed_nodes[fields[0]].node_kind else {
                panic!("invalid field in union literal");
            };

            let NodeKind::Name { text: name_text } = &self.typed_nodes[*name].node_kind else {
                panic!("invalid field name text in union literal");
            };

            let Some(tag) = get_field_index_by_name(name_text, &self.typed_nodes, field_kinds)
            else {
                panic!("tag not found in union literal");
            };

            self.body_emitters.top().body.emit(".tag = ");
            self.body_emitters.top().body.emit(&tag.to_string());
            self.body_emitters.top().body.emitln(",");

            self.body_emitters.top().body.emit(".variant = ");
        }

        self.body_emitters.top().body.emitln("{");
        self.body_emitters.top().body.indent();

        if fields.is_empty() {
            // Since empty structs aren't allowed in C, we generate a placeholder field
            // in structs that would be empty otherwise. We also have to initialize it here.
            self.body_emitters.top().body.emitln("0,");
        }

        for field in fields.iter() {
            self.gen_node(*field);
            self.body_emitters.top().body.emitln(",");
        }

        self.body_emitters.top().body.unindent();
        self.body_emitters.top().body.emit("}");

        if is_union {
            self.body_emitters.top().body.emitln(",");
            self.body_emitters.top().body.unindent();
            self.body_emitters.top().body.emit("}");
        }
    }

    fn field_literal(&mut self, name: usize, expression: usize, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit(".");
        self.gen_node(name);
        self.body_emitters.top().body.emit(" = ");
        self.gen_node(expression);
    }

    fn type_size(&mut self, _type_name: usize, node_type: Option<Type>) {
        let type_kind_id = node_type.unwrap().type_kind_id;

        self.body_emitters.top().body.emit("sizeof(");
        self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, false);
        self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
        self.body_emitters.top().body.emit(")");
    }

    fn emit_memmove_expression_to_variable(
        &mut self,
        destination: usize,
        source: usize,
        type_kind_id: usize,
    ) {
        self.body_emitters.top().body.emit("memmove(");
        self.gen_node(destination);
        self.body_emitters.top().body.emit(", ");
        self.gen_node(source);
        self.body_emitters.top().body.emit(", ");
        self.emit_type_size(type_kind_id);
        self.body_emitters.top().body.emit(")");
    }

    fn emit_memmove_expression_to_name(
        &mut self,
        destination: &str,
        source: usize,
        type_kind_id: usize,
    ) {
        self.body_emitters.top().body.emit("memmove(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit(", ");
        self.gen_node(source);
        self.body_emitters.top().body.emit(", ");
        self.emit_type_size(type_kind_id);
        self.body_emitters.top().body.emit(")");
    }

    fn emit_memmove_name_to_name(&mut self, destination: &str, source: &str, type_kind_id: usize) {
        self.body_emitters.top().body.emit("memmove(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit(source);
        self.body_emitters.top().body.emit(", ");
        self.emit_type_size(type_kind_id);
        self.body_emitters.top().body.emit(")");
    }

    fn emit_type_size(&mut self, type_kind_id: usize) {
        match self.type_kinds.get_by_id(type_kind_id) {
            TypeKind::Array {
                element_type_kind_id,
                element_count,
            } => {
                self.emit_type_size(element_type_kind_id);
                self.body_emitters.top().body.emit(" * ");
                self.body_emitters
                    .top()
                    .body
                    .emit(&element_count.to_string());
            }
            _ => {
                self.body_emitters.top().body.emit("sizeof(");
                self.emit_type_kind_left(type_kind_id, EmitterKind::Body, false, true);
                self.emit_type_kind_right(type_kind_id, EmitterKind::Body, false);
                self.body_emitters.top().body.emit(")");
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
            TypeKind::String => self.emitter(kind).emit("const char*"),
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
            TypeKind::Struct { name, .. } => {
                self.emitter(kind).emit("struct ");
                let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
                    panic!("invalid struct name: {:?}", self.typed_nodes[name].node_kind);
                };
                self.emit_name(text, kind);
                self.emit_number_backwards(type_kind_id, kind);
            }
            TypeKind::Enum { name, .. } => {
                self.emitter(kind).emit("enum ");
                let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
                    panic!("invalid enum name");
                };
                self.emit_name(text, kind);
            }
            TypeKind::Array {
                element_type_kind_id,
                ..
            } => {
                self.emit_type_kind_left(
                    element_type_kind_id,
                    kind,
                    do_arrays_as_pointers,
                    true,
                );
                if do_arrays_as_pointers {
                    self.emitter(kind).emit("*");
                }
            }
            TypeKind::Pointer {
                inner_type_kind_id,
                is_inner_mutable,
            } => {
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

                self.emit_type_kind_left(
                    inner_type_kind_id,
                    kind,
                    do_arrays_as_pointers,
                    true,
                );
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
                self.emit_type_kind_right(
                    element_type_kind_id,
                    kind,
                    do_arrays_as_pointers,
                );
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

    fn emit_binary_op(&mut self, op: Op) {
        self.body_emitters.top().body.emit(match op {
            Op::Equal => " == ",
            Op::NotEqual => " != ",
            Op::Less => " < ",
            Op::Greater => " > ",
            Op::LessEqual => " <= ",
            Op::GreaterEqual => " >= ",
            Op::Plus => " + ",
            Op::Minus => " - ",
            Op::Multiply => " * ",
            Op::Divide => " / ",
            Op::Assign => " = ",
            Op::And => " && ",
            Op::Or => " || ",
            Op::PlusAssign => " += ",
            Op::MinusAssign => " -= ",
            Op::MultiplyAssign => " *= ",
            Op::DivideAssign => " /= ",
            _ => panic!("expected binary operator"),
        });
    }

    fn emit_function_declaration(
        &mut self,
        kind: EmitterKind,
        name: usize,
        params: &Arc<Vec<usize>>,
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
        self.emit_name_node(name, kind);
        self.emit_number_backwards(type_kind_id, kind);

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

    fn emit_param_node(&mut self, param: usize, kind: EmitterKind) {
        let type_kind_id = self.typed_nodes[param].node_type.as_ref().unwrap().type_kind_id;
        let NodeKind::Param { name, .. } = self.typed_nodes[param].node_kind else {
            panic!("invalid param");
        };

        self.emit_param(name, type_kind_id, kind);
    }

    fn emit_param(&mut self, name: usize, type_kind_id: usize, kind: EmitterKind) {
        self.emit_type_kind_left(type_kind_id, kind, false, true);
        self.emit_name_node(name, kind);
        self.emit_type_kind_right(type_kind_id, kind, false);
    }

    fn emit_param_string(&mut self, name: &str, type_kind_id: usize, kind: EmitterKind) {
        self.emit_type_kind_left(type_kind_id, kind, false, true);
        self.emitter(kind).emit(name);
        self.emit_type_kind_right(type_kind_id, kind, false);
    }

    fn emit_name_node(&mut self, name: usize, kind: EmitterKind) {
        let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
            panic!("invalid name");
        };

        self.emit_name(text, kind);
    }

    fn emit_name(&mut self, text: Arc<str>, kind: EmitterKind) {
        if reserved_names().contains(&text) {
            self.emitter(kind).emit("__");
        }

        self.emitter(kind).emit(&text);
    }

    // Used for name mangling, so that multiple versions of a generic function can be generated without colliding.
//     fn emit_generic_param_suffix(
//         &mut self,
//         generic_arg_type_kind_ids: &Option<Arc<Vec<usize>>>,
//         kind: EmitterKind,
//     ) {
//         let Some(generic_arg_type_kind_ids) = generic_arg_type_kind_ids else {
//             return;
//         };
//
//         if generic_arg_type_kind_ids.is_empty() {
//             return;
//         }
//
//         self.emitter(kind).emit("_");
//
//         for mut generic_arg_type_kind_id in generic_arg_type_kind_ids.iter().copied() {
//             self.emitter(kind).emit("_");
//
//             // This prints the number backwards, but it doesn't matter for the purpose of name mangling.
//             let mut number = generic_arg_type_kind_id;
//             let mut digit = 0;
//             while number > 0 || digit == 0 {
//                 self.emitter(kind)
//                     .emit_char(((number % 10) as u8 + b'0') as char);
//                 number /= 10;
//                 digit += 1;
//             }
//         }
//     }

    fn emit_number_backwards(&mut self, mut number: usize, kind: EmitterKind) {
        let mut digit = 0;
        while number > 0 || digit == 0 {
            self.emitter(kind)
                .emit_char(((number % 10) as u8 + b'0') as char);
            number /= 10;
            digit += 1;
        }
    }

    fn emit_scoped_statement(&mut self, statement: usize) {
        let NodeKind::Statement { inner } = self.typed_nodes[statement].node_kind else {
            panic!("invalid statement in scoped statement");
        };

        let needs_scope = inner.is_none()
            || !matches!(
                self.typed_nodes[inner.unwrap()].node_kind,
                NodeKind::Block { .. }
            );

        if needs_scope {
            self.body_emitters.top().body.emitln("{");
            self.body_emitters.top().body.indent();
        }

        self.gen_node(statement);

        if needs_scope {
            self.body_emitters.top().body.unindent();
            self.body_emitters.top().body.emitln("}");
        }
    }

    fn emit_bounds_check(&mut self) {
        self.function_prototype_emitter
            .emitln("intptr_t __BoundsCheck(intptr_t index, intptr_t count);");
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

    fn emitter(&mut self, kind: EmitterKind) -> &mut Emitter {
        match kind {
            EmitterKind::TypePrototype => &mut self.type_prototype_emitter,
            EmitterKind::FunctionPrototype => &mut self.function_prototype_emitter,
            EmitterKind::Top => &mut self.body_emitters.top().top,
            EmitterKind::Body => &mut self.body_emitters.top().body,
        }
    }

    fn temp_variable_name(&mut self, prefix: &str) -> String {
        let temp_variable_index = self.temp_variable_count;
        self.temp_variable_count += 1;

        format!("__{}{}", prefix, temp_variable_index)
    }
}
