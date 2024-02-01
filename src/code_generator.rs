use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};

use crate::{
    emitter::Emitter,
    emitter_stack::EmitterStack,
    parser::{NodeKind, Op, TypeKind},
    type_checker::TypedNode,
    types::{is_type_kind_array, is_typed_expression_array_literal},
};

#[derive(Clone, Copy, Debug)]
enum EmitterKind {
    Prototype,
    Body,
    Top,
}

fn reserved_names() -> &'static HashSet<String> {
    static NAMES: OnceLock<HashSet<String>> = OnceLock::new();
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
        .map(|s| s.to_string())
        .collect()
    })
}

pub struct CodeGenerator {
    pub typed_nodes: Vec<TypedNode>,
    pub types: Vec<TypeKind>,
    pub header_emitter: Emitter,
    pub prototype_emitter: Emitter,
    pub body_emitters: EmitterStack,
    function_declaration_needing_init: Option<usize>,
    temp_variable_count: usize,
}

impl CodeGenerator {
    pub fn new(typed_nodes: Vec<TypedNode>, types: Vec<TypeKind>) -> Self {
        let mut code_generator = Self {
            typed_nodes,
            types,
            header_emitter: Emitter::new(0),
            prototype_emitter: Emitter::new(0),
            body_emitters: EmitterStack::new(),
            function_declaration_needing_init: None,
            temp_variable_count: 0,
        };

        code_generator.header_emitter.emitln("#include <string.h>");
        code_generator
            .header_emitter
            .emitln("#include <inttypes.h>");
        code_generator.header_emitter.emitln("#include <stdbool.h>");
        code_generator.body_emitters.push(1);

        code_generator
    }

    pub fn gen(&mut self, start_index: usize) {
        self.gen_node(start_index);
    }

    fn gen_node(&mut self, index: usize) {
        match self.typed_nodes[index].clone() {
            TypedNode {
                node_kind: NodeKind::TopLevel { functions, structs },
                type_kind,
            } => self.top_level(functions, structs, type_kind),
            TypedNode {
                node_kind: NodeKind::StructDefinition { name, fields, .. },
                type_kind,
            } => self.struct_definition(name, fields, type_kind),
            TypedNode {
                node_kind: NodeKind::Field { name, type_name },
                type_kind,
            } => self.field(name, type_name, type_kind),
            TypedNode {
                node_kind: NodeKind::Function { declaration, block },
                type_kind,
            } => self.function(declaration, block, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::FunctionDeclaration {
                        name,
                        return_type_name,
                        params,
                        ..
                    },
                type_kind,
            } => self.function_declaration(name, return_type_name, params, type_kind),
            TypedNode {
                node_kind: NodeKind::ExternFunction { declaration },
                type_kind,
            } => self.extern_function(declaration, type_kind),
            TypedNode {
                node_kind: NodeKind::Param { name, type_name },
                type_kind,
            } => self.param(name, type_name, type_kind),
            TypedNode {
                node_kind: NodeKind::Block { statements },
                type_kind,
            } => self.block(statements, type_kind),
            TypedNode {
                node_kind: NodeKind::Statement { inner },
                type_kind,
            } => self.statement(inner, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::VariableDeclaration {
                        is_mutable,
                        name,
                        type_name,
                        expression,
                    },
                type_kind,
            } => self.variable_declaration(is_mutable, name, type_name, expression, type_kind),
            TypedNode {
                node_kind: NodeKind::ReturnStatement { expression },
                type_kind,
            } => self.return_statement(expression, type_kind),
            TypedNode {
                node_kind: NodeKind::DeferStatement { statement },
                type_kind,
            } => self.defer_statement(statement, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::IfStatement {
                        expression,
                        block,
                        next,
                    },
                type_kind,
            } => self.if_statement(expression, block, next, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::SwitchStatement {
                        expression,
                        case_block,
                    },
                type_kind,
            } => self.switch_statement(expression, case_block, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::CaseBlock {
                        expression,
                        block,
                        next,
                    },
                type_kind,
            } => self.case_block(expression, block, next, type_kind),
            TypedNode {
                node_kind: NodeKind::WhileLoop { expression, block },
                type_kind,
            } => self.while_loop(expression, block, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::ForLoop {
                        iterator,
                        op,
                        from,
                        to,
                        by,
                        block,
                    },
                type_kind,
            } => self.for_loop(iterator, op, from, to, by, block, type_kind),
            TypedNode {
                node_kind: NodeKind::Binary { left, op, right },
                type_kind,
            } => self.binary(left, op, right, type_kind),
            TypedNode {
                node_kind: NodeKind::UnaryPrefix { op, right },
                type_kind,
            } => self.unary_prefix(op, right, type_kind),
            TypedNode {
                node_kind: NodeKind::Call { left, args },
                type_kind,
            } => self.call(left, args, type_kind),
            TypedNode {
                node_kind: NodeKind::IndexAccess { left, expression },
                type_kind,
            } => self.index_access(left, expression, type_kind),
            TypedNode {
                node_kind: NodeKind::FieldAccess { left, name },
                type_kind,
            } => self.field_access(left, name, type_kind),
            TypedNode {
                node_kind: NodeKind::Cast { left, type_name },
                type_kind,
            } => self.cast(left, type_name, type_kind),
            TypedNode {
                node_kind: NodeKind::Name { text },
                type_kind,
            } => self.name(text, type_kind),
            TypedNode {
                node_kind: NodeKind::Identifier { name },
                type_kind,
            } => self.identifier(name, type_kind),
            TypedNode {
                node_kind: NodeKind::IntLiteral { text },
                type_kind,
            } => self.int_literal(text, type_kind),
            TypedNode {
                node_kind: NodeKind::Float32Literal { text },
                type_kind,
            } => self.float32_literal(text, type_kind),
            TypedNode {
                node_kind: NodeKind::CharLiteral { value },
                type_kind,
            } => self.char_literal(value, type_kind),
            TypedNode {
                node_kind: NodeKind::StringLiteral { text },
                type_kind,
            } => self.string_literal(text, type_kind),
            TypedNode {
                node_kind: NodeKind::BoolLiteral { value },
                type_kind,
            } => self.bool_literal(value, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::ArrayLiteral {
                        elements,
                        repeat_count,
                    },
                type_kind,
            } => self.array_literal(elements, repeat_count, type_kind),
            TypedNode {
                node_kind: NodeKind::StructLiteral { name, fields },
                type_kind,
            } => self.struct_literal(name, fields, type_kind),
            TypedNode {
                node_kind: NodeKind::FieldLiteral { name, expression },
                type_kind,
            } => self.field_literal(name, expression, type_kind),
            TypedNode {
                node_kind: NodeKind::TypeSize { type_name },
                type_kind,
            } => self.type_size(type_name, type_kind),
            TypedNode {
                node_kind: NodeKind::TypeName { .. },
                ..
            } => panic!("Cannot generate type name with gen_node"),
            TypedNode {
                node_kind: NodeKind::Error,
                ..
            } => panic!("Cannot generate error node"),
        }
    }

    fn gen_node_prototype(&mut self, index: usize) {
        match self.typed_nodes[index].clone() {
            TypedNode {
                node_kind:
                    NodeKind::FunctionDeclaration {
                        name,
                        return_type_name,
                        params,
                        ..
                    },
                type_kind,
            } => self.function_declaration_prototype(name, return_type_name, params, type_kind),
            TypedNode {
                node_kind: NodeKind::Param { name, type_name },
                type_kind,
            } => self.param_prototype(name, type_name, type_kind),
            TypedNode {
                node_kind: NodeKind::Name { text },
                type_kind,
            } => self.name_prototype(text, type_kind),
            _ => panic!(
                "Node cannot be generated as a prototype: {:?}",
                self.typed_nodes[index]
            ),
        }
    }

    fn top_level(
        &mut self,
        functions: Arc<Vec<usize>>,
        structs: Arc<Vec<usize>>,
        _type_kind: Option<usize>,
    ) {
        for (i, struct_definition) in structs.iter().enumerate() {
            if i > 0 {
                self.prototype_emitter.newline();
            }

            self.gen_node(*struct_definition);
        }

        let mut i = 0;
        for function in functions.iter() {
            if matches!(
                self.typed_nodes[*function],
                TypedNode {
                    node_kind: NodeKind::Function { .. },
                    ..
                }
            ) {
                if i > 0 {
                    self.body_emitters.top().body.newline();
                }

                i += 1;
            }

            self.gen_node(*function);
        }
    }

    fn struct_definition(
        &mut self,
        name: usize,
        fields: Arc<Vec<usize>>,
        _type_kind: Option<usize>,
    ) {
        self.prototype_emitter.emit("struct ");
        self.gen_node_prototype(name);
        self.prototype_emitter.emitln(" {");
        self.prototype_emitter.indent();

        for field in fields.iter() {
            self.gen_node(*field);
        }

        self.prototype_emitter.unindent();
        self.prototype_emitter.emitln("};");
    }

    fn field(&mut self, name: usize, _type_name: usize, type_kind: Option<usize>) {
        self.emit_type_kind_left(type_kind.unwrap(), EmitterKind::Prototype, false, true);
        self.gen_node_prototype(name);
        self.emit_type_kind_right(type_kind.unwrap(), EmitterKind::Prototype, false);
        self.prototype_emitter.emitln(";");
    }

    fn function_declaration_prototype(
        &mut self,
        name: usize,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
        type_kind: Option<usize>,
    ) {
        self.emit_function_declaration(
            EmitterKind::Prototype,
            name,
            return_type_name,
            &params,
            type_kind,
        );
        self.prototype_emitter.emitln(";");
    }

    fn param_prototype(&mut self, name: usize, type_name: usize, _type_kind: Option<usize>) {
        self.emit_param(name, type_name, EmitterKind::Prototype);
    }

    fn function(&mut self, declaration: usize, block: usize, _type_kind: Option<usize>) {
        self.gen_node(declaration);
        self.function_declaration_needing_init = Some(declaration);
        self.gen_node(block);
        self.body_emitters.top().body.newline();
    }

    fn function_declaration(
        &mut self,
        name: usize,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
        type_kind: Option<usize>,
    ) {
        self.emit_function_declaration(
            EmitterKind::Body,
            name,
            return_type_name,
            &params,
            type_kind,
        );
        self.body_emitters.top().body.emit(" ");

        self.function_declaration_prototype(name, return_type_name, params, type_kind);
    }

    fn extern_function(&mut self, declaration: usize, _type_kind: Option<usize>) {
        self.prototype_emitter.emit("extern ");
        self.gen_node_prototype(declaration);
    }

    fn param(&mut self, name: usize, type_name: usize, _type_kind: Option<usize>) {
        self.emit_param(name, type_name, EmitterKind::Body);
    }

    fn copy_array_params(&mut self, function_declaration: usize) {
        let TypedNode {
            node_kind: NodeKind::FunctionDeclaration { params, .. },
            ..
        } = self.typed_nodes[function_declaration].clone()
        else {
            panic!("Invalid function declaration needing init");
        };

        for param in params.iter() {
            let TypedNode {
                node_kind: NodeKind::Param { name, type_name },
                type_kind,
            } = self.typed_nodes[*param].clone()
            else {
                panic!("Invalid param in function declaration needing init");
            };

            if !is_type_kind_array(&self.types, type_kind.unwrap()) {
                continue;
            }

            let copy_name = format!("__{}", name);

            self.emit_type_name_left(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(&copy_name);
            self.emit_type_name_right(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emitln(";");

            let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
            else {
                panic!("Invalid parameter name");
            };
            self.emit_memmove_name_to_name(&copy_name, &name_text, type_kind.unwrap());
            self.body_emitters.top().body.emitln(";");

            self.gen_node(name);
            self.body_emitters.top().body.emit(" = ");
            self.body_emitters.top().body.emit(&copy_name);
            self.body_emitters.top().body.emitln(";");
        }
    }

    fn block(&mut self, statements: Arc<Vec<usize>>, _type_kind: Option<usize>) {
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

        let was_last_statement_return = if let Some(last_statement) = statements.last() {
            let TypedNode {
                node_kind: NodeKind::Statement { inner },
                ..
            } = self.typed_nodes[*last_statement]
            else {
                panic!("Last statement is not a statement");
            };
            matches!(
                self.typed_nodes[inner],
                TypedNode {
                    node_kind: NodeKind::ReturnStatement { .. },
                    ..
                }
            )
        } else {
            false
        };

        self.body_emitters.pop(!was_last_statement_return);
        self.body_emitters.top().body.emit("}");
    }

    fn statement(&mut self, inner: usize, _type_kind: Option<usize>) {
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
        is_mutable: bool,
        name: usize,
        _type_name: Option<usize>,
        expression: usize,
        type_kind: Option<usize>,
    ) {
        let Some(type_kind) = type_kind else {
            panic!("cannot generate variable declaration without a type");
        };

        let is_array = is_type_kind_array(&self.types, type_kind);
        let needs_const = !is_mutable && !is_array;

        self.emit_type_kind_left(type_kind, EmitterKind::Body, false, true);
        if needs_const {
            self.body_emitters.top().body.emit("const ");
        }
        self.gen_node(name);
        self.emit_type_kind_right(type_kind, EmitterKind::Body, false);

        if is_array && !is_typed_expression_array_literal(&self.typed_nodes, expression) {
            self.body_emitters.top().body.emitln(";");

            let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone() else {
                panic!("Invalid variable name");
            };
            self.emit_memmove_expression_to_name(&name_text, expression, type_kind);
        } else {
            self.body_emitters.top().body.emit(" = ");
            self.gen_node(expression);
        }
    }

    fn return_statement(&mut self, expression: Option<usize>, type_kind: Option<usize>) {
        self.body_emitters.exiting_all_scopes();

        let expression = if let Some(expression) = expression {
            expression
        } else {
            self.body_emitters.top().body.emit("return");
            return;
        };

        if is_type_kind_array(&self.types, type_kind.unwrap()) {
            if is_typed_expression_array_literal(&self.typed_nodes, expression) {
                let temp_name = self.temp_variable_name("temp");

                self.emit_type_kind_left(type_kind.unwrap(), EmitterKind::Body, false, true);
                self.body_emitters.top().body.emit(&temp_name);
                self.emit_type_kind_right(type_kind.unwrap(), EmitterKind::Body, false);
                self.body_emitters.top().body.emit(" = ");
                self.gen_node(expression);
                self.body_emitters.top().body.emitln(";");

                self.emit_memmove_name_to_name("__return", &temp_name, type_kind.unwrap());
                self.body_emitters.top().body.emitln(";");
            } else {
                self.emit_memmove_expression_to_name("__return", expression, type_kind.unwrap());
                self.body_emitters.top().body.emitln(";");
            }

            self.body_emitters.top().body.emit("return __return");
        } else {
            self.body_emitters.top().body.emit("return ");
            self.gen_node(expression);
        }
    }

    fn defer_statement(&mut self, statement: usize, _type_kind: Option<usize>) {
        self.body_emitters.push(0);
        self.gen_node(statement);
        self.body_emitters.pop_to_bottom();
    }

    fn if_statement(
        &mut self,
        expression: usize,
        block: usize,
        next: Option<usize>,
        _type_kind: Option<usize>,
    ) {
        self.body_emitters.top().body.emit("if (");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(") ");
        self.gen_node(block);

        if let Some(next) = next {
            self.body_emitters.top().body.emit(" else ");
            self.gen_node(next);

            if matches!(self.typed_nodes[next].node_kind, NodeKind::Block { .. }) {
                self.body_emitters.top().body.newline();
            }
        } else {
            self.body_emitters.top().body.newline();
        }
    }

    fn switch_statement(
        &mut self,
        expression: usize,
        case_block: usize,
        _type_kind: Option<usize>,
    ) {
        self.body_emitters.top().body.emit("switch (");
        self.gen_node(expression);
        self.body_emitters.top().body.emitln(") {");
        self.gen_node(case_block);
        self.body_emitters.top().body.emitln("}");
    }

    fn case_block(
        &mut self,
        expression: usize,
        block: usize,
        next: Option<usize>,
        _type_kind: Option<usize>,
    ) {
        self.body_emitters.top().body.emit("case ");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(": ");
        self.gen_node(block);
        self.body_emitters.top().body.emitln(" break;");

        if let Some(next) = next {
            if matches!(self.typed_nodes[next].node_kind, NodeKind::Block { .. }) {
                self.body_emitters.top().body.emit("default: ");
                self.gen_node(next);
                self.body_emitters.top().body.emitln(" break;");
            } else {
                self.gen_node(next);
            }
        }
    }

    fn while_loop(&mut self, expression: usize, block: usize, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("while (");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(") ");
        self.gen_node(block);
        self.body_emitters.top().body.newline();
    }

    #[allow(clippy::too_many_arguments)]
    fn for_loop(
        &mut self,
        iterator: usize,
        op: Op,
        from: usize,
        to: usize,
        by: Option<usize>,
        block: usize,
        _type_kind: Option<usize>,
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

        self.gen_node(block);
        self.body_emitters.top().body.newline();
    }

    fn binary(&mut self, left: usize, op: Op, right: usize, type_kind: Option<usize>) {
        if op == Op::Assign {
            let is_array = is_type_kind_array(&self.types, type_kind.unwrap());

            if is_array && !is_typed_expression_array_literal(&self.typed_nodes, left) {
                self.emit_memmove_expression_to_variable(left, right, type_kind.unwrap());
            } else {
                self.gen_node(left);
                self.body_emitters.top().body.emit(" = ");
                self.gen_node(right);
            }

            return;
        }

        self.gen_node(left);
        self.emit_binary_op(op);
        self.gen_node(right);
    }

    fn unary_prefix(&mut self, op: Op, right: usize, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit(match op {
            Op::Plus => "+",
            Op::Minus => "-",
            Op::Not => "!",
            Op::Reference => "&",
            Op::Dereference => "*",
            _ => panic!("Expected unary operator"),
        });

        self.gen_node(right);
    }

    fn call(&mut self, left: usize, args: Arc<Vec<usize>>, type_kind: Option<usize>) {
        self.gen_node(left);

        self.body_emitters.top().body.emit("(");
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.gen_node(*arg);
        }

        if is_type_kind_array(&self.types, type_kind.unwrap()) {
            if args.len() > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            let return_array_name = self.temp_variable_name("returnArray");

            self.emit_type_kind_left(type_kind.unwrap(), EmitterKind::Top, false, true);
            self.body_emitters.top().top.emit(&return_array_name);
            self.emit_type_kind_right(type_kind.unwrap(), EmitterKind::Top, false);
            self.body_emitters.top().top.emitln(";");

            self.body_emitters.top().body.emit(&return_array_name);
        }
        self.body_emitters.top().body.emit(")");
    }

    fn index_access(&mut self, left: usize, expression: usize, _type_kind: Option<usize>) {
        self.gen_node(left);
        self.body_emitters.top().body.emit("[");
        self.gen_node(expression);
        self.body_emitters.top().body.emit("]");
    }

    fn field_access(&mut self, left: usize, name: usize, _type_kind: Option<usize>) {
        self.gen_node(left);
        self.body_emitters.top().body.emit(".");
        self.gen_node(name);
    }

    fn cast(&mut self, left: usize, type_name: usize, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("((");
        self.emit_type_name_left(type_name, EmitterKind::Body, false);
        self.emit_type_name_right(type_name, EmitterKind::Body, false);
        self.body_emitters.top().body.emit(")");
        self.gen_node(left);
        self.body_emitters.top().body.emit(")");
    }

    fn name(&mut self, text: String, _type_kind: Option<usize>) {
        self.emit_name(text, EmitterKind::Body);
    }

    fn identifier(&mut self, name: usize, _type_kind: Option<usize>) {
        self.gen_node(name);
    }

    fn name_prototype(&mut self, text: String, _type_kind: Option<usize>) {
        self.emit_name(text, EmitterKind::Prototype);
    }

    fn int_literal(&mut self, text: String, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit(&text);
    }

    fn float32_literal(&mut self, text: String, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit(&text);
        self.body_emitters.top().body.emit("f");
    }

    fn char_literal(&mut self, value: char, _type_kind: Option<usize>) {
        let mut char_buffer = [0u8];

        self.body_emitters.top().body.emit("'");
        self.body_emitters
            .top()
            .body
            .emit(value.encode_utf8(&mut char_buffer));
        self.body_emitters.top().body.emit("'");
    }

    fn string_literal(&mut self, text: String, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("\"");
        for (i, line) in text.lines().enumerate() {
            if i > 0 {
                self.body_emitters.top().body.emit("\\n");
            }

            self.body_emitters.top().body.emit(line);
        }
        self.body_emitters.top().body.emit("\"");
    }

    fn bool_literal(&mut self, value: bool, _type_kind: Option<usize>) {
        if value {
            self.body_emitters.top().body.emit("true");
        } else {
            self.body_emitters.top().body.emit("false");
        }
    }

    fn array_literal(
        &mut self,
        elements: Arc<Vec<usize>>,
        repeat_count: usize,
        _type_kind: Option<usize>,
    ) {
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

    fn struct_literal(&mut self, _name: usize, fields: Arc<Vec<usize>>, type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("(");
        if let Some(type_kind) = type_kind {
            self.emit_type_kind_left(type_kind, EmitterKind::Body, false, true);
            self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
        } else {
            panic!("Can't generate struct literal without a type");
        }
        self.body_emitters.top().body.emit(") ");

        self.body_emitters.top().body.emitln("{");
        self.body_emitters.top().body.indent();

        for field in fields.iter() {
            self.gen_node(*field);
            self.body_emitters.top().body.emitln(",");
        }

        self.body_emitters.top().body.unindent();
        self.body_emitters.top().body.emit("}");
    }

    fn field_literal(&mut self, _name: usize, expression: usize, _type_kind: Option<usize>) {
        self.gen_node(expression);
    }

    fn type_size(&mut self, type_name: usize, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("sizeof(");
        self.emit_type_name_left(type_name, EmitterKind::Body, false);
        self.emit_type_name_right(type_name, EmitterKind::Body, false);
        self.body_emitters.top().body.emit(")");
    }

    fn emit_memmove_expression_to_variable(
        &mut self,
        destination: usize,
        source: usize,
        type_kind: usize,
    ) {
        self.body_emitters.top().body.emit("memmove(");
        self.gen_node(destination);
        self.body_emitters.top().body.emit(", ");
        self.gen_node(source);
        self.body_emitters.top().body.emit(", ");
        self.emit_type_size(type_kind);
        self.body_emitters.top().body.emit(")");
    }

    fn emit_memmove_expression_to_name(
        &mut self,
        destination: &str,
        source: usize,
        type_kind: usize,
    ) {
        self.body_emitters.top().body.emit("memmove(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit(", ");
        self.gen_node(source);
        self.body_emitters.top().body.emit(", ");
        self.emit_type_size(type_kind);
        self.body_emitters.top().body.emit(")");
    }

    fn emit_memmove_name_to_name(&mut self, destination: &str, source: &str, type_kind: usize) {
        self.body_emitters.top().body.emit("memmove(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit(source);
        self.body_emitters.top().body.emit(", ");
        self.emit_type_size(type_kind);
        self.body_emitters.top().body.emit(")");
    }

    fn emit_type_size(&mut self, type_kind: usize) {
        match self.types[type_kind] {
            TypeKind::Array {
                element_type_kind,
                element_count,
            } => {
                self.emit_type_size(element_type_kind);
                self.body_emitters.top().body.emit(" * ");
                self.body_emitters
                    .top()
                    .body
                    .emit(&element_count.to_string());
            }
            _ => {
                self.body_emitters.top().body.emit("sizeof(");
                self.emit_type_kind_left(type_kind, EmitterKind::Body, false, true);
                self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
                self.body_emitters.top().body.emit(")");
            }
        };
    }

    fn emit_type_name_left(
        &mut self,
        type_name: usize,
        emitter_kind: EmitterKind,
        do_arrays_as_pointers: bool,
    ) {
        let TypedNode {
            node_kind: NodeKind::TypeName { type_kind },
            ..
        } = self.typed_nodes[type_name]
        else {
            panic!("Tried to emit node that wasn't a type name");
        };
        self.emit_type_kind_left(type_kind, emitter_kind, do_arrays_as_pointers, true);
    }

    fn emit_type_name_right(
        &mut self,
        type_name: usize,
        emitter_kind: EmitterKind,
        do_arrays_as_pointers: bool,
    ) {
        let TypedNode {
            node_kind: NodeKind::TypeName { type_kind },
            ..
        } = self.typed_nodes[type_name]
        else {
            panic!("Tried to emit node that wasn't a type name");
        };
        self.emit_type_kind_right(type_kind, emitter_kind, do_arrays_as_pointers);
    }

    fn emit_type_kind_left(
        &mut self,
        type_kind: usize,
        emitter_kind: EmitterKind,
        do_arrays_as_pointers: bool,
        is_prefix: bool,
    ) {
        let type_kind = &self.types[type_kind];
        let needs_trailing_space = is_prefix && !matches!(type_kind, TypeKind::Array { .. } | TypeKind::Pointer { .. } | TypeKind::Function { .. });

        match type_kind.clone() {
            TypeKind::Int => self.emitter(emitter_kind).emit("intptr_t"),
            TypeKind::String => self.emitter(emitter_kind).emit("const char*"),
            TypeKind::Bool => self.emitter(emitter_kind).emit("bool"),
            TypeKind::Char => self.emitter(emitter_kind).emit("char"),
            TypeKind::Void => self.emitter(emitter_kind).emit("void"),
            TypeKind::UInt => self.emitter(emitter_kind).emit("uintptr_t"),
            TypeKind::Int8 => self.emitter(emitter_kind).emit("int8_t"),
            TypeKind::UInt8 => self.emitter(emitter_kind).emit("uint8_t"),
            TypeKind::Int16 => self.emitter(emitter_kind).emit("int16_t"),
            TypeKind::UInt16 => self.emitter(emitter_kind).emit("uint16_t"),
            TypeKind::Int32 => self.emitter(emitter_kind).emit("int32_t"),
            TypeKind::UInt32 => self.emitter(emitter_kind).emit("uint32_t"),
            TypeKind::Int64 => self.emitter(emitter_kind).emit("int64_t"),
            TypeKind::UInt64 => self.emitter(emitter_kind).emit("uint64_t"),
            TypeKind::Float32 => self.emitter(emitter_kind).emit("float"),
            TypeKind::Float64 => self.emitter(emitter_kind).emit("double"),
            TypeKind::Struct { name, .. } => {
                self.emitter(emitter_kind).emit("struct ");
                self.gen_node(name);
                self.emitter(emitter_kind).emit(" ");
            }
            TypeKind::Array {
                element_type_kind, ..
            } => {
                self.emit_type_kind_left(element_type_kind, emitter_kind, do_arrays_as_pointers, true);
                if do_arrays_as_pointers {
                    self.emitter(emitter_kind).emit("*");
                }
            }
            TypeKind::Pointer { inner_type_kind } => {
                self.emit_type_kind_left(inner_type_kind, emitter_kind, do_arrays_as_pointers, true);
                self.emitter(emitter_kind).emit("*");
            }
            TypeKind::PartialStruct => panic!("Can't emit partial struct"),
            TypeKind::Function { return_type_kind, .. } => {
                self.emit_type_kind_left(return_type_kind, emitter_kind, true, true);
                self.emit_type_kind_right(return_type_kind, emitter_kind, true);
                self.emitter(emitter_kind).emit("(");
            }
        };

        if needs_trailing_space {
            self.emitter(emitter_kind).emit(" ");
        }
    }

    fn emit_type_kind_right(
        &mut self,
        type_kind: usize,
        emitter_kind: EmitterKind,
        do_arrays_as_pointers: bool,
    ) {
        let type_kind = self.types[type_kind].clone();

        match type_kind {
            TypeKind::Array { element_type_kind, element_count } => {
                if !do_arrays_as_pointers {
                    self.emitter(emitter_kind).emit("[");
                    self.emitter(emitter_kind).emit(&element_count.to_string());
                    self.emitter(emitter_kind).emit("]");
                }
                self.emit_type_kind_right(element_type_kind, emitter_kind, do_arrays_as_pointers);
            }
            TypeKind::Pointer { inner_type_kind } => {
                self.emit_type_kind_right(inner_type_kind, emitter_kind, do_arrays_as_pointers);
            }
            TypeKind::Function { param_kinds, .. } => {
                self.emitter(emitter_kind).emit(")(");
                for (i, param_kind) in param_kinds.iter().enumerate() {
                    if i > 0 {
                        self.emitter(emitter_kind).emit(", ");
                    }

                    self.emit_type_kind_left(*param_kind, emitter_kind, false, false);
                    self.emit_type_kind_right(*param_kind, emitter_kind, false);
                }
                self.emitter(emitter_kind).emit(")");
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
            _ => panic!("Expected binary operator"),
        });
    }

    fn emit_function_declaration(
        &mut self,
        kind: EmitterKind,
        name: usize,
        return_type_name: usize,
        params: &Arc<Vec<usize>>,
        type_kind: Option<usize>,
    ) {
        self.emit_type_name_left(return_type_name, kind, true);

        match kind {
            EmitterKind::Prototype => self.gen_node_prototype(name),
            EmitterKind::Body => self.gen_node(name),
            _ => panic!("Unexpected emitter kind for function declaration"),
        }

        let mut param_count = 0;

        self.emitter(kind).emit("(");
        for param in params.iter() {
            if param_count > 0 {
                self.emitter(kind).emit(", ");
            }

            param_count += 1;

            match kind {
                EmitterKind::Prototype => self.gen_node_prototype(*param),
                EmitterKind::Body => self.gen_node(*param),
                _ => panic!("Unexpected emitter kind for function declaration"),
            }
        }

        let TypeKind::Function { return_type_kind, .. } = self.types[type_kind.unwrap()] else {
            panic!("Tried to emit function declaration for non-function type");
        };

        if is_type_kind_array(&self.types, return_type_kind) {
            if param_count > 0 {
                self.emitter(kind).emit(", ");
            }

            param_count += 1;

            self.emit_param_string("__return", return_type_name, kind);
        }

        if param_count == 0 {
            self.emitter(kind).emit("void");
        }

        self.emitter(kind).emit(")");

        self.emit_type_name_right(return_type_name, kind, true);
    }

    fn emit_param(&mut self, name: usize, type_name: usize, kind: EmitterKind) {
        self.emit_type_name_left(type_name, kind, false);
        match kind {
            EmitterKind::Prototype => self.gen_node_prototype(name),
            EmitterKind::Body => self.gen_node(name),
            _ => panic!("Unexpected emitter kind for parameter"),
        }
        self.emit_type_name_right(type_name, kind, false);
    }

    fn emit_param_string(&mut self, name: &str, type_name: usize, kind: EmitterKind) {
        self.emit_type_name_left(type_name, kind, false);
        self.emitter(kind).emit(name);
        self.emit_type_name_right(type_name, kind, false);
    }

    fn emit_name(&mut self, text: String, kind: EmitterKind) {
        if reserved_names().contains(&text) {
            self.emitter(kind).emit("__");
        }

        self.emitter(kind).emit(&text);
    }

    fn emitter(&mut self, kind: EmitterKind) -> &mut Emitter {
        match kind {
            EmitterKind::Prototype => &mut self.prototype_emitter,
            EmitterKind::Body => &mut self.body_emitters.top().body,
            EmitterKind::Top => &mut self.body_emitters.top().top,
        }
    }

    fn temp_variable_name(&mut self, prefix: &str) -> String {
        let temp_variable_index = self.temp_variable_count;
        self.temp_variable_count += 1;

        format!("__{}{}", prefix, temp_variable_index)
    }
}
