use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, OnceLock},
};

use crate::{
    emitter::Emitter,
    emitter_stack::EmitterStack,
    parser::{NodeKind, Op, TypeKind},
    type_checker::{InstanceKind, Type, TypedNode},
    types::{is_type_kind_array, is_typed_expression_array_literal},
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

#[derive(Clone)]
struct NamespaceName {
    name: Arc<str>,
    generic_param_type_kinds: Option<Arc<Vec<usize>>>,
}

pub struct CodeGenerator {
    pub typed_nodes: Vec<TypedNode>,
    pub type_kinds: Vec<TypeKind>,
    pub generic_usages: HashMap<usize, HashSet<Arc<Vec<usize>>>>,
    pub header_emitter: Emitter,
    pub type_prototype_emitter: Emitter,
    pub function_prototype_emitter: Emitter,
    pub body_emitters: EmitterStack,
    function_declaration_needing_init: Option<usize>,
    temp_variable_count: usize,
    current_namespace_names: Vec<NamespaceName>,
}

impl CodeGenerator {
    pub fn new(
        typed_nodes: Vec<TypedNode>,
        type_kinds: Vec<TypeKind>,
        generic_usages: HashMap<usize, HashSet<Arc<Vec<usize>>>>,
    ) -> Self {
        let mut code_generator = Self {
            typed_nodes,
            type_kinds,
            generic_usages,
            header_emitter: Emitter::new(0),
            type_prototype_emitter: Emitter::new(0),
            function_prototype_emitter: Emitter::new(0),
            body_emitters: EmitterStack::new(),
            function_declaration_needing_init: None,
            temp_variable_count: 0,
            current_namespace_names: Vec::new(),
        };

        code_generator.header_emitter.emitln("#include <string.h>");
        code_generator
            .header_emitter
            .emitln("#include <inttypes.h>");
        code_generator.header_emitter.emitln("#include <stdbool.h>");
        code_generator.header_emitter.newline();
        code_generator.body_emitters.push(1);

        code_generator
    }

    pub fn gen(&mut self, start_index: usize) {
        self.gen_node(start_index);
    }

    fn gen_node(&mut self, index: usize) {
        match self.typed_nodes[index].clone() {
            TypedNode {
                node_kind:
                    NodeKind::TopLevel {
                        functions,
                        structs,
                        enums,
                    },
                node_type,
            } => self.top_level(functions, structs, enums, node_type),
            TypedNode {
                node_kind:
                    NodeKind::StructDefinition {
                        name,
                        fields,
                        functions,
                        ..
                    },
                node_type,
            } => self.struct_definition(name, fields, functions, index, node_type),
            TypedNode {
                node_kind:
                    NodeKind::EnumDefinition {
                        name,
                        variant_names,
                        ..
                    },
                node_type,
            } => self.enum_definition(name, variant_names, node_type),
            TypedNode {
                node_kind: NodeKind::Field { name, type_name },
                node_type,
            } => self.field(name, type_name, node_type),
            TypedNode {
                node_kind: NodeKind::Function { declaration, block },
                node_type,
            } => self.function(declaration, block, node_type),
            TypedNode {
                node_kind:
                    NodeKind::FunctionDeclaration {
                        name,
                        params,
                        return_type_name,
                        ..
                    },
                node_type,
            } => self.function_declaration(name, params, None, return_type_name, node_type),
            TypedNode {
                node_kind: NodeKind::ExternFunction { declaration },
                node_type,
            } => self.extern_function(declaration, node_type),
            TypedNode {
                node_kind: NodeKind::Param { name, type_name },
                node_type,
            } => self.param(name, type_name, node_type),
            TypedNode {
                node_kind: NodeKind::Block { statements },
                node_type,
            } => self.block(statements, node_type),
            TypedNode {
                node_kind: NodeKind::Statement { inner },
                node_type,
            } => self.statement(inner, node_type),
            TypedNode {
                node_kind:
                    NodeKind::VariableDeclaration {
                        is_mutable,
                        name,
                        type_name,
                        expression,
                    },
                node_type,
            } => self.variable_declaration(is_mutable, name, type_name, expression, node_type),
            TypedNode {
                node_kind: NodeKind::ReturnStatement { expression },
                node_type,
            } => self.return_statement(expression, node_type),
            TypedNode {
                node_kind: NodeKind::DeferStatement { statement },
                node_type,
            } => self.defer_statement(statement, node_type),
            TypedNode {
                node_kind:
                    NodeKind::IfStatement {
                        expression,
                        block,
                        next,
                    },
                node_type,
            } => self.if_statement(expression, block, next, node_type),
            TypedNode {
                node_kind:
                    NodeKind::SwitchStatement {
                        expression,
                        case_block,
                    },
                node_type,
            } => self.switch_statement(expression, case_block, node_type),
            TypedNode {
                node_kind:
                    NodeKind::CaseBlock {
                        expression,
                        block,
                        next,
                    },
                node_type,
            } => self.case_block(expression, block, next, node_type),
            TypedNode {
                node_kind: NodeKind::WhileLoop { expression, block },
                node_type,
            } => self.while_loop(expression, block, node_type),
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
                node_type,
            } => self.for_loop(iterator, op, from, to, by, block, node_type),
            TypedNode {
                node_kind: NodeKind::Binary { left, op, right },
                node_type,
            } => self.binary(left, op, right, node_type),
            TypedNode {
                node_kind: NodeKind::UnaryPrefix { op, right },
                node_type,
            } => self.unary_prefix(op, right, node_type),
            TypedNode {
                node_kind: NodeKind::UnarySuffix { left, op },
                node_type,
            } => self.unary_suffix(left, op, node_type),
            TypedNode {
                node_kind: NodeKind::Call { left, args },
                node_type,
            } => self.call(left, args, node_type),
            TypedNode {
                node_kind: NodeKind::IndexAccess { left, expression },
                node_type,
            } => self.index_access(left, expression, node_type),
            TypedNode {
                node_kind: NodeKind::FieldAccess { left, name },
                node_type,
            } => self.field_access(left, name, node_type),
            TypedNode {
                node_kind: NodeKind::Cast { left, type_name },
                node_type,
            } => self.cast(left, type_name, node_type),
            TypedNode {
                node_kind:
                    NodeKind::GenericSpecifier {
                        left,
                        generic_param_type_kinds,
                    },
                node_type,
            } => self.generic_specifier(left, generic_param_type_kinds, node_type),
            TypedNode {
                node_kind: NodeKind::Name { text },
                node_type,
            } => self.name(text, node_type),
            TypedNode {
                node_kind: NodeKind::Identifier { name },
                node_type,
            } => self.identifier(name, node_type),
            TypedNode {
                node_kind: NodeKind::IntLiteral { text },
                node_type,
            } => self.int_literal(text, node_type),
            TypedNode {
                node_kind: NodeKind::Float32Literal { text },
                node_type,
            } => self.float32_literal(text, node_type),
            TypedNode {
                node_kind: NodeKind::CharLiteral { value },
                node_type,
            } => self.char_literal(value, node_type),
            TypedNode {
                node_kind: NodeKind::StringLiteral { text },
                node_type,
            } => self.string_literal(text, node_type),
            TypedNode {
                node_kind: NodeKind::BoolLiteral { value },
                node_type,
            } => self.bool_literal(value, node_type),
            TypedNode {
                node_kind:
                    NodeKind::ArrayLiteral {
                        elements,
                        repeat_count,
                    },
                node_type,
            } => self.array_literal(elements, repeat_count, node_type),
            TypedNode {
                node_kind: NodeKind::StructLiteral { left, fields },
                node_type,
            } => self.struct_literal(left, fields, node_type),
            TypedNode {
                node_kind: NodeKind::FieldLiteral { name, expression },
                node_type,
            } => self.field_literal(name, expression, node_type),
            TypedNode {
                node_kind: NodeKind::TypeSize { type_name },
                node_type,
            } => self.type_size(type_name, node_type),
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

    fn top_level(
        &mut self,
        functions: Arc<Vec<usize>>,
        structs: Arc<Vec<usize>>,
        enums: Arc<Vec<usize>>,
        _node_type: Option<Type>,
    ) {
        for struct_definition in structs.iter() {
            self.gen_node(*struct_definition);
        }

        for enum_definition in enums.iter() {
            self.gen_node(*enum_definition);
        }

        for function in functions.iter() {
            self.gen_node(*function);
        }
    }

    fn struct_definition(
        &mut self,
        name: usize,
        fields: Arc<Vec<usize>>,
        functions: Arc<Vec<usize>>,
        index: usize,
        node_type: Option<Type>,
    ) {
        let TypeKind::Struct {
            generic_type_kinds, ..
        } = self.type_kinds[node_type.unwrap().type_kind].clone()
        else {
            panic!("Invalid function type");
        };

        let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone() else {
            panic!("Invalid name in generic struct");
        };

        if let Some(generic_usages) = self.generic_usages.get(&index) {
            let generic_usages: Vec<Arc<Vec<usize>>> = generic_usages.iter().cloned().collect();

            for generic_usage in generic_usages {
                // Replace generic types with their concrete types for this usage.
                for (generic_param_type_kind, generic_type_kind) in
                    generic_usage.iter().zip(generic_type_kinds.iter())
                {
                    self.type_kinds[*generic_type_kind] = TypeKind::Alias {
                        inner_type_kind: *generic_param_type_kind,
                    };
                }

                // TODO: Duplication.
                self.current_namespace_names.push(NamespaceName {
                    name: name_text.clone(),
                    generic_param_type_kinds: Some(generic_usage.clone()),
                });

                for function in functions.iter() {
                    self.gen_node(*function);
                }

                self.current_namespace_names.pop();

                self.type_prototype_emitter.emit("struct ");
                self.emit_name_node(name, EmitterKind::TypePrototype);
                self.emit_generic_param_suffix(generic_usage, EmitterKind::TypePrototype);

                self.type_prototype_emitter.emitln(" {");
                self.type_prototype_emitter.indent();

                for field in fields.iter() {
                    self.gen_node(*field);
                }

                self.type_prototype_emitter.unindent();
                self.type_prototype_emitter.emitln("};");
                self.type_prototype_emitter.newline();
            }
        } else if generic_type_kinds.is_empty() {
            // TODO: Duplication.
            self.current_namespace_names.push(NamespaceName {
                name: name_text,
                generic_param_type_kinds: None,
            });

            for function in functions.iter() {
                self.gen_node(*function);
            }

            self.current_namespace_names.pop();

            self.type_prototype_emitter.emit("struct ");
            self.emit_name_node(name, EmitterKind::TypePrototype);
            self.type_prototype_emitter.emitln(" {");
            self.type_prototype_emitter.indent();

            for field in fields.iter() {
                self.gen_node(*field);
            }

            self.type_prototype_emitter.unindent();
            self.type_prototype_emitter.emitln("};");
            self.type_prototype_emitter.newline();
        }
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

        let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone() else {
            panic!("Invalid name in enum");
        };

        self.current_namespace_names.push(NamespaceName {
            name: name_text,
            generic_param_type_kinds: None,
        });

        for variant_name in variant_names.iter() {
            self.emit_namespace_prefix(EmitterKind::TypePrototype);
            self.emit_name_node(*variant_name, EmitterKind::TypePrototype);
            self.type_prototype_emitter.emitln(",");
        }

        self.current_namespace_names.pop();

        self.type_prototype_emitter.unindent();
        self.type_prototype_emitter.emitln("};");
        self.type_prototype_emitter.newline();
    }

    fn field(&mut self, name: usize, _type_name: usize, node_type: Option<Type>) {
        self.emit_type_kind_left(
            node_type.unwrap().type_kind,
            EmitterKind::TypePrototype,
            false,
            true,
        );
        self.emit_name_node(name, EmitterKind::TypePrototype);
        self.emit_type_kind_right(
            node_type.unwrap().type_kind,
            EmitterKind::TypePrototype,
            false,
        );
        self.type_prototype_emitter.emitln(";");
    }

    fn function(&mut self, declaration: usize, block: usize, node_type: Option<Type>) {
        let NodeKind::FunctionDeclaration {
            name,
            params,
            return_type_name,
            ..
        } = self.typed_nodes[declaration].node_kind.clone()
        else {
            panic!("Invalid function declaration");
        };

        let TypeKind::Function {
            generic_type_kinds, ..
        } = self.type_kinds[node_type.unwrap().type_kind].clone()
        else {
            panic!("Invalid function type");
        };

        if let Some(generic_usages) = self.generic_usages.get(&declaration) {
            let generic_usages: Vec<Arc<Vec<usize>>> = generic_usages.iter().cloned().collect();

            for generic_usage in generic_usages {
                // Replace generic types with their concrete types for this usage.
                for (generic_param_type_kind, generic_type_kind) in
                    generic_usage.iter().zip(generic_type_kinds.iter())
                {
                    self.type_kinds[*generic_type_kind] = TypeKind::Alias {
                        inner_type_kind: *generic_param_type_kind,
                    };
                }

                self.function_declaration(
                    name,
                    params.clone(),
                    Some(generic_usage),
                    return_type_name,
                    node_type,
                );
                self.function_declaration_needing_init = Some(declaration);
                self.gen_node(block);
                self.body_emitters.top().body.newline();
                self.body_emitters.top().body.newline();
            }
        } else if generic_type_kinds.is_empty() {
            self.function_declaration(name, params, None, return_type_name, node_type);
            self.function_declaration_needing_init = Some(declaration);
            self.gen_node(block);
            self.body_emitters.top().body.newline();
            self.body_emitters.top().body.newline();
        }
    }

    fn function_declaration(
        &mut self,
        name: usize,
        params: Arc<Vec<usize>>,
        generic_usage: Option<Arc<Vec<usize>>>,
        return_type_name: usize,
        node_type: Option<Type>,
    ) {
        self.emit_function_declaration(
            EmitterKind::Body,
            name,
            &params,
            generic_usage.clone(),
            return_type_name,
            node_type.unwrap().type_kind,
        );
        self.body_emitters.top().body.emit(" ");

        self.emit_function_declaration(
            EmitterKind::FunctionPrototype,
            name,
            &params,
            generic_usage,
            return_type_name,
            node_type.unwrap().type_kind,
        );
        self.function_prototype_emitter.emitln(";");
        self.function_prototype_emitter.newline();
    }

    fn extern_function(&mut self, declaration: usize, _node_type: Option<Type>) {
        self.function_prototype_emitter.emit("extern ");
        let NodeKind::FunctionDeclaration {
            name,
            params,
            return_type_name,
            type_kind,
            ..
        } = self.typed_nodes[declaration].node_kind.clone()
        else {
            panic!("Invalid function declaration");
        };
        self.emit_function_declaration(
            EmitterKind::FunctionPrototype,
            name,
            &params,
            None,
            return_type_name,
            type_kind,
        );
        self.function_prototype_emitter.emitln(";");
        self.function_prototype_emitter.newline();
    }

    fn param(&mut self, name: usize, type_name: usize, _node_type: Option<Type>) {
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
                node_type,
            } = self.typed_nodes[*param].clone()
            else {
                panic!("Invalid param in function declaration needing init");
            };

            let type_kind = node_type.unwrap().type_kind;

            if !is_type_kind_array(&self.type_kinds, type_kind) {
                continue;
            }

            let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
            else {
                panic!("Invalid parameter name");
            };
            let copy_name = format!("__{}", &name_text);

            self.emit_type_name_left(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(&copy_name);
            self.emit_type_name_right(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emitln(";");

            self.emit_memmove_name_to_name(&copy_name, &name_text, type_kind);
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

    fn statement(&mut self, inner: usize, _node_type: Option<Type>) {
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
        node_type: Option<Type>,
    ) {
        let type_kind = node_type.unwrap().type_kind;

        let is_array = is_type_kind_array(&self.type_kinds, type_kind);
        let needs_const = !is_mutable && !is_array;

        self.emit_type_kind_left(type_kind, EmitterKind::Body, false, true);
        if needs_const {
            self.body_emitters.top().body.emit("const ");
        }
        self.gen_node(name);
        self.emit_type_kind_right(type_kind, EmitterKind::Body, false);

        if is_array && !is_typed_expression_array_literal(&self.typed_nodes, expression) {
            self.body_emitters.top().body.emitln(";");

            let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
            else {
                panic!("Invalid variable name");
            };
            self.emit_memmove_expression_to_name(&name_text, expression, type_kind);
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

        let type_kind = node_type.unwrap().type_kind;

        if is_type_kind_array(&self.type_kinds, type_kind) {
            if is_typed_expression_array_literal(&self.typed_nodes, expression) {
                let temp_name = self.temp_variable_name("temp");

                self.emit_type_kind_left(type_kind, EmitterKind::Body, false, true);
                self.body_emitters.top().body.emit(&temp_name);
                self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
                self.body_emitters.top().body.emit(" = ");
                self.gen_node(expression);
                self.body_emitters.top().body.emitln(";");

                self.emit_memmove_name_to_name("__return", &temp_name, type_kind);
                self.body_emitters.top().body.emitln(";");
            } else {
                self.emit_memmove_expression_to_name("__return", expression, type_kind);
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
        block: usize,
        next: Option<usize>,
        _node_type: Option<Type>,
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

    fn switch_statement(&mut self, expression: usize, case_block: usize, _node_type: Option<Type>) {
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
        _node_type: Option<Type>,
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

    fn while_loop(&mut self, expression: usize, block: usize, _node_type: Option<Type>) {
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

        self.gen_node(block);
        self.body_emitters.top().body.newline();
    }

    fn binary(&mut self, left: usize, op: Op, right: usize, node_type: Option<Type>) {
        if op == Op::Assign {
            let type_kind = node_type.unwrap().type_kind;
            let is_array = is_type_kind_array(&self.type_kinds, type_kind);

            if is_array && !is_typed_expression_array_literal(&self.typed_nodes, left) {
                self.emit_memmove_expression_to_variable(left, right, type_kind);
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

    fn unary_prefix(&mut self, op: Op, right: usize, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit(match op {
            Op::Plus => "+",
            Op::Minus => "-",
            Op::Not => "!",
            Op::Reference => "&",
            _ => panic!("Expected unary prefix operator"),
        });

        self.gen_node(right);
    }

    fn unary_suffix(&mut self, left: usize, op: Op, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit("(");
        self.body_emitters.top().body.emit(match op {
            Op::Dereference => "*",
            _ => panic!("Expected unary suffix operator"),
        });

        self.gen_node(left);
        self.body_emitters.top().body.emit(")");
    }

    fn call(&mut self, mut left: usize, args: Arc<Vec<usize>>, node_type: Option<Type>) {
        self.gen_node(left);

        self.body_emitters.top().body.emit("(");
        let mut i = 0;

        if let NodeKind::GenericSpecifier { left: new_left, .. } = &self.typed_nodes[left].node_kind
        {
            left = *new_left;
        }

        // If the left node is a field access, then we must be calling a method. If we were calling a function pointer,
        // the left node wouldn't be a field access because the function pointer would need to be dereferenced before calling.
        if let NodeKind::FieldAccess {
            left: field_access_left,
            ..
        } = &self.typed_nodes[left].node_kind
        {
            let field_access_left_type = self.typed_nodes[*field_access_left].node_type.unwrap();

            if field_access_left_type.instance_kind == InstanceKind::Variable {
                // We have a variable to pass as the first parameter.
                if !matches!(
                    self.type_kinds[field_access_left_type.type_kind],
                    TypeKind::Pointer { .. }
                ) {
                    self.body_emitters.top().body.emit("&");
                }

                self.gen_node(*field_access_left);

                i += 1;
            }
        };

        for arg in args.iter() {
            if i > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.gen_node(*arg);

            i += 1;
        }

        let type_kind = node_type.unwrap().type_kind;
        if is_type_kind_array(&self.type_kinds, type_kind) {
            if args.len() > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            let return_array_name = self.temp_variable_name("returnArray");

            self.emit_type_kind_left(type_kind, EmitterKind::Top, false, true);
            self.body_emitters.top().top.emit(&return_array_name);
            self.emit_type_kind_right(type_kind, EmitterKind::Top, false);
            self.body_emitters.top().top.emitln(";");

            self.body_emitters.top().body.emit(&return_array_name);
        }
        self.body_emitters.top().body.emit(")");
    }

    fn index_access(&mut self, left: usize, expression: usize, _node_type: Option<Type>) {
        self.gen_node(left);
        self.body_emitters.top().body.emit("[");
        self.gen_node(expression);
        self.body_emitters.top().body.emit("]");
    }

    fn field_access(&mut self, left: usize, name: usize, node_type: Option<Type>) {
        let left_type = self.typed_nodes[left].node_type.unwrap();

        if matches!(
            self.type_kinds[node_type.unwrap().type_kind],
            TypeKind::Function { .. }
        ) {
            let TypeKind::Struct {
                name: struct_name, ..
            } = self.type_kinds[left_type.type_kind]
            else {
                panic!("Expected function field to be part of a struct");
            };

            let NodeKind::Name {
                text: struct_name_text,
            } = self.typed_nodes[struct_name].node_kind.clone()
            else {
                panic!("Invalid name in struct field access");
            };

            self.current_namespace_names.push(NamespaceName {
                name: struct_name_text,
                generic_param_type_kinds: None,
            });
            self.emit_namespace_prefix(EmitterKind::Body);
            self.gen_node(name);
            self.current_namespace_names.pop();

            return;
        }

        match self.type_kinds[left_type.type_kind] {
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
            _ => panic!("Tried to access type that cannot be accessed"),
        }

        self.gen_node(name);
    }

    fn cast(&mut self, left: usize, type_name: usize, _node_type: Option<Type>) {
        self.body_emitters.top().body.emit("((");
        self.emit_type_name_left(type_name, EmitterKind::Body, false);
        self.emit_type_name_right(type_name, EmitterKind::Body, false);
        self.body_emitters.top().body.emit(")");
        self.gen_node(left);
        self.body_emitters.top().body.emit(")");
    }

    fn generic_specifier(
        &mut self,
        left: usize,
        generic_param_type_kinds: Arc<Vec<usize>>,
        _node_type: Option<Type>,
    ) {
        self.gen_node(left);
        self.emit_generic_param_suffix(generic_param_type_kinds, EmitterKind::Body);
    }

    fn name(&mut self, text: Arc<str>, _node_type: Option<Type>) {
        self.emit_name(text, EmitterKind::Body);
    }

    fn identifier(&mut self, name: usize, _node_type: Option<Type>) {
        self.gen_node(name);
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
        repeat_count: usize,
        _node_type: Option<Type>,
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

    fn struct_literal(&mut self, _left: usize, fields: Arc<Vec<usize>>, node_type: Option<Type>) {
        let type_kind = node_type.unwrap().type_kind;

        self.body_emitters.top().body.emit("(");
        self.emit_type_kind_left(type_kind, EmitterKind::Body, false, false);
        self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
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

    fn field_literal(&mut self, _name: usize, expression: usize, _node_type: Option<Type>) {
        self.gen_node(expression);
    }

    fn type_size(&mut self, type_name: usize, _node_type: Option<Type>) {
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
        match self.type_kinds[type_kind] {
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
            node_type: Some(Type { type_kind, .. }),
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
            node_type: Some(Type { type_kind, .. }),
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
        let type_kind = &self.type_kinds[type_kind];
        let needs_trailing_space = is_prefix
            && !matches!(
                type_kind,
                TypeKind::Array { .. } | TypeKind::Pointer { .. } | TypeKind::Function { .. }
            );

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
            TypeKind::Struct {
                name,
                generic_param_type_kinds,
                ..
            } => {
                self.emitter(emitter_kind).emit("struct ");
                let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
                    panic!("Invalid struct name");
                };
                self.emit_name(text, emitter_kind);
                self.emit_generic_param_suffix(generic_param_type_kinds, emitter_kind);
            }
            TypeKind::Enum { name, .. } => {
                self.emitter(emitter_kind).emit("enum ");
                let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
                    panic!("Invalid enum name");
                };
                self.emit_name(text, emitter_kind);
            }
            TypeKind::Array {
                element_type_kind, ..
            } => {
                self.emit_type_kind_left(
                    element_type_kind,
                    emitter_kind,
                    do_arrays_as_pointers,
                    true,
                );
                if do_arrays_as_pointers {
                    self.emitter(emitter_kind).emit("*");
                }
            }
            TypeKind::Pointer { inner_type_kind } => {
                self.emit_type_kind_left(
                    inner_type_kind,
                    emitter_kind,
                    do_arrays_as_pointers,
                    true,
                );
                self.emitter(emitter_kind).emit("*");
            }
            TypeKind::Alias { inner_type_kind } => {
                self.emit_type_kind_left(
                    inner_type_kind,
                    emitter_kind,
                    do_arrays_as_pointers,
                    is_prefix,
                );
            }
            TypeKind::Partial | TypeKind::PartialGeneric { .. } => {
                panic!("Can't emit partial type: {:?}", type_kind)
            }
            TypeKind::Function {
                return_type_kind, ..
            } => {
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
        let type_kind = self.type_kinds[type_kind].clone();

        match type_kind {
            TypeKind::Array {
                element_type_kind,
                element_count,
            } => {
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
            TypeKind::Function {
                param_type_kinds, ..
            } => {
                self.emitter(emitter_kind).emit(")(");
                for (i, param_kind) in param_type_kinds.iter().enumerate() {
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
        params: &Arc<Vec<usize>>,
        generic_usage: Option<Arc<Vec<usize>>>,
        return_type_name: usize,
        type_kind: usize,
    ) {
        self.emit_type_name_left(return_type_name, kind, true);
        self.emit_function_name(name, generic_usage, kind);

        let mut param_count = 0;

        self.emitter(kind).emit("(");
        for param in params.iter() {
            if param_count > 0 {
                self.emitter(kind).emit(", ");
            }

            param_count += 1;

            self.emit_param_node(*param, kind);
        }

        let TypeKind::Function {
            return_type_kind, ..
        } = self.type_kinds[type_kind]
        else {
            panic!("Tried to emit function declaration for non-function type");
        };

        if is_type_kind_array(&self.type_kinds, return_type_kind) {
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

    fn emit_param_node(&mut self, param: usize, kind: EmitterKind) {
        let NodeKind::Param { name, type_name } = self.typed_nodes[param].node_kind else {
            panic!("Invalid param");
        };

        self.emit_param(name, type_name, kind);
    }

    fn emit_param(&mut self, name: usize, type_name: usize, kind: EmitterKind) {
        self.emit_type_name_left(type_name, kind, false);
        self.emit_name_node(name, kind);
        self.emit_type_name_right(type_name, kind, false);
    }

    fn emit_param_string(&mut self, name: &str, type_name: usize, kind: EmitterKind) {
        self.emit_type_name_left(type_name, kind, false);
        self.emitter(kind).emit(name);
        self.emit_type_name_right(type_name, kind, false);
    }

    fn emit_name_node(&mut self, name: usize, kind: EmitterKind) {
        let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
            panic!("Invalid name");
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
    fn emit_generic_param_suffix(
        &mut self,
        generic_param_type_kinds: Arc<Vec<usize>>,
        kind: EmitterKind,
    ) {
        if generic_param_type_kinds.is_empty() {
            return;
        }

        self.emitter(kind).emit("_");

        for mut generic_param_type_kind in generic_param_type_kinds.iter().copied() {
            if let TypeKind::Alias { inner_type_kind } = self.type_kinds[generic_param_type_kind] {
                generic_param_type_kind = inner_type_kind;
            }

            self.emitter(kind).emit("_");

            // This prints the number backwards, but it doesn't matter for the purpose of name mangling.
            let mut number = generic_param_type_kind;
            let mut digit = 0;
            while number > 0 || digit == 0 {
                self.emitter(kind)
                    .emit_char(((number % 10) as u8 + b'0') as char);
                number /= 10;
                digit += 1;
            }
        }
    }

    fn emit_namespace_prefix(&mut self, kind: EmitterKind) {
        if self.current_namespace_names.is_empty() {
            return;
        }

        self.emitter(kind).emit("__");

        for i in 0..self.current_namespace_names.len() {
            let namespace_name = self.current_namespace_names[i].clone();
            self.emitter(kind).emit(&namespace_name.name);

            if let Some(generic_param_type_kinds) = namespace_name.generic_param_type_kinds {
                self.emit_generic_param_suffix(generic_param_type_kinds, kind);
            }
        }
    }

    fn emit_function_name(
        &mut self,
        name: usize,
        generic_usage: Option<Arc<Vec<usize>>>,
        kind: EmitterKind,
    ) {
        self.emit_namespace_prefix(kind);
        self.emit_name_node(name, kind);
        if let Some(generic_usage) = generic_usage {
            self.emit_generic_param_suffix(generic_usage, kind);
        }
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
