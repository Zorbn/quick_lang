use std::sync::Arc;

use crate::{
    emitter::Emitter,
    emitter_stack::EmitterStack,
    parser::{
        NodeKind, Op, TrailingBinary, TrailingComparison, TrailingTerm, TrailingUnary, TypeKind,
    },
    type_checker::TypedNode,
    types::{is_type_kind_array, is_typed_expression_array_literal},
};

#[derive(Clone, Copy, Debug)]
enum EmitterKind {
    Prototype,
    Body,
    Top,
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
        code_generator.header_emitter.emitln("#include <inttypes.h>");
        code_generator.body_emitters.push();

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
                node_kind:
                    NodeKind::VariableAssignment {
                        variable,
                        expression,
                    },
                type_kind,
            } => self.variable_assignment(variable, expression, type_kind),
            TypedNode {
                node_kind: NodeKind::ReturnStatement { expression },
                type_kind,
            } => self.return_statement(expression, type_kind),
            TypedNode {
                node_kind: NodeKind::IfStatement { expression, block },
                type_kind,
            } => self.if_statement(expression, block, type_kind),
            TypedNode {
                node_kind: NodeKind::WhileLoop { expression, block },
                type_kind,
            } => self.while_loop(expression, block, type_kind),
            TypedNode {
                node_kind: NodeKind::ForLoop { iterator, inclusive, from, to, by, block },
                type_kind,
            } => self.for_loop(iterator, inclusive, from, to, by, block, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::Expression {
                        comparison,
                        trailing_comparisons,
                    },
                type_kind,
            } => self.expression(comparison, trailing_comparisons, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::Comparision {
                        binary,
                        trailing_binary,
                    },
                type_kind,
            } => self.comparison(binary, trailing_binary, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::Binary {
                        term,
                        trailing_terms,
                    },
                type_kind,
            } => self.binary(term, trailing_terms, type_kind),
            TypedNode {
                node_kind:
                    NodeKind::Term {
                        unary,
                        trailing_unaries,
                    },
                type_kind,
            } => self.term(unary, trailing_unaries, type_kind),
            TypedNode {
                node_kind: NodeKind::Unary { op, primary },
                type_kind,
            } => self.unary(op, primary, type_kind),
            TypedNode {
                node_kind: NodeKind::Primary { inner },
                type_kind,
            } => self.primary(inner, type_kind),
            TypedNode {
                node_kind: NodeKind::ParenthesizedExpression { expression },
                type_kind,
            } => self.parenthesized_expression(expression, type_kind),
            TypedNode {
                node_kind: NodeKind::Variable { inner },
                type_kind,
            } => self.variable(inner, type_kind),
            TypedNode {
                node_kind: NodeKind::VariableName { name },
                type_kind,
            } => self.variable_name(name, type_kind),
            TypedNode {
                node_kind: NodeKind::VariableIndex { parent, expression },
                type_kind,
            } => self.variable_index(parent, expression, type_kind),
            TypedNode {
                node_kind: NodeKind::VariableField { parent, name },
                type_kind,
            } => self.variable_field(parent, name, type_kind),
            TypedNode {
                node_kind: NodeKind::FunctionCall { name, args },
                type_kind,
            } => self.function_call(name, args, type_kind),
            TypedNode {
                node_kind: NodeKind::IntLiteral { text },
                type_kind,
            } => self.int_literal(text, type_kind),
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
                node_kind: NodeKind::TypeName { .. },
                ..
            } => panic!("Cannot generate type name with gen_node"),
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
                    },
                type_kind,
            } => self.function_declaration_prototype(name, return_type_name, params, type_kind),
            TypedNode {
                node_kind: NodeKind::Param { name, type_name },
                type_kind,
            } => self.param_prototype(name, type_name, type_kind),
            // NodeKind::TypeName { type_kind } => self.type_name_prototype(type_kind),
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
            if i > 0 {
                self.body_emitters.top().body.newline();
            }

            self.gen_node(*function);

            if matches!(self.typed_nodes[*function], TypedNode { node_kind: NodeKind::Function { .. }, .. }) {
                i += 1;
            }
        }
    }

    fn struct_definition(
        &mut self,
        name: String,
        fields: Arc<Vec<usize>>,
        _type_kind: Option<usize>,
    ) {
        self.prototype_emitter.emit("struct ");
        self.prototype_emitter.emit(&name);
        self.prototype_emitter.emitln(" {");
        self.prototype_emitter.indent();

        for field in fields.iter() {
            self.gen_node(*field);
        }

        self.prototype_emitter.unindent();
        self.prototype_emitter.emitln("};");
    }

    fn field(&mut self, name: String, _type_name: usize, type_kind: Option<usize>) {
        self.emit_type_kind_left(type_kind.unwrap(), EmitterKind::Prototype, false);
        self.prototype_emitter.emit(" ");
        self.prototype_emitter.emit(&name);
        self.emit_type_kind_right(type_kind.unwrap(), EmitterKind::Prototype, false);
        self.prototype_emitter.emitln(";");
    }

    fn function_declaration_prototype(
        &mut self,
        name: String,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
        type_kind: Option<usize>,
    ) {
        self.emit_type_name_left(return_type_name, EmitterKind::Prototype, true);
        self.emit_type_name_right(return_type_name, EmitterKind::Prototype, true);

        self.prototype_emitter.emit(" ");
        self.prototype_emitter.emit(&name);

        self.prototype_emitter.emit("(");
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                self.prototype_emitter.emit(", ");
            }

            self.gen_node_prototype(*param);
        }

        if is_type_kind_array(&self.types, type_kind.unwrap()) {
            if params.len() > 0 {
                self.prototype_emitter.emit(", ");
            }

            self.param_prototype("__return".into(), return_type_name, type_kind);
        }
        self.prototype_emitter.emitln(");");
    }

    fn param_prototype(&mut self, name: String, type_name: usize, _type_kind: Option<usize>) {
        self.emit_type_name_left(type_name, EmitterKind::Prototype, false);
        self.prototype_emitter.emit(" ");
        self.prototype_emitter.emit(&name);
        self.emit_type_name_right(type_name, EmitterKind::Prototype, false);
    }

    fn function(&mut self, declaration: usize, block: usize, _type_kind: Option<usize>) {
        self.gen_node(declaration);
        self.function_declaration_needing_init = Some(declaration);
        self.gen_node(block);
    }

    fn function_declaration(
        &mut self,
        name: String,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
        type_kind: Option<usize>,
    ) {
        self.emit_type_name_left(return_type_name, EmitterKind::Body, true);
        self.emit_type_name_right(return_type_name, EmitterKind::Body, true);
        self.body_emitters.top().body.emit(" ");
        self.body_emitters.top().body.emit(&name);

        self.body_emitters.top().body.emit("(");
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.gen_node(*param);
        }

        if is_type_kind_array(&self.types, type_kind.unwrap()) {
            if params.len() > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.param("__return".into(), return_type_name, type_kind);
        }
        self.body_emitters.top().body.emit(") ");

        self.function_declaration_prototype(name, return_type_name, params, type_kind);
    }

    fn extern_function(&mut self, declaration: usize, _type_kind: Option<usize>) {
        self.prototype_emitter.emit("extern ");
        self.gen_node_prototype(declaration);
    }

    fn param(&mut self, name: String, type_name: usize, _type_kind: Option<usize>) {
        self.emit_type_name_left(type_name, EmitterKind::Body, false);
        self.body_emitters.top().body.emit(" ");
        self.body_emitters.top().body.emit(&name);
        self.emit_type_name_right(type_name, EmitterKind::Body, false);
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
            self.body_emitters.top().body.emit(" ");
            self.body_emitters.top().body.emit(&copy_name);
            self.emit_type_name_right(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emitln(";");

            self.emit_memmove_name_to_name(&copy_name, &name, type_kind.unwrap());
            self.body_emitters.top().body.emitln(";");

            self.body_emitters.top().body.emit(&name);
            self.body_emitters.top().body.emit(" = ");
            self.body_emitters.top().body.emit(&copy_name);
            self.body_emitters.top().body.emitln(";");
        }
    }

    fn block(&mut self, statements: Arc<Vec<usize>>, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emitln("{");
        self.body_emitters.push();

        // Make copies of any parameters that are arrays, because arrays are supposed to be passed by value.
        if let Some(function_declaration) = self.function_declaration_needing_init {
            self.copy_array_params(function_declaration);
            self.function_declaration_needing_init = None;
        }

        for statement in statements.iter() {
            self.gen_node(*statement);
        }

        self.body_emitters.pop();
        self.body_emitters.top().body.emitln("}");
    }

    fn statement(&mut self, inner: usize, _type_kind: Option<usize>) {
        let needs_semicolon = !matches!(
            self.typed_nodes[inner],
            TypedNode {
                node_kind: NodeKind::IfStatement { .. },
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

        self.gen_node(inner);

        if needs_semicolon {
            self.body_emitters.top().body.emitln(";");
        }
    }

    fn variable_declaration(
        &mut self,
        is_mutable: bool,
        name: String,
        type_name: usize,
        expression: usize,
        type_kind: Option<usize>,
    ) {
        let is_array = is_type_kind_array(&self.types, type_kind.unwrap());

        if !is_mutable && !is_array {
            self.body_emitters.top().body.emit("const ");
        }

        if is_array && !is_typed_expression_array_literal(&self.typed_nodes, expression) {
            self.emit_type_name_left(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(" ");
            self.body_emitters.top().body.emit(&name);
            self.emit_type_name_right(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emitln(";");

            self.emit_memmove_expression_to_name(&name, expression, type_kind.unwrap());
        } else {
            self.emit_type_name_left(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(" ");
            self.body_emitters.top().body.emit(&name);
            self.emit_type_name_right(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(" = ");
            self.gen_node(expression);
        }
    }

    fn variable_assignment(
        &mut self,
        variable: usize,
        expression: usize,
        type_kind: Option<usize>,
    ) {
        let is_array = is_type_kind_array(&self.types, type_kind.unwrap());

        if is_array && !is_typed_expression_array_literal(&self.typed_nodes, expression) {
            self.emit_memmove_expression_to_variable(variable, expression, type_kind.unwrap());
        } else {
            self.gen_node(variable);
            self.body_emitters.top().body.emit(" = ");
            self.gen_node(expression);
        }
    }

    fn return_statement(&mut self, expression: usize, type_kind: Option<usize>) {
        if is_type_kind_array(&self.types, type_kind.unwrap()) {
            if is_typed_expression_array_literal(&self.typed_nodes, expression) {
                let temp_name = self.temp_variable_name("temp");

                self.emit_type_kind_left(type_kind.unwrap(), EmitterKind::Body, false);
                self.body_emitters.top().body.emit(" ");
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

    fn if_statement(&mut self, expression: usize, block: usize, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("if (");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(") ");
        self.gen_node(block);
    }

    fn while_loop(&mut self, expression: usize, block: usize, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("while (");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(") ");
        self.gen_node(block);
    }

    #[allow(clippy::too_many_arguments)]
    fn for_loop(&mut self, iterator: String, inclusive: bool, from: i64, to: i64, by: i64, block: usize, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("for (intptr_t ");
        self.body_emitters.top().body.emit(&iterator);
        self.body_emitters.top().body.emit(" = ");
        self.body_emitters.top().body.emit(&from.to_string());
        self.body_emitters.top().body.emit("; ");

        self.body_emitters.top().body.emit(&iterator);
        if from < to {
            if inclusive {
                self.body_emitters.top().body.emit(" <= ");
            } else {
                self.body_emitters.top().body.emit(" < ");
            }
        } else if inclusive {
            self.body_emitters.top().body.emit(" >= ");
        } else {
            self.body_emitters.top().body.emit(" > ");
        }
        self.body_emitters.top().body.emit(&to.to_string());
        self.body_emitters.top().body.emit("; ");

        self.body_emitters.top().body.emit(&iterator);
        if by >= 0 {
            self.body_emitters.top().body.emit(" += ");
            self.body_emitters.top().body.emit(&by.to_string());
        } else {
            self.body_emitters.top().body.emit(" -= ");
            self.body_emitters.top().body.emit(&(-by).to_string());
        }
        self.body_emitters.top().body.emit(") ");

        self.gen_node(block);
    }

    fn expression(
        &mut self,
        comparison: usize,
        trailing_comparisons: Arc<Vec<TrailingComparison>>,
        _type_kind: Option<usize>,
    ) {
        self.gen_node(comparison);

        for trailing_comparison in trailing_comparisons.iter() {
            if trailing_comparison.op == Op::And {
                self.body_emitters.top().body.emit(" && ");
            } else {
                self.body_emitters.top().body.emit(" || ");
            }

            self.gen_node(trailing_comparison.comparison);
        }
    }

    fn comparison(
        &mut self,
        binary: usize,
        trailing_binary: Option<TrailingBinary>,
        _type_kind: Option<usize>,
    ) {
        self.gen_node(binary);

        if let Some(trailing_binary) = trailing_binary {
            match trailing_binary.op {
                Op::EqualEqual => self.body_emitters.top().body.emit(" == "),
                Op::NotEqual => self.body_emitters.top().body.emit(" != "),
                Op::Less => self.body_emitters.top().body.emit(" < "),
                Op::Greater => self.body_emitters.top().body.emit(" > "),
                Op::LessEqual => self.body_emitters.top().body.emit(" <= "),
                Op::GreaterEqual => self.body_emitters.top().body.emit(" >= "),
                _ => panic!("Unexpected operator in comparison"),
            }

            self.gen_node(trailing_binary.binary);
        }
    }

    fn binary(
        &mut self,
        term: usize,
        trailing_terms: Arc<Vec<TrailingTerm>>,
        _type_kind: Option<usize>,
    ) {
        self.gen_node(term);

        for trailing_term in trailing_terms.iter() {
            if trailing_term.op == Op::Add {
                self.body_emitters.top().body.emit(" + ");
            } else {
                self.body_emitters.top().body.emit(" - ");
            }

            self.gen_node(trailing_term.term);
        }
    }

    fn term(
        &mut self,
        unary: usize,
        trailing_unaries: Arc<Vec<TrailingUnary>>,
        _type_kind: Option<usize>,
    ) {
        self.gen_node(unary);

        for trailing_unary in trailing_unaries.iter() {
            if trailing_unary.op == Op::Multiply {
                self.body_emitters.top().body.emit(" * ");
            } else {
                self.body_emitters.top().body.emit(" / ");
            }

            self.gen_node(trailing_unary.unary);
        }
    }

    fn unary(&mut self, op: Option<Op>, primary: usize, _type_kind: Option<usize>) {
        match op {
            Some(Op::Add) => self.body_emitters.top().body.emit(" + "),
            Some(Op::Subtract) => self.body_emitters.top().body.emit(" - "),
            None => {}
            _ => panic!("Unexpected operator in unary"),
        }

        self.gen_node(primary);
    }

    fn primary(&mut self, inner: usize, _type_kind: Option<usize>) {
        self.gen_node(inner);
    }

    fn parenthesized_expression(&mut self, expression: usize, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("(");
        self.gen_node(expression);
        self.body_emitters.top().body.emit(")");
    }

    fn variable(&mut self, inner: usize, _type_kind: Option<usize>) {
        self.gen_node(inner);
    }

    fn variable_name(&mut self, name: String, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit(&name);
    }

    fn variable_index(&mut self, parent: usize, expression: usize, _type_kind: Option<usize>) {
        self.gen_node(parent);
        self.body_emitters.top().body.emit("[");
        self.gen_node(expression);
        self.body_emitters.top().body.emit("]");
    }

    fn variable_field(&mut self, parent: usize, name: String, _type_kind: Option<usize>) {
        self.gen_node(parent);
        self.body_emitters.top().body.emit(".");
        self.body_emitters.top().body.emit(&name);
    }

    fn function_call(&mut self, name: String, args: Arc<Vec<usize>>, type_kind: Option<usize>) {
        self.body_emitters.top().body.emit(&name);

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

            self.emit_type_kind_left(type_kind.unwrap(), EmitterKind::Top, false);
            self.body_emitters.top().top.emit(" ");
            self.body_emitters.top().top.emit(&return_array_name);
            self.emit_type_kind_right(type_kind.unwrap(), EmitterKind::Top, false);
            self.body_emitters.top().top.emitln(";");

            self.body_emitters.top().body.emit(&return_array_name);
        }
        self.body_emitters.top().body.emit(")");
    }

    fn int_literal(&mut self, text: String, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit(&text);
    }

    fn string_literal(&mut self, text: String, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("\"");
        self.body_emitters.top().body.emit(&text);
        self.body_emitters.top().body.emit("\"");
    }

    fn bool_literal(&mut self, value: bool, _type_kind: Option<usize>) {
        if value {
            self.body_emitters.top().body.emit("1");
        } else {
            self.body_emitters.top().body.emit("0");
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

    fn struct_literal(
        &mut self,
        _name: String,
        fields: Arc<Vec<usize>>,
        _type_kind: Option<usize>,
    ) {
        self.body_emitters.top().body.emitln("{");
        self.body_emitters.top().body.indent();

        for field in fields.iter() {
            self.gen_node(*field);
            self.body_emitters.top().body.emitln(",");
        }

        self.body_emitters.top().body.unindent();
        self.body_emitters.top().body.emit("}");
    }

    fn field_literal(&mut self, _name: String, expression: usize, _type_kind: Option<usize>) {
        self.gen_node(expression);
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
                self.emit_type_kind_left(type_kind, EmitterKind::Body, false);
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
        self.emit_type_kind_left(type_kind, emitter_kind, do_arrays_as_pointers);
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
    ) {
        let type_kind = &self.types[type_kind];

        match type_kind.clone() {
            TypeKind::Int => self.emitter(emitter_kind).emit("int32_t"),
            TypeKind::String => self.emitter(emitter_kind).emit("char*"),
            TypeKind::Bool => self.emitter(emitter_kind).emit("int32_t"),
            TypeKind::Struct { name, .. } => {
                self.emitter(emitter_kind).emit("struct ");
                self.emitter(emitter_kind).emit(&name)
            },
            TypeKind::Array {
                element_type_kind, ..
            } => {
                self.emit_type_kind_left(element_type_kind, emitter_kind, do_arrays_as_pointers);
                if do_arrays_as_pointers {
                    self.emitter(emitter_kind).emit("*");
                }
            }
            TypeKind::PartialStruct { .. } => panic!("Can't emit partial struct"),
        };
    }

    fn emit_type_kind_right(
        &mut self,
        type_kind: usize,
        emitter_kind: EmitterKind,
        do_arrays_as_pointers: bool,
    ) {
        let type_kind = self.types[type_kind].clone();

        if let TypeKind::Array {
            element_type_kind,
            element_count,
        } = type_kind
        {
            if !do_arrays_as_pointers {
                self.emitter(emitter_kind).emit("[");
                self.emitter(emitter_kind).emit(&element_count.to_string());
                self.emitter(emitter_kind).emit("]");
            }
            self.emit_type_kind_right(element_type_kind, emitter_kind, do_arrays_as_pointers);
        };
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
