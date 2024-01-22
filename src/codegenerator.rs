use std::sync::Arc;

use crate::{parser::{NodeKind, TrailingTerm, TrailingUnary, Op, TypeKind}, emitter::Emitter};

pub struct CodeGenerator {
    pub nodes: Vec<NodeKind>,
    pub types: Vec<TypeKind>,
    pub header_emitter: Emitter,
    pub prototype_emitter: Emitter,
    pub body_emitter: Emitter,
}

impl CodeGenerator {
    pub fn new(nodes: Vec<NodeKind>, types: Vec<TypeKind>) -> Self {
        Self {
            nodes,
            types,
            header_emitter: Emitter::new(),
            prototype_emitter: Emitter::new(),
            body_emitter: Emitter::new(),
        }
    }

    pub fn gen(&mut self, start_index: usize) {
        self.gen_node(start_index);
    }

    fn gen_node(&mut self, index: usize) {
        match self.nodes[index].clone() {
            NodeKind::TopLevel { functions } => self.top_level(functions),
            NodeKind::Function { name, return_type_name, params, block } => self.function(name, return_type_name, params, block),
            NodeKind::Param { name, type_name } => self.param(name, type_name),
            NodeKind::Block { statements } => self.block(statements),
            NodeKind::Statement { inner } => self.statement(inner),
            NodeKind::VariableDeclaration { is_mutable, name, type_name, expression } => self.variable_declaration(is_mutable, name, type_name, expression),
            NodeKind::VariableAssignment { variable, expression } => self.variable_assignment(variable, expression),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression),
            NodeKind::Expression { term, trailing_terms } => self.expression(term, trailing_terms),
            NodeKind::Term { unary, trailing_unaries } => self.term(unary, trailing_unaries),
            NodeKind::Unary { op, primary } => self.unary(op, primary),
            NodeKind::Primary { inner } => self.primary(inner),
            NodeKind::Variable { name } => self.variable(name),
            NodeKind::FunctionCall { name, args } => self.function_call(name, args),
            NodeKind::IntLiteral { text } => self.int_literal(text),
            NodeKind::StringLiteral { text } => self.string_literal(text),
            NodeKind::TypeName { type_kind } => self.type_name(type_kind),
        }
    }

    fn gen_node_prototype(&mut self, index: usize) {
        match self.nodes[index].clone() {
            NodeKind::Function { name, return_type_name, params, block } => self.function_prototype(name, return_type_name, params, block),
            NodeKind::Param { name, type_name } => self.param_prototype(name, type_name),
            NodeKind::TypeName { type_kind } => self.type_name_prototype(type_kind),
            _ => panic!("Node cannot be generated as a prototype: {:?}", self.nodes[index]),
        }
    }

    fn top_level(&mut self, functions: Arc<Vec<usize>>) {
        for (i, function) in functions.iter().enumerate() {
            if i > 0 {
                self.body_emitter.newline();
            }

            self.gen_node(*function);
        }
    }

    fn function_prototype(&mut self, name: String, return_type_name: usize, params: Arc<Vec<usize>>, _block: usize) {
        self.gen_node_prototype(return_type_name);
        self.prototype_emitter.emit(" ");
        self.prototype_emitter.emit(&name);

        self.prototype_emitter.emit("(");
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                self.prototype_emitter.emit(", ");
            }

            self.gen_node_prototype(*param);
        }
        self.prototype_emitter.emitln(");");
    }

    fn param_prototype(&mut self, name: String, type_name: usize) {
        self.gen_node_prototype(type_name);
        self.prototype_emitter.emit(" ");
        self.prototype_emitter.emit(&name);
    }

    fn function(&mut self, name: String, return_type_name: usize, params: Arc<Vec<usize>>, block: usize) {
        self.gen_node(return_type_name);
        self.body_emitter.emit(" ");
        self.body_emitter.emit(&name);

        self.body_emitter.emit("(");
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                self.body_emitter.emit(", ");
            }

            self.gen_node(*param);
        }
        self.body_emitter.emit(") ");

        self.gen_node(block);

        self.function_prototype(name, return_type_name, params, block);
    }

    fn param(&mut self, name: String, type_name: usize) {
        self.gen_node(type_name);
        self.body_emitter.emit(" ");
        self.body_emitter.emit(&name);
    }

    fn block(&mut self, statements: Arc<Vec<usize>>) {
        self.body_emitter.emitln("{");
        self.body_emitter.indent();

        for statement in statements.iter() {
            self.gen_node(*statement);
        }

        self.body_emitter.unindent();
        self.body_emitter.emitln("}");
    }

    fn statement(&mut self, inner: usize) {
        self.gen_node(inner);
        self.body_emitter.emitln(";");
    }

    fn variable_declaration(&mut self, is_mutable: bool, name: String, type_name: usize, expression: usize) {
        if !is_mutable {
            self.body_emitter.emit("const ");
        }

        self.gen_node(type_name);
        self.body_emitter.emit(" ");
        self.body_emitter.emit(&name);
        self.body_emitter.emit(" = ");
        self.gen_node(expression);
    }

    fn variable_assignment(&mut self, variable: usize, expression: usize) {
        self.gen_node(variable);
        self.body_emitter.emit(" = ");
        self.gen_node(expression);
    }

    fn return_statement(&mut self, expression: usize) {
        self.body_emitter.emit("return ");
        self.gen_node(expression);
    }

    fn expression(&mut self, term: usize, trailing_terms: Arc<Vec<TrailingTerm>>) {
        self.gen_node(term);

        for trailing_term in trailing_terms.iter() {
            if trailing_term.op == Op::Add {
                self.body_emitter.emit(" + ");
            } else {
                self.body_emitter.emit(" - ");
            }

            self.gen_node(trailing_term.term);
        }
    }

    fn term(&mut self, unary: usize, trailing_unaries: Arc<Vec<TrailingUnary>>) {
        self.gen_node(unary);

        for trailing_unary in trailing_unaries.iter() {
            if trailing_unary.op == Op::Multiply {
                self.body_emitter.emit(" * ");
            } else {
                self.body_emitter.emit(" / ");
            }

            self.gen_node(trailing_unary.unary);
        }
    }

    fn unary(&mut self, op: Option<Op>, primary: usize) {
        match op {
            Some(Op::Add) => self.body_emitter.emit(" + "),
            Some(Op::Subtract) => self.body_emitter.emit(" - "),
            None => {},
            _ => panic!("Unexpected operator in unary"),
        }

        self.gen_node(primary);
    }

    fn primary(&mut self, inner: usize) {
        self.gen_node(inner);
    }

    fn variable(&mut self, name: String) {
        self.body_emitter.emit(&name);
    }

    fn function_call(&mut self, name: String, args: Arc<Vec<usize>>) {
        self.body_emitter.emit(&name);

        self.body_emitter.emit("(");
        for (i, param) in args.iter().enumerate() {
            if i > 0 {
                self.body_emitter.emit(", ");
            }

            self.gen_node(*param);
        }
        self.body_emitter.emit(")");
    }

    fn int_literal(&mut self, text: String) {
        self.body_emitter.emit(&text);
    }

    fn string_literal(&mut self, text: String) {
        self.body_emitter.emit("\"");
        self.body_emitter.emit(&text);
        self.body_emitter.emit("\"");
    }

    fn type_name(&mut self, type_name: usize) {
        CodeGenerator::emit_type_name(&self.types[type_name], &mut self.body_emitter);
    }

    fn type_name_prototype(&mut self, type_name: usize) {
        CodeGenerator::emit_type_name(&self.types[type_name], &mut self.prototype_emitter);
    }

    fn emit_type_name(type_kind: &TypeKind, emitter: &mut Emitter) {
        match type_kind {
            TypeKind::Int => emitter.emit("int"),
            TypeKind::String => emitter.emit("char*"),
        };
    }
}