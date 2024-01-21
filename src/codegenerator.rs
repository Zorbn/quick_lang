use std::sync::Arc;

use crate::{parser::{NodeKind, TrailingTerm, TrailingUnary, Op}, emitter::Emitter};

pub struct CodeGenerator {
    pub nodes: Vec<NodeKind>,
    pub emitter: Emitter,
}

impl CodeGenerator {
    pub fn new(nodes: Vec<NodeKind>) -> Self {
        Self {
            nodes,
            emitter: Emitter::new(),
        }
    }

    pub fn gen(&mut self, start_index: usize) {
        self.gen_node(start_index);
    }

    fn gen_node(&mut self, index: usize) {
        match self.nodes[index].clone() {
            NodeKind::TopLevel { functions } => self.top_level(functions),
            NodeKind::Function { name, params, block } => self.function(name, params, block),
            NodeKind::Param { name } => self.param(name),
            NodeKind::Block { statements } => self.block(statements),
            NodeKind::Statement { inner } => self.statement(inner),
            NodeKind::VariableDeclaration { name, expression } => self.variable_declaration(name, expression),
            NodeKind::Expression { term, trailing_terms } => self.expression(term, trailing_terms),
            NodeKind::Term { unary, trailing_unaries } => self.term(unary, trailing_unaries),
            NodeKind::Unary { op, primary } => self.unary(op, primary),
            NodeKind::Primary { inner } => self.primary(inner),
            NodeKind::Variable { name } => self.variable(name),
            NodeKind::IntLiteral { text } => self.int_literal(text),
        }
    }

    fn top_level(&mut self, functions: Arc<Vec<usize>>) {
        for function in functions.iter() {
            self.gen_node(*function);
        }
    }

    fn function(&mut self, name: String, params: Arc<Vec<usize>>, block: usize) {
        self.emitter.emit("int ");
        self.emitter.emit(&name);

        self.emitter.emit("(");
        for param in params.iter() {
            self.gen_node(*param);
            self.emitter.emit(",");
        }
        self.emitter.emit(") ");

        self.gen_node(block);
    }

    fn param(&mut self, name: String) {
        self.emitter.emit(&name);
    }

    fn block(&mut self, statements: Arc<Vec<usize>>) {
        self.emitter.emitln("{");
        self.emitter.indent();

        for statement in statements.iter() {
            self.gen_node(*statement);
        }

        self.emitter.unindent();
        self.emitter.emitln("}");
    }

    fn statement(&mut self, inner: usize) {
        self.gen_node(inner);
        self.emitter.emitln(";");
    }

    fn variable_declaration(&mut self, name: String, expression: usize) {
        self.emitter.emit("int ");
        self.emitter.emit(&name);
        self.emitter.emit(" = ");
        self.gen_node(expression);
    }

    fn expression(&mut self, term: usize, trailing_terms: Arc<Vec<TrailingTerm>>) {
        self.gen_node(term);

        for trailing_term in trailing_terms.iter() {
            if trailing_term.op == Op::Add {
                self.emitter.emit(" + ");
            } else {
                self.emitter.emit(" - ");
            }

            self.gen_node(trailing_term.term);
        }
    }

    fn term(&mut self, unary: usize, trailing_unaries: Arc<Vec<TrailingUnary>>) {
        self.gen_node(unary);

        for trailing_unary in trailing_unaries.iter() {
            if trailing_unary.op == Op::Multiply {
                self.emitter.emit(" * ");
            } else {
                self.emitter.emit(" / ");
            }

            self.gen_node(trailing_unary.unary);
        }
    }

    fn unary(&mut self, op: Option<Op>, primary: usize) {
        match op {
            Some(Op::Add) => self.emitter.emit(" + "),
            Some(Op::Subtract) => self.emitter.emit(" - "),
            None => {},
            _ => panic!("Unexpected operator in unary"),
        }

        self.gen_node(primary);
    }

    fn primary(&mut self, inner: usize) {
        self.gen_node(inner);
    }

    fn variable(&mut self, name: String) {
        self.emitter.emit(&name);
    }

    fn int_literal(&mut self, text: String) {
        self.emitter.emit(&text);
    }
}