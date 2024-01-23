use std::{collections::HashMap, sync::Arc};

use crate::{
    emitter::Emitter, emitter_stack::EmitterStack, parser::{NodeKind, Op, TrailingTerm, TrailingUnary, TypeKind}, types::is_type_name_array
};

#[derive(Clone, Copy, Debug)]
enum EmitterKind {
    Prototype,
    Body,
    Top,
}

pub struct CodeGenerator {
    pub nodes: Vec<NodeKind>,
    pub types: Vec<TypeKind>,
    pub header_emitter: Emitter,
    pub prototype_emitter: Emitter,
    pub body_emitters: EmitterStack,
    function_declaration_indices: HashMap<String, usize>,
    current_function_return_type_name: Option<usize>,
}

impl CodeGenerator {
    pub fn new(
        nodes: Vec<NodeKind>,
        types: Vec<TypeKind>,
        function_declaration_indices: HashMap<String, usize>,
    ) -> Self {
        let mut code_generator = Self {
            nodes,
            types,
            header_emitter: Emitter::new(0),
            prototype_emitter: Emitter::new(0),
            body_emitters: EmitterStack::new(),
            function_declaration_indices,
            current_function_return_type_name: None,
        };

        code_generator.header_emitter.emitln("#include <string.h>");
        code_generator.body_emitters.push();

        code_generator
    }

    pub fn gen(&mut self, start_index: usize) {
        self.gen_node(start_index);
    }

    fn gen_node(&mut self, index: usize) {
        match self.nodes[index].clone() {
            NodeKind::TopLevel { functions } => self.top_level(functions),
            NodeKind::Function { declaration, block } => self.function(declaration, block),
            NodeKind::FunctionDeclaration {
                name,
                return_type_name,
                params,
            } => self.function_declaration(name, return_type_name, params),
            NodeKind::ExternFunction { declaration } => self.extern_function(declaration),
            NodeKind::Param { name, type_name } => self.param(name, type_name),
            NodeKind::Block { statements } => self.block(statements),
            NodeKind::Statement { inner } => self.statement(inner),
            NodeKind::VariableDeclaration {
                is_mutable,
                is_copy,
                name,
                type_name,
                expression,
            } => self.variable_declaration(is_mutable, is_copy, name, type_name, expression),
            NodeKind::VariableAssignment {
                is_copy,
                variable,
                expression,
            } => self.variable_assignment(is_copy, variable, expression),
            NodeKind::ReturnStatement { expression } => self.return_statement(expression),
            NodeKind::Expression {
                term,
                trailing_terms,
            } => self.expression(term, trailing_terms),
            NodeKind::Term {
                unary,
                trailing_unaries,
            } => self.term(unary, trailing_unaries),
            NodeKind::Unary { op, primary } => self.unary(op, primary),
            NodeKind::Primary { inner } => self.primary(inner),
            NodeKind::Variable { name } => self.variable(name),
            NodeKind::FunctionCall { name, args } => self.function_call(name, args),
            NodeKind::IntLiteral { text } => self.int_literal(text),
            NodeKind::StringLiteral { text } => self.string_literal(text),
            NodeKind::ArrayLiteral { elements } => self.array_literal(elements),
            // NodeKind::TypeName { type_kind } => self.type_name(type_kind),
            NodeKind::TypeName { .. } => panic!("Cannot generate type name with gen_node"),
        }
    }

    fn gen_node_prototype(&mut self, index: usize) {
        match self.nodes[index].clone() {
            NodeKind::FunctionDeclaration {
                name,
                return_type_name,
                params,
            } => self.function_declaration_prototype(name, return_type_name, params),
            NodeKind::Param { name, type_name } => self.param_prototype(name, type_name),
            // NodeKind::TypeName { type_kind } => self.type_name_prototype(type_kind),
            _ => panic!(
                "Node cannot be generated as a prototype: {:?}",
                self.nodes[index]
            ),
        }
    }

    fn top_level(&mut self, functions: Arc<Vec<usize>>) {
        for (i, function) in functions.iter().enumerate() {
            if i > 0 {
                self.body_emitters.top().body.newline();
            }

            self.gen_node(*function);
        }
    }

    fn function_declaration_prototype(
        &mut self,
        name: String,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
    ) {
        // TODO: Need special case for functions that return arrays, since that's not allowed in C.
        // TODO: The return value should be a pointer to the "fake return" arg that was passed in,
        // with an extra argument for returning the array, then returning should memcpy to the destination if it isn't null.
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

        if is_type_name_array(&self.nodes, &self.types, return_type_name) {
            if params.len() > 0 {
                self.prototype_emitter.emit(", ");
            }

            self.param_prototype("__return".into(), return_type_name);
        }
        self.prototype_emitter.emitln(");");
    }

    fn param_prototype(&mut self, name: String, type_name: usize) {
        self.emit_type_name_left(type_name, EmitterKind::Prototype, false);
        self.prototype_emitter.emit(" ");
        self.prototype_emitter.emit(&name);
        self.emit_type_name_right(type_name, EmitterKind::Prototype, false);
    }

    fn function(&mut self, declaration: usize, block: usize) {
        self.gen_node(declaration);
        let NodeKind::FunctionDeclaration { return_type_name, .. } = self.nodes[declaration] else {
            panic!("Invalid function declaration");
        };
        self.current_function_return_type_name = Some(return_type_name);
        self.gen_node(block);
        self.current_function_return_type_name = None;
    }

    fn function_declaration(
        &mut self,
        name: String,
        return_type_name: usize,
        params: Arc<Vec<usize>>,
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

        if is_type_name_array(&self.nodes, &self.types, return_type_name) {
            if params.len() > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.param("__return".into(), return_type_name);
        }
        self.body_emitters.top().body.emit(") ");

        self.function_declaration_prototype(name, return_type_name, params);
    }

    fn extern_function(&mut self, declaration: usize) {
        self.prototype_emitter.emit("extern ");
        self.gen_node_prototype(declaration);
    }

    fn param(&mut self, name: String, type_name: usize) {
        self.emit_type_name_left(type_name, EmitterKind::Body, false);
        self.body_emitters.top().body.emit(" ");
        self.body_emitters.top().body.emit(&name);
        self.emit_type_name_right(type_name, EmitterKind::Body, false);
    }

    fn block(&mut self, statements: Arc<Vec<usize>>) {
        self.body_emitters.top().body.emitln("{");
        self.body_emitters.push();

        for statement in statements.iter() {
            self.gen_node(*statement);
        }

        self.body_emitters.pop();
        self.body_emitters.top().body.emitln("}");
    }

    fn statement(&mut self, inner: usize) {
        self.gen_node(inner);
        self.body_emitters.top().body.emitln(";");
    }

    fn variable_declaration(
        &mut self,
        is_mutable: bool,
        is_copy: bool,
        name: String,
        type_name: usize,
        expression: usize,
    ) {
        if !is_mutable && !is_copy {
            self.body_emitters.top().body.emit("const ");
        }

        if is_copy {
            self.emit_type_name_left(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(" ");
            self.body_emitters.top().body.emit(&name);
            self.emit_type_name_right(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emitln(";");

            self.emit_memmove_expression_to_name(&name, expression);
        } else {
            self.emit_type_name_left(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(" ");
            self.body_emitters.top().body.emit(&name);
            self.emit_type_name_right(type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(" = ");
            self.gen_node(expression);
        }
    }

    fn variable_assignment(&mut self, is_copy: bool, variable: usize, expression: usize) {
        if is_copy {
            self.emit_memmove_expression_to_variable(variable, expression);
        } else {
            self.gen_node(variable);
            self.body_emitters.top().body.emit(" = ");
            self.gen_node(expression);
        }
    }

    fn return_statement(&mut self, expression: usize) {
        let return_type_name = self.current_function_return_type_name.unwrap();

        if is_type_name_array(&self.nodes, &self.types, return_type_name) {
            self.emit_type_name_left(return_type_name, EmitterKind::Body, false);
            // TODO: Make sure temp names are  distinct.
            self.body_emitters.top().body.emit(" __temp");
            self.emit_type_name_right(return_type_name, EmitterKind::Body, false);
            self.body_emitters.top().body.emit(" = ");
            self.gen_node(expression);
            self.body_emitters.top().body.emitln(";");

            self.emit_memmove_name_to_name("__return", "__temp");
            self.body_emitters.top().body.emitln(";");

            self.body_emitters.top().body.emit("return __return");
        } else {
            self.body_emitters.top().body.emit("return ");
            self.gen_node(expression);
        }
    }

    fn expression(&mut self, term: usize, trailing_terms: Arc<Vec<TrailingTerm>>) {
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

    fn term(&mut self, unary: usize, trailing_unaries: Arc<Vec<TrailingUnary>>) {
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

    fn unary(&mut self, op: Option<Op>, primary: usize) {
        match op {
            Some(Op::Add) => self.body_emitters.top().body.emit(" + "),
            Some(Op::Subtract) => self.body_emitters.top().body.emit(" - "),
            None => {}
            _ => panic!("Unexpected operator in unary"),
        }

        self.gen_node(primary);
    }

    fn primary(&mut self, inner: usize) {
        self.gen_node(inner);
    }

    fn variable(&mut self, name: String) {
        self.body_emitters.top().body.emit(&name);
    }

    fn function_call(&mut self, name: String, args: Arc<Vec<usize>>) {
        let NodeKind::FunctionDeclaration {
            return_type_name,
            ..
        } = self.nodes[self.function_declaration_indices[&name]].clone() else {
            panic!("Function call to nonexistant function");
        };

        self.body_emitters.top().body.emit(&name);

        self.body_emitters.top().body.emit("(");
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.gen_node(*arg);
        }

        if is_type_name_array(&self.nodes, &self.types, return_type_name) {
            if args.len() > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.emit_type_name_left(return_type_name, EmitterKind::Top, false);
            // TODO: Make this name different each time it is generated.
            self.body_emitters.top().top.emit(" ");
            self.body_emitters.top().top.emit("__return_array");
            self.emit_type_name_right(return_type_name, EmitterKind::Top, false);
            self.body_emitters.top().top.emitln(";");

            // TODO: Make this name different each time it is generated.
            self.body_emitters.top().body.emit("__return_array");
        }
        self.body_emitters.top().body.emit(")");
    }

    fn int_literal(&mut self, text: String) {
        self.body_emitters.top().body.emit(&text);
    }

    fn string_literal(&mut self, text: String) {
        self.body_emitters.top().body.emit("\"");
        self.body_emitters.top().body.emit(&text);
        self.body_emitters.top().body.emit("\"");
    }

    fn array_literal(&mut self, elements: Arc<Vec<usize>>) {
        self.body_emitters.top().body.emit("{");
        for (i, element) in elements.iter().enumerate() {
            if i > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.gen_node(*element);
        }
        self.body_emitters.top().body.emit("}");
    }

    fn emit_memmove_expression_to_variable(&mut self, destination: usize, source: usize) {
        self.body_emitters.top().body.emit("memmove(");
        self.gen_node(destination);
        self.body_emitters.top().body.emit(", ");
        self.gen_node(source);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit("sizeof(");
        self.gen_node(destination);
        self.body_emitters.top().body.emit("))");
    }

    fn emit_memmove_expression_to_name(&mut self, destination: &str, source: usize) {
        self.body_emitters.top().body.emit("memmove(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit(", ");
        self.gen_node(source);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit("sizeof(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit("))");
    }

    fn emit_memmove_name_to_name(&mut self, destination: &str, source: &str) {
        self.body_emitters.top().body.emit("memmove(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit(source);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit("sizeof(");
        self.body_emitters.top().body.emit(source);
        self.body_emitters.top().body.emit("))");
    }

    // fn emit_memmove(&mut self, destination: &str, source: usize, type_kind: usize) {
    //     self.body_emitters.top().body.emit("memcpy(");
    //     self.body_emitters.top().body.emit(destination);
    //     self.body_emitters.top().body.emit(", ");
    //     self.gen_node(source);
    //     self.body_emitters.top().body.emit(", ");
    //     self.emit_type_size(type_kind);
    //     self.body_emitters.top().body.emit(")");
    // }

    // fn emit_type_size(&mut self, type_kind: usize) {
    //     match self.types[type_kind] {
    //         TypeKind::Array {
    //             element_type_kind, element_count,
    //         } => {
    //             self.emit_type_size(element_type_kind);
    //             self.body_emitters.top().body.emit(" * ");
    //             self.body_emitters.top().body.emit(&element_count.to_string());
    //         },
    //         _ => {
    //             self.body_emitters.top().body.emit("sizeof(");
    //             self.emit_type_kind_left(type_kind, EmitterKind::Body, false);
    //             self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
    //             self.body_emitters.top().body.emit(")");
    //         }
    //     };
    // }

    fn emit_type_name_left(
        &mut self,
        type_name: usize,
        emitter_kind: EmitterKind,
        do_arrays_as_pointers: bool,
    ) {
        let NodeKind::TypeName { type_kind } = self.nodes[type_name] else {
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
        let NodeKind::TypeName { type_kind } = self.nodes[type_name] else {
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

        match type_kind {
            TypeKind::Int => self.emitter(emitter_kind).emit("int"),
            TypeKind::String => self.emitter(emitter_kind).emit("char*"),
            TypeKind::Array {
                element_type_kind, ..
            } => {
                self.emit_type_kind_left(*element_type_kind, emitter_kind, do_arrays_as_pointers);
                if do_arrays_as_pointers {
                    self.emitter(emitter_kind).emit("*");
                }
            }
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
            element_type_kind, element_count
        } = type_kind
        {
            self.emit_type_name_right(element_type_kind, emitter_kind, do_arrays_as_pointers);
            if !do_arrays_as_pointers {
                self.emitter(emitter_kind).emit("[");
                self.emitter(emitter_kind).emit(&element_count.to_string());
                self.emitter(emitter_kind).emit("]");
            }
        };
    }

    fn emitter(&mut self, kind: EmitterKind) -> &mut Emitter {
        match kind {
            EmitterKind::Prototype => &mut self.prototype_emitter,
            EmitterKind::Body => &mut self.body_emitters.top().body,
            EmitterKind::Top => &mut self.body_emitters.top().top,
        }
    }
}
