use std::sync::Arc;

use crate::{
    emitter::Emitter,
    emitter_stack::EmitterStack,
    parser::{NodeKind, Op, TrailingTerm, TrailingUnary, TypeKind},
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
    pub fn new(
        typed_nodes: Vec<TypedNode>,
        types: Vec<TypeKind>,
    ) -> Self {
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
        code_generator.body_emitters.push();

        code_generator
    }

    pub fn gen(&mut self, start_index: usize) {
        self.gen_node(start_index);
    }

    fn gen_node(&mut self, index: usize) {
        match self.typed_nodes[index].clone() {
            TypedNode { node_kind: NodeKind::TopLevel { functions }, type_kind } => self.top_level(functions, type_kind),
            TypedNode { node_kind: NodeKind::Function { declaration, block }, type_kind } => self.function(declaration, block, type_kind),
            TypedNode { node_kind: NodeKind::FunctionDeclaration {
                name,
                return_type_name,
                params,
            }, type_kind } => self.function_declaration(name, return_type_name, params, type_kind),
            TypedNode { node_kind: NodeKind::ExternFunction { declaration }, type_kind } => self.extern_function(declaration, type_kind),
            TypedNode { node_kind: NodeKind::Param { name, type_name }, type_kind } => self.param(name, type_name, type_kind),
            TypedNode { node_kind: NodeKind::Block { statements }, type_kind } => self.block(statements, type_kind),
            TypedNode { node_kind: NodeKind::Statement { inner }, type_kind } => self.statement(inner, type_kind),
            TypedNode { node_kind: NodeKind::VariableDeclaration {
                is_mutable,
                name,
                type_name,
                expression,
            }, type_kind } => self.variable_declaration(is_mutable, name, type_name, expression, type_kind),
            TypedNode { node_kind: NodeKind::VariableAssignment {
                variable,
                expression,
            }, type_kind } => self.variable_assignment(variable, expression, type_kind),
            TypedNode { node_kind: NodeKind::ReturnStatement { expression }, type_kind } => self.return_statement(expression, type_kind),
            TypedNode { node_kind: NodeKind::Expression {
                term,
                trailing_terms,
            }, type_kind } => self.expression(term, trailing_terms, type_kind),
            TypedNode { node_kind: NodeKind::Term {
                unary,
                trailing_unaries,
            }, type_kind } => self.term(unary, trailing_unaries, type_kind),
            TypedNode { node_kind: NodeKind::Unary { op, primary }, type_kind } => self.unary(op, primary, type_kind),
            TypedNode { node_kind: NodeKind::Primary { inner }, type_kind } => self.primary(inner, type_kind),
            TypedNode { node_kind: NodeKind::Variable { name }, type_kind } => self.variable(name, type_kind),
            TypedNode { node_kind: NodeKind::FunctionCall { name, args }, type_kind } => self.function_call(name, args, type_kind),
            TypedNode { node_kind: NodeKind::IntLiteral { text }, type_kind } => self.int_literal(text, type_kind),
            TypedNode { node_kind: NodeKind::StringLiteral { text }, type_kind } => self.string_literal(text, type_kind),
            TypedNode { node_kind: NodeKind::ArrayLiteral { elements }, type_kind } => self.array_literal(elements, type_kind),
            TypedNode { node_kind: NodeKind::TypeName { .. }, .. } => panic!("Cannot generate type name with gen_node"),
        }
    }

    fn gen_node_prototype(&mut self, index: usize) {
        match self.typed_nodes[index].clone() {
            TypedNode { node_kind: NodeKind::FunctionDeclaration {
                name,
                return_type_name,
                params,
            }, type_kind } => self.function_declaration_prototype(name, return_type_name, params, type_kind),
            TypedNode { node_kind: NodeKind::Param { name, type_name }, type_kind } => self.param_prototype(name, type_name, type_kind),
            // NodeKind::TypeName { type_kind } => self.type_name_prototype(type_kind),
            _ => panic!(
                "Node cannot be generated as a prototype: {:?}",
                self.typed_nodes[index]
            ),
        }
    }

    fn top_level(&mut self, functions: Arc<Vec<usize>>, _type_kind: Option<usize>) {
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
        let TypedNode { node_kind: NodeKind::FunctionDeclaration { params, .. }, .. } = self.typed_nodes[function_declaration].clone()
        else {
            panic!("Invalid function declaration needing init");
        };

        for param in params.iter() {
            let TypedNode { node_kind: NodeKind::Param { name, type_name }, type_kind } = self.typed_nodes[*param].clone() else {
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
        self.gen_node(inner);
        self.body_emitters.top().body.emitln(";");
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

    fn variable_assignment(&mut self, variable: usize, expression: usize, type_kind: Option<usize>) {
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

    fn expression(&mut self, term: usize, trailing_terms: Arc<Vec<TrailingTerm>>, _type_kind: Option<usize>) {
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

    fn term(&mut self, unary: usize, trailing_unaries: Arc<Vec<TrailingUnary>>, _type_kind: Option<usize>) {
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

    fn variable(&mut self, name: String, _type_kind: Option<usize>) {
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

    fn array_literal(&mut self, elements: Arc<Vec<usize>>, _type_kind: Option<usize>) {
        self.body_emitters.top().body.emit("{");
        for (i, element) in elements.iter().enumerate() {
            if i > 0 {
                self.body_emitters.top().body.emit(", ");
            }

            self.gen_node(*element);
        }
        self.body_emitters.top().body.emit("}");
    }

    fn emit_memmove_expression_to_variable(&mut self, destination: usize, source: usize, type_kind: usize) {
        self.body_emitters.top().body.emit("memmove(");
        self.gen_node(destination);
        self.body_emitters.top().body.emit(", ");
        self.gen_node(source);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit("sizeof(");
        self.emit_type_size(type_kind);
        self.body_emitters.top().body.emit("))");
    }

    fn emit_memmove_expression_to_name(&mut self, destination: &str, source: usize, type_kind: usize) {
        self.body_emitters.top().body.emit("memmove(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit(", ");
        self.gen_node(source);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit("sizeof(");
        self.emit_type_size(type_kind);
        self.body_emitters.top().body.emit("))");
    }

    fn emit_memmove_name_to_name(
        &mut self,
        destination: &str,
        source: &str,
        type_kind: usize,
    ) {
        self.body_emitters.top().body.emit("memmove(");
        self.body_emitters.top().body.emit(destination);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit(source);
        self.body_emitters.top().body.emit(", ");
        self.body_emitters.top().body.emit("sizeof(");
        self.emit_type_size(type_kind);
        self.body_emitters.top().body.emit("))");
    }

    fn emit_type_size(&mut self, type_kind: usize) {
        match self.types[type_kind] {
            TypeKind::Array {
                element_type_kind, element_count,
            } => {
                self.emit_type_size(element_type_kind);
                self.body_emitters.top().body.emit(" * ");
                self.body_emitters.top().body.emit(&element_count.to_string());
            },
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
        let TypedNode { node_kind: NodeKind::TypeName { type_kind }, .. } = self.typed_nodes[type_name] else {
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
        let TypedNode { node_kind: NodeKind::TypeName { type_kind }, .. } = self.typed_nodes[type_name] else {
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
            element_type_kind,
            element_count,
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

    fn temp_variable_name(&mut self, prefix: &str) -> String {
        let temp_variable_index = self.temp_variable_count;
        self.temp_variable_count += 1;

        format!("__{}{}", prefix, temp_variable_index)
    }
}
