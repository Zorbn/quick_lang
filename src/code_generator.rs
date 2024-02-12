// use std::{
//     collections::{HashMap, HashSet},
//     sync::{Arc, OnceLock},
// };
//
// use crate::{
//     const_value::ConstValue,
//     emitter::Emitter,
//     emitter_stack::EmitterStack,
//     parser::{DeclarationKind, NodeKind, Op, TypeKind},
//     type_checker::{InstanceKind, Type, TypedNode},
//     types::{
//         get_field_index_by_name, get_method_subject, is_type_kind_array, is_typed_expression_array_literal, replace_generic_type_kinds
//     },
// };
//
// #[derive(Clone, Copy, Debug)]
// enum EmitterKind {
//     TypePrototype,
//     FunctionPrototype,
//     Top,
//     Body,
// }
//
// fn reserved_names() -> &'static HashSet<Arc<str>> {
//     static NAMES: OnceLock<HashSet<Arc<str>>> = OnceLock::new();
//     NAMES.get_or_init(|| {
//         [
//             "alignas",
//             "alignof",
//             "auto",
//             "bool",
//             "break",
//             "case",
//             "char",
//             "const",
//             "constexpr",
//             "continue",
//             "default",
//             "do",
//             "double",
//             "else",
//             "enum",
//             "extern",
//             "false",
//             "float",
//             "for",
//             "goto",
//             "if",
//             "inline",
//             "int",
//             "long",
//             "nullptr",
//             "register",
//             "restrict",
//             "return",
//             "short",
//             "signed",
//             "sizeof",
//             "static",
//             "static_assert",
//             "struct",
//             "switch",
//             "thread_local",
//             "true",
//             "typedef",
//             "typeof",
//             "typeof_unqual",
//             "union",
//             "unsigned",
//             "void",
//             "volatile",
//             "while",
//             "_Alignas",
//             "_Alignof",
//             "_Atomic",
//             "_BitInt",
//             "_Bool",
//             "_Complex",
//             "_Decimal128",
//             "_Decimal32",
//             "_Decimal64",
//             "_Generic",
//             "_Imaginary",
//             "_Noreturn",
//             "_Static_assert",
//             "_Thread_local",
//         ]
//         .iter()
//         .map(|s| Arc::from(*s))
//         .collect()
//     })
// }
//
// #[derive(Clone)]
// struct NamespaceName {
//     name: Arc<str>,
//     generic_param_type_kinds: Option<Arc<Vec<usize>>>,
// }
//
// pub struct CodeGenerator {
//     pub typed_nodes: Vec<TypedNode>,
//     pub type_kinds: Vec<TypeKind>,
//     pub generic_usages: HashMap<usize, HashSet<Arc<Vec<usize>>>>,
//     pub header_emitter: Emitter,
//     pub type_prototype_emitter: Emitter,
//     pub function_prototype_emitter: Emitter,
//     pub body_emitters: EmitterStack,
//     function_declaration_needing_init: Option<usize>,
//     temp_variable_count: usize,
//     current_namespace_names: Vec<NamespaceName>,
//     is_debug_mode: bool,
// }
//
// impl CodeGenerator {
//     pub fn new(
//         typed_nodes: Vec<TypedNode>,
//         type_kinds: Vec<TypeKind>,
//         generic_usages: HashMap<usize, HashSet<Arc<Vec<usize>>>>,
//         is_debug_mode: bool,
//     ) -> Self {
//         let mut code_generator = Self {
//             typed_nodes,
//             type_kinds,
//             generic_usages,
//             header_emitter: Emitter::new(0),
//             type_prototype_emitter: Emitter::new(0),
//             function_prototype_emitter: Emitter::new(0),
//             body_emitters: EmitterStack::new(),
//             function_declaration_needing_init: None,
//             temp_variable_count: 0,
//             current_namespace_names: Vec::new(),
//             is_debug_mode,
//         };
//
//         code_generator.header_emitter.emitln("#include <string.h>");
//         code_generator
//             .header_emitter
//             .emitln("#include <inttypes.h>");
//         code_generator.header_emitter.emitln("#include <stdbool.h>");
//         code_generator.header_emitter.emitln("#include <assert.h>");
//         code_generator.header_emitter.newline();
//         code_generator.body_emitters.push(1);
//
//         if is_debug_mode {
//             code_generator.emit_bounds_check();
//         }
//
//         code_generator
//     }
//
//     pub fn gen(&mut self, start_index: usize) {
//         self.gen_node(start_index);
//     }
//
//     fn gen_node(&mut self, index: usize) {
//         match self.typed_nodes[index].clone() {
//             TypedNode {
//                 node_kind:
//                     NodeKind::TopLevel {
//                         functions,
//                         structs,
//                         enums,
//                     },
//                 node_type,
//             } => self.top_level(functions, structs, enums, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::StructDefinition {
//                         name,
//                         fields,
//                         functions,
//                         ..
//                     },
//                 node_type,
//             } => self.struct_definition(name, fields, functions, index, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::EnumDefinition {
//                         name,
//                         variant_names,
//                         ..
//                     },
//                 node_type,
//             } => self.enum_definition(name, variant_names, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Field { name, type_name },
//                 node_type,
//             } => self.field(name, type_name, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::Function {
//                         declaration,
//                         statement,
//                     },
//                 node_type,
//             } => self.function(declaration, statement, index, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::FunctionDeclaration {
//                         name,
//                         params,
//                         return_type_name,
//                         ..
//                     },
//                 node_type,
//             } => self.function_declaration(name, params, None, return_type_name, node_type),
//             TypedNode {
//                 node_kind: NodeKind::ExternFunction { declaration },
//                 node_type,
//             } => self.extern_function(declaration, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Param { name, type_name },
//                 node_type,
//             } => self.param(name, type_name, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Block { statements },
//                 node_type,
//             } => self.block(statements, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Statement { inner },
//                 node_type,
//             } => self.statement(inner, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::VariableDeclaration {
//                         declaration_kind,
//                         name,
//                         type_name,
//                         expression,
//                     },
//                 node_type,
//             } => {
//                 self.variable_declaration(declaration_kind, name, type_name, expression, node_type)
//             }
//             TypedNode {
//                 node_kind: NodeKind::ReturnStatement { expression },
//                 node_type,
//             } => self.return_statement(expression, node_type),
//             TypedNode {
//                 node_kind: NodeKind::DeferStatement { statement },
//                 node_type,
//             } => self.defer_statement(statement, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::IfStatement {
//                         expression,
//                         statement,
//                         next,
//                     },
//                 node_type,
//             } => self.if_statement(expression, statement, next, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::SwitchStatement {
//                         expression,
//                         case_statement,
//                     },
//                 node_type,
//             } => self.switch_statement(expression, case_statement, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::CaseStatement {
//                         expression,
//                         statement,
//                         next,
//                     },
//                 node_type,
//             } => self.case_statement(expression, statement, next, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::WhileLoop {
//                         expression,
//                         statement,
//                     },
//                 node_type,
//             } => self.while_loop(expression, statement, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::ForLoop {
//                         iterator,
//                         op,
//                         from,
//                         to,
//                         by,
//                         statement,
//                     },
//                 node_type,
//             } => self.for_loop(iterator, op, from, to, by, statement, node_type),
//             TypedNode {
//                 node_kind: NodeKind::ConstExpression { inner },
//                 node_type,
//             } => self.const_expression(inner, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Binary { left, op, right },
//                 node_type,
//             } => self.binary(left, op, right, node_type),
//             TypedNode {
//                 node_kind: NodeKind::UnaryPrefix { op, right },
//                 node_type,
//             } => self.unary_prefix(op, right, node_type),
//             TypedNode {
//                 node_kind: NodeKind::UnarySuffix { left, op },
//                 node_type,
//             } => self.unary_suffix(left, op, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Call { left, args },
//                 node_type,
//             } => self.call(left, args, node_type),
//             TypedNode {
//                 node_kind: NodeKind::IndexAccess { left, expression },
//                 node_type,
//             } => self.index_access(left, expression, node_type),
//             TypedNode {
//                 node_kind: NodeKind::FieldAccess { left, name },
//                 node_type,
//             } => self.field_access(left, name, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Cast { left, type_name },
//                 node_type,
//             } => self.cast(left, type_name, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::GenericSpecifier {
//                         left,
//                         generic_param_type_kinds,
//                     },
//                 node_type,
//             } => self.generic_specifier(left, generic_param_type_kinds, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Name { text },
//                 node_type,
//             } => self.name(text, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Identifier { name },
//                 node_type,
//             } => self.identifier(name, node_type),
//             TypedNode {
//                 node_kind: NodeKind::IntLiteral { text },
//                 node_type,
//             } => self.int_literal(text, node_type),
//             TypedNode {
//                 node_kind: NodeKind::Float32Literal { text },
//                 node_type,
//             } => self.float32_literal(text, node_type),
//             TypedNode {
//                 node_kind: NodeKind::CharLiteral { value },
//                 node_type,
//             } => self.char_literal(value, node_type),
//             TypedNode {
//                 node_kind: NodeKind::StringLiteral { text },
//                 node_type,
//             } => self.string_literal(text, node_type),
//             TypedNode {
//                 node_kind: NodeKind::BoolLiteral { value },
//                 node_type,
//             } => self.bool_literal(value, node_type),
//             TypedNode {
//                 node_kind:
//                     NodeKind::ArrayLiteral {
//                         elements,
//                         repeat_count_const_expression,
//                     },
//                 node_type,
//             } => self.array_literal(elements, repeat_count_const_expression, node_type),
//             TypedNode {
//                 node_kind: NodeKind::StructLiteral { left, fields },
//                 node_type,
//             } => self.struct_literal(left, fields, node_type),
//             TypedNode {
//                 node_kind: NodeKind::FieldLiteral { name, expression },
//                 node_type,
//             } => self.field_literal(name, expression, node_type),
//             TypedNode {
//                 node_kind: NodeKind::TypeSize { type_name },
//                 node_type,
//             } => self.type_size(type_name, node_type),
//             TypedNode {
//                 node_kind: NodeKind::TypeName { .. },
//                 ..
//             } => panic!("cannot generate type name with gen_node"),
//             TypedNode {
//                 node_kind: NodeKind::Error,
//                 ..
//             } => panic!("cannot generate error node"),
//         }
//     }
//
//     fn top_level(
//         &mut self,
//         functions: Arc<Vec<usize>>,
//         structs: Arc<Vec<usize>>,
//         enums: Arc<Vec<usize>>,
//         _node_type: Option<Type>,
//     ) {
//         for struct_definition in structs.iter() {
//             self.gen_node(*struct_definition);
//         }
//
//         for enum_definition in enums.iter() {
//             self.gen_node(*enum_definition);
//         }
//
//         for function in functions.iter() {
//             self.gen_node(*function);
//         }
//     }
//
//     fn emit_struct_definition(
//         &mut self,
//         name: usize,
//         fields: &Arc<Vec<usize>>,
//         functions: &Arc<Vec<usize>>,
//         generic_usage: Option<Arc<Vec<usize>>>,
//         struct_type: Type,
//     ) {
//         let TypeKind::Struct { is_union, .. } = self.type_kinds[struct_type.type_kind].clone()
//         else {
//             panic!("invalid struct");
//         };
//
//         let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone() else {
//             panic!("invalid name in struct");
//         };
//
//         self.current_namespace_names.push(NamespaceName {
//             name: name_text.clone(),
//             generic_param_type_kinds: generic_usage.clone(),
//         });
//
//         for function in functions.iter() {
//             self.gen_node(*function);
//         }
//
//         if is_union {
//             // TODO: Refactor, emit_union_check_tag
//             self.emit_type_kind_left(
//                 struct_type.type_kind,
//                 EmitterKind::FunctionPrototype,
//                 true,
//                 false,
//             );
//             self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::FunctionPrototype);
//             self.emit_type_kind_right(struct_type.type_kind, EmitterKind::FunctionPrototype, true);
//             self.function_prototype_emitter.emit("* ");
//             self.emit_namespace_prefix(EmitterKind::FunctionPrototype);
//             self.function_prototype_emitter.emit("__CheckTag(");
//             self.emit_type_kind_left(
//                 struct_type.type_kind,
//                 EmitterKind::FunctionPrototype,
//                 false,
//                 false,
//             );
//             self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::FunctionPrototype);
//             self.function_prototype_emitter.emit(" *self");
//             self.emit_type_kind_right(struct_type.type_kind, EmitterKind::FunctionPrototype, false);
//             self.function_prototype_emitter.emitln(", intptr_t tag);");
//             self.function_prototype_emitter.newline();
//
//             self.emit_type_kind_left(struct_type.type_kind, EmitterKind::Body, true, false);
//             self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::Body);
//             self.emit_type_kind_right(struct_type.type_kind, EmitterKind::Body, true);
//             self.body_emitters.top().body.emit("* ");
//             self.emit_namespace_prefix(EmitterKind::Body);
//             self.body_emitters.top().body.emit("__CheckTag(");
//             self.emit_type_kind_left(struct_type.type_kind, EmitterKind::Body, false, false);
//             self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::Body);
//             self.body_emitters.top().body.emit(" *self");
//             self.emit_type_kind_right(struct_type.type_kind, EmitterKind::Body, false);
//             self.body_emitters.top().body.emitln(", intptr_t tag) {");
//             self.body_emitters.top().body.indent();
//             self.body_emitters
//                 .top()
//                 .body
//                 .emitln("assert(self->tag == tag);");
//             self.body_emitters.top().body.emitln("return self;");
//             self.body_emitters.top().body.unindent();
//             self.body_emitters.top().body.emitln("}");
//             self.body_emitters.top().body.newline();
//
//             // TODO: Refactor, emit_union_set_tag
//             self.emit_type_kind_left(
//                 struct_type.type_kind,
//                 EmitterKind::FunctionPrototype,
//                 true,
//                 false,
//             );
//             self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::FunctionPrototype);
//             self.emit_type_kind_right(struct_type.type_kind, EmitterKind::FunctionPrototype, true);
//             self.function_prototype_emitter.emit("* ");
//             self.emit_namespace_prefix(EmitterKind::FunctionPrototype);
//             self.function_prototype_emitter.emit("__WithTag(");
//             self.emit_type_kind_left(
//                 struct_type.type_kind,
//                 EmitterKind::FunctionPrototype,
//                 false,
//                 false,
//             );
//             self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::FunctionPrototype);
//             self.function_prototype_emitter.emit(" *self");
//             self.emit_type_kind_right(struct_type.type_kind, EmitterKind::FunctionPrototype, false);
//             self.function_prototype_emitter.emitln(", intptr_t tag);");
//             self.function_prototype_emitter.newline();
//
//             self.emit_type_kind_left(struct_type.type_kind, EmitterKind::Body, true, false);
//             self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::Body);
//             self.emit_type_kind_right(struct_type.type_kind, EmitterKind::Body, true);
//             self.body_emitters.top().body.emit("* ");
//             self.emit_namespace_prefix(EmitterKind::Body);
//             self.body_emitters.top().body.emit("__WithTag(");
//             self.emit_type_kind_left(struct_type.type_kind, EmitterKind::Body, false, false);
//             self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::Body);
//             self.body_emitters.top().body.emit(" *self");
//             self.emit_type_kind_right(struct_type.type_kind, EmitterKind::Body, false);
//             self.body_emitters.top().body.emitln(", intptr_t tag) {");
//             self.body_emitters.top().body.indent();
//             self.body_emitters.top().body.emitln("self->tag = tag;");
//             self.body_emitters.top().body.emitln("return self;");
//             self.body_emitters.top().body.unindent();
//             self.body_emitters.top().body.emitln("}");
//             self.body_emitters.top().body.newline();
//         }
//
//         self.current_namespace_names.pop();
//
//         self.type_prototype_emitter.emit("struct ");
//         self.emit_name_node(name, EmitterKind::TypePrototype);
//         self.emit_generic_param_suffix(generic_usage.as_ref(), EmitterKind::TypePrototype);
//
//         self.type_prototype_emitter.emit(" ");
//
//         if is_union {
//             self.type_prototype_emitter.emitln("{");
//             self.type_prototype_emitter.indent();
//             self.type_prototype_emitter.emitln("intptr_t tag;");
//             self.type_prototype_emitter.emit("union ");
//         }
//
//         self.type_prototype_emitter.emitln("{");
//         self.type_prototype_emitter.indent();
//
//         if fields.is_empty() {
//             // C doesn't allow empty structs.
//             self.type_prototype_emitter.emitln("bool placeholder;");
//         }
//
//         for field in fields.iter() {
//             self.gen_node(*field);
//         }
//
//         self.type_prototype_emitter.unindent();
//         self.type_prototype_emitter.emit("}");
//
//         if is_union {
//             self.type_prototype_emitter.emitln(" variant;");
//             self.type_prototype_emitter.unindent();
//             self.type_prototype_emitter.emit("}");
//         }
//
//         self.type_prototype_emitter.emitln(";");
//         self.type_prototype_emitter.newline();
//     }
//
//     fn struct_definition(
//         &mut self,
//         name: usize,
//         fields: Arc<Vec<usize>>,
//         functions: Arc<Vec<usize>>,
//         index: usize,
//         node_type: Option<Type>,
//     ) {
//         let Some(node_type) = node_type else {
//             return;
//         };
//
//         let TypeKind::Struct {
//             generic_type_kinds, ..
//         } = self.type_kinds[node_type.type_kind].clone()
//         else {
//             panic!("invalid function type");
//         };
//
//         if let Some(generic_usages) = self.generic_usages.get(&index) {
//             let generic_usages: Vec<Arc<Vec<usize>>> = generic_usages.iter().cloned().collect();
//
//             for generic_usage in generic_usages {
//                 // Replace generic types with their concrete types for this usage.
//                 replace_generic_type_kinds(
//                     &mut self.type_kinds,
//                     &generic_type_kinds,
//                     &generic_usage,
//                 );
//                 self.emit_struct_definition(
//                     name,
//                     &fields,
//                     &functions,
//                     Some(generic_usage),
//                     node_type.clone(),
//                 );
//             }
//         } else if generic_type_kinds.is_empty() {
//             self.emit_struct_definition(name, &fields, &functions, None, node_type);
//         }
//     }
//
//     fn enum_definition(
//         &mut self,
//         name: usize,
//         variant_names: Arc<Vec<usize>>,
//         _node_type: Option<Type>,
//     ) {
//         self.type_prototype_emitter.emit("enum ");
//         self.emit_name_node(name, EmitterKind::TypePrototype);
//         self.type_prototype_emitter.emitln(" {");
//         self.type_prototype_emitter.indent();
//
//         let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone() else {
//             panic!("invalid name in enum");
//         };
//
//         self.current_namespace_names.push(NamespaceName {
//             name: name_text,
//             generic_param_type_kinds: None,
//         });
//
//         for variant_name in variant_names.iter() {
//             self.emit_namespace_prefix(EmitterKind::TypePrototype);
//             self.emit_name_node(*variant_name, EmitterKind::TypePrototype);
//             self.type_prototype_emitter.emitln(",");
//         }
//
//         self.current_namespace_names.pop();
//
//         self.type_prototype_emitter.unindent();
//         self.type_prototype_emitter.emitln("};");
//         self.type_prototype_emitter.newline();
//     }
//
//     fn field(&mut self, name: usize, _type_name: usize, node_type: Option<Type>) {
//         self.emit_type_kind_left(
//             node_type.clone().unwrap().type_kind,
//             EmitterKind::TypePrototype,
//             false,
//             true,
//         );
//         self.emit_name_node(name, EmitterKind::TypePrototype);
//         self.emit_type_kind_right(
//             node_type.unwrap().type_kind,
//             EmitterKind::TypePrototype,
//             false,
//         );
//         self.type_prototype_emitter.emitln(";");
//     }
//
//     fn function(
//         &mut self,
//         declaration: usize,
//         statement: usize,
//         index: usize,
//         node_type: Option<Type>,
//     ) {
//         let Some(node_type) = node_type else {
//             return;
//         };
//
//         let NodeKind::FunctionDeclaration {
//             name,
//             params,
//             return_type_name,
//             ..
//         } = self.typed_nodes[declaration].node_kind.clone()
//         else {
//             panic!("invalid function declaration");
//         };
//
//         let TypeKind::Function {
//             generic_type_kinds, ..
//         } = self.type_kinds[node_type.type_kind].clone()
//         else {
//             panic!("invalid function type");
//         };
//
//         if let Some(generic_usages) = self.generic_usages.get(&index) {
//             let generic_usages: Vec<Arc<Vec<usize>>> = generic_usages.iter().cloned().collect();
//
//             for generic_usage in generic_usages {
//                 // Replace generic types with their concrete types for this usage.
//                 replace_generic_type_kinds(
//                     &mut self.type_kinds,
//                     &generic_type_kinds,
//                     &generic_usage,
//                 );
//
//                 self.function_declaration(
//                     name,
//                     params.clone(),
//                     Some(generic_usage),
//                     return_type_name,
//                     Some(node_type.clone()),
//                 );
//                 self.function_declaration_needing_init = Some(declaration);
//                 self.emit_scoped_statement(statement);
//                 self.body_emitters.top().body.newline();
//             }
//         } else if generic_type_kinds.is_empty() {
//             self.function_declaration(name, params, None, return_type_name, Some(node_type));
//             self.function_declaration_needing_init = Some(declaration);
//             self.emit_scoped_statement(statement);
//             self.body_emitters.top().body.newline();
//         }
//     }
//
//     fn function_declaration(
//         &mut self,
//         name: usize,
//         params: Arc<Vec<usize>>,
//         generic_usage: Option<Arc<Vec<usize>>>,
//         return_type_name: usize,
//         node_type: Option<Type>,
//     ) {
//         self.emit_function_declaration(
//             EmitterKind::Body,
//             name,
//             &params,
//             generic_usage.clone(),
//             return_type_name,
//             node_type.clone().unwrap().type_kind,
//         );
//         self.body_emitters.top().body.emit(" ");
//
//         self.emit_function_declaration(
//             EmitterKind::FunctionPrototype,
//             name,
//             &params,
//             generic_usage,
//             return_type_name,
//             node_type.unwrap().type_kind,
//         );
//         self.function_prototype_emitter.emitln(";");
//         self.function_prototype_emitter.newline();
//     }
//
//     fn extern_function(&mut self, declaration: usize, _node_type: Option<Type>) {
//         self.function_prototype_emitter.emit("extern ");
//         let NodeKind::FunctionDeclaration {
//             name,
//             params,
//             return_type_name,
//             type_kind,
//             ..
//         } = self.typed_nodes[declaration].node_kind.clone()
//         else {
//             panic!("invalid function declaration");
//         };
//         self.emit_function_declaration(
//             EmitterKind::FunctionPrototype,
//             name,
//             &params,
//             None,
//             return_type_name,
//             type_kind,
//         );
//         self.function_prototype_emitter.emitln(";");
//         self.function_prototype_emitter.newline();
//     }
//
//     fn param(&mut self, name: usize, type_name: usize, _node_type: Option<Type>) {
//         self.emit_param(name, type_name, EmitterKind::Body);
//     }
//
//     fn copy_array_params(&mut self, function_declaration: usize) {
//         let TypedNode {
//             node_kind: NodeKind::FunctionDeclaration { params, .. },
//             ..
//         } = self.typed_nodes[function_declaration].clone()
//         else {
//             panic!("invalid function declaration needing init");
//         };
//
//         for param in params.iter() {
//             let TypedNode {
//                 node_kind: NodeKind::Param { name, type_name },
//                 node_type,
//             } = self.typed_nodes[*param].clone()
//             else {
//                 panic!("invalid param in function declaration needing init");
//             };
//
//             let type_kind = node_type.unwrap().type_kind;
//
//             if !is_type_kind_array(&self.type_kinds, type_kind) {
//                 continue;
//             }
//
//             let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
//             else {
//                 panic!("invalid parameter name");
//             };
//             let copy_name = format!("__{}", &name_text);
//
//             self.emit_type_name_left(type_name, EmitterKind::Body, false, true);
//             self.body_emitters.top().body.emit(&copy_name);
//             self.emit_type_name_right(type_name, EmitterKind::Body, false);
//             self.body_emitters.top().body.emitln(";");
//
//             self.emit_memmove_name_to_name(&copy_name, &name_text, type_kind);
//             self.body_emitters.top().body.emitln(";");
//
//             self.gen_node(name);
//             self.body_emitters.top().body.emit(" = ");
//             self.body_emitters.top().body.emit(&copy_name);
//             self.body_emitters.top().body.emitln(";");
//         }
//     }
//
//     fn block(&mut self, statements: Arc<Vec<usize>>, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emitln("{");
//         self.body_emitters.push(1);
//
//         // Make copies of any parameters that are arrays, because arrays are supposed to be passed by value.
//         if let Some(function_declaration) = self.function_declaration_needing_init {
//             self.copy_array_params(function_declaration);
//             self.function_declaration_needing_init = None;
//         }
//
//         for statement in statements.iter() {
//             self.gen_node(*statement);
//         }
//
//         let mut was_last_statement_return = false;
//         for statement in statements.iter().rev() {
//             let TypedNode {
//                 node_kind: NodeKind::Statement { inner },
//                 ..
//             } = self.typed_nodes[*statement]
//             else {
//                 panic!("last statement is not a statement");
//             };
//
//             let Some(inner) = inner else {
//                 continue;
//             };
//
//             was_last_statement_return = matches!(
//                 self.typed_nodes[inner],
//                 TypedNode {
//                     node_kind: NodeKind::ReturnStatement { .. },
//                     ..
//                 }
//             );
//
//             break;
//         }
//
//         self.body_emitters.pop(!was_last_statement_return);
//         self.body_emitters.top().body.emit("}");
//     }
//
//     fn statement(&mut self, inner: Option<usize>, _node_type: Option<Type>) {
//         let Some(inner) = inner else {
//             self.body_emitters.top().body.emitln(";");
//             return;
//         };
//
//         let needs_semicolon = !matches!(
//             self.typed_nodes[inner],
//             TypedNode {
//                 node_kind: NodeKind::DeferStatement { .. },
//                 ..
//             } | TypedNode {
//                 node_kind: NodeKind::IfStatement { .. },
//                 ..
//             } | TypedNode {
//                 node_kind: NodeKind::SwitchStatement { .. },
//                 ..
//             } | TypedNode {
//                 node_kind: NodeKind::WhileLoop { .. },
//                 ..
//             } | TypedNode {
//                 node_kind: NodeKind::ForLoop { .. },
//                 ..
//             } | TypedNode {
//                 node_kind: NodeKind::Block { .. },
//                 ..
//             }
//         );
//
//         let needs_newline = matches!(
//             self.typed_nodes[inner],
//             TypedNode {
//                 node_kind: NodeKind::Block { .. },
//                 ..
//             }
//         );
//
//         self.gen_node(inner);
//
//         if needs_semicolon {
//             self.body_emitters.top().body.emitln(";");
//         }
//
//         if needs_newline {
//             self.body_emitters.top().body.newline();
//         }
//     }
//
//     fn variable_declaration(
//         &mut self,
//         declaration_kind: DeclarationKind,
//         name: usize,
//         _type_name: Option<usize>,
//         expression: usize,
//         node_type: Option<Type>,
//     ) {
//         let type_kind = node_type.unwrap().type_kind;
//
//         let is_array = is_type_kind_array(&self.type_kinds, type_kind);
//         let needs_const = declaration_kind != DeclarationKind::Var && !is_array;
//
//         self.emit_type_kind_left(type_kind, EmitterKind::Body, false, true);
//         if needs_const {
//             self.body_emitters.top().body.emit("const ");
//         }
//         self.gen_node(name);
//         self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
//
//         if is_array && !is_typed_expression_array_literal(&self.typed_nodes, expression) {
//             self.body_emitters.top().body.emitln(";");
//
//             let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
//             else {
//                 panic!("invalid variable name");
//             };
//             self.emit_memmove_expression_to_name(&name_text, expression, type_kind);
//         } else {
//             self.body_emitters.top().body.emit(" = ");
//             self.gen_node(expression);
//         }
//     }
//
//     fn return_statement(&mut self, expression: Option<usize>, node_type: Option<Type>) {
//         self.body_emitters.exiting_all_scopes();
//
//         let expression = if let Some(expression) = expression {
//             expression
//         } else {
//             self.body_emitters.top().body.emit("return");
//             return;
//         };
//
//         let type_kind = node_type.unwrap().type_kind;
//
//         if is_type_kind_array(&self.type_kinds, type_kind) {
//             if is_typed_expression_array_literal(&self.typed_nodes, expression) {
//                 let temp_name = self.temp_variable_name("temp");
//
//                 self.emit_type_kind_left(type_kind, EmitterKind::Body, false, true);
//                 self.body_emitters.top().body.emit(&temp_name);
//                 self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
//                 self.body_emitters.top().body.emit(" = ");
//                 self.gen_node(expression);
//                 self.body_emitters.top().body.emitln(";");
//
//                 self.emit_memmove_name_to_name("__return", &temp_name, type_kind);
//                 self.body_emitters.top().body.emitln(";");
//             } else {
//                 self.emit_memmove_expression_to_name("__return", expression, type_kind);
//                 self.body_emitters.top().body.emitln(";");
//             }
//
//             self.body_emitters.top().body.emit("return __return");
//         } else {
//             self.body_emitters.top().body.emit("return ");
//             self.gen_node(expression);
//         }
//     }
//
//     fn defer_statement(&mut self, statement: usize, _node_type: Option<Type>) {
//         self.body_emitters.push(0);
//         self.gen_node(statement);
//         self.body_emitters.pop_to_bottom();
//     }
//
//     fn if_statement(
//         &mut self,
//         expression: usize,
//         statement: usize,
//         next: Option<usize>,
//         _node_type: Option<Type>,
//     ) {
//         self.body_emitters.top().body.emit("if (");
//         self.gen_node(expression);
//         self.body_emitters.top().body.emit(") ");
//         self.gen_node(statement);
//
//         if let Some(next) = next {
//             self.body_emitters.top().body.emit("else ");
//             self.gen_node(next);
//         }
//     }
//
//     fn switch_statement(
//         &mut self,
//         expression: usize,
//         case_statement: usize,
//         _node_type: Option<Type>,
//     ) {
//         self.body_emitters.top().body.emit("switch (");
//         self.gen_node(expression);
//         self.body_emitters.top().body.emitln(") {");
//         self.gen_node(case_statement);
//         self.body_emitters.top().body.emitln("}");
//     }
//
//     fn case_statement(
//         &mut self,
//         expression: usize,
//         statement: usize,
//         next: Option<usize>,
//         _node_type: Option<Type>,
//     ) {
//         self.body_emitters.top().body.emit("case ");
//         self.gen_node(expression);
//         self.body_emitters.top().body.emit(": ");
//         self.emit_scoped_statement(statement);
//         self.body_emitters.top().body.emitln("break;");
//
//         if let Some(next) = next {
//             if !matches!(
//                 self.typed_nodes[next].node_kind,
//                 NodeKind::CaseStatement { .. }
//             ) {
//                 self.body_emitters.top().body.emit("default: ");
//                 self.emit_scoped_statement(next);
//                 self.body_emitters.top().body.emitln("break;");
//             } else {
//                 self.gen_node(next);
//             }
//         }
//     }
//
//     fn while_loop(&mut self, expression: usize, statement: usize, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emit("while (");
//         self.gen_node(expression);
//         self.body_emitters.top().body.emit(") ");
//         self.gen_node(statement);
//     }
//
//     #[allow(clippy::too_many_arguments)]
//     fn for_loop(
//         &mut self,
//         iterator: usize,
//         op: Op,
//         from: usize,
//         to: usize,
//         by: Option<usize>,
//         statement: usize,
//         _node_type: Option<Type>,
//     ) {
//         self.body_emitters.top().body.emit("for (intptr_t ");
//         self.gen_node(iterator);
//         self.body_emitters.top().body.emit(" = ");
//         self.gen_node(from);
//         self.body_emitters.top().body.emit("; ");
//
//         self.gen_node(iterator);
//         self.emit_binary_op(op);
//         self.gen_node(to);
//         self.body_emitters.top().body.emit("; ");
//
//         self.gen_node(iterator);
//         self.body_emitters.top().body.emit(" += ");
//         if let Some(by) = by {
//             self.gen_node(by);
//         } else {
//             self.body_emitters.top().body.emit("1");
//         }
//         self.body_emitters.top().body.emit(") ");
//
//         self.gen_node(statement);
//     }
//
//     fn const_expression(&mut self, _inner: usize, node_type: Option<Type>) {
//         let Some(Type {
//             instance_kind: InstanceKind::Const(const_value),
//             ..
//         }) = node_type
//         else {
//             panic!("invalid node type of const expression");
//         };
//
//         match const_value {
//             ConstValue::Int { value } => self.body_emitters.top().body.emit(&value.to_string()),
//             ConstValue::UInt { value } => self.body_emitters.top().body.emit(&value.to_string()),
//             ConstValue::Float32 { value } => self.body_emitters.top().body.emit(&value.to_string()),
//             ConstValue::String { value } => {
//                 self.body_emitters.top().body.emit_char('"');
//                 self.body_emitters.top().body.emit(&value);
//                 self.body_emitters.top().body.emit_char('"');
//             }
//             ConstValue::Char { value } => self.body_emitters.top().body.emit_char(value),
//             ConstValue::Bool { value } => {
//                 if value {
//                     self.body_emitters.top().body.emit("true");
//                 } else {
//                     self.body_emitters.top().body.emit("false");
//                 }
//             }
//         }
//     }
//
//     fn binary(&mut self, left: usize, op: Op, right: usize, node_type: Option<Type>) {
//         if op == Op::Assign {
//             let type_kind = node_type.unwrap().type_kind;
//             let is_array = is_type_kind_array(&self.type_kinds, type_kind);
//
//             if is_array && !is_typed_expression_array_literal(&self.typed_nodes, left) {
//                 self.emit_memmove_expression_to_variable(left, right, type_kind);
//                 return;
//             }
//
//             // TODO: Needs refactor into it's own fn, also consider the similar code that exists in field access.
//             if let NodeKind::FieldAccess { left, name } = self.typed_nodes[left].node_kind {
//                 let left_type = self.typed_nodes[left].node_type.as_ref().unwrap();
//
//                 let (dereferenced_left_type_kind, is_left_pointer) =
//                     if let TypeKind::Pointer { inner_type_kind, .. } =
//                         self.type_kinds[left_type.type_kind]
//                     {
//                         (inner_type_kind, true)
//                     } else {
//                         (left_type.type_kind, false)
//                     };
//
//                 if let TypeKind::Struct {
//                     name: struct_name,
//                     field_kinds,
//                     generic_param_type_kinds,
//                     is_union,
//                     ..
//                 } = &self.type_kinds[dereferenced_left_type_kind]
//                 {
//                     if *is_union {
//                         let NodeKind::Name { text: name_text } =
//                             self.typed_nodes[name].node_kind.clone()
//                         else {
//                             panic!("invalid field name in field access");
//                         };
//
//                         let Some(tag) =
//                             get_field_index_by_name(&name_text, &self.typed_nodes, field_kinds)
//                         else {
//                             panic!("tag not found in union assignment");
//                         };
//
//                         let NodeKind::Name {
//                             text: struct_name_text,
//                         } = self.typed_nodes[*struct_name].node_kind.clone()
//                         else {
//                             panic!("invalid name in struct field access");
//                         };
//
//                         self.current_namespace_names.push(NamespaceName {
//                             name: struct_name_text,
//                             generic_param_type_kinds: Some(generic_param_type_kinds.clone()),
//                         });
//                         self.emit_namespace_prefix(EmitterKind::Body);
//                         self.current_namespace_names.pop();
//
//                         self.body_emitters.top().body.emit("__WithTag((");
//                         self.emit_type_kind_left(dereferenced_left_type_kind, EmitterKind::Body, false, false);
//                         self.emit_type_kind_right(dereferenced_left_type_kind, EmitterKind::Body, false);
//                         self.body_emitters.top().body.emit("*)");
//
//                         if !is_left_pointer {
//                             self.body_emitters.top().body.emit("&");
//                         }
//
//                         self.gen_node(left);
//
//                         self.body_emitters.top().body.emit(", ");
//                         self.body_emitters.top().body.emit(&tag.to_string());
//
//                         self.body_emitters.top().body.emit(")->variant.");
//                         self.gen_node(name);
//
//                         self.emit_binary_op(op);
//                         self.gen_node(right);
//
//                         return;
//                     }
//                 }
//             }
//         }
//
//         self.gen_node(left);
//         self.emit_binary_op(op);
//         self.gen_node(right);
//     }
//
//     fn unary_prefix(&mut self, op: Op, right: usize, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emit(match op {
//             Op::Plus => "+",
//             Op::Minus => "-",
//             Op::Not => "!",
//             Op::Reference => "&",
//             _ => panic!("expected unary prefix operator"),
//         });
//
//         self.gen_node(right);
//     }
//
//     fn unary_suffix(&mut self, left: usize, op: Op, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emit("(");
//         self.body_emitters.top().body.emit(match op {
//             Op::Dereference => "*",
//             _ => panic!("expected unary suffix operator"),
//         });
//
//         self.gen_node(left);
//         self.body_emitters.top().body.emit(")");
//     }
//
//     fn call(&mut self, mut left: usize, args: Arc<Vec<usize>>, node_type: Option<Type>) {
//         self.gen_node(left);
//
//         self.body_emitters.top().body.emit("(");
//         let mut i = 0;
//
//         if let NodeKind::GenericSpecifier { left: new_left, .. } = &self.typed_nodes[left].node_kind
//         {
//             left = *new_left;
//         }
//
//         // If the left node is a field access, then we must be calling a method. If we were calling a function pointer,
//         // the left node wouldn't be a field access because the function pointer would need to be dereferenced before calling.
//         if let Some(method_subject) = get_method_subject(&self.typed_nodes, left) {
//             if !matches!(
//                 self.type_kinds[method_subject.type_kind],
//                 TypeKind::Pointer { .. }
//             ) {
//                 self.body_emitters.top().body.emit("&");
//             }
//
//             self.gen_node(method_subject.node);
//
//             i += 1;
//         }
//
//         for arg in args.iter() {
//             if i > 0 {
//                 self.body_emitters.top().body.emit(", ");
//             }
//
//             self.gen_node(*arg);
//
//             i += 1;
//         }
//
//         let type_kind = node_type.unwrap().type_kind;
//         if is_type_kind_array(&self.type_kinds, type_kind) {
//             if args.len() > 0 {
//                 self.body_emitters.top().body.emit(", ");
//             }
//
//             let return_array_name = self.temp_variable_name("returnArray");
//
//             self.emit_type_kind_left(type_kind, EmitterKind::Top, false, true);
//             self.body_emitters.top().top.emit(&return_array_name);
//             self.emit_type_kind_right(type_kind, EmitterKind::Top, false);
//             self.body_emitters.top().top.emitln(";");
//
//             self.body_emitters.top().body.emit(&return_array_name);
//         }
//         self.body_emitters.top().body.emit(")");
//     }
//
//     fn index_access(&mut self, left: usize, expression: usize, _node_type: Option<Type>) {
//         self.gen_node(left);
//         self.body_emitters.top().body.emit("[");
//
//         if self.is_debug_mode {
//             let left_type = self.typed_nodes[left].node_type.as_ref().unwrap();
//             let TypeKind::Array { element_count, .. } = &self.type_kinds[left_type.type_kind] else {
//                 panic!("tried to perform an index access on a non-array type");
//             };
//             let element_count = *element_count;
//
//             self.body_emitters.top().body.emit("__BoundsCheck(");
//             self.gen_node(expression);
//             self.body_emitters.top().body.emit(", ");
//             self.body_emitters.top().body.emit(&element_count.to_string());
//             self.body_emitters.top().body.emit(")");
//         } else {
//             self.gen_node(expression);
//         }
//
//         self.body_emitters.top().body.emit("]");
//     }
//
//     fn field_access(&mut self, left: usize, name: usize, node_type: Option<Type>) {
//         let left_type = self.typed_nodes[left].node_type.as_ref().unwrap();
//
//         match self.type_kinds[node_type.unwrap().type_kind] {
//             TypeKind::Function { .. } => {
//                 let TypeKind::Struct {
//                     name: struct_name, ..
//                 } = self.type_kinds[left_type.type_kind]
//                 else {
//                     panic!("expected function field to be part of a struct");
//                 };
//
//                 let NodeKind::Name {
//                     text: struct_name_text,
//                 } = self.typed_nodes[struct_name].node_kind.clone()
//                 else {
//                     panic!("invalid name in struct field access");
//                 };
//
//                 self.current_namespace_names.push(NamespaceName {
//                     name: struct_name_text,
//                     generic_param_type_kinds: None,
//                 });
//                 self.emit_namespace_prefix(EmitterKind::Body);
//                 self.gen_node(name);
//                 self.current_namespace_names.pop();
//
//                 return;
//             }
//             TypeKind::Tag => {
//                 let TypeKind::Struct {
//                     field_kinds,
//                     is_union,
//                     ..
//                 } = &self.type_kinds[left_type.type_kind]
//                 else {
//                     panic!("expected tag field to be part of a struct");
//                 };
//
//                 if left_type.instance_kind == InstanceKind::Name && *is_union {
//                     let NodeKind::Name { text: name_text } =
//                         self.typed_nodes[name].node_kind.clone()
//                     else {
//                         panic!("invalid tag name in tag access");
//                     };
//
//                     let Some(tag) =
//                         get_field_index_by_name(&name_text, &self.typed_nodes, field_kinds)
//                     else {
//                         panic!("tag not found in field access");
//                     };
//
//                     self.body_emitters.top().body.emit(&tag.to_string());
//
//                     return;
//                 }
//             }
//             _ => {}
//         }
//
//         let (dereferenced_left_type_kind, is_left_pointer) =
//             if let TypeKind::Pointer { inner_type_kind, .. } = self.type_kinds[left_type.type_kind] {
//                 (inner_type_kind, true)
//             } else {
//                 (left_type.type_kind, false)
//             };
//
//         if let TypeKind::Struct {
//             name: struct_name,
//             field_kinds,
//             generic_param_type_kinds,
//             is_union,
//             ..
//         } = &self.type_kinds[dereferenced_left_type_kind]
//         {
//             if *is_union {
//                 let NodeKind::Name { text: name_text } = self.typed_nodes[name].node_kind.clone()
//                 else {
//                     panic!("invalid field name in field access");
//                 };
//
//                 let Some(tag) = get_field_index_by_name(&name_text, &self.typed_nodes, field_kinds)
//                 else {
//                     panic!("tag not found in field access");
//                 };
//
//                 let NodeKind::Name {
//                     text: struct_name_text,
//                 } = self.typed_nodes[*struct_name].node_kind.clone()
//                 else {
//                     panic!("invalid name in struct field access");
//                 };
//
//                 self.current_namespace_names.push(NamespaceName {
//                     name: struct_name_text,
//                     generic_param_type_kinds: Some(generic_param_type_kinds.clone()),
//                 });
//                 self.emit_namespace_prefix(EmitterKind::Body);
//                 self.current_namespace_names.pop();
//
//                 self.body_emitters.top().body.emit("__CheckTag((");
//                 self.emit_type_kind_left(dereferenced_left_type_kind, EmitterKind::Body, false, false);
//                 self.emit_type_kind_right(dereferenced_left_type_kind, EmitterKind::Body, false);
//                 self.body_emitters.top().body.emit("*)");
//
//                 if !is_left_pointer {
//                     self.body_emitters.top().body.emit("&");
//                 }
//
//                 self.gen_node(left);
//
//                 self.body_emitters.top().body.emit(", ");
//                 self.body_emitters.top().body.emit(&tag.to_string());
//
//                 self.body_emitters.top().body.emit(")->variant.");
//                 self.gen_node(name);
//
//                 return;
//             }
//         }
//
//         match self.type_kinds[left_type.type_kind] {
//             TypeKind::Pointer { .. } => {
//                 self.gen_node(left);
//                 self.body_emitters.top().body.emit("->")
//             }
//             TypeKind::Struct { .. } => {
//                 self.gen_node(left);
//                 self.body_emitters.top().body.emit(".")
//             }
//             TypeKind::Enum {
//                 name: enum_name, ..
//             } => {
//                 self.body_emitters.top().body.emit("__");
//                 self.gen_node(enum_name);
//             }
//             TypeKind::Array { element_count, .. } => {
//                 // On arrays, only the "count" field is allowed.
//                 self.body_emitters
//                     .top()
//                     .body
//                     .emit(&element_count.to_string());
//                 return;
//             }
//             _ => panic!("tried to access type that cannot be accessed"),
//         }
//
//         self.gen_node(name);
//     }
//
//     fn cast(&mut self, left: usize, type_name: usize, node_type: Option<Type>) {
//         if let TypeKind::Tag { .. } = &self.type_kinds[node_type.unwrap().type_kind] {
//             let left_type_kind = self.typed_nodes[left].node_type.as_ref().unwrap().type_kind;
//
//             let TypeKind::Struct { is_union, .. } = &self.type_kinds[left_type_kind] else {
//                 panic!("casting to a tag is not allowed for this value");
//             };
//
//             if !is_union {
//                 panic!("casting to a tag is not allowed for this value");
//             }
//
//             self.gen_node(left);
//             self.body_emitters.top().body.emit(".tag");
//
//             return;
//         }
//
//         self.body_emitters.top().body.emit("((");
//         self.emit_type_name_left(type_name, EmitterKind::Body, false, false);
//         self.emit_type_name_right(type_name, EmitterKind::Body, false);
//         self.body_emitters.top().body.emit(")");
//         self.gen_node(left);
//         self.body_emitters.top().body.emit(")");
//     }
//
//     fn generic_specifier(
//         &mut self,
//         left: usize,
//         generic_param_type_kinds: Arc<Vec<usize>>,
//         _node_type: Option<Type>,
//     ) {
//         self.gen_node(left);
//         self.emit_generic_param_suffix(Some(&generic_param_type_kinds), EmitterKind::Body);
//     }
//
//     fn name(&mut self, text: Arc<str>, _node_type: Option<Type>) {
//         self.emit_name(text, EmitterKind::Body);
//     }
//
//     fn identifier(&mut self, name: usize, _node_type: Option<Type>) {
//         self.gen_node(name);
//     }
//
//     fn int_literal(&mut self, text: Arc<str>, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emit(&text);
//     }
//
//     fn float32_literal(&mut self, text: Arc<str>, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emit(&text);
//         self.body_emitters.top().body.emit("f");
//     }
//
//     fn char_literal(&mut self, value: char, _node_type: Option<Type>) {
//         let mut char_buffer = [0u8];
//
//         self.body_emitters.top().body.emit("'");
//         self.body_emitters
//             .top()
//             .body
//             .emit(value.encode_utf8(&mut char_buffer));
//         self.body_emitters.top().body.emit("'");
//     }
//
//     fn string_literal(&mut self, text: Arc<str>, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emit("\"");
//         for (i, line) in text.lines().enumerate() {
//             if i > 0 {
//                 self.body_emitters.top().body.emit("\\n");
//             }
//
//             self.body_emitters.top().body.emit(line);
//         }
//         self.body_emitters.top().body.emit("\"");
//     }
//
//     fn bool_literal(&mut self, value: bool, _node_type: Option<Type>) {
//         if value {
//             self.body_emitters.top().body.emit("true");
//         } else {
//             self.body_emitters.top().body.emit("false");
//         }
//     }
//
//     fn array_literal(
//         &mut self,
//         elements: Arc<Vec<usize>>,
//         _repeat_count_const_expression: Option<usize>,
//         node_type: Option<Type>,
//     ) {
//         let TypeKind::Array { element_count, .. } = self.type_kinds[node_type.unwrap().type_kind]
//         else {
//             panic!("invalid type for array literal");
//         };
//
//         let repeat_count = element_count / elements.len();
//
//         self.body_emitters.top().body.emit("{");
//         let mut i = 0;
//         for _ in 0..repeat_count {
//             for element in elements.iter() {
//                 if i > 0 {
//                     self.body_emitters.top().body.emit(", ");
//                 }
//
//                 self.gen_node(*element);
//
//                 i += 1;
//             }
//         }
//         self.body_emitters.top().body.emit("}");
//     }
//
//     fn struct_literal(&mut self, _left: usize, fields: Arc<Vec<usize>>, node_type: Option<Type>) {
//         let type_kind = node_type.unwrap().type_kind;
//
//         self.body_emitters.top().body.emit("(");
//         self.emit_type_kind_left(type_kind, EmitterKind::Body, false, false);
//         self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
//         self.body_emitters.top().body.emit(") ");
//
//         let TypeKind::Struct {
//             field_kinds,
//             is_union,
//             ..
//         } = &self.type_kinds[type_kind]
//         else {
//             panic!("struct literal does not have a struct type");
//         };
//         let is_union = *is_union;
//
//         if is_union {
//             self.body_emitters.top().body.emitln("{");
//             self.body_emitters.top().body.indent();
//
//             if fields.len() != 1 {
//                 panic!("expected union literal to contain a single field");
//             }
//
//             let NodeKind::FieldLiteral { name, .. } = &self.typed_nodes[fields[0]].node_kind else {
//                 panic!("invalid field in union literal");
//             };
//
//             let NodeKind::Name { text: name_text } = &self.typed_nodes[*name].node_kind else {
//                 panic!("invalid field name text in union literal");
//             };
//
//             let Some(tag) = get_field_index_by_name(name_text, &self.typed_nodes, field_kinds)
//             else {
//                 panic!("tag not found in union literal");
//             };
//
//             self.body_emitters.top().body.emit(".tag = ");
//             self.body_emitters.top().body.emit(&tag.to_string());
//             self.body_emitters.top().body.emitln(",");
//
//             self.body_emitters.top().body.emit(".variant = ");
//         }
//
//         self.body_emitters.top().body.emitln("{");
//         self.body_emitters.top().body.indent();
//
//         if fields.is_empty() {
//             // Since empty structs aren't allowed in C, we generate a placeholder field
//             // in structs that would be empty otherwise. We also have to initialize it here.
//             self.body_emitters.top().body.emitln("0,");
//         }
//
//         for field in fields.iter() {
//             self.gen_node(*field);
//             self.body_emitters.top().body.emitln(",");
//         }
//
//         self.body_emitters.top().body.unindent();
//         self.body_emitters.top().body.emit("}");
//
//         if is_union {
//             self.body_emitters.top().body.emitln(",");
//             self.body_emitters.top().body.unindent();
//             self.body_emitters.top().body.emit("}");
//         }
//     }
//
//     fn field_literal(&mut self, name: usize, expression: usize, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emit(".");
//         self.gen_node(name);
//         self.body_emitters.top().body.emit(" = ");
//         self.gen_node(expression);
//     }
//
//     fn type_size(&mut self, type_name: usize, _node_type: Option<Type>) {
//         self.body_emitters.top().body.emit("sizeof(");
//         self.emit_type_name_left(type_name, EmitterKind::Body, false, false);
//         self.emit_type_name_right(type_name, EmitterKind::Body, false);
//         self.body_emitters.top().body.emit(")");
//     }
//
//     fn emit_memmove_expression_to_variable(
//         &mut self,
//         destination: usize,
//         source: usize,
//         type_kind: usize,
//     ) {
//         self.body_emitters.top().body.emit("memmove(");
//         self.gen_node(destination);
//         self.body_emitters.top().body.emit(", ");
//         self.gen_node(source);
//         self.body_emitters.top().body.emit(", ");
//         self.emit_type_size(type_kind);
//         self.body_emitters.top().body.emit(")");
//     }
//
//     fn emit_memmove_expression_to_name(
//         &mut self,
//         destination: &str,
//         source: usize,
//         type_kind: usize,
//     ) {
//         self.body_emitters.top().body.emit("memmove(");
//         self.body_emitters.top().body.emit(destination);
//         self.body_emitters.top().body.emit(", ");
//         self.gen_node(source);
//         self.body_emitters.top().body.emit(", ");
//         self.emit_type_size(type_kind);
//         self.body_emitters.top().body.emit(")");
//     }
//
//     fn emit_memmove_name_to_name(&mut self, destination: &str, source: &str, type_kind: usize) {
//         self.body_emitters.top().body.emit("memmove(");
//         self.body_emitters.top().body.emit(destination);
//         self.body_emitters.top().body.emit(", ");
//         self.body_emitters.top().body.emit(source);
//         self.body_emitters.top().body.emit(", ");
//         self.emit_type_size(type_kind);
//         self.body_emitters.top().body.emit(")");
//     }
//
//     fn emit_type_size(&mut self, type_kind: usize) {
//         match self.type_kinds[type_kind] {
//             TypeKind::Array {
//                 element_type_kind,
//                 element_count,
//             } => {
//                 self.emit_type_size(element_type_kind);
//                 self.body_emitters.top().body.emit(" * ");
//                 self.body_emitters
//                     .top()
//                     .body
//                     .emit(&element_count.to_string());
//             }
//             _ => {
//                 self.body_emitters.top().body.emit("sizeof(");
//                 self.emit_type_kind_left(type_kind, EmitterKind::Body, false, true);
//                 self.emit_type_kind_right(type_kind, EmitterKind::Body, false);
//                 self.body_emitters.top().body.emit(")");
//             }
//         };
//     }
//
//     fn emit_type_name_left(
//         &mut self,
//         type_name: usize,
//         emitter_kind: EmitterKind,
//         do_arrays_as_pointers: bool,
//         is_prefix: bool,
//     ) {
//         let TypedNode {
//             node_type: Some(Type { type_kind, .. }),
//             ..
//         } = self.typed_nodes[type_name]
//         else {
//             panic!("tried to emit node that wasn't a type name");
//         };
//         self.emit_type_kind_left(type_kind, emitter_kind, do_arrays_as_pointers, is_prefix);
//     }
//
//     fn emit_type_name_right(
//         &mut self,
//         type_name: usize,
//         emitter_kind: EmitterKind,
//         do_arrays_as_pointers: bool,
//     ) {
//         let TypedNode {
//             node_type: Some(Type { type_kind, .. }),
//             ..
//         } = self.typed_nodes[type_name]
//         else {
//             panic!("tried to emit node that wasn't a type name");
//         };
//         self.emit_type_kind_right(type_kind, emitter_kind, do_arrays_as_pointers);
//     }
//
//     fn emit_type_kind_left(
//         &mut self,
//         type_kind: usize,
//         emitter_kind: EmitterKind,
//         do_arrays_as_pointers: bool,
//         is_prefix: bool,
//     ) {
//         let type_kind = &self.type_kinds[type_kind];
//         let needs_trailing_space = is_prefix
//             && !matches!(
//                 type_kind,
//                 TypeKind::Array { .. }
//                     | TypeKind::Pointer { .. }
//                     | TypeKind::Function { .. }
//                     | TypeKind::Alias { .. }
//             );
//
//         match type_kind.clone() {
//             TypeKind::Int | TypeKind::Tag { .. } => self.emitter(emitter_kind).emit("intptr_t"),
//             TypeKind::String => self.emitter(emitter_kind).emit("const char*"),
//             TypeKind::Bool => self.emitter(emitter_kind).emit("bool"),
//             TypeKind::Char => self.emitter(emitter_kind).emit("char"),
//             TypeKind::Void => self.emitter(emitter_kind).emit("void"),
//             TypeKind::UInt => self.emitter(emitter_kind).emit("uintptr_t"),
//             TypeKind::Int8 => self.emitter(emitter_kind).emit("int8_t"),
//             TypeKind::UInt8 => self.emitter(emitter_kind).emit("uint8_t"),
//             TypeKind::Int16 => self.emitter(emitter_kind).emit("int16_t"),
//             TypeKind::UInt16 => self.emitter(emitter_kind).emit("uint16_t"),
//             TypeKind::Int32 => self.emitter(emitter_kind).emit("int32_t"),
//             TypeKind::UInt32 => self.emitter(emitter_kind).emit("uint32_t"),
//             TypeKind::Int64 => self.emitter(emitter_kind).emit("int64_t"),
//             TypeKind::UInt64 => self.emitter(emitter_kind).emit("uint64_t"),
//             TypeKind::Float32 => self.emitter(emitter_kind).emit("float"),
//             TypeKind::Float64 => self.emitter(emitter_kind).emit("double"),
//             TypeKind::Struct {
//                 name,
//                 generic_param_type_kinds,
//                 ..
//             } => {
//                 self.emitter(emitter_kind).emit("struct ");
//                 let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
//                     panic!("invalid struct name");
//                 };
//                 self.emit_name(text, emitter_kind);
//                 self.emit_generic_param_suffix(Some(&generic_param_type_kinds), emitter_kind);
//             }
//             TypeKind::Enum { name, .. } => {
//                 self.emitter(emitter_kind).emit("enum ");
//                 let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
//                     panic!("invalid enum name");
//                 };
//                 self.emit_name(text, emitter_kind);
//             }
//             TypeKind::Array {
//                 element_type_kind, ..
//             } => {
//                 self.emit_type_kind_left(
//                     element_type_kind,
//                     emitter_kind,
//                     do_arrays_as_pointers,
//                     true,
//                 );
//                 if do_arrays_as_pointers {
//                     self.emitter(emitter_kind).emit("*");
//                 }
//             }
//             TypeKind::Pointer { inner_type_kind, is_inner_mutable } => {
//                 // If the pointer points to an immutable value, then add a const to the generated code.
//                 // Except for functions, because a const function has no meaning in C.
//                 if !is_inner_mutable && !matches!(self.type_kinds[inner_type_kind], TypeKind::Function { .. }) {
//                     self.emitter(emitter_kind).emit("const ");
//                 }
//
//                 self.emit_type_kind_left(
//                     inner_type_kind,
//                     emitter_kind,
//                     do_arrays_as_pointers,
//                     true,
//                 );
//                 self.emitter(emitter_kind).emit("*");
//             }
//             TypeKind::Alias { inner_type_kind } => {
//                 self.emit_type_kind_left(
//                     inner_type_kind,
//                     emitter_kind,
//                     do_arrays_as_pointers,
//                     is_prefix,
//                 );
//             }
//             TypeKind::Partial | TypeKind::PartialGeneric { .. } | TypeKind::PartialArray { .. } => {
//                 panic!("can't emit partial type: {:?}", type_kind)
//             }
//             TypeKind::Function {
//                 return_type_kind, ..
//             } => {
//                 self.emit_type_kind_left(return_type_kind, emitter_kind, true, true);
//                 self.emit_type_kind_right(return_type_kind, emitter_kind, true);
//                 self.emitter(emitter_kind).emit("(");
//             }
//         };
//
//         if needs_trailing_space {
//             self.emitter(emitter_kind).emit(" ");
//         }
//     }
//
//     fn emit_type_kind_right(
//         &mut self,
//         type_kind: usize,
//         emitter_kind: EmitterKind,
//         do_arrays_as_pointers: bool,
//     ) {
//         let type_kind = self.type_kinds[type_kind].clone();
//
//         match type_kind {
//             TypeKind::Array {
//                 element_type_kind,
//                 element_count,
//             } => {
//                 if !do_arrays_as_pointers {
//                     self.emitter(emitter_kind).emit("[");
//                     self.emitter(emitter_kind).emit(&element_count.to_string());
//                     self.emitter(emitter_kind).emit("]");
//                 }
//                 self.emit_type_kind_right(element_type_kind, emitter_kind, do_arrays_as_pointers);
//             }
//             TypeKind::Pointer { inner_type_kind, .. } => {
//                 self.emit_type_kind_right(inner_type_kind, emitter_kind, do_arrays_as_pointers);
//             }
//             TypeKind::Function {
//                 param_type_kinds, ..
//             } => {
//                 self.emitter(emitter_kind).emit(")(");
//                 for (i, param_kind) in param_type_kinds.iter().enumerate() {
//                     if i > 0 {
//                         self.emitter(emitter_kind).emit(", ");
//                     }
//
//                     self.emit_type_kind_left(*param_kind, emitter_kind, false, false);
//                     self.emit_type_kind_right(*param_kind, emitter_kind, false);
//                 }
//                 self.emitter(emitter_kind).emit(")");
//             }
//             TypeKind::Alias { inner_type_kind } => {
//                 self.emit_type_kind_right(inner_type_kind, emitter_kind, do_arrays_as_pointers);
//             }
//             _ => {}
//         }
//     }
//
//     fn emit_binary_op(&mut self, op: Op) {
//         self.body_emitters.top().body.emit(match op {
//             Op::Equal => " == ",
//             Op::NotEqual => " != ",
//             Op::Less => " < ",
//             Op::Greater => " > ",
//             Op::LessEqual => " <= ",
//             Op::GreaterEqual => " >= ",
//             Op::Plus => " + ",
//             Op::Minus => " - ",
//             Op::Multiply => " * ",
//             Op::Divide => " / ",
//             Op::Assign => " = ",
//             Op::And => " && ",
//             Op::Or => " || ",
//             Op::PlusAssign => " += ",
//             Op::MinusAssign => " -= ",
//             Op::MultiplyAssign => " *= ",
//             Op::DivideAssign => " /= ",
//             _ => panic!("expected binary operator"),
//         });
//     }
//
//     fn emit_function_declaration(
//         &mut self,
//         kind: EmitterKind,
//         name: usize,
//         params: &Arc<Vec<usize>>,
//         generic_usage: Option<Arc<Vec<usize>>>,
//         return_type_name: usize,
//         type_kind: usize,
//     ) {
//         self.emit_type_name_left(return_type_name, kind, true, true);
//         self.emit_function_name(name, generic_usage, kind);
//
//         let mut param_count = 0;
//
//         self.emitter(kind).emit("(");
//         for param in params.iter() {
//             if param_count > 0 {
//                 self.emitter(kind).emit(", ");
//             }
//
//             param_count += 1;
//
//             self.emit_param_node(*param, kind);
//         }
//
//         let TypeKind::Function {
//             return_type_kind, ..
//         } = self.type_kinds[type_kind]
//         else {
//             panic!("tried to emit function declaration for non-function type");
//         };
//
//         if is_type_kind_array(&self.type_kinds, return_type_kind) {
//             if param_count > 0 {
//                 self.emitter(kind).emit(", ");
//             }
//
//             param_count += 1;
//
//             self.emit_param_string("__return", return_type_name, kind);
//         }
//
//         if param_count == 0 {
//             self.emitter(kind).emit("void");
//         }
//
//         self.emitter(kind).emit(")");
//
//         self.emit_type_name_right(return_type_name, kind, true);
//     }
//
//     fn emit_param_node(&mut self, param: usize, kind: EmitterKind) {
//         let NodeKind::Param { name, type_name } = self.typed_nodes[param].node_kind else {
//             panic!("invalid param");
//         };
//
//         self.emit_param(name, type_name, kind);
//     }
//
//     fn emit_param(&mut self, name: usize, type_name: usize, kind: EmitterKind) {
//         self.emit_type_name_left(type_name, kind, false, true);
//         self.emit_name_node(name, kind);
//         self.emit_type_name_right(type_name, kind, false);
//     }
//
//     fn emit_param_string(&mut self, name: &str, type_name: usize, kind: EmitterKind) {
//         self.emit_type_name_left(type_name, kind, false, true);
//         self.emitter(kind).emit(name);
//         self.emit_type_name_right(type_name, kind, false);
//     }
//
//     fn emit_name_node(&mut self, name: usize, kind: EmitterKind) {
//         let NodeKind::Name { text } = self.typed_nodes[name].node_kind.clone() else {
//             panic!("invalid name");
//         };
//
//         self.emit_name(text, kind);
//     }
//
//     fn emit_name(&mut self, text: Arc<str>, kind: EmitterKind) {
//         if reserved_names().contains(&text) {
//             self.emitter(kind).emit("__");
//         }
//
//         self.emitter(kind).emit(&text);
//     }
//
//     // Used for name mangling, so that multiple versions of a generic function can be generated without colliding.
//     fn emit_generic_param_suffix(
//         &mut self,
//         generic_param_type_kinds: Option<&Arc<Vec<usize>>>,
//         kind: EmitterKind,
//     ) {
//         let Some(generic_param_type_kinds) = generic_param_type_kinds else {
//             return;
//         };
//
//         if generic_param_type_kinds.is_empty() {
//             return;
//         }
//
//         self.emitter(kind).emit("_");
//
//         for mut generic_param_type_kind in generic_param_type_kinds.iter().copied() {
//             if let TypeKind::Alias { inner_type_kind } = self.type_kinds[generic_param_type_kind] {
//                 generic_param_type_kind = inner_type_kind;
//             }
//
//             self.emitter(kind).emit("_");
//
//             // This prints the number backwards, but it doesn't matter for the purpose of name mangling.
//             let mut number = generic_param_type_kind;
//             let mut digit = 0;
//             while number > 0 || digit == 0 {
//                 self.emitter(kind)
//                     .emit_char(((number % 10) as u8 + b'0') as char);
//                 number /= 10;
//                 digit += 1;
//             }
//         }
//     }
//
//     fn emit_namespace_prefix(&mut self, kind: EmitterKind) {
//         if self.current_namespace_names.is_empty() {
//             return;
//         }
//
//         self.emitter(kind).emit("__");
//
//         for i in 0..self.current_namespace_names.len() {
//             let namespace_name = self.current_namespace_names[i].clone();
//             self.emitter(kind).emit(&namespace_name.name);
//             self.emit_generic_param_suffix(namespace_name.generic_param_type_kinds.as_ref(), kind);
//         }
//     }
//
//     fn emit_function_name(
//         &mut self,
//         name: usize,
//         generic_usage: Option<Arc<Vec<usize>>>,
//         kind: EmitterKind,
//     ) {
//         self.emit_namespace_prefix(kind);
//         self.emit_name_node(name, kind);
//         self.emit_generic_param_suffix(generic_usage.as_ref(), kind);
//     }
//
//     fn emit_scoped_statement(&mut self, statement: usize) {
//         let NodeKind::Statement { inner } = self.typed_nodes[statement].node_kind else {
//             panic!("invalid statement in scoped statement");
//         };
//
//         let needs_scope = inner.is_none()
//             || !matches!(
//                 self.typed_nodes[inner.unwrap()].node_kind,
//                 NodeKind::Block { .. }
//             );
//
//         if needs_scope {
//             self.body_emitters.top().body.emitln("{");
//             self.body_emitters.top().body.indent();
//         }
//
//         self.gen_node(statement);
//
//         if needs_scope {
//             self.body_emitters.top().body.unindent();
//             self.body_emitters.top().body.emitln("}");
//         }
//     }
//
//     fn emit_bounds_check(&mut self) {
//         self.function_prototype_emitter.emitln("intptr_t __BoundsCheck(intptr_t index, intptr_t count);");
//         self.function_prototype_emitter.newline();
//
//         self.body_emitters.top().body.emitln("intptr_t __BoundsCheck(intptr_t index, intptr_t count) {");
//         self.body_emitters.top().body.indent();
//         self.body_emitters.top().body.emitln("assert(index >= 0 && index < count);");
//         self.body_emitters.top().body.emitln("return index;");
//         self.body_emitters.top().body.unindent();
//         self.body_emitters.top().body.emitln("}");
//         self.body_emitters.top().body.newline();
//     }
//
//     fn emitter(&mut self, kind: EmitterKind) -> &mut Emitter {
//         match kind {
//             EmitterKind::TypePrototype => &mut self.type_prototype_emitter,
//             EmitterKind::FunctionPrototype => &mut self.function_prototype_emitter,
//             EmitterKind::Top => &mut self.body_emitters.top().top,
//             EmitterKind::Body => &mut self.body_emitters.top().body,
//         }
//     }
//
//     fn temp_variable_name(&mut self, prefix: &str) -> String {
//         let temp_variable_index = self.temp_variable_count;
//         self.temp_variable_count += 1;
//
//         format!("__{}{}", prefix, temp_variable_index)
//     }
// }
