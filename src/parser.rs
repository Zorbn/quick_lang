use std::{collections::HashMap, hash::Hash, sync::Arc};

use crate::{
    environment::Environment,
    file_data::FileData,
    lexer::{Token, TokenKind},
    position::Position,
    types::{add_type, get_function_type_kind, get_type_kind_as_pointer},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeclarationKind {
    Var,
    Val,
    Const,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Plus,
    Minus,
    Multiply,
    Divide,
    Not,
    Assign,
    PlusAssign,
    MinusAssign,
    MultiplyAssign,
    DivideAssign,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    And,
    Or,
    Reference,
    Dereference,
}

#[derive(Debug)]
pub struct Field {
    pub name: usize,
    pub type_kind: usize,
}

// Used to search for the index of an array type by its layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayLayout {
    pub element_type_kind: usize,
    pub element_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionLayout {
    pub param_type_kinds: Arc<Vec<usize>>,
    pub generic_type_kinds: Arc<Vec<usize>>,
    pub return_type_kind: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructLayout {
    pub name: Arc<str>,
    pub generic_param_type_kinds: Arc<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PointerLayout {
    pub inner_type_kind: usize,
    pub is_inner_mutable: bool,
}

#[derive(Clone, Debug)]
pub enum TypeKind {
    Int,
    String,
    Bool,
    Char,
    Void,
    UInt,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
    Tag,
    Pointer {
        inner_type_kind: usize,
        is_inner_mutable: bool,
    },
    Array {
        element_type_kind: usize,
        element_count: usize,
    },
    Struct {
        name: usize,
        field_kinds: Arc<Vec<Field>>,
        generic_type_kinds: Arc<Vec<usize>>,
        generic_param_type_kinds: Arc<Vec<usize>>,
        is_union: bool,
    },
    Partial,
    PartialGeneric {
        inner_type_kind: usize,
        generic_param_type_kinds: Arc<Vec<usize>>,
    },
    PartialArray {
        element_type_kind: usize,
        const_expression: usize,
    },
    Function {
        param_type_kinds: Arc<Vec<usize>>,
        generic_type_kinds: Arc<Vec<usize>>,
        return_type_kind: usize,
    },
    Enum {
        name: usize,
        variant_names: Arc<Vec<usize>>,
    },
    Alias {
        inner_type_kind: usize,
    },
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    Name {
        text: Arc<str>,
    },
    TopLevel {
        functions: Arc<Vec<usize>>,
        structs: Arc<Vec<usize>>,
        enums: Arc<Vec<usize>>,
    },
    StructDefinition {
        name: usize,
        fields: Arc<Vec<usize>>,
        generic_params: Arc<Vec<usize>>,
        functions: Arc<Vec<usize>>,
        type_kind: usize,
    },
    EnumDefinition {
        name: usize,
        variant_names: Arc<Vec<usize>>,
        type_kind: usize,
    },
    Field {
        name: usize,
        type_name: usize,
    },
    FunctionDeclaration {
        name: usize,
        params: Arc<Vec<usize>>,
        generic_params: Arc<Vec<usize>>,
        return_type_name: usize,
        type_kind: usize,
    },
    Function {
        declaration: usize,
        statement: usize,
    },
    ExternFunction {
        declaration: usize,
    },
    Param {
        name: usize,
        type_name: usize,
    },
    Block {
        statements: Arc<Vec<usize>>,
    },
    Statement {
        inner: Option<usize>,
    },
    VariableDeclaration {
        declaration_kind: DeclarationKind,
        name: usize,
        type_name: Option<usize>,
        expression: usize,
    },
    ReturnStatement {
        expression: Option<usize>,
    },
    DeferStatement {
        statement: usize,
    },
    IfStatement {
        expression: usize,
        statement: usize,
        next: Option<usize>,
    },
    SwitchStatement {
        expression: usize,
        case_statement: usize,
    },
    CaseStatement {
        expression: usize,
        statement: usize,
        next: Option<usize>,
    },
    WhileLoop {
        expression: usize,
        statement: usize,
    },
    ForLoop {
        iterator: usize,
        op: Op,
        from: usize,
        to: usize,
        by: Option<usize>,
        statement: usize,
    },
    Binary {
        left: usize,
        op: Op,
        right: usize,
    },
    UnaryPrefix {
        op: Op,
        right: usize,
    },
    UnarySuffix {
        left: usize,
        op: Op,
    },
    Call {
        left: usize,
        args: Arc<Vec<usize>>,
    },
    IndexAccess {
        left: usize,
        expression: usize,
    },
    FieldAccess {
        left: usize,
        name: usize,
    },
    Cast {
        left: usize,
        type_name: usize,
    },
    GenericSpecifier {
        left: usize,
        generic_param_type_kinds: Arc<Vec<usize>>,
    },
    Identifier {
        name: usize,
    },
    IntLiteral {
        text: Arc<str>,
    },
    Float32Literal {
        text: Arc<str>,
    },
    StringLiteral {
        text: Arc<str>,
    },
    BoolLiteral {
        value: bool,
    },
    CharLiteral {
        value: char,
    },
    ArrayLiteral {
        elements: Arc<Vec<usize>>,
        repeat_count_const_expression: Option<usize>,
    },
    StructLiteral {
        left: usize,
        // TODO: Should these be renamed field_literals? They are FieldLiterals, not Fields...
        fields: Arc<Vec<usize>>,
    },
    FieldLiteral {
        name: usize,
        expression: usize,
    },
    TypeSize {
        type_name: usize,
    },
    TypeName {
        type_kind: usize,
    },
    ConstExpression {
        inner: usize,
    },
    Error,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub start: Position,
    pub end: Position,
}

pub const INT_INDEX: usize = 0;
pub const STRING_INDEX: usize = 1;
pub const BOOL_INDEX: usize = 2;
pub const CHAR_INDEX: usize = 3;
pub const VOID_INDEX: usize = 4;
pub const UINT_INDEX: usize = 5;
pub const INT8_INDEX: usize = 6;
pub const UINT8_INDEX: usize = 7;
pub const INT16_INDEX: usize = 8;
pub const UINT16_INDEX: usize = 9;
pub const INT32_INDEX: usize = 10;
pub const UINT32_INDEX: usize = 11;
pub const INT64_INDEX: usize = 12;
pub const UINT64_INDEX: usize = 13;
pub const FLOAT32_INDEX: usize = 14;
pub const FLOAT64_INDEX: usize = 15;
pub const TAG_INDEX: usize = 16;

macro_rules! assert_token {
    ($self:ident, $token:expr, $start:expr, $end:expr) => {
        if let Some(node) = $self.assert_token($token, $start, $end) {
            return node;
        }
    };
}

macro_rules! parse_error {
    ($self:ident, $message:expr, $start:expr, $end:expr) => {{
        return $self.parse_error($message, $start, $end);
    }};
}

pub struct Parser {
    pub nodes: Vec<Node>,
    pub type_kinds: Vec<TypeKind>,
    pub array_type_kinds: HashMap<ArrayLayout, usize>,
    pub pointer_type_kinds: HashMap<PointerLayout, usize>,
    pub named_type_kinds: Environment<usize>,
    pub function_type_kinds: HashMap<FunctionLayout, usize>,
    pub struct_type_kinds: HashMap<StructLayout, usize>,
    pub definition_indices: HashMap<Vec<Arc<str>>, usize>,
    pub had_error: bool,

    pub tokens: Option<Vec<Token>>,
    pub position: usize,

    files: Arc<Vec<FileData>>,
    current_namespace_names: Vec<Arc<str>>,
}

impl Parser {
    pub fn new(files: Arc<Vec<FileData>>) -> Self {
        let mut parser = Self {
            files,
            tokens: None,
            nodes: Vec::new(),
            type_kinds: Vec::new(),
            array_type_kinds: HashMap::new(),
            pointer_type_kinds: HashMap::new(),
            named_type_kinds: Environment::new(),
            function_type_kinds: HashMap::new(),
            struct_type_kinds: HashMap::new(),
            definition_indices: HashMap::new(),
            had_error: false,
            position: 0,
            current_namespace_names: Vec::new(),
        };

        parser.named_type_kinds.push();

        parser.add_type(TypeKind::Int);
        parser.add_type(TypeKind::String);
        parser.add_type(TypeKind::Bool);
        parser.add_type(TypeKind::Char);
        parser.add_type(TypeKind::Void);
        parser.add_type(TypeKind::UInt);
        parser.add_type(TypeKind::Int8);
        parser.add_type(TypeKind::UInt8);
        parser.add_type(TypeKind::Int16);
        parser.add_type(TypeKind::UInt16);
        parser.add_type(TypeKind::Int32);
        parser.add_type(TypeKind::UInt32);
        parser.add_type(TypeKind::Int64);
        parser.add_type(TypeKind::UInt64);
        parser.add_type(TypeKind::Float32);
        parser.add_type(TypeKind::Float64);
        parser.add_type(TypeKind::Tag);

        parser
    }

    fn token_kind(&self) -> &TokenKind {
        let tokens = self.tokens.as_ref().unwrap();
        if let Some(token) = tokens.get(self.position) {
            &token.kind
        } else {
            &TokenKind::Eof
        }
    }

    fn node_end(&self, index: usize) -> Position {
        self.nodes[index].end
    }

    fn token_start(&self) -> Position {
        let tokens = self.tokens.as_ref().unwrap();

        if tokens.len() <= self.position {
            return Position::new(0);
        }

        tokens[self.position].start
    }

    fn last_token_end(&self) -> Position {
        if self.position < 1 {
            return self.token_end();
        }

        let tokens = self.tokens.as_ref().unwrap();

        if tokens.len() <= self.position {
            return Position::new(0);
        }

        tokens[self.position - 1].end
    }

    fn token_end(&self) -> Position {
        let tokens = self.tokens.as_ref().unwrap();

        if tokens.len() <= self.position {
            return Position::new(0);
        }

        tokens[self.position].end
    }

    fn assert_token(
        &mut self,
        token_kind: TokenKind,
        start: Position,
        end: Position,
    ) -> Option<usize> {
        if *self.token_kind() != token_kind {
            self.error(&format!(
                "expected \"{}\" but got \"{}\"",
                token_kind,
                self.token_kind()
            ));
            self.position += 1;

            return Some(self.add_node(Node {
                kind: NodeKind::Error,
                start,
                end,
            }));
        }

        None
    }

    fn parse_error(&mut self, message: &str, start: Position, end: Position) -> usize {
        self.error(message);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::Error,
            start,
            end,
        })
    }

    fn add_node(&mut self, node: Node) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    fn add_type(&mut self, type_kind: TypeKind) -> usize {
        add_type(&mut self.type_kinds, type_kind)
    }

    fn error(&mut self, message: &str) {
        self.had_error = true;
        self.token_start().error("Syntax", message, &self.files);
    }

    pub fn parse(&mut self, tokens: Vec<Token>) -> usize {
        self.position = 0;
        self.tokens = Some(tokens);
        self.top_level()
    }

    fn top_level(&mut self) -> usize {
        let mut functions = Vec::new();
        let mut structs = Vec::new();
        let mut enums = Vec::new();
        let start = self.token_start();
        let mut end = self.token_end();

        while *self.token_kind() != TokenKind::Eof {
            let index;

            match *self.token_kind() {
                TokenKind::Func => {
                    index = self.function();
                    functions.push(index);
                }
                TokenKind::Extern => {
                    index = self.extern_function();
                    functions.push(index);
                }
                TokenKind::Struct | TokenKind::Union => {
                    index = self.struct_definition();
                    structs.push(index);
                }
                TokenKind::Enum => {
                    index = self.enum_definition();
                    enums.push(index);
                }
                _ => parse_error!(self, "unexpected token at top level", start, end),
            }

            end = self.node_end(index);
        }

        self.add_node(Node {
            kind: NodeKind::TopLevel {
                functions: Arc::new(functions),
                structs: Arc::new(structs),
                enums: Arc::new(enums),
            },
            start,
            end,
        })
    }

    fn struct_definition(&mut self) -> usize {
        let start = self.token_start();

        let is_union = match *self.token_kind() {
            TokenKind::Struct => false,
            TokenKind::Union => true,
            _ => parse_error!(
                self,
                "unexpected token at the start of struct definition",
                start,
                self.token_end()
            ),
        };
        self.position += 1;

        let name = self.name();

        let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
            parse_error!(self, "invalid struct name", start, self.node_end(name));
        };

        self.named_type_kinds.push();
        self.current_namespace_names.push(name_text.clone());

        let mut generic_params = Vec::new();
        let mut generic_type_kinds = Vec::new();

        if *self.token_kind() == TokenKind::Less {
            if let Some(error_node) =
                self.parse_generic_params(start, &mut generic_params, &mut generic_type_kinds)
            {
                return error_node;
            }
        }

        assert_token!(self, TokenKind::LBrace, start, self.token_end());
        self.position += 1;

        let mut fields = Vec::new();
        let mut functions = Vec::new();
        let mut field_kinds = Vec::new();

        while *self.token_kind() != TokenKind::RBrace {
            if *self.token_kind() == TokenKind::Func {
                let function = self.function();
                functions.push(function);

                let NodeKind::Function { declaration, .. } = &self.nodes[function].kind else {
                    parse_error!(self, "invalid struct function", start, self.token_end());
                };

                let NodeKind::FunctionDeclaration {
                    name, type_kind, ..
                } = &self.nodes[*declaration].kind
                else {
                    parse_error!(
                        self,
                        "invalid struct function declaration",
                        start,
                        self.token_end()
                    );
                };

                field_kinds.push(Field {
                    name: *name,
                    type_kind: *type_kind,
                });

                continue;
            }

            let field = self.field();
            fields.push(field);

            if *self.token_kind() != TokenKind::Semicolon {
                break;
            }

            assert_token!(self, TokenKind::Semicolon, start, self.token_end());
            self.position += 1;

            let NodeKind::Field { name, type_name } = &self.nodes[field].kind else {
                parse_error!(self, "invalid struct field", start, self.token_end());
            };

            let NodeKind::TypeName { type_kind } = &self.nodes[*type_name].kind else {
                parse_error!(
                    self,
                    "invalid struct field type name",
                    start,
                    self.token_end()
                );
            };

            field_kinds.push(Field {
                name: *name,
                type_kind: *type_kind,
            });
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBrace, start, end);
        self.position += 1;

        self.current_namespace_names.pop();
        self.named_type_kinds.pop();

        let generic_param_type_kinds = Arc::new(Vec::new());

        let type_kind = self.add_named_type(
            &name_text,
            TypeKind::Struct {
                name,
                field_kinds: Arc::new(field_kinds),
                generic_type_kinds: Arc::new(generic_type_kinds),
                generic_param_type_kinds: generic_param_type_kinds.clone(),
                is_union,
            },
            false,
        );

        let struct_layout = StructLayout {
            name: name_text.clone(),
            generic_param_type_kinds,
        };
        self.struct_type_kinds.insert(struct_layout, type_kind);

        let index = self.add_node(Node {
            kind: NodeKind::StructDefinition {
                name,
                fields: Arc::new(fields),
                generic_params: Arc::new(generic_params),
                functions: Arc::new(functions),
                type_kind,
            },
            start,
            end,
        });

        let namespaced_name = self.get_namespaced_name(name_text);
        self.definition_indices.insert(namespaced_name, index);

        index
    }

    fn enum_definition(&mut self) -> usize {
        let start = self.token_start();

        assert_token!(self, TokenKind::Enum, start, self.token_end());
        self.position += 1;

        let name = self.name();

        assert_token!(self, TokenKind::LBrace, start, self.token_end());
        self.position += 1;

        let mut variant_names = Vec::new();

        while *self.token_kind() != TokenKind::RBrace {
            variant_names.push(self.name());

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            assert_token!(self, TokenKind::Comma, start, self.token_end());
            self.position += 1;
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBrace, start, end);
        self.position += 1;

        let variant_names = Arc::new(variant_names);

        let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
            parse_error!(self, "invalid enum name", start, end);
        };

        let type_kind = self.add_named_type(
            &name_text,
            TypeKind::Enum {
                name,
                variant_names: variant_names.clone(),
            },
            false,
        );

        self.add_node(Node {
            kind: NodeKind::EnumDefinition {
                name,
                variant_names,
                type_kind,
            },
            start,
            end,
        })
    }

    fn field(&mut self) -> usize {
        let start = self.token_start();
        let name = self.name();

        let type_name = self.type_name();
        let end = self.node_end(type_name);

        self.add_node(Node {
            kind: NodeKind::Field { name, type_name },
            start,
            end,
        })
    }

    fn parse_function_params(
        &mut self,
        start: Position,
        params: &mut Vec<usize>,
        param_type_kinds: &mut Vec<usize>,
    ) -> Option<usize> {
        if let Some(error_node) = self.assert_token(TokenKind::LParen, start, self.token_end()) {
            return Some(error_node);
        }
        self.position += 1;

        while *self.token_kind() != TokenKind::RParen {
            let param = self.param();
            params.push(param);

            let NodeKind::Param {
                type_name: param_type_name,
                ..
            } = self.nodes[param].kind
            else {
                return Some(self.parse_error("invalid parameter", start, self.token_end()));
            };

            let NodeKind::TypeName {
                type_kind: param_type_kind,
            } = self.nodes[param_type_name].kind
            else {
                return Some(self.parse_error(
                    "invalid parameter type kind",
                    start,
                    self.token_end(),
                ));
            };

            param_type_kinds.push(param_type_kind);

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            if let Some(error_node) = self.assert_token(TokenKind::Comma, start, self.token_end()) {
                return Some(error_node);
            }
            self.position += 1;
        }

        if let Some(error_node) = self.assert_token(TokenKind::RParen, start, self.token_end()) {
            return Some(error_node);
        }
        self.position += 1;

        None
    }

    fn parse_generic_params(
        &mut self,
        start: Position,
        generic_params: &mut Vec<usize>,
        generic_type_kinds: &mut Vec<usize>,
    ) -> Option<usize> {
        if let Some(error_node) = self.assert_token(TokenKind::Less, start, self.token_end()) {
            return Some(error_node);
        }
        self.position += 1;

        while *self.token_kind() != TokenKind::Greater {
            let name = self.name();
            generic_params.push(name);

            let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
                return Some(self.parse_error("invalid generic name", start, self.token_end()));
            };

            let type_kind = self.add_named_type(&name_text, TypeKind::Partial, true);
            generic_type_kinds.push(type_kind);

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            if let Some(error_node) = self.assert_token(TokenKind::Comma, start, self.token_end()) {
                return Some(error_node);
            }
            self.position += 1;
        }

        if let Some(error_node) = self.assert_token(TokenKind::Greater, start, self.token_end()) {
            return Some(error_node);
        }
        self.position += 1;

        None
    }

    fn function_declaration(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::Func, start, self.token_end());
        self.position += 1;

        let name = self.name();

        let mut generic_params = Vec::new();
        let mut generic_type_kinds = Vec::new();

        if *self.token_kind() == TokenKind::Less {
            if let Some(error_node) =
                self.parse_generic_params(start, &mut generic_params, &mut generic_type_kinds)
            {
                return error_node;
            }
        }

        let mut params = Vec::new();
        let mut param_type_kinds = Vec::new();

        if let Some(error_node) =
            self.parse_function_params(start, &mut params, &mut param_type_kinds)
        {
            return error_node;
        }

        let return_type_name = self.type_name();
        let end = self.node_end(return_type_name);

        let NodeKind::TypeName {
            type_kind: return_type_kind,
        } = self.nodes[return_type_name].kind
        else {
            parse_error!(self, "invalid parameter type kind", start, self.token_end());
        };

        let function_layout = FunctionLayout {
            param_type_kinds: Arc::new(param_type_kinds),
            generic_type_kinds: Arc::new(generic_type_kinds),
            return_type_kind,
        };
        let type_kind = get_function_type_kind(
            &mut self.type_kinds,
            &mut self.function_type_kinds,
            function_layout,
        );

        self.add_node(Node {
            kind: NodeKind::FunctionDeclaration {
                name,
                params: Arc::new(params),
                generic_params: Arc::new(generic_params),
                return_type_name,
                type_kind,
            },
            start,
            end,
        })
    }

    fn function(&mut self) -> usize {
        self.named_type_kinds.push();

        let start = self.token_start();
        let declaration = self.function_declaration();
        let statement = self.statement();
        let end = self.node_end(statement);

        self.named_type_kinds.pop();

        let NodeKind::FunctionDeclaration { name, .. } = self.nodes[declaration].kind else {
            parse_error!(self, "invalid function declaration", start, end);
        };

        let NodeKind::Name { text: name_text } = self.nodes[name].kind.clone() else {
            parse_error!(self, "invalid function name", start, self.node_end(name));
        };

        let index = self.add_node(Node {
            kind: NodeKind::Function {
                declaration,
                statement,
            },
            start,
            end,
        });

        let namespaced_name = self.get_namespaced_name(name_text);
        self.definition_indices.insert(namespaced_name, index);

        index
    }

    fn extern_function(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::Extern, start, self.token_end());
        self.position += 1;

        let declaration = self.function_declaration();
        let end = self.node_end(declaration);

        self.add_node(Node {
            kind: NodeKind::ExternFunction { declaration },
            start,
            end,
        })
    }

    fn param(&mut self) -> usize {
        let start = self.token_start();
        let name = self.name();

        let type_name = self.type_name();
        let end = self.node_end(type_name);

        self.add_node(Node {
            kind: NodeKind::Param { name, type_name },
            start,
            end,
        })
    }

    fn block(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::LBrace, start, self.token_end());
        self.position += 1;

        let mut statements = Vec::new();

        while *self.token_kind() != TokenKind::RBrace {
            statements.push(self.statement());
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBrace, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::Block {
                statements: Arc::new(statements),
            },
            start,
            end,
        })
    }

    fn statement(&mut self) -> usize {
        let start = self.token_start();
        let needs_semicolon = !matches!(
            self.token_kind(),
            TokenKind::Defer
                | TokenKind::If
                | TokenKind::Switch
                | TokenKind::While
                | TokenKind::For
                | TokenKind::LBrace
        );

        let inner = match self.token_kind() {
            TokenKind::Semicolon => None,
            TokenKind::Var | TokenKind::Val | TokenKind::Const => Some(self.variable_declaration()),
            TokenKind::Return => Some(self.return_statement()),
            TokenKind::Defer => Some(self.defer_statement()),
            TokenKind::If => Some(self.if_statement()),
            TokenKind::Switch => Some(self.switch_statement()),
            TokenKind::While => Some(self.while_loop()),
            TokenKind::For => Some(self.for_loop()),
            TokenKind::LBrace => Some(self.block()),
            _ => Some(self.expression()),
        };
        let mut end = if let Some(inner) = inner {
            self.node_end(inner)
        } else {
            start
        };

        if needs_semicolon {
            end = self.token_end();
            assert_token!(self, TokenKind::Semicolon, start, end);
            self.position += 1;
        }

        self.add_node(Node {
            kind: NodeKind::Statement { inner },
            start,
            end,
        })
    }

    fn variable_declaration(&mut self) -> usize {
        let start = self.token_start();
        let declaration_kind = match self.token_kind() {
            TokenKind::Var => DeclarationKind::Var,
            TokenKind::Val => DeclarationKind::Val,
            TokenKind::Const => DeclarationKind::Const,
            _ => parse_error!(
                self,
                "expected var, val, or const keyword in declaration",
                start,
                self.token_end()
            ),
        };
        self.position += 1;

        let name = self.name();

        let type_name = if *self.token_kind() != TokenKind::Equal {
            Some(self.type_name())
        } else {
            None
        };

        assert_token!(self, TokenKind::Equal, start, self.token_end());
        self.position += 1;

        let expression = if declaration_kind == DeclarationKind::Const {
            self.const_expression()
        } else {
            self.expression()
        };
        let end = self.node_end(expression);

        self.add_node(Node {
            kind: NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                type_name,
                expression,
            },
            start,
            end,
        })
    }

    fn return_statement(&mut self) -> usize {
        let start = self.token_start();
        let mut end = self.token_end();
        assert_token!(self, TokenKind::Return, start, end);
        self.position += 1;

        let mut expression = None;
        if *self.token_kind() != TokenKind::Semicolon {
            let index = self.expression();
            end = self.node_end(index);
            expression = Some(index);
        }

        self.add_node(Node {
            kind: NodeKind::ReturnStatement { expression },
            start,
            end,
        })
    }

    fn defer_statement(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::Defer, start, self.token_end());
        self.position += 1;

        let statement = self.statement();
        let end = self.node_end(statement);

        self.add_node(Node {
            kind: NodeKind::DeferStatement { statement },
            start,
            end,
        })
    }

    fn if_statement(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::If, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        let statement = self.statement();
        let mut end = self.node_end(statement);

        let next = if *self.token_kind() == TokenKind::Else {
            self.position += 1;

            let next = if *self.token_kind() == TokenKind::If {
                self.if_statement()
            } else {
                self.statement()
            };

            end = self.node_end(next);

            Some(next)
        } else {
            None
        };

        self.add_node(Node {
            kind: NodeKind::IfStatement {
                expression,
                statement,
                next,
            },
            start,
            end,
        })
    }

    fn switch_statement(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::Switch, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        let case_statement = self.case_statement();
        let end = self.node_end(case_statement);

        self.add_node(Node {
            kind: NodeKind::SwitchStatement {
                expression,
                case_statement,
            },
            start,
            end,
        })
    }

    fn case_statement(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::Case, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        let statement = self.statement();
        let mut end = self.node_end(statement);

        let next = if *self.token_kind() == TokenKind::Case {
            let next = self.case_statement();
            end = self.node_end(next);

            Some(next)
        } else if *self.token_kind() == TokenKind::Else {
            self.position += 1;

            let next = self.statement();
            end = self.node_end(next);

            Some(next)
        } else {
            None
        };

        self.add_node(Node {
            kind: NodeKind::CaseStatement {
                expression,
                statement,
                next,
            },
            start,
            end,
        })
    }

    fn while_loop(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::While, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        let statement = self.statement();
        let end = self.node_end(statement);

        self.add_node(Node {
            kind: NodeKind::WhileLoop {
                expression,
                statement,
            },
            start,
            end,
        })
    }

    fn for_loop(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::For, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let iterator = self.name();

        assert_token!(self, TokenKind::Of, start, self.token_end());
        self.position += 1;

        let from = self.term();

        let op = match *self.token_kind() {
            TokenKind::Less => Op::Less,
            TokenKind::LessEqual => Op::LessEqual,
            TokenKind::Greater => Op::Greater,
            TokenKind::GreaterEqual => Op::GreaterEqual,
            _ => parse_error!(
                self,
                "expected comparison operator",
                start,
                self.token_end()
            ),
        };
        self.position += 1;

        let to = self.term();

        let mut by = None;

        if *self.token_kind() == TokenKind::By {
            self.position += 1;

            by = Some(self.term());
        }

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        let statement = self.statement();
        let end = self.node_end(statement);

        self.add_node(Node {
            kind: NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                statement,
            },
            start,
            end,
        })
    }

    /*
     * Precedence (highest to lowest):
     *
     * Primary: Literals, identifiers, parenthesized expressions
     *
     * (Nestable, eg. &pointer^)
     * UnarySuffix: .<, .*, [], (), ., as, {}
     * UnaryPrefix: &, !, +, -
     *
     * (Chainable, eg. a * b / c)
     * Factor: *, /
     * Term: +, -
     * Inequality: <, <=, >, >=
     * Equality: ==, !=
     * Compound: &&, ||
     * Assignment: =, +=, -=, /=, *=
     */
    fn expression(&mut self) -> usize {
        self.assignment()
    }

    fn const_expression(&mut self) -> usize {
        let start = self.token_start();

        // Assignment is not allowed in const expressions.
        let inner = self.compound();
        let end = self.node_end(inner);

        self.add_node(Node {
            kind: NodeKind::ConstExpression { inner },
            start,
            end,
        })
    }

    fn assignment(&mut self) -> usize {
        let start = self.token_start();
        let mut left = self.compound();

        loop {
            let op = match *self.token_kind() {
                TokenKind::Equal => Op::Assign,
                TokenKind::PlusEqual => Op::PlusAssign,
                TokenKind::MinusEqual => Op::MinusAssign,
                TokenKind::MultiplyEqual => Op::MultiplyAssign,
                TokenKind::DivideEqual => Op::DivideAssign,
                _ => break,
            };

            self.position += 1;
            let right = self.compound();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary { left, op, right },
                start,
                end,
            })
        }

        left
    }

    fn compound(&mut self) -> usize {
        let start = self.token_start();
        let mut left = self.equality();

        loop {
            let op = match *self.token_kind() {
                TokenKind::And => Op::And,
                TokenKind::Or => Op::Or,
                _ => break,
            };
            self.position += 1;

            let right = self.equality();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary { left, op, right },
                start,
                end,
            })
        }

        left
    }

    fn equality(&mut self) -> usize {
        let start = self.token_start();
        let mut left = self.inequality();

        loop {
            let op = match *self.token_kind() {
                TokenKind::EqualEqual => Op::Equal,
                TokenKind::NotEqual => Op::NotEqual,
                _ => break,
            };
            self.position += 1;

            let right = self.inequality();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary { left, op, right },
                start,
                end,
            })
        }

        left
    }

    fn inequality(&mut self) -> usize {
        let start = self.token_start();
        let mut left = self.term();

        loop {
            let op = match *self.token_kind() {
                TokenKind::Less => Op::Less,
                TokenKind::LessEqual => Op::LessEqual,
                TokenKind::Greater => Op::Greater,
                TokenKind::GreaterEqual => Op::GreaterEqual,
                _ => break,
            };
            self.position += 1;

            let right = self.term();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary { left, op, right },
                start,
                end,
            })
        }

        left
    }

    fn term(&mut self) -> usize {
        let start = self.token_start();
        let mut left = self.factor();

        loop {
            let op = match *self.token_kind() {
                TokenKind::Plus => Op::Plus,
                TokenKind::Minus => Op::Minus,
                _ => break,
            };
            self.position += 1;

            let right = self.factor();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary { left, op, right },
                start,
                end,
            })
        }

        left
    }

    fn factor(&mut self) -> usize {
        let start = self.token_start();
        let mut left = self.unary_prefix();

        loop {
            let op = match *self.token_kind() {
                TokenKind::Asterisk => Op::Multiply,
                TokenKind::Divide => Op::Divide,
                _ => break,
            };
            self.position += 1;

            let right = self.unary_prefix();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary { left, op, right },
                start,
                end,
            })
        }

        left
    }

    fn unary_prefix(&mut self) -> usize {
        let start = self.token_start();
        let op = match *self.token_kind() {
            TokenKind::Not => Op::Not,
            TokenKind::Plus => Op::Plus,
            TokenKind::Minus => Op::Minus,
            TokenKind::Ampersand => Op::Reference,
            _ => return self.unary_suffix(),
        };
        self.position += 1;

        let right = self.unary_prefix();
        let end = self.node_end(right);
        self.add_node(Node {
            kind: NodeKind::UnaryPrefix { op, right },
            start,
            end,
        })
    }

    fn parse_generic_specifier_list(
        &mut self,
        start: Position,
        end: &mut Position,
        generic_param_type_kinds: &mut Vec<usize>,
    ) -> Option<usize> {
        if let Some(error_node) =
            self.assert_token(TokenKind::GenericSpecifier, start, self.token_end())
        {
            return Some(error_node);
        }
        self.position += 1;

        while *self.token_kind() != TokenKind::Greater {
            let mut type_kind = 0;
            if let Some(error_node) = self.type_kind(&mut type_kind) {
                return Some(error_node);
            }

            generic_param_type_kinds.push(type_kind);

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            if let Some(error_node) = self.assert_token(TokenKind::Comma, start, self.token_end()) {
                return Some(error_node);
            }
            self.position += 1;
        }

        *end = self.token_end();
        if let Some(error_node) = self.assert_token(TokenKind::Greater, start, *end) {
            return Some(error_node);
        }
        self.position += 1;

        None
    }

    fn unary_suffix(&mut self) -> usize {
        let start = self.token_start();
        let mut left = self.primary();

        loop {
            match *self.token_kind() {
                TokenKind::LParen => {
                    self.position += 1;

                    let mut args = Vec::new();

                    while *self.token_kind() != TokenKind::RParen {
                        args.push(self.expression());

                        if *self.token_kind() != TokenKind::Comma {
                            break;
                        }

                        assert_token!(self, TokenKind::Comma, start, self.token_end());
                        self.position += 1;
                    }

                    let end = self.token_end();
                    assert_token!(self, TokenKind::RParen, start, end);
                    self.position += 1;

                    left = self.add_node(Node {
                        kind: NodeKind::Call {
                            left,
                            args: Arc::new(args),
                        },
                        start,
                        end,
                    });
                }
                TokenKind::GenericSpecifier => {
                    let mut generic_param_type_kinds = Vec::new();
                    let mut end = self.token_end();
                    if let Some(error_node) = self.parse_generic_specifier_list(
                        start,
                        &mut end,
                        &mut generic_param_type_kinds,
                    ) {
                        return error_node;
                    }

                    left = self.add_node(Node {
                        kind: NodeKind::GenericSpecifier {
                            left,
                            generic_param_type_kinds: Arc::new(generic_param_type_kinds),
                        },
                        start,
                        end,
                    })
                }
                TokenKind::LBrace => {
                    left = self.struct_literal(left, start);
                }
                TokenKind::Period => {
                    self.position += 1;

                    let name = self.name();
                    let end = self.node_end(name);

                    left = self.add_node(Node {
                        kind: NodeKind::FieldAccess { left, name },
                        start,
                        end,
                    });
                }
                TokenKind::LBracket => {
                    self.position += 1;

                    let expression = self.expression();

                    let end = self.token_end();
                    assert_token!(self, TokenKind::RBracket, start, end);
                    self.position += 1;

                    left = self.add_node(Node {
                        kind: NodeKind::IndexAccess { left, expression },
                        start,
                        end,
                    });
                }
                TokenKind::As => {
                    self.position += 1;

                    let type_name = self.type_name();
                    let end = self.node_end(type_name);

                    left = self.add_node(Node {
                        kind: NodeKind::Cast { left, type_name },
                        start,
                        end,
                    });
                }
                TokenKind::Dereference => {
                    self.position += 1;

                    let end = self.token_end();

                    left = self.add_node(Node {
                        kind: NodeKind::UnarySuffix {
                            left,
                            op: Op::Dereference,
                        },
                        start,
                        end,
                    });
                }
                _ => break,
            }
        }

        left
    }

    fn struct_literal(&mut self, left: usize, start: Position) -> usize {
        assert_token!(self, TokenKind::LBrace, start, self.token_end());
        self.position += 1;

        let mut fields = Vec::new();

        while *self.token_kind() != TokenKind::RBrace {
            fields.push(self.field_literal());

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            assert_token!(self, TokenKind::Comma, start, self.token_end());
            self.position += 1;
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBrace, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::StructLiteral {
                left,
                fields: Arc::new(fields),
            },
            start,
            end,
        })
    }

    fn primary(&mut self) -> usize {
        let start = self.token_start();
        match *self.token_kind() {
            TokenKind::LParen => self.parenthesized_expression(),
            TokenKind::Identifier { .. } => self.identifier(),
            TokenKind::IntLiteral { .. } => self.int_literal(),
            TokenKind::Float32Literal { .. } => self.float32_literal(),
            TokenKind::CharLiteral { .. } => self.char_literal(),
            TokenKind::StringLiteral { .. } => self.string_literal(),
            TokenKind::True | TokenKind::False => self.bool_literal(),
            TokenKind::LBracket { .. } => self.array_literal(),
            TokenKind::Sizeof { .. } => self.type_size(),
            _ => parse_error!(
                self,
                "invalid token in primary value",
                start,
                self.token_end()
            ),
        }
    }

    fn parenthesized_expression(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        let end = self.token_end();
        assert_token!(self, TokenKind::RParen, start, end);
        self.position += 1;

        expression
    }

    fn identifier(&mut self) -> usize {
        let start = self.token_start();
        let name = self.name();
        let end = self.node_end(name);

        self.add_node(Node {
            kind: NodeKind::Identifier { name },
            start,
            end,
        })
    }

    fn name(&mut self) -> usize {
        let start = self.token_start();
        let end = self.token_end();
        let TokenKind::Identifier { text } = self.token_kind().clone() else {
            parse_error!(self, "expected identifier", start, end);
        };
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::Name { text },
            start,
            end,
        })
    }

    fn int_literal(&mut self) -> usize {
        let start = self.token_start();
        let end = self.token_end();
        let text = match self.token_kind() {
            TokenKind::IntLiteral { text } => text.clone(),
            _ => parse_error!(self, "expected int literal", start, end),
        };
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::IntLiteral { text },
            start,
            end,
        })
    }

    fn float32_literal(&mut self) -> usize {
        let start = self.token_start();
        let end = self.token_end();
        let text = match self.token_kind() {
            TokenKind::Float32Literal { text } => text.clone(),
            _ => parse_error!(self, "expected float32 literal", start, end),
        };
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::Float32Literal { text },
            start,
            end,
        })
    }

    fn string_literal(&mut self) -> usize {
        let start = self.token_start();
        let end = self.token_end();
        let text = match self.token_kind() {
            TokenKind::StringLiteral { text } => text.clone(),
            _ => parse_error!(self, "expected string literal", start, end),
        };
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::StringLiteral { text },
            start,
            end,
        })
    }

    fn bool_literal(&mut self) -> usize {
        let start = self.token_start();
        let end = self.token_end();
        let value = match self.token_kind() {
            TokenKind::True => true,
            TokenKind::False => false,
            _ => parse_error!(self, "expected bool literal", start, end),
        };
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::BoolLiteral { value },
            start,
            end,
        })
    }

    fn char_literal(&mut self) -> usize {
        let start = self.token_start();
        let end = self.token_end();
        let value = match self.token_kind() {
            TokenKind::CharLiteral { value } => *value,
            _ => parse_error!(self, "expected char literal", start, self.token_end()),
        };
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::CharLiteral { value },
            start,
            end,
        })
    }

    fn array_literal(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::LBracket, start, self.token_end());
        self.position += 1;

        let mut elements = Vec::new();

        while *self.token_kind() != TokenKind::RBracket {
            elements.push(self.expression());

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            assert_token!(self, TokenKind::Comma, start, self.token_end());
            self.position += 1;
        }

        let mut repeat_count_const_expression = None;

        if *self.token_kind() == TokenKind::Semicolon {
            self.position += 1;
            repeat_count_const_expression = Some(self.const_expression());
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBracket, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::ArrayLiteral {
                elements: Arc::new(elements),
                repeat_count_const_expression,
            },
            start,
            end,
        })
    }

    fn field_literal(&mut self) -> usize {
        let start = self.token_start();
        let name = self.name();

        assert_token!(self, TokenKind::Equal, start, self.token_end());
        self.position += 1;

        let expression = self.expression();
        let end = self.node_end(expression);

        self.add_node(Node {
            kind: NodeKind::FieldLiteral { name, expression },
            start,
            end,
        })
    }

    fn type_size(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::Sizeof, start, self.token_end());
        self.position += 1;

        let type_name = self.type_name();
        let end = self.node_end(type_name);

        self.add_node(Node {
            kind: NodeKind::TypeSize { type_name },
            start,
            end,
        })
    }

    fn type_name(&mut self) -> usize {
        let start = self.token_start();
        let mut type_kind = 0;
        if let Some(error_node) = self.type_kind(&mut type_kind) {
            return error_node;
        }
        let end = self.last_token_end();

        self.add_node(Node {
            kind: NodeKind::TypeName { type_kind },
            start,
            end,
        })
    }

    fn type_kind(&mut self, type_kind: &mut usize) -> Option<usize> {
        let start = self.token_start();
        let token_kind = self.token_kind().clone();

        match token_kind {
            TokenKind::Func => {
                self.position += 1;

                if let Some(error_node) =
                    self.assert_token(TokenKind::LParen, start, self.token_end())
                {
                    return Some(error_node);
                }
                self.position += 1;

                let mut param_type_kinds = Vec::new();

                while *self.token_kind() != TokenKind::RParen {
                    let mut param_type_kind = 0;
                    if let Some(error_node) = self.type_kind(&mut param_type_kind) {
                        return Some(error_node);
                    }

                    param_type_kinds.push(param_type_kind);

                    if *self.token_kind() != TokenKind::Comma {
                        break;
                    }

                    if let Some(error_node) =
                        self.assert_token(TokenKind::Comma, start, self.token_end())
                    {
                        return Some(error_node);
                    }
                    self.position += 1;
                }

                if let Some(error_node) =
                    self.assert_token(TokenKind::RParen, start, self.token_end())
                {
                    return Some(error_node);
                }
                self.position += 1;

                let mut return_type_kind = 0;
                if let Some(error_node) = self.type_kind(&mut return_type_kind) {
                    return Some(error_node);
                }

                let function_layout = FunctionLayout {
                    param_type_kinds: Arc::new(param_type_kinds),
                    generic_type_kinds: Arc::new(Vec::new()),
                    return_type_kind,
                };
                *type_kind = get_function_type_kind(
                    &mut self.type_kinds,
                    &mut self.function_type_kinds,
                    function_layout,
                );

                return None;
            }
            TokenKind::LBracket => {
                self.position += 1;

                let const_expression = self.const_expression();

                if let Some(error_node) =
                    self.assert_token(TokenKind::RBracket, start, self.token_end())
                {
                    return Some(error_node);
                }
                self.position += 1;

                let mut element_type_kind = 0;
                if let Some(error_node) = self.type_kind(&mut element_type_kind) {
                    return Some(error_node);
                }

                *type_kind = self.add_type(TypeKind::PartialArray {
                    element_type_kind,
                    const_expression,
                });

                return None;
            }
            TokenKind::Asterisk => {
                self.position += 1;

                let is_mutable = match *self.token_kind() {
                    TokenKind::Val => false,
                    TokenKind::Var => true,
                    _ => return Some(self.parse_error("expected val or var to be specified for pointer", start, self.token_end()))
                };
                self.position += 1;

                let mut inner_type_kind = 0;
                if let Some(error_node) = self.type_kind(&mut inner_type_kind) {
                    return Some(error_node);
                }

                *type_kind = get_type_kind_as_pointer(
                    &mut self.type_kinds,
                    &mut self.pointer_type_kinds,
                    inner_type_kind,
                    is_mutable,
                );

                return None;
            }
            TokenKind::Identifier { text } => {
                *type_kind = self.get_named_type(&text);
                self.position += 1;

                if *self.token_kind() == TokenKind::GenericSpecifier {
                    let mut generic_param_type_kinds = Vec::new();
                    let mut end = self.token_end();

                    if let Some(error_node) = self.parse_generic_specifier_list(
                        start,
                        &mut end,
                        &mut generic_param_type_kinds,
                    ) {
                        return Some(error_node);
                    }

                    *type_kind = self.add_type(TypeKind::PartialGeneric {
                        inner_type_kind: *type_kind,
                        generic_param_type_kinds: Arc::new(generic_param_type_kinds),
                    });
                }

                return None;
            }
            _ => {}
        }

        *type_kind = match token_kind {
            TokenKind::Int => INT_INDEX,
            TokenKind::String => STRING_INDEX,
            TokenKind::Bool => BOOL_INDEX,
            TokenKind::Char => CHAR_INDEX,
            TokenKind::Void => VOID_INDEX,
            TokenKind::UInt => UINT_INDEX,
            TokenKind::Int8 => INT8_INDEX,
            TokenKind::UInt8 => UINT8_INDEX,
            TokenKind::Int16 => INT16_INDEX,
            TokenKind::UInt16 => UINT16_INDEX,
            TokenKind::Int32 => INT32_INDEX,
            TokenKind::UInt32 => UINT32_INDEX,
            TokenKind::Int64 => INT64_INDEX,
            TokenKind::UInt64 => UINT64_INDEX,
            TokenKind::Float32 => FLOAT32_INDEX,
            TokenKind::Float64 => FLOAT64_INDEX,
            TokenKind::Tag => TAG_INDEX,
            _ => return Some(self.parse_error("expected type name", start, self.token_end())),
        };
        self.position += 1;

        None
    }

    fn add_named_type(
        &mut self,
        name_text: &Arc<str>,
        new_type_kind: TypeKind,
        is_generic: bool,
    ) -> usize {
        if let Some(type_kind) = self.named_type_kinds.get(name_text) {
            self.type_kinds[type_kind] = new_type_kind;
            type_kind
        } else {
            let type_kind = self.add_type(new_type_kind);
            if is_generic {
                self.named_type_kinds.insert(name_text.clone(), type_kind);
            } else {
                self.named_type_kinds
                    .insert_global(name_text.clone(), type_kind);
            }
            type_kind
        }
    }

    fn get_named_type(&mut self, name_text: &Arc<str>) -> usize {
        if let Some(type_kind) = self.named_type_kinds.get(name_text) {
            type_kind
        } else {
            let type_kind = self.add_type(TypeKind::Partial);
            self.named_type_kinds
                .insert_global(name_text.clone(), type_kind);
            type_kind
        }
    }

    fn get_namespaced_name(&mut self, name_text: Arc<str>) -> Vec<Arc<str>> {
        let mut namespaced_name = Vec::new();

        for namespace in &self.current_namespace_names {
            namespaced_name.push(namespace.clone());
        }

        namespaced_name.push(name_text);

        namespaced_name
    }
}
