use std::{collections::HashMap, hash::Hash, sync::Arc};

use crate::{
    lexer::{Token, TokenKind},
    position::Position,
};

// TODO: Should strings be refcounted strs instead?

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Plus,
    Minus,
    Multiply,
    Divide,
    Not,
    EqualEqual,
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

#[derive(Clone, Copy, Debug)]
pub struct TrailingComparison {
    pub op: Op,
    pub comparison: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TrailingBinary {
    pub op: Op,
    pub binary: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TrailingTerm {
    pub op: Op,
    pub term: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TrailingUnary {
    pub op: Op,
    pub unary: usize,
}

#[derive(Debug)]
pub struct Field {
    pub name: String,
    pub type_kind: usize,
}

// Used to search for the index of an array type by its layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArrayLayout {
    pub element_type_kind: usize,
    pub element_count: usize,
}

#[derive(Clone, Debug)]
pub enum TypeKind {
    Int,
    String,
    Bool,
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
    Pointer {
        inner_type_kind: usize,
    },
    Array {
        element_type_kind: usize,
        element_count: usize,
    },
    Struct {
        name: String,
        field_kinds: Arc<Vec<Field>>,
    },
    PartialStruct,
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    TopLevel {
        functions: Arc<Vec<usize>>,
        structs: Arc<Vec<usize>>,
    },
    StructDefinition {
        name: String,
        fields: Arc<Vec<usize>>,
        type_kind: usize,
    },
    Field {
        name: String,
        type_name: usize,
    },
    FunctionDeclaration {
        name: String,
        params: Arc<Vec<usize>>,
        return_type_name: usize,
    },
    Function {
        declaration: usize,
        block: usize,
    },
    ExternFunction {
        declaration: usize,
    },
    Param {
        name: String,
        type_name: usize,
    },
    Block {
        statements: Arc<Vec<usize>>,
    },
    Statement {
        inner: usize,
    },
    VariableDeclaration {
        is_mutable: bool,
        name: String,
        type_name: usize,
        expression: usize,
    },
    VariableAssignment {
        dereference_count: usize,
        variable: usize,
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
        block: usize,
    },
    WhileLoop {
        expression: usize,
        block: usize,
    },
    ForLoop {
        iterator: String,
        op: Op,
        from: usize,
        to: usize,
        by: Option<usize>,
        block: usize,
    },
    Expression {
        comparison: usize,
        trailing_comparisons: Arc<Vec<TrailingComparison>>,
    },
    Comparision {
        binary: usize,
        trailing_binary: Option<TrailingBinary>,
    },
    Binary {
        term: usize,
        trailing_terms: Arc<Vec<TrailingTerm>>,
    },
    Term {
        unary: usize,
        trailing_unaries: Arc<Vec<TrailingUnary>>,
    },
    Unary {
        op: Option<Op>,
        primary: usize,
    },
    Primary {
        inner: usize,
    },
    ParenthesizedExpression {
        expression: usize,
    },
    Variable {
        inner: usize,
    },
    VariableName {
        name: String,
    },
    VariableIndex {
        parent: usize,
        expression: usize,
    },
    VariableField {
        parent: usize,
        name: String,
    },
    FunctionCall {
        name: String,
        args: Arc<Vec<usize>>,
    },
    IntLiteral {
        text: String,
    },
    Float32Literal {
        text: String,
    },
    StringLiteral {
        text: String,
    },
    BoolLiteral {
        value: bool,
    },
    ArrayLiteral {
        elements: Arc<Vec<usize>>,
        repeat_count: usize,
    },
    StructLiteral {
        name: String,
        fields: Arc<Vec<usize>>,
    },
    FieldLiteral {
        name: String,
        expression: usize,
    },
    TypeSize {
        type_name: usize,
    },
    TypeName {
        type_kind: usize,
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
pub const VOID_INDEX: usize = 3;
pub const UINT_INDEX: usize = 4;
pub const INT8_INDEX: usize = 5;
pub const UINT8_INDEX: usize = 6;
pub const INT16_INDEX: usize = 7;
pub const UINT16_INDEX: usize = 8;
pub const INT32_INDEX: usize = 9;
pub const UINT32_INDEX: usize = 10;
pub const INT64_INDEX: usize = 11;
pub const UINT64_INDEX: usize = 12;
pub const FLOAT32_INDEX: usize = 13;
pub const FLOAT64_INDEX: usize = 14;

macro_rules! assert_token {
    ($self:ident, $token:expr, $start:expr, $end:expr) => {
        if *$self.token_kind() != $token {
            $self.print_error(&format!(
                "expected token {:?} but got token {:?}",
                $token,
                $self.token_kind()
            ));

            return $self.add_node(Node {
                kind: NodeKind::Error,
                start: $start,
                end: $end,
            });
        }
    };
}

macro_rules! parse_error {
    ($self:ident, $message:expr, $start:expr, $end:expr) => {{
        $self.print_error($message);

        return $self.add_node(Node {
            kind: NodeKind::Error,
            start: $start,
            end: $end,
        });
    }};
}

pub struct Parser {
    pub nodes: Vec<Node>,
    pub types: Vec<TypeKind>,
    pub function_declaration_indices: HashMap<String, usize>,
    pub struct_definition_indices: HashMap<String, usize>,
    pub array_type_kinds: HashMap<ArrayLayout, usize>,
    pub pointer_type_kinds: HashMap<usize, usize>,
    pub struct_type_kinds: HashMap<String, usize>,
    pub had_error: bool,

    pub tokens: Option<Vec<Token>>,
    pub position: usize,
}

impl Parser {
    pub fn new() -> Self {
        let mut parser = Self {
            tokens: None,
            nodes: Vec::new(),
            types: Vec::new(),
            function_declaration_indices: HashMap::new(),
            struct_definition_indices: HashMap::new(),
            array_type_kinds: HashMap::new(),
            pointer_type_kinds: HashMap::new(),
            struct_type_kinds: HashMap::new(),
            had_error: true,
            position: 0,
        };

        parser.add_type(TypeKind::Int);
        parser.add_type(TypeKind::String);
        parser.add_type(TypeKind::Bool);
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

    fn peek_token_kind(&self) -> &TokenKind {
        let tokens = self.tokens.as_ref().unwrap();
        if let Some(token) = tokens.get(self.position + 1) {
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
        tokens[self.position].start
    }

    fn token_end(&self) -> Position {
        let tokens = self.tokens.as_ref().unwrap();
        tokens[self.position].end
    }

    fn parse_uint_literal(&mut self) -> Option<usize> {
        let int_string = match self.token_kind() {
            TokenKind::IntLiteral { text } => text,
            _ => {
                return None;
            }
        };

        let Ok(value) = int_string.parse::<usize>() else {
            return None;
        };

        Some(value)
    }

    fn add_node(&mut self, node: Node) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    fn add_type(&mut self, type_kind: TypeKind) -> usize {
        let index = self.types.len();
        self.types.push(type_kind);
        index
    }

    // TODO: Rename to error().
    fn print_error(&mut self, message: &str) {
        self.had_error = true;
        println!(
            "Syntax error at line {}, column {}: {}",
            self.token_start().line,
            self.token_start().column,
            message,
        );
    }

    pub fn parse(&mut self, tokens: Vec<Token>) -> usize {
        self.position = 0;
        self.tokens = Some(tokens);
        self.top_level()
    }

    fn top_level(&mut self) -> usize {
        let mut functions = Vec::new();
        let mut structs = Vec::new();
        let start = self.token_start();
        let mut end = self.token_end();

        while *self.token_kind() != TokenKind::Eof {
            let index;

            match *self.token_kind() {
                TokenKind::Fun => {
                    index = self.function();
                    functions.push(index);
                }
                TokenKind::Extern => {
                    index = self.extern_function();
                    functions.push(index);
                }
                TokenKind::Struct => {
                    index = self.struct_definition();
                    structs.push(index);
                }
                _ => parse_error!(self, "unexpected token at top level", start, end),
            }

            end = self.node_end(index);
        }

        self.add_node(Node {
            kind: NodeKind::TopLevel {
                functions: Arc::new(functions),
                structs: Arc::new(structs),
            },
            start,
            end,
        })
    }

    fn struct_definition(&mut self) -> usize {
        let start = self.token_start();

        assert_token!(self, TokenKind::Struct, start, self.token_end());
        self.position += 1;

        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected struct name", start, self.token_end()),
        };
        self.position += 1;

        assert_token!(self, TokenKind::LBrace, start, self.token_end());
        self.position += 1;

        let mut fields = Vec::new();

        while *self.token_kind() != TokenKind::RBrace {
            fields.push(self.field());

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            assert_token!(self, TokenKind::Comma, start, self.token_end());
            self.position += 1;
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBrace, start, end);
        self.position += 1;

        let mut field_kinds = Vec::new();

        for field in &fields {
            let Node {
                kind: NodeKind::Field { name, type_name },
                ..
            } = &self.nodes[*field]
            else {
                parse_error!(self, "invalid struct field", start, end);
            };

            let Node {
                kind: NodeKind::TypeName { type_kind },
                ..
            } = &self.nodes[*type_name]
            else {
                parse_error!(self, "invalid struct field type name", start, end);
            };

            field_kinds.push(Field {
                name: name.clone(),
                type_kind: *type_kind,
            });
        }

        let type_kind_struct = TypeKind::Struct {
            name: name.clone(),
            field_kinds: Arc::new(field_kinds),
        };

        let type_kind = if let Some(type_kind) = self.struct_type_kinds.get(&name) {
            self.types[*type_kind] = type_kind_struct;
            *type_kind
        } else {
            let type_kind = self.add_type(type_kind_struct);
            self.struct_type_kinds.insert(name.clone(), type_kind);
            type_kind
        };

        let index = self.add_node(Node {
            kind: NodeKind::StructDefinition {
                name: name.clone(),
                fields: Arc::new(fields),
                type_kind,
            },
            start,
            end,
        });
        self.struct_definition_indices.insert(name, index);

        index
    }

    fn field(&mut self) -> usize {
        let start = self.token_start();
        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected field name", start, self.token_end()),
        };
        self.position += 1;

        assert_token!(self, TokenKind::Colon, start, self.token_end());
        self.position += 1;

        let type_name = self.type_name();
        let end = self.node_end(type_name);

        self.add_node(Node {
            kind: NodeKind::Field { name, type_name },
            start,
            end,
        })
    }

    fn function_declaration(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::Fun, start, self.token_end());
        self.position += 1;

        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected function name", start, self.token_end()),
        };
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let mut params = Vec::new();

        while *self.token_kind() != TokenKind::RParen {
            params.push(self.param());

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            assert_token!(self, TokenKind::Comma, start, self.token_end());
            self.position += 1;
        }

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::Colon, start, self.token_end());
        self.position += 1;

        let return_type_name = self.type_name();
        let end = self.node_end(return_type_name);

        let index = self.add_node(Node {
            kind: NodeKind::FunctionDeclaration {
                name: name.clone(),
                return_type_name,
                params: Arc::new(params),
            },
            start,
            end,
        });
        self.function_declaration_indices.insert(name, index);

        index
    }

    fn function(&mut self) -> usize {
        let start = self.token_start();
        let declaration = self.function_declaration();
        let block = self.block();
        let end = self.node_end(block);

        self.add_node(Node {
            kind: NodeKind::Function { declaration, block },
            start,
            end,
        })
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
        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected param name", start, self.token_end()),
        };
        self.position += 1;

        assert_token!(self, TokenKind::Colon, start, self.token_end());
        self.position += 1;

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
                | TokenKind::While
                | TokenKind::For
                | TokenKind::LBrace
        );

        let inner = match self.token_kind() {
            TokenKind::Var | TokenKind::Val => self.variable_declaration(),
            TokenKind::Identifier { .. } if *self.peek_token_kind() == TokenKind::LParen => {
                self.function_call()
            }
            TokenKind::Return => self.return_statement(),
            TokenKind::Defer => self.defer_statement(),
            TokenKind::If => self.if_statement(),
            TokenKind::While => self.while_loop(),
            TokenKind::For => self.for_loop(),
            TokenKind::LBrace => self.block(),
            _ => self.variable_assignment(),
        };
        let mut end = self.node_end(inner);

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
        let is_mutable = match self.token_kind() {
            TokenKind::Var => true,
            TokenKind::Val => false,
            _ => parse_error!(self, "expected var or val keyword in declaration", start, self.token_end()),
        };
        self.position += 1;

        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected variable name in declaration", start, self.token_end()),
        };
        self.position += 1;

        assert_token!(self, TokenKind::Colon, start, self.token_end());
        self.position += 1;

        let type_name = self.type_name();

        assert_token!(self, TokenKind::Equal, start, self.token_end());
        self.position += 1;

        let expression = self.expression(true);
        let end = self.node_end(expression);

        self.add_node(Node {
            kind: NodeKind::VariableDeclaration {
                is_mutable,
                name,
                type_name,
                expression,
            },
            start,
            end,
        })
    }

    fn variable_assignment(&mut self) -> usize {
        let start = self.token_start();
        let mut dereference_count = 0;
        while *self.token_kind() == TokenKind::Asterisk {
            dereference_count += 1;
            self.position += 1;
        }

        let variable = self.variable();

        assert_token!(self, TokenKind::Equal, start, self.token_end());
        self.position += 1;

        let expression = self.expression(true);
        let end = self.node_end(expression);

        self.add_node(Node {
            kind: NodeKind::VariableAssignment {
                dereference_count,
                variable,
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
            let index = self.expression(true);
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

        let expression = self.expression(false);
        let block = self.block();
        let end = self.node_end(block);

        self.add_node(Node {
            kind: NodeKind::IfStatement { expression, block },
            start,
            end,
        })
    }

    fn while_loop(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::While, start, self.token_end());
        self.position += 1;

        let expression = self.expression(false);
        let block = self.block();
        let end = self.node_end(block);

        self.add_node(Node {
            kind: NodeKind::WhileLoop { expression, block },
            start,
            end,
        })
    }

    fn for_loop(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::For, start, self.token_end());
        self.position += 1;

        let iterator = if let TokenKind::Identifier { text } = self.token_kind() {
            text.clone()
        } else {
            parse_error!(self, "expected iterator name", start, self.token_end());
        };
        self.position += 1;

        assert_token!(self, TokenKind::In, start, self.token_end());
        self.position += 1;

        let from = self.binary(true);

        let op = match *self.token_kind() {
            TokenKind::Less => Op::Less,
            TokenKind::LessEqual => Op::LessEqual,
            TokenKind::Greater => Op::Greater,
            TokenKind::GreaterEqual => Op::GreaterEqual,
            _ => parse_error!(self, "expected comparison operator", start, self.token_end()),
        };
        self.position += 1;

        let to = self.binary(true);

        let mut by = None;

        if *self.token_kind() == TokenKind::By {
            self.position += 1;

            by = Some(self.binary(true));
        }

        let block = self.block();
        let end = self.node_end(block);

        self.add_node(Node {
            kind: NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                block,
            },
            start,
            end,
        })
    }

    fn expression(&mut self, allow_struct_literal: bool) -> usize {
        let start = self.token_start();
        let comparison = self.comparison(allow_struct_literal);
        let mut end = self.node_end(comparison);
        let mut trailing_comparisons = Vec::new();

        while *self.token_kind() == TokenKind::And || *self.token_kind() == TokenKind::Or {
            let op = if *self.token_kind() == TokenKind::And {
                Op::And
            } else {
                Op::Or
            };
            self.position += 1;

            let comparison = self.comparison(true);
            end = self.node_end(comparison);
            trailing_comparisons.push(TrailingComparison { op, comparison });
        }

        self.add_node(Node {
            kind: NodeKind::Expression {
                comparison,
                trailing_comparisons: Arc::new(trailing_comparisons),
            },
            start,
            end,
        })
    }

    fn comparison(&mut self, allow_struct_literal: bool) -> usize {
        let start = self.token_start();
        let binary = self.binary(allow_struct_literal);
        let mut end = self.node_end(binary);
        let mut trailing_binary = None;

        let op = match *self.token_kind() {
            TokenKind::Less => Some(Op::Less),
            TokenKind::Greater => Some(Op::Greater),
            TokenKind::EqualEqual => Some(Op::EqualEqual),
            TokenKind::NotEqual => Some(Op::NotEqual),
            TokenKind::LessEqual => Some(Op::LessEqual),
            TokenKind::GreaterEqual => Some(Op::GreaterEqual),
            _ => None,
        };

        if let Some(op) = op {
            self.position += 1;
            let binary = self.binary(true);
            end = self.node_end(binary);
            trailing_binary = Some(TrailingBinary { op, binary });
        }

        self.add_node(Node {
            kind: NodeKind::Comparision {
                binary,
                trailing_binary,
            },
            start,
            end,
        })
    }

    fn binary(&mut self, allow_struct_literal: bool) -> usize {
        let start = self.token_start();
        let term = self.term(allow_struct_literal);
        let mut end = self.node_end(term);
        let mut trailing_terms = Vec::new();

        while *self.token_kind() == TokenKind::Plus || *self.token_kind() == TokenKind::Minus {
            let op = if *self.token_kind() == TokenKind::Plus {
                Op::Plus
            } else {
                Op::Minus
            };
            self.position += 1;

            let term = self.term(true);
            end = self.node_end(term);
            trailing_terms.push(TrailingTerm { op, term });
        }

        self.add_node(Node {
            kind: NodeKind::Binary {
                term,
                trailing_terms: Arc::new(trailing_terms),
            },
            start,
            end,
        })
    }

    fn term(&mut self, allow_struct_literal: bool) -> usize {
        let start = self.token_start();
        let unary = self.unary(allow_struct_literal);
        let mut end = self.node_end(unary);
        let mut trailing_unaries = Vec::new();

        while *self.token_kind() == TokenKind::Asterisk || *self.token_kind() == TokenKind::Divide {
            let op = if *self.token_kind() == TokenKind::Asterisk {
                Op::Multiply
            } else {
                Op::Divide
            };
            self.position += 1;

            let unary = self.unary(true);
            end = self.node_end(unary);
            trailing_unaries.push(TrailingUnary { op, unary });
        }

        self.add_node(Node {
            kind: NodeKind::Term {
                unary,
                trailing_unaries: Arc::new(trailing_unaries),
            },
            start,
            end,
        })
    }

    fn unary(&mut self, allow_struct_literal: bool) -> usize {
        let start = self.token_start();
        let op = match *self.token_kind() {
            TokenKind::Plus => {
                self.position += 1;
                Some(Op::Plus)
            }
            TokenKind::Minus => {
                self.position += 1;
                Some(Op::Minus)
            }
            TokenKind::Not => {
                self.position += 1;
                Some(Op::Not)
            }
            TokenKind::Ampersand => {
                self.position += 1;
                Some(Op::Reference)
            }
            TokenKind::Asterisk => {
                self.position += 1;
                Some(Op::Dereference)
            }
            _ => None,
        };

        let primary = self.primary(allow_struct_literal);
        let end = self.node_end(primary);

        self.add_node(Node {
            kind: NodeKind::Unary { op, primary },
            start,
            end,
        })
    }

    fn primary(&mut self, allow_struct_literal: bool) -> usize {
        let start = self.token_start();
        let inner = match *self.token_kind() {
            TokenKind::LParen => self.parenthesized_expression(),
            TokenKind::Identifier { .. } if *self.peek_token_kind() == TokenKind::LParen => {
                self.function_call()
            }
            TokenKind::Identifier { .. }
                if allow_struct_literal && *self.peek_token_kind() == TokenKind::LBrace =>
            {
                self.struct_literal()
            }
            TokenKind::Identifier { .. } => self.variable(),
            TokenKind::IntLiteral { .. } => self.int_literal(),
            TokenKind::Float32Literal { .. } => self.float32_literal(),
            TokenKind::StringLiteral { .. } => self.string_literal(),
            TokenKind::True | TokenKind::False => self.bool_literal(),
            TokenKind::LBracket { .. } => self.array_literal(),
            TokenKind::Sizeof { .. } => self.type_size(),
            _ => parse_error!(self, "invalid token in primary value", start, self.token_end()),
        };
        let end = self.node_end(inner);

        self.add_node(Node {
            kind: NodeKind::Primary { inner },
            start,
            end,
        })
    }

    fn parenthesized_expression(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression(true);

        let end = self.token_end();
        assert_token!(self, TokenKind::RParen, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::ParenthesizedExpression { expression },
            start,
            end,
        })
    }

    fn variable(&mut self) -> usize {
        let start = self.token_start();
        let mut end = self.token_end();
        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected variable name", start, end),
        };
        self.position += 1;

        let mut inner = self.add_node(Node {
            kind: NodeKind::VariableName { name },
            start,
            end,
        });

        loop {
            if *self.token_kind() == TokenKind::LBracket {
                self.position += 1;
                let expression = self.expression(true);
                end = self.token_end();
                assert_token!(self, TokenKind::RBracket, start, end);
                self.position += 1;

                inner = self.add_node(Node {
                    kind: NodeKind::VariableIndex {
                        parent: inner,
                        expression,
                    },
                    start,
                    end,
                });
                continue;
            }

            if *self.token_kind() == TokenKind::Period {
                self.position += 1;
                end = self.token_end();

                let field_name = match self.token_kind() {
                    TokenKind::Identifier { text } => text.clone(),
                    _ => parse_error!(self, "expected field name", start, end),
                };
                self.position += 1;

                inner = self.add_node(Node {
                    kind: NodeKind::VariableField {
                        parent: inner,
                        name: field_name,
                    },
                    start,
                    end,
                });
                continue;
            }

            break;
        }

        self.add_node(Node {
            kind: NodeKind::Variable { inner },
            start,
            end,
        })
    }

    fn function_call(&mut self) -> usize {
        let start = self.token_start();
        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected function name", start, self.token_end()),
        };
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let mut args = Vec::new();

        while *self.token_kind() != TokenKind::RParen {
            args.push(self.expression(true));

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            assert_token!(self, TokenKind::Comma, start, self.token_end());
            self.position += 1;
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RParen, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::FunctionCall {
                name,
                args: Arc::new(args),
            },
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

    fn array_literal(&mut self) -> usize {
        let start = self.token_start();
        assert_token!(self, TokenKind::LBracket, start, self.token_end());
        self.position += 1;

        let mut elements = Vec::new();

        while *self.token_kind() != TokenKind::RBracket {
            elements.push(self.expression(true));

            if *self.token_kind() != TokenKind::Comma {
                break;
            }

            assert_token!(self, TokenKind::Comma, start, self.token_end());
            self.position += 1;
        }

        let mut repeat_count = 1;

        if *self.token_kind() == TokenKind::Semicolon {
            self.position += 1;

            repeat_count = self.parse_uint_literal().unwrap_or_else(|| parse_error!(self, "expected uint literal", start, self.token_end()));
            self.position += 1;
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBracket, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::ArrayLiteral {
                elements: Arc::new(elements),
                repeat_count,
            },
            start,
            end,
        })
    }

    fn struct_literal(&mut self) -> usize {
        let start = self.token_start();
        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected struct name", start, self.token_end()),
        };
        self.position += 1;

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
                name,
                fields: Arc::new(fields),
            },
            start,
            end,
        })
    }

    fn field_literal(&mut self) -> usize {
        let start = self.token_start();
        let name = match self.token_kind() {
            TokenKind::Identifier { text } => text.clone(),
            _ => parse_error!(self, "expected field name", start, self.token_end()),
        };
        self.position += 1;

        assert_token!(self, TokenKind::Colon, start, self.token_end());
        self.position += 1;

        let expression = self.expression(true);
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
        let mut end = self.token_end();
        let mut type_kind = match self.token_kind().clone() {
            TokenKind::Int => INT_INDEX,
            TokenKind::String => STRING_INDEX,
            TokenKind::Bool => BOOL_INDEX,
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
            TokenKind::Identifier { text } => {
                if let Some(type_kind) = self.struct_type_kinds.get(&text) {
                    *type_kind
                } else {
                    let type_kind = self.add_type(TypeKind::PartialStruct);
                    self.struct_type_kinds.insert(text.clone(), type_kind);
                    type_kind
                }
            }
            _ => parse_error!(self, "expected type name", start, end),
        };
        self.position += 1;

        loop {
            if *self.token_kind() == TokenKind::LBracket {
                self.position += 1;

                let length = self.parse_uint_literal().unwrap_or_else(|| parse_error!(self, "expected uint literal", start, end));
                self.position += 1;

                end = self.token_end();
                assert_token!(self, TokenKind::RBracket, start, end);
                self.position += 1;

                let array_layout = ArrayLayout {
                    element_type_kind: type_kind,
                    element_count: length,
                };

                type_kind = if let Some(index) = self.array_type_kinds.get(&array_layout) {
                    *index
                } else {
                    let index = self.add_type(TypeKind::Array {
                        element_type_kind: type_kind,
                        element_count: length,
                    });
                    self.array_type_kinds.insert(array_layout, index);
                    index
                };

                continue;
            }

            if *self.token_kind() == TokenKind::Asterisk {
                end = self.token_end();
                self.position += 1;

                type_kind = if let Some(index) = self.pointer_type_kinds.get(&type_kind) {
                    *index
                } else {
                    let index = self.add_type(TypeKind::Pointer {
                        inner_type_kind: type_kind,
                    });
                    self.pointer_type_kinds.insert(type_kind, index);
                    index
                };

                continue;
            }

            break;
        }

        self.add_node(Node {
            kind: NodeKind::TypeName { type_kind },
            start,
            end,
        })
    }
}
