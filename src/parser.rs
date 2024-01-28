use std::{collections::HashMap, hash::Hash, sync::Arc};

use crate::lexer::TokenKind;

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
    Array {
        element_type_kind: usize,
        element_count: usize,
    },
    Struct {
        name: String,
        fields_kinds: Arc<Vec<Field>>,
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
    TypeName {
        type_kind: usize,
    },
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

pub struct Parser {
    pub nodes: Vec<NodeKind>,
    pub types: Vec<TypeKind>,
    pub function_declaration_indices: HashMap<String, usize>,
    pub struct_definition_indices: HashMap<String, usize>,
    pub array_type_kinds: HashMap<ArrayLayout, usize>,
    pub struct_type_kinds: HashMap<String, usize>,

    pub tokens: Option<Vec<TokenKind>>,
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
            struct_type_kinds: HashMap::new(),
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

    fn token(&self) -> &TokenKind {
        let tokens = self.tokens.as_ref().unwrap();
        tokens.get(self.position).unwrap_or(&TokenKind::Eof)
    }

    fn peek_token(&self) -> &TokenKind {
        let tokens = self.tokens.as_ref().unwrap();
        tokens
            .get(self.position + 1)
            .unwrap_or(&TokenKind::Eof)
    }

    fn assert_token(&self, token: TokenKind) {
        if *self.token() != token {
            panic!(
                "Expected token {:?} but got token {:?}",
                token,
                self.token()
            );
        }
    }

    fn parse_uint_literal(&self) -> usize {
        let int_string = match self.token() {
            TokenKind::IntLiteral { text } => text,
            _ => panic!("Expected int literal"),
        };

        int_string.parse::<usize>().expect("Expected unsigned integer")
    }

    fn add_node(&mut self, node: NodeKind) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);
        index
    }

    fn add_type(&mut self, type_kind: TypeKind) -> usize {
        let index = self.types.len();
        self.types.push(type_kind);
        index
    }

    pub fn parse(&mut self, tokens: Vec<TokenKind>) -> usize {
        self.position = 0;
        self.tokens = Some(tokens);
        self.top_level()
    }

    fn top_level(&mut self) -> usize {
        let mut functions = Vec::new();
        let mut structs = Vec::new();

        while *self.token() != TokenKind::Eof {
            match *self.token() {
                TokenKind::Fun => functions.push(self.function()),
                TokenKind::Extern => functions.push(self.extern_function()),
                TokenKind::Struct => structs.push(self.struct_definition()),
                _ => panic!("Unexpected token at top level"),
            }
        }

        self.add_node(NodeKind::TopLevel {
            functions: Arc::new(functions),
            structs: Arc::new(structs),
        })
    }

    fn struct_definition(&mut self) -> usize {
        self.assert_token(TokenKind::Struct);
        self.position += 1;

        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected struct name"),
        };
        self.position += 1;

        self.assert_token(TokenKind::LBrace);
        self.position += 1;

        let mut fields = Vec::new();

        while *self.token() != TokenKind::RBrace {
            fields.push(self.field());

            if *self.token() != TokenKind::Comma {
                break;
            }

            self.assert_token(TokenKind::Comma);
            self.position += 1;
        }

        self.assert_token(TokenKind::RBrace);
        self.position += 1;

        let mut field_kinds = Vec::new();

        for field in &fields {
            let NodeKind::Field { name, type_name } = &self.nodes[*field] else {
                panic!("Invalid struct field");
            };

            let NodeKind::TypeName { type_kind } = &self.nodes[*type_name] else {
                panic!("Invalid struct field type name");
            };

            field_kinds.push(Field {
                name: name.clone(),
                type_kind: *type_kind,
            });
        }

        let type_kind_struct = TypeKind::Struct {
            name: name.clone(),
            fields_kinds: Arc::new(field_kinds),
        };

        let type_kind = if let Some(type_kind) = self.struct_type_kinds.get(&name) {
            self.types[*type_kind] = type_kind_struct;
            *type_kind
        } else {
            let type_kind = self.add_type(type_kind_struct);
            self.struct_type_kinds.insert(name.clone(), type_kind);
            type_kind
        };

        let index = self.add_node(NodeKind::StructDefinition {
            name: name.clone(),
            fields: Arc::new(fields),
            type_kind,
        });
        self.struct_definition_indices.insert(name, index);

        index
    }

    fn field(&mut self) -> usize {
        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected field name"),
        };
        self.position += 1;

        self.assert_token(TokenKind::Colon);
        self.position += 1;

        let type_name = self.type_name();

        self.add_node(NodeKind::Field { name, type_name })
    }

    fn function_declaration(&mut self) -> usize {
        self.assert_token(TokenKind::Fun);
        self.position += 1;

        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected function name"),
        };
        self.position += 1;

        self.assert_token(TokenKind::LParen);
        self.position += 1;

        let mut params = Vec::new();

        while *self.token() != TokenKind::RParen {
            params.push(self.param());

            if *self.token() != TokenKind::Comma {
                break;
            }

            self.assert_token(TokenKind::Comma);
            self.position += 1;
        }

        self.assert_token(TokenKind::RParen);
        self.position += 1;

        self.assert_token(TokenKind::Colon);
        self.position += 1;

        let return_type_name = self.type_name();

        let index = self.add_node(NodeKind::FunctionDeclaration {
            name: name.clone(),
            return_type_name,
            params: Arc::new(params),
        });
        self.function_declaration_indices.insert(name, index);

        index
    }

    fn function(&mut self) -> usize {
        let declaration = self.function_declaration();
        let block = self.block();

        self.add_node(NodeKind::Function { declaration, block })
    }

    fn extern_function(&mut self) -> usize {
        self.assert_token(TokenKind::Extern);
        self.position += 1;

        let declaration = self.function_declaration();

        self.add_node(NodeKind::ExternFunction { declaration })
    }

    fn param(&mut self) -> usize {
        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected param name"),
        };
        self.position += 1;

        self.assert_token(TokenKind::Colon);
        self.position += 1;

        let type_name = self.type_name();

        self.add_node(NodeKind::Param { name, type_name })
    }

    fn block(&mut self) -> usize {
        self.assert_token(TokenKind::LBrace);
        self.position += 1;

        let mut statements = Vec::new();

        while *self.token() != TokenKind::RBrace {
            statements.push(self.statement());
        }

        self.assert_token(TokenKind::RBrace);
        self.position += 1;

        self.add_node(NodeKind::Block {
            statements: Arc::new(statements),
        })
    }

    fn statement(&mut self) -> usize {
        let needs_semicolon = !matches!(
            self.token(),
            TokenKind::Defer | TokenKind::If | TokenKind::While | TokenKind::For | TokenKind::LBrace
        );

        let inner = match self.token() {
            TokenKind::Var | TokenKind::Val => self.variable_declaration(),
            TokenKind::Identifier { .. } if *self.peek_token() == TokenKind::LParen => {
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

        if needs_semicolon {
            self.assert_token(TokenKind::Semicolon);
            self.position += 1;
        }

        self.add_node(NodeKind::Statement { inner })
    }

    fn variable_declaration(&mut self) -> usize {
        let is_mutable = match self.token() {
            TokenKind::Var => true,
            TokenKind::Val => false,
            _ => panic!("Expected var or val keyword in declaration"),
        };
        self.position += 1;

        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected variable name in declaration"),
        };
        self.position += 1;

        self.assert_token(TokenKind::Colon);
        self.position += 1;

        let type_name = self.type_name();

        self.assert_token(TokenKind::Equal);
        self.position += 1;

        let expression = self.expression(true);

        self.add_node(NodeKind::VariableDeclaration {
            is_mutable,
            name,
            type_name,
            expression,
        })
    }

    fn variable_assignment(&mut self) -> usize {
        let variable = self.variable();

        self.assert_token(TokenKind::Equal);
        self.position += 1;

        let expression = self.expression(true);

        self.add_node(NodeKind::VariableAssignment {
            variable,
            expression,
        })
    }

    fn return_statement(&mut self) -> usize {
        self.assert_token(TokenKind::Return);
        self.position += 1;

        let mut expression = None;
        if *self.token() != TokenKind::Semicolon {
            expression = Some(self.expression(true));
        }

        self.add_node(NodeKind::ReturnStatement { expression })
    }

    fn defer_statement(&mut self) -> usize {
        self.assert_token(TokenKind::Defer);
        self.position += 1;

        let statement = self.statement();

        self.add_node(NodeKind::DeferStatement { statement })
    }

    fn if_statement(&mut self) -> usize {
        self.assert_token(TokenKind::If);
        self.position += 1;

        let expression = self.expression(false);
        let block = self.block();

        self.add_node(NodeKind::IfStatement { expression, block })
    }

    fn while_loop(&mut self) -> usize {
        self.assert_token(TokenKind::While);
        self.position += 1;

        let expression = self.expression(false);
        let block = self.block();

        self.add_node(NodeKind::WhileLoop { expression, block })
    }

    fn for_loop(&mut self) -> usize {
        self.assert_token(TokenKind::For);
        self.position += 1;

        let iterator = if let TokenKind::Identifier { text } = self.token() {
            text.clone()
        } else {
            panic!("Expected iterator name");
        };
        self.position += 1;

        self.assert_token(TokenKind::In);
        self.position += 1;

        let from = self.binary(true);

        let op = match *self.token() {
            TokenKind::Less => Op::Less,
            TokenKind::LessEqual => Op::LessEqual,
            TokenKind::Greater => Op::Greater,
            TokenKind::GreaterEqual => Op::GreaterEqual,
            _ => panic!("Expected comparison operator")
        };
        self.position += 1;

        let to = self.binary(true);

        let mut by = None;

        if *self.token() == TokenKind::By {
            self.position += 1;

            by = Some(self.binary(true));
        }

        let block = self.block();

        self.add_node(NodeKind::ForLoop { iterator, op, from, to, by, block })
    }

    fn expression(&mut self, allow_struct_literal: bool) -> usize {
        let comparison = self.comparison(allow_struct_literal);
        let mut trailing_comparisons = Vec::new();

        while *self.token() == TokenKind::And || *self.token() == TokenKind::Or {
            let op = if *self.token() == TokenKind::And {
                Op::And
            } else {
                Op::Or
            };
            self.position += 1;

            trailing_comparisons.push(TrailingComparison {
                op,
                comparison: self.comparison(true),
            });
        }

        self.add_node(NodeKind::Expression {
            comparison,
            trailing_comparisons: Arc::new(trailing_comparisons),
        })
    }

    fn comparison(&mut self, allow_struct_literal: bool) -> usize {
        let binary = self.binary(allow_struct_literal);
        let mut trailing_binary = None;

        let op = match *self.token() {
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
            trailing_binary = Some(TrailingBinary {
                op,
                binary: self.binary(true),
            });
        }

        self.add_node(NodeKind::Comparision {
            binary,
            trailing_binary,
        })
    }

    fn binary(&mut self, allow_struct_literal: bool) -> usize {
        let term = self.term(allow_struct_literal);
        let mut trailing_terms = Vec::new();

        while *self.token() == TokenKind::Plus || *self.token() == TokenKind::Minus {
            let op = if *self.token() == TokenKind::Plus {
                Op::Plus
            } else {
                Op::Minus
            };
            self.position += 1;

            trailing_terms.push(TrailingTerm {
                op,
                term: self.term(true),
            });
        }

        self.add_node(NodeKind::Binary {
            term,
            trailing_terms: Arc::new(trailing_terms),
        })
    }

    fn term(&mut self, allow_struct_literal: bool) -> usize {
        let unary = self.unary(allow_struct_literal);
        let mut trailing_unaries = Vec::new();

        while *self.token() == TokenKind::Multiply || *self.token() == TokenKind::Divide {
            let op = if *self.token() == TokenKind::Multiply {
                Op::Multiply
            } else {
                Op::Divide
            };
            self.position += 1;

            trailing_unaries.push(TrailingUnary {
                op,
                unary: self.unary(true),
            });
        }

        self.add_node(NodeKind::Term {
            unary,
            trailing_unaries: Arc::new(trailing_unaries),
        })
    }

    fn unary(&mut self, allow_struct_literal: bool) -> usize {
        let op = match *self.token() {
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
            _ => None,
        };

        let primary = self.primary(allow_struct_literal);

        self.add_node(NodeKind::Unary { op, primary })
    }

    fn primary(&mut self, allow_struct_literal: bool) -> usize {
        let inner = match *self.token() {
            TokenKind::LParen => self.parenthesized_expression(),
            TokenKind::Identifier { .. } if *self.peek_token() == TokenKind::LParen => {
                self.function_call()
            }
            TokenKind::Identifier { .. }
                if allow_struct_literal && *self.peek_token() == TokenKind::LBrace =>
            {
                self.struct_literal()
            }
            TokenKind::Identifier { .. } => self.variable(),
            TokenKind::IntLiteral { .. } => self.int_literal(),
            TokenKind::Float32Literal { .. } => self.float32_literal(),
            TokenKind::StringLiteral { .. } => self.string_literal(),
            TokenKind::True | TokenKind::False => self.bool_literal(),
            TokenKind::LBracket { .. } => self.array_literal(),
            _ => panic!("Invalid token in primary value"),
        };

        self.add_node(NodeKind::Primary { inner })
    }

    fn parenthesized_expression(&mut self) -> usize {
        self.assert_token(TokenKind::LParen);
        self.position += 1;

        let expression = self.expression(true);

        self.assert_token(TokenKind::RParen);
        self.position += 1;

        self.add_node(NodeKind::ParenthesizedExpression { expression })
    }

    fn variable(&mut self) -> usize {
        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected variable name"),
        };
        self.position += 1;

        let mut inner = self.add_node(NodeKind::VariableName { name });

        loop {
            if *self.token() == TokenKind::LBracket {
                self.position += 1;
                let expression = self.expression(true);
                self.assert_token(TokenKind::RBracket);
                self.position += 1;

                inner = self.add_node(NodeKind::VariableIndex {
                    parent: inner,
                    expression,
                });
                continue;
            }

            if *self.token() == TokenKind::Period {
                self.position += 1;

                let field_name = match self.token() {
                    TokenKind::Identifier { text } => text.clone(),
                    _ => panic!("Expected field name"),
                };
                self.position += 1;

                inner = self.add_node(NodeKind::VariableField {
                    parent: inner,
                    name: field_name,
                });
                continue;
            }

            break;
        }

        self.add_node(NodeKind::Variable { inner })
    }

    fn function_call(&mut self) -> usize {
        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected function name"),
        };
        self.position += 1;

        self.assert_token(TokenKind::LParen);
        self.position += 1;

        let mut args = Vec::new();

        while *self.token() != TokenKind::RParen {
            args.push(self.expression(true));

            if *self.token() != TokenKind::Comma {
                break;
            }

            self.assert_token(TokenKind::Comma);
            self.position += 1;
        }

        self.assert_token(TokenKind::RParen);
        self.position += 1;

        self.add_node(NodeKind::FunctionCall {
            name,
            args: Arc::new(args),
        })
    }

    fn int_literal(&mut self) -> usize {
        let text = match self.token() {
            TokenKind::IntLiteral { text } => text.clone(),
            _ => panic!("Expected int literal"),
        };
        self.position += 1;

        self.add_node(NodeKind::IntLiteral { text })
    }

    fn float32_literal(&mut self) -> usize {
        let text = match self.token() {
            TokenKind::Float32Literal { text } => text.clone(),
            _ => panic!("Expected float32 literal"),
        };
        self.position += 1;

        self.add_node(NodeKind::Float32Literal { text })
    }

    fn string_literal(&mut self) -> usize {
        let text = match self.token() {
            TokenKind::StringLiteral { text } => text.clone(),
            _ => panic!("Expected string literal"),
        };
        self.position += 1;

        self.add_node(NodeKind::StringLiteral { text })
    }

    fn bool_literal(&mut self) -> usize {
        let value = match self.token() {
            TokenKind::True => true,
            TokenKind::False => false,
            _ => panic!("Expected bool literal"),
        };
        self.position += 1;

        self.add_node(NodeKind::BoolLiteral { value })
    }

    fn array_literal(&mut self) -> usize {
        self.assert_token(TokenKind::LBracket);
        self.position += 1;

        let mut elements = Vec::new();

        while *self.token() != TokenKind::RBracket {
            elements.push(self.expression(true));

            if *self.token() != TokenKind::Comma {
                break;
            }

            self.assert_token(TokenKind::Comma);
            self.position += 1;
        }

        let mut repeat_count = 1;

        if *self.token() == TokenKind::Semicolon {
            self.position += 1;

            repeat_count = self.parse_uint_literal();
            self.position += 1;
        }

        self.assert_token(TokenKind::RBracket);
        self.position += 1;

        self.add_node(NodeKind::ArrayLiteral {
            elements: Arc::new(elements),
            repeat_count,
        })
    }

    fn struct_literal(&mut self) -> usize {
        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected struct name"),
        };
        self.position += 1;

        self.assert_token(TokenKind::LBrace);
        self.position += 1;

        let mut fields = Vec::new();

        while *self.token() != TokenKind::RBrace {
            fields.push(self.field_literal());

            if *self.token() != TokenKind::Comma {
                break;
            }

            self.assert_token(TokenKind::Comma);
            self.position += 1;
        }

        self.assert_token(TokenKind::RBrace);
        self.position += 1;

        self.add_node(NodeKind::StructLiteral {
            name,
            fields: Arc::new(fields),
        })
    }

    fn field_literal(&mut self) -> usize {
        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected field name"),
        };
        self.position += 1;

        self.assert_token(TokenKind::Colon);
        self.position += 1;

        let expression = self.expression(true);

        self.add_node(NodeKind::FieldLiteral { name, expression })
    }

    fn type_name(&mut self) -> usize {
        let mut type_kind = match self.token().clone() {
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
            _ => panic!("Expected type name"),
        };
        self.position += 1;

        loop {
            if *self.token() == TokenKind::LBracket {
                self.position += 1;

                let length = self.parse_uint_literal();
                self.position += 1;

                self.assert_token(TokenKind::RBracket);
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

            break;
        }

        self.add_node(NodeKind::TypeName { type_kind })
    }
}
