use std::{collections::HashMap, hash::Hash, sync::Arc};

use crate::lexer::TokenKind;

// TODO: Should strings be refcounted strs instead?

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Add,
    Subtract,
    Multiply,
    Divide,
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
    Array {
        element_type_kind: usize,
        element_count: usize,
    },
    Struct {
        name: String,
        fields_kinds: Arc<Vec<Field>>,
    },
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
        expression: usize,
    },
    Expression {
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
    StringLiteral {
        text: String,
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

pub struct Parser {
    pub tokens: Vec<TokenKind>,
    pub nodes: Vec<NodeKind>,
    pub types: Vec<TypeKind>,
    pub function_declaration_indices: HashMap<String, usize>,
    pub struct_definition_indices: HashMap<String, usize>,
    pub array_type_kinds: HashMap<ArrayLayout, usize>,
    position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<TokenKind>) -> Self {
        let mut parser = Self {
            tokens,
            nodes: Vec::new(),
            types: Vec::new(),
            array_type_kinds: HashMap::new(),
            function_declaration_indices: HashMap::new(),
            struct_definition_indices: HashMap::new(),
            position: 0,
        };

        parser.add_type(TypeKind::Int);
        parser.add_type(TypeKind::String);

        parser
    }

    fn token(&self) -> &TokenKind {
        self.tokens.get(self.position).unwrap_or(&TokenKind::Eof)
    }

    fn peek_token(&self) -> &TokenKind {
        self.tokens
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

    pub fn parse(&mut self) -> usize {
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
        
        let type_kind = self.add_type(TypeKind::Struct { name: name.clone(), fields_kinds: Arc::new(field_kinds) });
        let index = self.add_node(NodeKind::StructDefinition { name: name.clone(), fields: Arc::new(fields), type_kind });
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
        let inner = match self.token() {
            TokenKind::Var | TokenKind::Val => self.variable_declaration(),
            TokenKind::Identifier { .. } if *self.peek_token() == TokenKind::LParen => {
                self.function_call()
            }
            TokenKind::Return => self.return_statement(),
            _ => self.variable_assignment(),
        };

        self.assert_token(TokenKind::Semicolon);
        self.position += 1;
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

        self.assert_token(TokenKind::Equals);
        self.position += 1;

        let expression = self.expression();

        self.add_node(NodeKind::VariableDeclaration {
            is_mutable,
            name,
            type_name,
            expression,
        })
    }

    fn variable_assignment(&mut self) -> usize {
        let variable = self.variable();

        self.assert_token(TokenKind::Equals);
        self.position += 1;

        let expression = self.expression();

        self.add_node(NodeKind::VariableAssignment {
            variable,
            expression,
        })
    }

    fn return_statement(&mut self) -> usize {
        self.assert_token(TokenKind::Return);
        self.position += 1;

        let expression = self.expression();

        self.add_node(NodeKind::ReturnStatement { expression })
    }

    fn expression(&mut self) -> usize {
        let term = self.term();
        let mut trailing_terms = Vec::new();

        while *self.token() == TokenKind::Plus || *self.token() == TokenKind::Minus {
            let op = if *self.token() == TokenKind::Plus {
                Op::Add
            } else {
                Op::Subtract
            };
            self.position += 1;

            trailing_terms.push(TrailingTerm {
                op,
                term: self.term(),
            });
        }

        self.add_node(NodeKind::Expression {
            term,
            trailing_terms: Arc::new(trailing_terms),
        })
    }

    fn term(&mut self) -> usize {
        let unary = self.unary();
        let mut trailing_unaries = Vec::new();

        while *self.token() == TokenKind::Asterisk || *self.token() == TokenKind::ForwardSlash {
            let op = if *self.token() == TokenKind::Asterisk {
                Op::Multiply
            } else {
                Op::Divide
            };
            self.position += 1;

            trailing_unaries.push(TrailingUnary {
                op,
                unary: self.unary(),
            });
        }

        self.add_node(NodeKind::Term {
            unary,
            trailing_unaries: Arc::new(trailing_unaries),
        })
    }

    fn unary(&mut self) -> usize {
        let op = match *self.token() {
            TokenKind::Plus => {
                self.position += 1;
                Some(Op::Add)
            }
            TokenKind::Minus => {
                self.position += 1;
                Some(Op::Divide)
            }
            _ => None,
        };

        let primary = self.primary();

        self.add_node(NodeKind::Unary { op, primary })
    }

    fn primary(&mut self) -> usize {
        let inner = match *self.token() {
            TokenKind::Identifier { .. } if *self.peek_token() == TokenKind::LParen => {
                self.function_call()
            }
            TokenKind::Identifier { .. } if *self.peek_token() == TokenKind::LBrace => {
                self.struct_literal()
            }
            TokenKind::Identifier { .. } => self.variable(),
            TokenKind::IntLiteral { .. } => self.int_literal(),
            TokenKind::StringLiteral { .. } => self.string_literal(),
            TokenKind::LBracket { .. } => self.array_literal(),
            _ => panic!("Invalid token in primary value"),
        };

        self.add_node(NodeKind::Primary { inner })
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
                let expression = self.expression();
                self.assert_token(TokenKind::RBracket);
                self.position += 1;
                
                inner = self.add_node(NodeKind::VariableIndex { parent: inner, expression });
                continue;
            }

            if *self.token() == TokenKind::Period {
                self.position += 1;

                let field_name = match self.token() {
                    TokenKind::Identifier { text } => text.clone(),
                    _ => panic!("Expected field name"),
                };
                self.position += 1;

                inner = self.add_node(NodeKind::VariableField { parent: inner, name: field_name });
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
            args.push(self.expression());

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

    fn string_literal(&mut self) -> usize {
        let text = match self.token() {
            TokenKind::StringLiteral { text } => text.clone(),
            _ => panic!("Expected string literal"),
        };
        self.position += 1;

        self.add_node(NodeKind::StringLiteral { text })
    }

    fn array_literal(&mut self) -> usize {
        self.assert_token(TokenKind::LBracket);
        self.position += 1;

        let mut elements = Vec::new();

        while *self.token() != TokenKind::RParen {
            elements.push(self.expression());

            if *self.token() != TokenKind::Comma {
                break;
            }

            self.assert_token(TokenKind::Comma);
            self.position += 1;
        }
        
        let mut repeat_count = 1;

        if *self.token() == TokenKind::Semicolon {
            self.position += 1;

            let repeat_count_string = match self.token() {
                TokenKind::IntLiteral { text } => text,
                _ => panic!("Expected int literal for array repeat count"),
            };
            repeat_count = repeat_count_string.parse::<usize>().unwrap();
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
        
        self.add_node(NodeKind::StructLiteral { name, fields: Arc::new(fields) })
    }

    fn field_literal(&mut self) -> usize {
        let name = match self.token() {
            TokenKind::Identifier { text } => text.clone(),
            _ => panic!("Expected field name"),
        };
        self.position += 1;

        self.assert_token(TokenKind::Colon);
        self.position += 1;

        let expression = self.expression();

        self.add_node(NodeKind::FieldLiteral { name, expression })
    }

    fn type_name(&mut self) -> usize {
        let mut type_kind = match self.token() {
            TokenKind::Int => INT_INDEX,
            TokenKind::String => STRING_INDEX,
            TokenKind::Identifier { text } => {
                if let NodeKind::StructDefinition { type_kind, .. } = self.nodes[self.struct_definition_indices[text]] {
                    type_kind
                } else {
                    panic!("Expected struct name");
                }
            },
            _ => panic!("Expected type name"),
        };
        self.position += 1;

        loop {
            if *self.token() == TokenKind::LBracket {
                self.position += 1;

                let length_string = match self.token() {
                    TokenKind::IntLiteral { text } => text,
                    _ => panic!("Expected int literal in array type"),
                };
                let length = length_string.parse::<usize>().unwrap();
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
