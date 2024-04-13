use std::{collections::HashMap, sync::Arc};

use crate::{
    file_data::FileData,
    lexer::{Token, TokenKind},
    position::Position,
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
    BitwiseOr,
    Xor,
    Multiply,
    Divide,
    Modulo,
    BitwiseAnd,
    LeftShift,
    RightShift,
    Not,
    BitwiseNot,
    Assign,
    PlusAssign,
    MinusAssign,
    MultiplyAssign,
    DivideAssign,
    LeftShiftAssign,
    RightShiftAssign,
    ModuloAssign,
    BitwiseAndAssign,
    BitwiseOrAssign,
    XorAssign,
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MethodKind {
    Unknown,
    None,
    ByValue,
    ByReference,
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    Name {
        text: Arc<str>,
    },
    TopLevel {
        functions: Arc<Vec<NodeIndex>>,
        structs: Arc<Vec<NodeIndex>>,
        enums: Arc<Vec<NodeIndex>>,
        aliases: Arc<Vec<NodeIndex>>,
        variable_declarations: Arc<Vec<NodeIndex>>,
        usings: Arc<Vec<NodeIndex>>,
        definition_indices: Arc<HashMap<Arc<str>, NodeIndex>>,
    },
    StructDefinition {
        name: NodeIndex,
        fields: Arc<Vec<NodeIndex>>,
        functions: Arc<Vec<NodeIndex>>,
        generic_params: Arc<Vec<NodeIndex>>,
        definition_indices: Arc<HashMap<Arc<str>, NodeIndex>>,
        is_union: bool,
    },
    EnumDefinition {
        name: NodeIndex,
        variant_names: Arc<Vec<NodeIndex>>,
    },
    Field {
        name: NodeIndex,
        type_name: NodeIndex,
    },
    FunctionDeclaration {
        name: NodeIndex,
        params: Arc<Vec<NodeIndex>>,
        generic_params: Arc<Vec<NodeIndex>>,
        return_type_name: NodeIndex,
    },
    Function {
        declaration: NodeIndex,
        scoped_statement: NodeIndex,
    },
    ExternFunction {
        declaration: NodeIndex,
    },
    Using {
        namespace_type_name: NodeIndex,
    },
    Alias {
        aliased_type_name: NodeIndex,
        alias_name: NodeIndex,
    },
    Param {
        name: NodeIndex,
        type_name: NodeIndex,
    },
    Block {
        statements: Arc<Vec<NodeIndex>>,
    },
    Statement {
        inner: Option<NodeIndex>,
    },
    VariableDeclaration {
        declaration_kind: DeclarationKind,
        name: NodeIndex,
        type_name: Option<NodeIndex>,
        expression: Option<NodeIndex>,
    },
    ReturnStatement {
        expression: Option<NodeIndex>,
    },
    BreakStatement,
    ContinueStatement,
    DeferStatement {
        statement: NodeIndex,
    },
    CrashStatement {
        position: Position,
    },
    IfStatement {
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        next: Option<NodeIndex>,
    },
    SwitchStatement {
        expression: NodeIndex,
        case_statement: NodeIndex,
    },
    CaseStatement {
        expression: NodeIndex,
        scoped_statement: NodeIndex,
        next: Option<NodeIndex>,
    },
    WhileLoop {
        expression: NodeIndex,
        scoped_statement: NodeIndex,
    },
    ForLoop {
        iterator: NodeIndex,
        op: Op,
        from: NodeIndex,
        to: NodeIndex,
        by: Option<NodeIndex>,
        scoped_statement: NodeIndex,
    },
    Binary {
        left: NodeIndex,
        op: Op,
        right: NodeIndex,
    },
    UnaryPrefix {
        op: Op,
        right: NodeIndex,
    },
    UnarySuffix {
        left: NodeIndex,
        op: Op,
    },
    Call {
        left: NodeIndex,
        args: Arc<Vec<NodeIndex>>,
        method_kind: MethodKind,
    },
    IndexAccess {
        left: NodeIndex,
        expression: NodeIndex,
        position: Position,
    },
    FieldAccess {
        left: NodeIndex,
        name: NodeIndex,
        position: Position,
    },
    Cast {
        left: NodeIndex,
        type_name: NodeIndex,
    },
    GenericSpecifier {
        left: NodeIndex,
        generic_arg_type_names: Arc<Vec<NodeIndex>>,
    },
    Identifier {
        name: NodeIndex,
    },
    IntLiteral {
        text: Arc<str>,
    },
    FloatLiteral {
        text: Arc<str>,
    },
    StringLiteral {
        text: Arc<String>,
    },
    BoolLiteral {
        value: bool,
    },
    CharLiteral {
        value: char,
    },
    ArrayLiteral {
        elements: Arc<Vec<NodeIndex>>,
        repeat_count_const_expression: Option<NodeIndex>,
    },
    StructLiteral {
        left: NodeIndex,
        field_literals: Arc<Vec<NodeIndex>>,
    },
    FieldLiteral {
        name: NodeIndex,
        expression: NodeIndex,
    },
    TypeSize {
        type_name: NodeIndex,
    },
    ConstExpression {
        inner: NodeIndex,
    },
    TypeName {
        name: NodeIndex,
    },
    TypeNamePointer {
        inner: NodeIndex,
        is_inner_mutable: bool,
    },
    TypeNameArray {
        inner: NodeIndex,
        element_count_const_expression: NodeIndex,
    },
    TypeNameFunction {
        param_type_names: Arc<Vec<NodeIndex>>,
        return_type_name: NodeIndex,
    },
    TypeNameFieldAccess {
        left: NodeIndex,
        name: NodeIndex,
    },
    TypeNameGenericSpecifier {
        left: NodeIndex,
        generic_arg_type_names: Arc<Vec<NodeIndex>>,
    },
    Error,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub start: Position,
    pub end: Position,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIndex {
    pub node_index: usize,
    pub file_index: usize,
}

pub struct Parser {
    tokens: Option<Vec<Token>>,

    pub nodes: Vec<Node>,
    pub start_index: NodeIndex,
    pub error_count: usize,

    files: Arc<Vec<FileData>>,
    position: usize,
    file_index: usize,
}

impl Parser {
    pub fn new(file_index: usize, files: Arc<Vec<FileData>>) -> Self {
        Self {
            files,
            tokens: None,
            nodes: Vec::new(),
            error_count: 0,
            start_index: NodeIndex {
                node_index: 0,
                file_index: 0,
            },
            position: 0,
            file_index,
        }
    }

    fn token_kind(&self) -> &TokenKind {
        let tokens = self.tokens.as_ref().unwrap();
        if let Some(token) = tokens.get(self.position) {
            &token.kind
        } else {
            &TokenKind::Eof
        }
    }

    fn node_end(&self, index: NodeIndex) -> Position {
        self.nodes[index.node_index].end
    }

    fn token_start(&self) -> Position {
        let tokens = self.tokens.as_ref().unwrap();

        if tokens.len() <= self.position {
            return Position::new(0);
        }

        tokens[self.position].start
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
    ) -> Option<NodeIndex> {
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

    fn parse_error(&mut self, message: &str, start: Position, end: Position) -> NodeIndex {
        self.error(message);

        self.add_node(Node {
            kind: NodeKind::Error,
            start,
            end,
        })
    }

    fn add_node(&mut self, node: Node) -> NodeIndex {
        let node_index = self.nodes.len();
        self.nodes.push(node);

        NodeIndex {
            node_index,
            file_index: self.file_index,
        }
    }

    fn error(&mut self, message: &str) {
        self.error_count += 1;
        self.token_start().error("Syntax", message, &self.files);
    }

    pub fn parse(&mut self, tokens: Vec<Token>) {
        self.position = 0;
        self.tokens = Some(tokens);
        self.start_index = self.top_level();
    }

    // TODO: Functions, structs, and enums could probably all be stored in the same vec.
    fn top_level(&mut self) -> NodeIndex {
        let mut functions = Vec::new();
        let mut structs = Vec::new();
        let mut enums = Vec::new();
        let mut aliases = Vec::new();
        let mut variable_declarations = Vec::new();
        let mut usings = Vec::new();

        let mut definition_indices = HashMap::new();

        let start = self.token_start();
        let mut end = self.token_end();

        while *self.token_kind() != TokenKind::Eof {
            let index;

            match *self.token_kind() {
                TokenKind::Func => {
                    index = self.function(&mut definition_indices);
                    functions.push(index);
                }
                TokenKind::Extern => {
                    index = self.extern_function(&mut definition_indices);
                    functions.push(index);
                }
                TokenKind::Struct | TokenKind::Union => {
                    index = self.struct_definition(&mut definition_indices);
                    structs.push(index);
                }
                TokenKind::Enum => {
                    index = self.enum_definition(&mut definition_indices);
                    enums.push(index);
                }
                TokenKind::Using => {
                    index = self.using();
                    usings.push(index);
                }
                TokenKind::Alias => {
                    index = self.alias(&mut definition_indices);
                    aliases.push(index);
                }
                TokenKind::Val | TokenKind::Const => {
                    index = self.variable_declaration(Some(&mut definition_indices));

                    assert_token!(self, TokenKind::Semicolon, start, self.token_end());
                    self.position += 1;

                    variable_declarations.push(index);
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
                aliases: Arc::new(aliases),
                variable_declarations: Arc::new(variable_declarations),
                usings: Arc::new(usings),
                definition_indices: Arc::new(definition_indices),
            },
            start,
            end,
        })
    }

    fn struct_definition(
        &mut self,
        definition_indices: &mut HashMap<Arc<str>, NodeIndex>,
    ) -> NodeIndex {
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

        let NodeKind::Name { text: name_text } = self.nodes[name.node_index].kind.clone() else {
            parse_error!(self, "invalid struct name", start, self.node_end(name));
        };

        let mut generic_params = Vec::new();

        if *self.token_kind() == TokenKind::Less {
            if let Some(error_node) = self.parse_generic_params(start, &mut generic_params) {
                return error_node;
            }
        }

        assert_token!(self, TokenKind::LBrace, start, self.token_end());
        self.position += 1;

        let mut fields = Vec::new();
        let mut functions = Vec::new();
        let mut inner_definition_indices = HashMap::new();

        while *self.token_kind() != TokenKind::RBrace {
            if *self.token_kind() == TokenKind::Func {
                let function = self.function(&mut inner_definition_indices);
                functions.push(function);
                continue;
            }

            let field = self.field();
            fields.push(field);

            if *self.token_kind() != TokenKind::Semicolon {
                break;
            }

            assert_token!(self, TokenKind::Semicolon, start, self.token_end());
            self.position += 1;
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBrace, start, end);
        self.position += 1;

        let index = self.add_node(Node {
            kind: NodeKind::StructDefinition {
                name,
                fields: Arc::new(fields),
                functions: Arc::new(functions),
                generic_params: Arc::new(generic_params),
                definition_indices: Arc::new(inner_definition_indices),
                is_union,
            },
            start,
            end,
        });

        definition_indices.insert(name_text, index);

        index
    }

    fn enum_definition(
        &mut self,
        definition_indices: &mut HashMap<Arc<str>, NodeIndex>,
    ) -> NodeIndex {
        let start = self.token_start();

        assert_token!(self, TokenKind::Enum, start, self.token_end());
        self.position += 1;

        let name = self.name();

        let NodeKind::Name { text: name_text } = self.nodes[name.node_index].kind.clone() else {
            parse_error!(self, "invalid struct name", start, self.node_end(name));
        };

        assert_token!(self, TokenKind::LBrace, start, self.token_end());
        self.position += 1;

        let mut variant_names = Vec::new();

        while *self.token_kind() != TokenKind::RBrace {
            variant_names.push(self.name());

            assert_token!(self, TokenKind::Semicolon, start, self.token_end());
            self.position += 1;
        }

        let end = self.token_end();
        assert_token!(self, TokenKind::RBrace, start, end);
        self.position += 1;

        let variant_names = Arc::new(variant_names);

        let index = self.add_node(Node {
            kind: NodeKind::EnumDefinition {
                name,
                variant_names,
            },
            start,
            end,
        });

        definition_indices.insert(name_text, index);

        index
    }

    fn using(&mut self) -> NodeIndex {
        let start = self.token_start();

        assert_token!(self, TokenKind::Using, start, self.token_end());
        self.position += 1;

        let namespace_type_name = self.type_name();

        let end = self.token_end();
        assert_token!(self, TokenKind::Semicolon, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::Using {
                namespace_type_name,
            },
            start,
            end,
        })
    }

    fn alias(&mut self, definition_indices: &mut HashMap<Arc<str>, NodeIndex>) -> NodeIndex {
        let start = self.token_start();

        assert_token!(self, TokenKind::Alias, start, self.token_end());
        self.position += 1;

        let alias_name = self.name();

        assert_token!(self, TokenKind::Equal, start, self.token_end());
        self.position += 1;

        let aliased_type_name = self.type_name();

        let end = self.token_end();
        assert_token!(self, TokenKind::Semicolon, start, end);
        self.position += 1;

        let index = self.add_node(Node {
            kind: NodeKind::Alias {
                aliased_type_name,
                alias_name,
            },
            start,
            end,
        });

        let NodeKind::Name { text: name_text } = self.nodes[alias_name.node_index].kind.clone()
        else {
            panic!("invalid alias name");
        };

        definition_indices.insert(name_text, index);

        index
    }

    fn field(&mut self) -> NodeIndex {
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
        params: &mut Vec<NodeIndex>,
    ) -> Option<NodeIndex> {
        if let Some(error_node) = self.assert_token(TokenKind::LParen, start, self.token_end()) {
            return Some(error_node);
        }
        self.position += 1;

        while *self.token_kind() != TokenKind::RParen {
            let param = self.param();
            params.push(param);

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
        generic_params: &mut Vec<NodeIndex>,
    ) -> Option<NodeIndex> {
        if let Some(error_node) = self.assert_token(TokenKind::Less, start, self.token_end()) {
            return Some(error_node);
        }
        self.position += 1;

        while *self.token_kind() != TokenKind::Greater {
            let name = self.name();
            generic_params.push(name);

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

    fn function_declaration(&mut self) -> NodeIndex {
        let start = self.token_start();
        assert_token!(self, TokenKind::Func, start, self.token_end());
        self.position += 1;

        let name = self.name();

        let mut generic_params = Vec::new();

        if *self.token_kind() == TokenKind::Less {
            if let Some(error_node) = self.parse_generic_params(start, &mut generic_params) {
                return error_node;
            }
        }

        let mut params = Vec::new();

        if let Some(error_node) = self.parse_function_params(start, &mut params) {
            return error_node;
        }

        let return_type_name = self.type_name();
        let end = self.node_end(return_type_name);

        self.add_node(Node {
            kind: NodeKind::FunctionDeclaration {
                name,
                params: Arc::new(params),
                generic_params: Arc::new(generic_params),
                return_type_name,
            },
            start,
            end,
        })
    }

    fn function(&mut self, definition_indices: &mut HashMap<Arc<str>, NodeIndex>) -> NodeIndex {
        let start = self.token_start();
        let declaration = self.function_declaration();
        let scoped_statement = self.scoped_statement();
        let end = self.node_end(scoped_statement);

        let NodeKind::FunctionDeclaration { name, .. } = self.nodes[declaration.node_index].kind
        else {
            parse_error!(self, "invalid function declaration", start, end);
        };

        let NodeKind::Name { text: name_text } = self.nodes[name.node_index].kind.clone() else {
            parse_error!(self, "invalid function name", start, self.node_end(name));
        };

        let index = self.add_node(Node {
            kind: NodeKind::Function {
                declaration,
                scoped_statement,
            },
            start,
            end,
        });

        definition_indices.insert(name_text, index);

        index
    }

    fn extern_function(
        &mut self,
        definition_indices: &mut HashMap<Arc<str>, NodeIndex>,
    ) -> NodeIndex {
        let start = self.token_start();
        assert_token!(self, TokenKind::Extern, start, self.token_end());
        self.position += 1;

        let declaration = self.function_declaration();

        let end = self.token_end();
        assert_token!(self, TokenKind::Semicolon, start, end);
        self.position += 1;

        let NodeKind::FunctionDeclaration { name, .. } = self.nodes[declaration.node_index].kind
        else {
            parse_error!(self, "invalid function declaration", start, end);
        };

        let NodeKind::Name { text: name_text } = self.nodes[name.node_index].kind.clone() else {
            parse_error!(self, "invalid function name", start, self.node_end(name));
        };

        let index = self.add_node(Node {
            kind: NodeKind::ExternFunction { declaration },
            start,
            end,
        });

        definition_indices.insert(name_text, index);

        index
    }

    fn param(&mut self) -> NodeIndex {
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

    fn block(&mut self) -> NodeIndex {
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

    fn scoped_statement(&mut self) -> NodeIndex {
        let start = self.token_start();
        let statement = self.statement();
        let end = self.node_end(statement);

        let NodeKind::Statement { inner } = &self.nodes[statement.node_index].kind else {
            parse_error!(self, "invalid statement", start, end);
        };

        if let Some(inner) = inner {
            if matches!(self.nodes[inner.node_index].kind, NodeKind::Block { .. }) {
                return *inner;
            }
        }

        let statements = vec![statement];

        self.add_node(Node {
            kind: NodeKind::Block {
                statements: Arc::new(statements),
            },
            start,
            end,
        })
    }

    fn statement(&mut self) -> NodeIndex {
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
            TokenKind::Var | TokenKind::Val | TokenKind::Const => {
                Some(self.variable_declaration(None))
            }
            TokenKind::Return => Some(self.return_statement()),
            TokenKind::Break => Some(self.break_statement()),
            TokenKind::Continue => Some(self.continue_statement()),
            TokenKind::Defer => Some(self.defer_statement()),
            TokenKind::Crash => Some(self.crash_statement()),
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

    fn variable_declaration(
        &mut self,
        definition_indices: Option<&mut HashMap<Arc<str>, NodeIndex>>,
    ) -> NodeIndex {
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

        let (expression, end) = if *self.token_kind() == TokenKind::QuestionMark {
            let end = self.token_end();
            self.position += 1;

            if declaration_kind != DeclarationKind::Var {
                parse_error!(self, "uninitialized variable must be a var", start, end);
            }

            if type_name.is_none() {
                parse_error!(
                    self,
                    "uninitialized variable must have a specified type",
                    start,
                    end
                );
            }

            (None, end)
        } else if declaration_kind == DeclarationKind::Const {
            let expression = self.const_expression();
            let end = self.node_end(expression);

            (Some(expression), end)
        } else {
            let expression = self.expression();
            let end = self.node_end(expression);

            (Some(expression), end)
        };

        let index = self.add_node(Node {
            kind: NodeKind::VariableDeclaration {
                declaration_kind,
                name,
                type_name,
                expression,
            },
            start,
            end,
        });

        if let Some(definition_indices) = definition_indices {
            let NodeKind::Name { text: name_text } = self.nodes[name.node_index].kind.clone()
            else {
                parse_error!(self, "invalid variable name", start, end);
            };

            definition_indices.insert(name_text, index);
        }

        index
    }

    fn return_statement(&mut self) -> NodeIndex {
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

    fn break_statement(&mut self) -> NodeIndex {
        let start = self.token_start();
        let end = self.token_end();
        assert_token!(self, TokenKind::Break, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::BreakStatement,
            start,
            end,
        })
    }

    fn continue_statement(&mut self) -> NodeIndex {
        let start = self.token_start();
        let end = self.token_end();
        assert_token!(self, TokenKind::Continue, start, end);
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::ContinueStatement,
            start,
            end,
        })
    }

    fn defer_statement(&mut self) -> NodeIndex {
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

    fn crash_statement(&mut self) -> NodeIndex {
        let start = self.token_start();
        let end = self.token_end();
        assert_token!(self, TokenKind::Crash, start, self.token_end());
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::CrashStatement { position: start },
            start,
            end,
        })
    }

    fn if_statement(&mut self) -> NodeIndex {
        let start = self.token_start();
        assert_token!(self, TokenKind::If, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        let scoped_statement = self.scoped_statement();
        let mut end = self.node_end(scoped_statement);

        let next = if *self.token_kind() == TokenKind::Else {
            self.position += 1;

            let next = if *self.token_kind() == TokenKind::If {
                self.if_statement()
            } else {
                self.scoped_statement()
            };

            end = self.node_end(next);

            Some(next)
        } else {
            None
        };

        self.add_node(Node {
            kind: NodeKind::IfStatement {
                expression,
                scoped_statement,
                next,
            },
            start,
            end,
        })
    }

    fn switch_statement(&mut self) -> NodeIndex {
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

    fn case_statement(&mut self) -> NodeIndex {
        let start = self.token_start();
        assert_token!(self, TokenKind::Case, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        let scoped_statement = self.scoped_statement();
        let mut end = self.node_end(scoped_statement);

        let next = if *self.token_kind() == TokenKind::Case {
            let next = self.case_statement();
            end = self.node_end(next);

            Some(next)
        } else if *self.token_kind() == TokenKind::Else {
            self.position += 1;

            let next = self.scoped_statement();
            end = self.node_end(next);

            Some(next)
        } else {
            None
        };

        self.add_node(Node {
            kind: NodeKind::CaseStatement {
                expression,
                scoped_statement,
                next,
            },
            start,
            end,
        })
    }

    fn while_loop(&mut self) -> NodeIndex {
        let start = self.token_start();
        assert_token!(self, TokenKind::While, start, self.token_end());
        self.position += 1;

        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        assert_token!(self, TokenKind::RParen, start, self.token_end());
        self.position += 1;

        let scoped_statement = self.scoped_statement();
        let end = self.node_end(scoped_statement);

        self.add_node(Node {
            kind: NodeKind::WhileLoop {
                expression,
                scoped_statement,
            },
            start,
            end,
        })
    }

    fn for_loop(&mut self) -> NodeIndex {
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

        let scoped_statement = self.scoped_statement();
        let end = self.node_end(scoped_statement);

        self.add_node(Node {
            kind: NodeKind::ForLoop {
                iterator,
                op,
                from,
                to,
                by,
                scoped_statement,
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
     * UnaryPrefix: *, !, ~, +, -
     *
     * (Chainable, eg. a * b / c)
     * Factor: *, /, %, &, <<, >>
     * Term: +, -, |, ^
     * Inequality: <, <=, >, >=
     * Equality: ==, !=
     * BooleanAnd: &&
     * BooleanOr: ||
     * Assignment: =, +=, -=, /=, *=, %=, <<=, >>=, &=, ^=, |=
     */
    fn expression(&mut self) -> NodeIndex {
        self.assignment()
    }

    fn const_expression(&mut self) -> NodeIndex {
        let start = self.token_start();

        // Assignment is not allowed in const expressions.
        let inner = self.boolean_or();
        let end = self.node_end(inner);

        self.add_node(Node {
            kind: NodeKind::ConstExpression { inner },
            start,
            end,
        })
    }

    fn assignment(&mut self) -> NodeIndex {
        let start = self.token_start();
        let mut left = self.boolean_or();

        loop {
            let op = match *self.token_kind() {
                TokenKind::Equal => Op::Assign,
                TokenKind::PlusEqual => Op::PlusAssign,
                TokenKind::MinusEqual => Op::MinusAssign,
                TokenKind::MultiplyEqual => Op::MultiplyAssign,
                TokenKind::DivideEqual => Op::DivideAssign,
                TokenKind::LessLessEqual => Op::LeftShiftAssign,
                TokenKind::GreaterGreaterEqual => Op::RightShiftAssign,
                TokenKind::PercentEqual => Op::ModuloAssign,
                TokenKind::AmpersandEqual => Op::BitwiseAndAssign,
                TokenKind::PipeEqual => Op::BitwiseOrAssign,
                TokenKind::CaretEqual => Op::XorAssign,
                _ => break,
            };

            self.position += 1;
            let right = self.boolean_or();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary { left, op, right },
                start,
                end,
            })
        }

        left
    }

    fn boolean_or(&mut self) -> NodeIndex {
        let start = self.token_start();
        let mut left = self.boolean_and();

        loop {
            let TokenKind::Or = *self.token_kind() else {
                break;
            };
            self.position += 1;

            let right = self.boolean_and();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary {
                    left,
                    op: Op::Or,
                    right,
                },
                start,
                end,
            })
        }

        left
    }

    fn boolean_and(&mut self) -> NodeIndex {
        let start = self.token_start();
        let mut left = self.equality();

        loop {
            let TokenKind::And = *self.token_kind() else {
                break;
            };
            self.position += 1;

            let right = self.equality();
            let end = self.node_end(right);
            left = self.add_node(Node {
                kind: NodeKind::Binary {
                    left,
                    op: Op::And,
                    right,
                },
                start,
                end,
            })
        }

        left
    }

    fn equality(&mut self) -> NodeIndex {
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

    fn inequality(&mut self) -> NodeIndex {
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

    fn term(&mut self) -> NodeIndex {
        let start = self.token_start();
        let mut left = self.factor();

        loop {
            let op = match *self.token_kind() {
                TokenKind::Plus => Op::Plus,
                TokenKind::Minus => Op::Minus,
                TokenKind::Pipe => Op::BitwiseOr,
                TokenKind::Caret => Op::Xor,
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

    fn factor(&mut self) -> NodeIndex {
        let start = self.token_start();
        let mut left = self.unary_prefix();

        loop {
            let op = match *self.token_kind() {
                TokenKind::Asterisk => Op::Multiply,
                TokenKind::Divide => Op::Divide,
                TokenKind::Percent => Op::Modulo,
                TokenKind::Ampersand => Op::BitwiseAnd,
                TokenKind::LessLess => Op::LeftShift,
                TokenKind::GreaterGreater => Op::RightShift,
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

    fn unary_prefix(&mut self) -> NodeIndex {
        let start = self.token_start();
        let op = match *self.token_kind() {
            TokenKind::Not => Op::Not,
            TokenKind::Tilde => Op::BitwiseNot,
            TokenKind::Plus => Op::Plus,
            TokenKind::Minus => Op::Minus,
            TokenKind::Asterisk => Op::Reference,
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
        generic_arg_type_names: &mut Vec<NodeIndex>,
    ) -> Option<NodeIndex> {
        if let Some(error_node) =
            self.assert_token(TokenKind::GenericSpecifier, start, self.token_end())
        {
            return Some(error_node);
        }
        self.position += 1;

        while *self.token_kind() != TokenKind::Greater {
            generic_arg_type_names.push(self.type_name());

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

    fn unary_suffix(&mut self) -> NodeIndex {
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
                            method_kind: MethodKind::Unknown,
                        },
                        start,
                        end,
                    });
                }
                TokenKind::GenericSpecifier => {
                    let mut generic_arg_type_names = Vec::new();
                    let mut end = self.token_end();
                    if let Some(error_node) = self.parse_generic_specifier_list(
                        start,
                        &mut end,
                        &mut generic_arg_type_names,
                    ) {
                        return error_node;
                    }

                    left = self.add_node(Node {
                        kind: NodeKind::GenericSpecifier {
                            left,
                            generic_arg_type_names: Arc::new(generic_arg_type_names),
                        },
                        start,
                        end,
                    });
                }
                TokenKind::LBrace => {
                    left = self.struct_literal(left, start);
                }
                TokenKind::Period => {
                    self.position += 1;

                    let name = self.name();
                    let end = self.node_end(name);

                    left = self.add_node(Node {
                        kind: NodeKind::FieldAccess {
                            left,
                            name,
                            position: start,
                        },
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
                        kind: NodeKind::IndexAccess {
                            left,
                            expression,
                            position: start,
                        },
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

    fn struct_literal(&mut self, left: NodeIndex, start: Position) -> NodeIndex {
        assert_token!(self, TokenKind::LBrace, start, self.token_end());
        self.position += 1;

        let mut field_literals = Vec::new();

        while *self.token_kind() != TokenKind::RBrace {
            field_literals.push(self.field_literal());

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
                field_literals: Arc::new(field_literals),
            },
            start,
            end,
        })
    }

    fn primary(&mut self) -> NodeIndex {
        let start = self.token_start();
        match *self.token_kind() {
            TokenKind::LParen => self.parenthesized_expression(),
            TokenKind::Identifier { .. } => self.identifier(),
            TokenKind::IntLiteral { .. } => self.int_literal(),
            TokenKind::Float32Literal { .. } => self.float_literal(),
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

    fn parenthesized_expression(&mut self) -> NodeIndex {
        let start = self.token_start();
        assert_token!(self, TokenKind::LParen, start, self.token_end());
        self.position += 1;

        let expression = self.expression();

        let end = self.token_end();
        assert_token!(self, TokenKind::RParen, start, end);
        self.position += 1;

        expression
    }

    fn identifier(&mut self) -> NodeIndex {
        let start = self.token_start();
        let name = self.name();
        let end = self.node_end(name);

        self.add_node(Node {
            kind: NodeKind::Identifier { name },
            start,
            end,
        })
    }

    fn name(&mut self) -> NodeIndex {
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

    fn int_literal(&mut self) -> NodeIndex {
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

    fn float_literal(&mut self) -> NodeIndex {
        let start = self.token_start();
        let end = self.token_end();
        let text = match self.token_kind() {
            TokenKind::Float32Literal { text } => text.clone(),
            _ => parse_error!(self, "expected float32 literal", start, end),
        };
        self.position += 1;

        self.add_node(Node {
            kind: NodeKind::FloatLiteral { text },
            start,
            end,
        })
    }

    fn string_literal(&mut self) -> NodeIndex {
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

    fn bool_literal(&mut self) -> NodeIndex {
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

    fn char_literal(&mut self) -> NodeIndex {
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

    fn array_literal(&mut self) -> NodeIndex {
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

    fn field_literal(&mut self) -> NodeIndex {
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

    fn type_size(&mut self) -> NodeIndex {
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

    fn type_name(&mut self) -> NodeIndex {
        let start = self.token_start();
        let token_kind = self.token_kind().clone();

        match token_kind {
            TokenKind::Func => {
                self.position += 1;
                assert_token!(self, TokenKind::LParen, start, self.token_end());
                self.position += 1;

                let mut param_type_names = Vec::new();

                while *self.token_kind() != TokenKind::RParen {
                    param_type_names.push(self.type_name());

                    if *self.token_kind() != TokenKind::Comma {
                        break;
                    }

                    assert_token!(self, TokenKind::Comma, start, self.token_end());
                    self.position += 1;
                }

                assert_token!(self, TokenKind::RParen, start, self.token_end());
                self.position += 1;

                let return_type_name = self.type_name();
                let end = self.node_end(return_type_name);

                self.add_node(Node {
                    kind: NodeKind::TypeNameFunction {
                        param_type_names: Arc::new(param_type_names),
                        return_type_name,
                    },
                    start,
                    end,
                })
            }
            TokenKind::LBracket => {
                self.position += 1;

                let element_count_const_expression = self.const_expression();

                assert_token!(self, TokenKind::RBracket, start, self.token_end());
                self.position += 1;

                let inner = self.type_name();
                let end = self.node_end(inner);

                self.add_node(Node {
                    kind: NodeKind::TypeNameArray {
                        inner,
                        element_count_const_expression,
                    },
                    start,
                    end,
                })
            }
            TokenKind::Asterisk => {
                self.position += 1;

                let is_inner_mutable = match *self.token_kind() {
                    TokenKind::Val => false,
                    TokenKind::Var => true,
                    _ => parse_error!(
                        self,
                        "expected val or var to be specified for pointer",
                        start,
                        self.token_end()
                    ),
                };
                self.position += 1;

                let inner = self.type_name();
                let end = self.node_end(inner);

                self.add_node(Node {
                    kind: NodeKind::TypeNamePointer {
                        inner,
                        is_inner_mutable,
                    },
                    start,
                    end,
                })
            }
            TokenKind::Identifier { .. } => {
                let name = self.name();
                let mut end = self.token_end();
                let mut left = self.add_node(Node {
                    kind: NodeKind::TypeName { name },
                    start,
                    end,
                });

                loop {
                    match *self.token_kind() {
                        TokenKind::Period => {
                            self.position += 1;
                            let name = self.name();
                            end = self.node_end(name);

                            left = self.add_node(Node {
                                kind: NodeKind::TypeNameFieldAccess { left, name },
                                start,
                                end,
                            });
                        }
                        TokenKind::GenericSpecifier => {
                            let mut generic_arg_type_names = Vec::new();
                            end = self.token_end();
                            if let Some(error_node) = self.parse_generic_specifier_list(
                                start,
                                &mut end,
                                &mut generic_arg_type_names,
                            ) {
                                return error_node;
                            }

                            return self.add_node(Node {
                                kind: NodeKind::TypeNameGenericSpecifier {
                                    left,
                                    generic_arg_type_names: Arc::new(generic_arg_type_names),
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
            _ => parse_error!(self, "expected type name", start, self.token_end()),
        }
    }
}
