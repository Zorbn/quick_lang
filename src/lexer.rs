use std::fmt::Display;

use crate::position::Position;

#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Period,
    Plus,
    Minus,
    Asterisk,
    Ampersand,
    Dereference,
    Divide,
    Equal,
    PlusEqual,
    MinusEqual,
    MultiplyEqual,
    DivideEqual,
    Less,
    Greater,
    Not,
    NotEqual,
    EqualEqual,
    LessEqual,
    GreaterEqual,
    And,
    Or,
    Var,
    Val,
    Fun,
    Struct,
    Return,
    Extern,
    If,
    Else,
    Switch,
    Case,
    While,
    For,
    Of,
    By,
    As,
    Defer,
    Sizeof,
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
    True,
    False,
    IntLiteral { text: String },
    Float32Literal { text: String },
    CharLiteral { value: char },
    StringLiteral { text: String },
    Identifier { text: String },
    Eof,
    Error,
}

impl Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut char_buffer = [0u8];

        let str = match self {
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::Comma => ",",
            TokenKind::Semicolon => ";",
            TokenKind::Period => ".",
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Asterisk => "*",
            TokenKind::Ampersand => "&",
            TokenKind::Dereference => ".*",
            TokenKind::Divide => "/",
            TokenKind::Equal => "=",
            TokenKind::PlusEqual => "+=",
            TokenKind::MinusEqual => "-=",
            TokenKind::MultiplyEqual => "*=",
            TokenKind::DivideEqual => "/=",
            TokenKind::Less => "<",
            TokenKind::Greater => ">",
            TokenKind::Not => "!",
            TokenKind::NotEqual => "!=",
            TokenKind::EqualEqual => "==",
            TokenKind::LessEqual => "<=",
            TokenKind::GreaterEqual => ">=",
            TokenKind::And => "&&",
            TokenKind::Or => "||",
            TokenKind::Var => "var",
            TokenKind::Val => "val",
            TokenKind::Fun => "fun",
            TokenKind::Struct => "struct",
            TokenKind::Return => "return",
            TokenKind::Extern => "extern",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::Switch => "switch",
            TokenKind::Case => "case",
            TokenKind::While => "while",
            TokenKind::For => "for",
            TokenKind::Of => "of",
            TokenKind::By => "by",
            TokenKind::As => "as",
            TokenKind::Defer => "defer",
            TokenKind::Sizeof => "sizeof",
            TokenKind::Int => "int",
            TokenKind::String => "string",
            TokenKind::Bool => "bool",
            TokenKind::Char => "char",
            TokenKind::Void => "void",
            TokenKind::UInt => "UInt",
            TokenKind::Int8 => "Int8",
            TokenKind::UInt8 => "UInt8",
            TokenKind::Int16 => "Int16",
            TokenKind::UInt16 => "UInt16",
            TokenKind::Int32 => "Int32",
            TokenKind::UInt32 => "UInt32",
            TokenKind::Int64 => "Int64",
            TokenKind::UInt64 => "UInt64",
            TokenKind::Float32 => "Float32",
            TokenKind::Float64 => "Float64",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::IntLiteral { text } => text,
            TokenKind::Float32Literal { text } => text,
            TokenKind::CharLiteral { value } => value.encode_utf8(&mut char_buffer),
            TokenKind::StringLiteral { text } => text,
            TokenKind::Identifier { text } => text,
            TokenKind::Eof => "EOF",
            TokenKind::Error => "error",
        };

        f.write_str(str)
    }
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub start: Position,
    pub end: Position,
}

pub struct Lexer {
    pub chars: Vec<char>,
    pub tokens: Vec<Token>,
    pub had_error: bool,
    position: Position,
}

impl Lexer {
    pub fn new(chars: Vec<char>) -> Self {
        Self {
            chars,
            tokens: Vec::new(),
            had_error: false,
            position: Position::new(),
        }
    }

    fn char(&self) -> char {
        return *self.chars.get(self.position.index).unwrap_or(&'\0');
    }

    fn try_consume_string(&mut self, str: &str) -> bool {
        if self.position.index + str.len() > self.chars.len() {
            return false;
        }

        let mut is_last_char_valid_in_identifier = false;
        for (i, c) in str.chars().enumerate() {
            is_last_char_valid_in_identifier = Lexer::is_char_valid_in_identifier(c);
            if self.chars[self.position.index + i] != c {
                return false;
            }
        }

        if is_last_char_valid_in_identifier {
            // The string needs to be a full word, rather than the start of another string,
            // eg, "for" but not "fortune".
            let next_char_index = self.position.index + str.len();
            if next_char_index < self.chars.len()
                && Lexer::is_char_valid_in_identifier(self.chars[next_char_index])
            {
                return false;
            }
        }

        self.position.advance_by(str.len());

        true
    }

    fn try_string_to_token(&mut self, str: &str, kind: TokenKind) -> bool {
        let start = self.position;
        let did_succeed = self.try_consume_string(str);
        let end = self.position;

        if did_succeed {
            self.tokens.push(Token { kind, start, end });
        }

        did_succeed
    }

    fn handle_single_line_comment(&mut self) {
        while self.char() != '\n' {
            self.position.advance();
        }

        self.position.newline();
    }

    fn handle_multi_line_comment(&mut self) {
        // Track the number of open multi-line comments, so that these comments can be nested.
        let mut open_count = 1;
        self.position.advance();

        while open_count > 0 {
            if self.try_consume_string("/*") {
                open_count += 1;
            } else if self.try_consume_string("*/") {
                open_count -= 1;
            }

            if self.char() == '\n' {
                self.position.newline();
            } else {
                self.position.advance();
            }
        }
    }

    fn is_char_valid_in_identifier(c: char) -> bool {
        c.is_alphabetic() || c.is_numeric() || c == '_'
    }

    fn error(&mut self, message: &str) {
        self.had_error = true;
        println!(
            "Syntax error at line {}, column {}: {}",
            self.position.line, self.position.column, message
        );
    }

    pub fn lex(&mut self) {
        while self.position.index < self.chars.len() {
            if self.try_string_to_token("!=", TokenKind::NotEqual) {
                continue;
            }

            if self.try_string_to_token("==", TokenKind::EqualEqual) {
                continue;
            }

            if self.try_string_to_token("<=", TokenKind::LessEqual) {
                continue;
            }

            if self.try_string_to_token(">=", TokenKind::GreaterEqual) {
                continue;
            }

            if self.try_string_to_token("+=", TokenKind::PlusEqual) {
                continue;
            }

            if self.try_string_to_token("-=", TokenKind::MinusEqual) {
                continue;
            }

            if self.try_string_to_token("*=", TokenKind::MultiplyEqual) {
                continue;
            }

            if self.try_string_to_token("/=", TokenKind::DivideEqual) {
                continue;
            }

            if self.try_string_to_token("&&", TokenKind::And) {
                continue;
            }

            if self.try_string_to_token("||", TokenKind::Or) {
                continue;
            }

            if self.try_string_to_token(".*", TokenKind::Dereference) {
                continue;
            }

            if self.try_consume_string("//") {
                self.handle_single_line_comment();
                continue;
            }

            if self.try_consume_string("/*") {
                self.handle_multi_line_comment();
                continue;
            }

            if self.try_string_to_token("var", TokenKind::Var) {
                continue;
            }

            if self.try_string_to_token("val", TokenKind::Val) {
                continue;
            }

            if self.try_string_to_token("fun", TokenKind::Fun) {
                continue;
            }

            if self.try_string_to_token("struct", TokenKind::Struct) {
                continue;
            }

            if self.try_string_to_token("return", TokenKind::Return) {
                continue;
            }

            if self.try_string_to_token("extern", TokenKind::Extern) {
                continue;
            }

            if self.try_string_to_token("if", TokenKind::If) {
                continue;
            }

            if self.try_string_to_token("else", TokenKind::Else) {
                continue;
            }

            if self.try_string_to_token("switch", TokenKind::Switch) {
                continue;
            }

            if self.try_string_to_token("case", TokenKind::Case) {
                continue;
            }

            if self.try_string_to_token("while", TokenKind::While) {
                continue;
            }

            if self.try_string_to_token("for", TokenKind::For) {
                continue;
            }

            if self.try_string_to_token("of", TokenKind::Of) {
                continue;
            }

            if self.try_string_to_token("by", TokenKind::By) {
                continue;
            }

            if self.try_string_to_token("as", TokenKind::As) {
                continue;
            }

            if self.try_string_to_token("defer", TokenKind::Defer) {
                continue;
            }

            if self.try_string_to_token("sizeof", TokenKind::Sizeof) {
                continue;
            }

            if self.try_string_to_token("String", TokenKind::String) {
                continue;
            }

            if self.try_string_to_token("Bool", TokenKind::Bool) {
                continue;
            }

            if self.try_string_to_token("Char", TokenKind::Char) {
                continue;
            }

            if self.try_string_to_token("Void", TokenKind::Void) {
                continue;
            }

            if self.try_string_to_token("Int8", TokenKind::Int8) {
                continue;
            }

            if self.try_string_to_token("UInt8", TokenKind::UInt8) {
                continue;
            }

            if self.try_string_to_token("Int16", TokenKind::Int16) {
                continue;
            }

            if self.try_string_to_token("UInt16", TokenKind::UInt16) {
                continue;
            }

            if self.try_string_to_token("Int32", TokenKind::Int32) {
                continue;
            }

            if self.try_string_to_token("UInt32", TokenKind::UInt32) {
                continue;
            }

            if self.try_string_to_token("Int64", TokenKind::Int64) {
                continue;
            }

            if self.try_string_to_token("UInt64", TokenKind::UInt64) {
                continue;
            }

            if self.try_string_to_token("UInt", TokenKind::UInt) {
                continue;
            }

            if self.try_string_to_token("Int", TokenKind::Int) {
                continue;
            }

            if self.try_string_to_token("Float32", TokenKind::Float32) {
                continue;
            }

            if self.try_string_to_token("Float64", TokenKind::Float64) {
                continue;
            }

            if self.try_string_to_token("true", TokenKind::True) {
                continue;
            }

            if self.try_string_to_token("false", TokenKind::False) {
                continue;
            }

            if self.char().is_alphabetic() || self.char() == '_' {
                let mut c = self.char();
                let start = self.position;

                while Lexer::is_char_valid_in_identifier(c) {
                    self.position.advance();
                    c = self.char();
                }

                let end = self.position;
                self.tokens.push(Token {
                    kind: TokenKind::Identifier {
                        text: self.chars[start.index..end.index].iter().collect(),
                    },
                    start,
                    end,
                });

                continue;
            }

            if self.char().is_numeric() {
                let mut c = self.char();
                let mut has_decimal_point = false;
                let start = self.position;

                while c.is_numeric() || (c == '.' && !has_decimal_point) {
                    if c == '.' {
                        has_decimal_point = true;
                    }

                    self.position.advance();
                    c = self.char();
                }

                let end = self.position;
                let text = self.chars[start.index..end.index].iter().collect();

                if has_decimal_point {
                    self.tokens.push(Token {
                        kind: TokenKind::Float32Literal { text },
                        start,
                        end,
                    });
                } else {
                    self.tokens.push(Token {
                        kind: TokenKind::IntLiteral { text },
                        start,
                        end,
                    });
                }

                continue;
            }

            if self.char().is_whitespace() {
                if self.char() == '\n' {
                    self.position.newline();
                } else {
                    self.position.advance();
                }

                continue;
            }

            match self.char() {
                '(' => {
                    self.tokens.push(Token {
                        kind: TokenKind::LParen,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                ')' => {
                    self.tokens.push(Token {
                        kind: TokenKind::RParen,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '{' => {
                    self.tokens.push(Token {
                        kind: TokenKind::LBrace,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '}' => {
                    self.tokens.push(Token {
                        kind: TokenKind::RBrace,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '[' => {
                    self.tokens.push(Token {
                        kind: TokenKind::LBracket,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                ']' => {
                    self.tokens.push(Token {
                        kind: TokenKind::RBracket,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                ',' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Comma,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                ';' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Semicolon,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '.' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Period,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '+' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Plus,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '-' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Minus,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '*' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Asterisk,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '&' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Ampersand,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '/' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Divide,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '=' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Equal,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '<' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Less,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '>' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Greater,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '!' => {
                    self.tokens.push(Token {
                        kind: TokenKind::Not,
                        start: self.position,
                        end: self.position,
                    });
                    self.position.advance();
                }
                '\'' => {
                    let start = self.position;
                    self.position.advance();
                    let value = self.char();
                    self.position.advance();

                    if self.char() != '\'' {
                        self.error("expected end of char literal");
                        self.tokens.push(Token {
                            kind: TokenKind::Error,
                            start,
                            end: self.position,
                        });
                        continue;
                    }

                    self.position.advance();

                    self.tokens.push(Token {
                        kind: TokenKind::CharLiteral { value },
                        start,
                        end: self.position,
                    });
                }
                '"' => {
                    self.position.advance();
                    let mut c = self.char();
                    let mut had_error = false;
                    let start = self.position;

                    loop {
                        match c {
                            '\0' => {
                                self.tokens.push(Token {
                                    kind: TokenKind::Error,
                                    start,
                                    end: self.position,
                                });
                                self.error("reached end of file during string literal");
                                self.position.advance();

                                had_error = true;
                                break;
                            }
                            '\"' => break,
                            '\n' => self.position.newline(),
                            _ => self.position.advance(),
                        }

                        c = self.char();
                    }

                    if had_error {
                        continue;
                    }

                    let end = self.position;
                    self.tokens.push(Token {
                        kind: TokenKind::StringLiteral {
                            text: self.chars[start.index..end.index].iter().collect(),
                        },
                        start,
                        end,
                    });
                    self.position.advance();
                }
                _ => {
                    self.tokens.push(Token {
                        kind: TokenKind::Error,
                        start: self.position,
                        end: self.position,
                    });
                    self.error(&format!("unexpected character \"{}\"", self.char()));
                    self.position.advance();
                }
            }
        }
    }
}
