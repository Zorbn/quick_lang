use std::{
    collections::HashMap,
    fmt::Display,
    sync::{Arc, OnceLock},
};

use crate::{error_bucket::ErrorBucket, file_data::FileData, position::Position};

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
    GenericSpecifier,
    Divide,
    LessLess,
    Percent,
    Pipe,
    Caret,
    Tilde,
    LessLessEqual,
    GreaterGreaterEqual,
    PercentEqual,
    AmpersandEqual,
    CaretEqual,
    PipeEqual,
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
    QuestionMark,
    Var,
    Val,
    Const,
    Func,
    Struct,
    Union,
    Enum,
    Return,
    Break,
    Continue,
    Extern,
    Using,
    Alias,
    If,
    Else,
    Switch,
    Case,
    While,
    For,
    Of,
    In,
    By,
    As,
    Defer,
    Sizeof,
    New,
    Delete,
    Scope,
    True,
    False,
    IntLiteral { text: Arc<str> },
    Float32Literal { text: Arc<str> },
    CharLiteral { value: char },
    StringLiteral { text: Arc<String> },
    Identifier { text: Arc<str> },
    StringInterpolationStart,
    StringInterpolationEnd,
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
            TokenKind::GenericSpecifier => ".<",
            TokenKind::Divide => "/",
            TokenKind::LessLess => "<<",
            TokenKind::Percent => "%",
            TokenKind::Pipe => "|",
            TokenKind::Caret => "^",
            TokenKind::Tilde => "~",
            TokenKind::LessLessEqual => "<<=",
            TokenKind::GreaterGreaterEqual => ">>=",
            TokenKind::PercentEqual => "%=",
            TokenKind::AmpersandEqual => "&=",
            TokenKind::CaretEqual => "^=",
            TokenKind::PipeEqual => "|=",
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
            TokenKind::QuestionMark => "?",
            TokenKind::Var => "var",
            TokenKind::Val => "val",
            TokenKind::Const => "const",
            TokenKind::Func => "func",
            TokenKind::Struct => "struct",
            TokenKind::Union => "union",
            TokenKind::Return => "return",
            TokenKind::Break => "break",
            TokenKind::Continue => "continue",
            TokenKind::Enum => "enum",
            TokenKind::Extern => "extern",
            TokenKind::Using => "using",
            TokenKind::Alias => "alias",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::Switch => "switch",
            TokenKind::Case => "case",
            TokenKind::While => "while",
            TokenKind::For => "for",
            TokenKind::Of => "of",
            TokenKind::In => "in",
            TokenKind::By => "by",
            TokenKind::As => "as",
            TokenKind::Defer => "defer",
            TokenKind::Sizeof => "sizeof",
            TokenKind::New => "new",
            TokenKind::Delete => "delete",
            TokenKind::Scope => "scope",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::IntLiteral { text } => text,
            TokenKind::Float32Literal { text } => text,
            TokenKind::CharLiteral { value } => value.encode_utf8(&mut char_buffer),
            TokenKind::StringLiteral { text } => text,
            TokenKind::Identifier { text } => text,
            TokenKind::StringInterpolationStart => "$\"",
            TokenKind::StringInterpolationEnd => "\"",
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

fn keywords() -> &'static HashMap<Arc<str>, TokenKind> {
    static KEYWORDS: OnceLock<HashMap<Arc<str>, TokenKind>> = OnceLock::new();
    KEYWORDS.get_or_init(|| {
        let mut keywords = HashMap::new();
        keywords.insert(Arc::from("var"), TokenKind::Var);
        keywords.insert(Arc::from("val"), TokenKind::Val);
        keywords.insert(Arc::from("const"), TokenKind::Const);
        keywords.insert(Arc::from("func"), TokenKind::Func);
        keywords.insert(Arc::from("struct"), TokenKind::Struct);
        keywords.insert(Arc::from("union"), TokenKind::Union);
        keywords.insert(Arc::from("enum"), TokenKind::Enum);
        keywords.insert(Arc::from("return"), TokenKind::Return);
        keywords.insert(Arc::from("break"), TokenKind::Break);
        keywords.insert(Arc::from("continue"), TokenKind::Continue);
        keywords.insert(Arc::from("extern"), TokenKind::Extern);
        keywords.insert(Arc::from("using"), TokenKind::Using);
        keywords.insert(Arc::from("alias"), TokenKind::Alias);
        keywords.insert(Arc::from("if"), TokenKind::If);
        keywords.insert(Arc::from("else"), TokenKind::Else);
        keywords.insert(Arc::from("switch"), TokenKind::Switch);
        keywords.insert(Arc::from("case"), TokenKind::Case);
        keywords.insert(Arc::from("while"), TokenKind::While);
        keywords.insert(Arc::from("for"), TokenKind::For);
        keywords.insert(Arc::from("of"), TokenKind::Of);
        keywords.insert(Arc::from("in"), TokenKind::In);
        keywords.insert(Arc::from("by"), TokenKind::By);
        keywords.insert(Arc::from("as"), TokenKind::As);
        keywords.insert(Arc::from("defer"), TokenKind::Defer);
        keywords.insert(Arc::from("sizeof"), TokenKind::Sizeof);
        keywords.insert(Arc::from("new"), TokenKind::New);
        keywords.insert(Arc::from("delete"), TokenKind::Delete);
        keywords.insert(Arc::from("scope"), TokenKind::Scope);
        keywords.insert(Arc::from("true"), TokenKind::True);
        keywords.insert(Arc::from("false"), TokenKind::False);
        keywords
    })
}

fn two_char_ops() -> &'static HashMap<String, TokenKind> {
    static TWO_CHAR_OPS: OnceLock<HashMap<String, TokenKind>> = OnceLock::new();
    TWO_CHAR_OPS.get_or_init(|| {
        let mut ops = HashMap::new();
        ops.insert("!=".into(), TokenKind::NotEqual);
        ops.insert("==".into(), TokenKind::EqualEqual);
        ops.insert("<=".into(), TokenKind::LessEqual);
        ops.insert(">=".into(), TokenKind::GreaterEqual);
        ops.insert("+=".into(), TokenKind::PlusEqual);
        ops.insert("-=".into(), TokenKind::MinusEqual);
        ops.insert("*=".into(), TokenKind::MultiplyEqual);
        ops.insert("/=".into(), TokenKind::DivideEqual);
        ops.insert("&&".into(), TokenKind::And);
        ops.insert("||".into(), TokenKind::Or);
        ops.insert(".*".into(), TokenKind::Dereference);
        ops.insert(".<".into(), TokenKind::GenericSpecifier);
        ops.insert("<<".into(), TokenKind::LessLess);
        ops.insert("<<=".into(), TokenKind::LessLessEqual);
        ops.insert(">>=".into(), TokenKind::GreaterGreaterEqual);
        ops.insert("%=".into(), TokenKind::PercentEqual);
        ops.insert("&=".into(), TokenKind::AmpersandEqual);
        ops.insert("^=".into(), TokenKind::CaretEqual);
        ops.insert("|=".into(), TokenKind::PipeEqual);
        ops
    })
}

pub struct Lexer {
    pub tokens: Vec<Token>,
    pub error_bucket: ErrorBucket,
    position: Position,
    files: Arc<Vec<FileData>>,
}

impl Lexer {
    pub fn new(file_index: usize, files: Arc<Vec<FileData>>) -> Self {
        Self {
            files,
            tokens: Vec::new(),
            error_bucket: ErrorBucket::new(),
            position: Position::new(file_index),
        }
    }

    fn chars(&self) -> &Vec<char> {
        &self.files[self.position.file_index].chars
    }

    fn char(&self) -> char {
        return *self.chars().get(self.position.index).unwrap_or(&'\0');
    }

    fn peek_char(&self) -> char {
        return *self.chars().get(self.position.index + 1).unwrap_or(&'\0');
    }

    fn try_consume_string(&mut self, str: &str) -> bool {
        if self.position.index + str.len() > self.chars().len() {
            return false;
        }

        let mut is_last_char_valid_in_identifier = false;
        for (i, c) in str.chars().enumerate() {
            is_last_char_valid_in_identifier = Lexer::is_char_valid_in_identifier(c);
            if self.chars()[self.position.index + i] != c {
                return false;
            }
        }

        if is_last_char_valid_in_identifier {
            // The string needs to be a full word, rather than the start of another string,
            // eg, "for" but not "fortune".
            let next_char_index = self.position.index + str.len();
            if next_char_index < self.chars().len()
                && Lexer::is_char_valid_in_identifier(self.chars()[next_char_index])
            {
                return false;
            }
        }

        self.position.advance_by(str.len());

        true
    }

    fn handle_single_line_comment(&mut self) {
        while self.char() != '\n' && self.char() != '\0' {
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

    fn handle_escape_sequence(&mut self) -> char {
        self.position.advance();
        let c = self.char();

        match c {
            '\'' => '\'',
            '"' => '\"',
            '\\' => '\\',
            '0' => '\0',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            '{' => '{',
            _ => {
                self.error("unexpected escape sequence in string literal");
                '\\'
            }
        }
    }

    fn handle_string_literal(&mut self, is_interpolated: bool) -> bool {
        let mut c = self.char();
        let start = self.position;
        let mut text = String::new();
        let mut did_string_end = false;

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

                    return false;
                }
                '\"' => {
                    did_string_end = true;

                    break
                },
                '{' if is_interpolated => {
                    break;
                }
                '\n' => {
                    text.push('\n');
                    self.position.newline()
                }
                '\\' => {
                    text.push(self.handle_escape_sequence());
                    self.position.advance();
                }
                _ => {
                    text.push(c);
                    self.position.advance()
                }
            }

            c = self.char();
        }

        let end = self.position;
        self.tokens.push(Token {
            kind: TokenKind::StringLiteral {
                text: Arc::new(text),
            },
            start,
            end,
        });
        self.position.advance();

        did_string_end
    }

    fn handle_string_interpolation(&mut self) {
        let start = self.position;

        self.tokens.push(Token {
            kind: TokenKind::StringInterpolationStart,
            start,
            end: self.position,
        });

        loop {
            let did_string_end = self.handle_string_literal(true);

            if did_string_end || self.char() == '\0' {
               break;
            }

            let mut open_bracket_count = 1;

            loop {
                if self.char() == '\0' {
                    self.tokens.push(Token {
                        kind: TokenKind::Error,
                        start,
                        end: self.position,
                    });
                    self.error("reached end of file during string literal");
                    self.position.advance();

                    return;
                }

                self.lex_token();

                match self.tokens.last() {
                    Some(Token { kind: TokenKind::LBrace, .. }) => {
                        open_bracket_count += 1;
                    }
                    Some(Token { kind: TokenKind::RBrace, .. }) => {
                        open_bracket_count -= 1;

                        if open_bracket_count <= 0 {
                            break
                        }
                    }
                    _ => {}
                }
            }
        }

        self.tokens.push(Token {
            kind: TokenKind::StringInterpolationEnd,
            start,
            end: self.position,
        });
    }

    fn is_char_valid_in_identifier(c: char) -> bool {
        c.is_alphabetic() || c.is_numeric() || c == '_'
    }

    fn error(&mut self, message: &str) {
        self.error_bucket
            .error(self.position, "Syntax", message, &self.files);
    }

    fn collect_chars(&self, start: Position, end: Position) -> Arc<str> {
        Arc::from(
            self.chars()[start.index..end.index]
                .iter()
                .collect::<String>(),
        )
    }

    fn lex_token(&mut self) {
        let mut two_chars = [0u8; 8];

        let first_char_len = self.char().encode_utf8(&mut two_chars).len();
        let second_char_len = self
            .peek_char()
            .encode_utf8(&mut two_chars[first_char_len..])
            .len();
        let two_chars_str =
            std::str::from_utf8(&two_chars[..(first_char_len + second_char_len)]).unwrap();

        if let Some(kind) = two_char_ops().get(two_chars_str) {
            let start = self.position;
            self.position.advance_by(2);
            let end = self.position;

            self.tokens.push(Token {
                kind: kind.clone(),
                start,
                end,
            });
            return;
        }

        if self.try_consume_string("//") {
            self.handle_single_line_comment();
            return;
        }

        if self.try_consume_string("/*") {
            self.handle_multi_line_comment();
            return;
        }

        if self.try_consume_string("$\"") {
            self.handle_string_interpolation();
            return;
        }

        let start = self.position;

        if self.try_consume_string("<<=") {
            self.tokens.push(Token {
                kind: TokenKind::LessLessEqual,
                start,
                end: self.position,
            });

            return;
        }

        if self.try_consume_string(">>=") {
            self.tokens.push(Token {
                kind: TokenKind::GreaterGreaterEqual,
                start,
                end: self.position,
            });

            return;
        }

        if self.char().is_alphabetic() || self.char() == '_' {
            let mut c = self.char();

            while Lexer::is_char_valid_in_identifier(c) {
                self.position.advance();
                c = self.char();
            }

            let end = self.position;
            let text = self.collect_chars(start, end);

            if let Some(keyword_kind) = keywords().get(&text) {
                self.tokens.push(Token {
                    kind: keyword_kind.clone(),
                    start,
                    end,
                });
                return;
            }

            self.tokens.push(Token {
                kind: TokenKind::Identifier { text },
                start,
                end,
            });

            return;
        }

        if self.char() == '0' && self.peek_char() == 'x' {
            self.position.advance_by(2);

            let mut has_contents = false;

            loop {
                match self.char() {
                    'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'A' | 'B' | 'C' | 'D' | 'E' | 'F' => {}
                    _ => {
                        if !self.char().is_numeric() {
                            break;
                        }
                    }
                }

                has_contents = true;
                self.position.advance();
            }

            let end = self.position;

            if !has_contents {
                self.error("malformed hex literal");
                self.tokens.push(Token {
                    kind: TokenKind::Error,
                    start,
                    end: self.position,
                });
                return;
            }

            let text = self.collect_chars(start, end);

            self.tokens.push(Token {
                kind: TokenKind::IntLiteral { text },
                start,
                end,
            });
        }

        if self.char().is_numeric() {
            let mut c = self.char();
            let mut has_decimal_point = false;

            while c.is_numeric() || (c == '.' && !has_decimal_point) {
                if c == '.' {
                    has_decimal_point = true;
                }

                self.position.advance();
                c = self.char();
            }

            let end = self.position;
            let text = self.collect_chars(start, end);

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

            return;
        }

        if self.char().is_whitespace() {
            if self.char() == '\n' {
                self.position.newline();
            } else {
                self.position.advance();
            }

            return;
        }

        let kind = match self.char() {
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            ',' => TokenKind::Comma,
            ';' => TokenKind::Semicolon,
            '.' => TokenKind::Period,
            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,
            '*' => TokenKind::Asterisk,
            '&' => TokenKind::Ampersand,
            '/' => TokenKind::Divide,
            '=' => TokenKind::Equal,
            '<' => TokenKind::Less,
            '>' => TokenKind::Greater,
            '!' => TokenKind::Not,
            '^' => TokenKind::Caret,
            '~' => TokenKind::Tilde,
            '%' => TokenKind::Percent,
            '|' => TokenKind::Pipe,
            '?' => TokenKind::QuestionMark,
            '\'' => {
                let start = self.position;
                self.position.advance();

                let mut value = self.char();
                if value == '\\' {
                    value = self.handle_escape_sequence();
                }

                self.position.advance();

                if self.char() != '\'' {
                    self.error("expected end of char literal");
                    self.tokens.push(Token {
                        kind: TokenKind::Error,
                        start,
                        end: self.position,
                    });
                    return;
                }

                self.position.advance();

                self.tokens.push(Token {
                    kind: TokenKind::CharLiteral { value },
                    start,
                    end: self.position,
                });

                return;
            }
            '"' => {
                self.position.advance();
                self.handle_string_literal(false);
                return;
            }
            _ => {
                self.error(&format!("unexpected character \"{}\"", self.char()));
                TokenKind::Error
            }
        };

        self.tokens.push(Token {
            kind,
            start: self.position,
            end: self.position,
        });
        self.position.advance();
    }

    pub fn lex(&mut self) {
        while self.position.index < self.chars().len() {
            self.lex_token();
        }
    }
}
