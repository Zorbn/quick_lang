#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semicolon,
    Period,
    Plus,
    Minus,
    Asterisk,
    Ampersand,
    Divide,
    Equal,
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
    While,
    For,
    In,
    By,
    Defer,
    Sizeof,
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
    True,
    False,
    IntLiteral { text: String },
    Float32Literal { text: String },
    StringLiteral { text: String },
    Identifier { text: String },
    Eof,
}

pub struct Lexer {
    pub chars: Vec<char>,
    pub tokens: Vec<TokenKind>,
    position: usize,
}

impl Lexer {
    pub fn new(chars: Vec<char>) -> Self {
        Self {
            chars,
            tokens: Vec::new(),
            position: 0,
        }
    }

    fn char(&self) -> char {
        return *self.chars.get(self.position).unwrap_or(&'\0');
    }

    fn try_consume_string(&mut self, str: &str) -> bool {
        if self.position + str.len() > self.chars.len() {
            return false;
        }

        let mut is_last_char_valid_in_identifier = false;
        for (i, c) in str.chars().enumerate() {
            is_last_char_valid_in_identifier = Lexer::is_char_valid_in_identifier(c);
            if self.chars[self.position + i] != c {
                return false;
            }
        }

        let next_char_index = self.position + str.len();
        if is_last_char_valid_in_identifier {
            // The string needs to be a full word, rather than the start of another string,
            // eg, "for" but not "fortune".
            if next_char_index < self.chars.len() && Lexer::is_char_valid_in_identifier(self.chars[next_char_index]) {
                return false;
            }
        }

        self.position = next_char_index;

        true
    }

    pub fn lex(&mut self) {
        while self.position < self.chars.len() {
            if self.try_consume_string("!=") {
                self.tokens.push(TokenKind::NotEqual);
                continue;
            }

            if self.try_consume_string("==") {
                self.tokens.push(TokenKind::EqualEqual);
                continue;
            }

            if self.try_consume_string("<=") {
                self.tokens.push(TokenKind::LessEqual);
                continue;
            }

            if self.try_consume_string(">=") {
                self.tokens.push(TokenKind::GreaterEqual);
                continue;
            }

            if self.try_consume_string("&&") {
                self.tokens.push(TokenKind::And);
                continue;
            }

            if self.try_consume_string("||") {
                self.tokens.push(TokenKind::Or);
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

            if self.try_consume_string("var") {
                self.tokens.push(TokenKind::Var);
                continue;
            }

            if self.try_consume_string("val") {
                self.tokens.push(TokenKind::Val);
                continue;
            }

            if self.try_consume_string("fun") {
                self.tokens.push(TokenKind::Fun);
                continue;
            }

            if self.try_consume_string("struct") {
                self.tokens.push(TokenKind::Struct);
                continue;
            }

            if self.try_consume_string("return") {
                self.tokens.push(TokenKind::Return);
                continue;
            }

            if self.try_consume_string("extern") {
                self.tokens.push(TokenKind::Extern);
                continue;
            }

            if self.try_consume_string("if") {
                self.tokens.push(TokenKind::If);
                continue;
            }

            if self.try_consume_string("while") {
                self.tokens.push(TokenKind::While);
                continue;
            }

            if self.try_consume_string("for") {
                self.tokens.push(TokenKind::For);
                continue;
            }

            if self.try_consume_string("in") {
                self.tokens.push(TokenKind::In);
                continue;
            }

            if self.try_consume_string("by") {
                self.tokens.push(TokenKind::By);
                continue;
            }

            if self.try_consume_string("defer") {
                self.tokens.push(TokenKind::Defer);
                continue;
            }

            if self.try_consume_string("sizeof") {
                self.tokens.push(TokenKind::Sizeof);
                continue;
            }

            if self.try_consume_string("String") {
                self.tokens.push(TokenKind::String);
                continue;
            }

            if self.try_consume_string("Bool") {
                self.tokens.push(TokenKind::Bool);
                continue;
            }

            if self.try_consume_string("Void") {
                self.tokens.push(TokenKind::Void);
                continue;
            }

            if self.try_consume_string("Int8") {
                self.tokens.push(TokenKind::Int8);
                continue;
            }

            if self.try_consume_string("UInt8") {
                self.tokens.push(TokenKind::UInt8);
                continue;
            }

            if self.try_consume_string("Int16") {
                self.tokens.push(TokenKind::Int16);
                continue;
            }

            if self.try_consume_string("UInt16") {
                self.tokens.push(TokenKind::UInt16);
                continue;
            }

            if self.try_consume_string("Int32") {
                self.tokens.push(TokenKind::Int32);
                continue;
            }

            if self.try_consume_string("UInt32") {
                self.tokens.push(TokenKind::UInt32);
                continue;
            }

            if self.try_consume_string("Int64") {
                self.tokens.push(TokenKind::Int64);
                continue;
            }

            if self.try_consume_string("UInt64") {
                self.tokens.push(TokenKind::UInt64);
                continue;
            }

            if self.try_consume_string("UInt") {
                self.tokens.push(TokenKind::UInt);
                continue;
            }

            if self.try_consume_string("Int") {
                self.tokens.push(TokenKind::Int);
                continue;
            }

            if self.try_consume_string("Float32") {
                self.tokens.push(TokenKind::Float32);
                continue;
            }

            if self.try_consume_string("Float64") {
                self.tokens.push(TokenKind::Float64);
                continue;
            }

            if self.try_consume_string("true") {
                self.tokens.push(TokenKind::True);
                continue;
            }

            if self.try_consume_string("false") {
                self.tokens.push(TokenKind::False);
                continue;
            }

            if self.char().is_alphabetic() || self.char() == '_' {
                let mut c = self.chars[self.position];
                let start = self.position;

                while Lexer::is_char_valid_in_identifier(c) {
                    self.position += 1;
                    c = self.char();
                }

                self.tokens.push(TokenKind::Identifier {
                    text: self.chars[start..self.position].iter().collect(),
                });

                continue;
            }

            if self.char().is_numeric() {
                let mut c = self.chars[self.position];
                let mut has_decimal_point = false;
                let start = self.position;

                while c.is_numeric() || (c == '.'  && !has_decimal_point) {
                    if c == '.' {
                        has_decimal_point = true;
                    }

                    self.position += 1;
                    c = self.char();
                }

                let text = self.chars[start..self.position].iter().collect();

                if has_decimal_point {
                    self.tokens.push(TokenKind::Float32Literal { text });
                } else {
                    self.tokens.push(TokenKind::IntLiteral { text });
                }

                continue;
            }

            if self.char().is_whitespace() {
                self.position += 1;
                continue;
            }

            match self.char() {
                '(' => {
                    self.tokens.push(TokenKind::LParen);
                    self.position += 1;
                }
                ')' => {
                    self.tokens.push(TokenKind::RParen);
                    self.position += 1;
                }
                '{' => {
                    self.tokens.push(TokenKind::LBrace);
                    self.position += 1;
                }
                '}' => {
                    self.tokens.push(TokenKind::RBrace);
                    self.position += 1;
                }
                '[' => {
                    self.tokens.push(TokenKind::LBracket);
                    self.position += 1;
                }
                ']' => {
                    self.tokens.push(TokenKind::RBracket);
                    self.position += 1;
                }
                ',' => {
                    self.tokens.push(TokenKind::Comma);
                    self.position += 1;
                }
                ':' => {
                    self.tokens.push(TokenKind::Colon);
                    self.position += 1;
                }
                ';' => {
                    self.tokens.push(TokenKind::Semicolon);
                    self.position += 1;
                }
                '.' => {
                    self.tokens.push(TokenKind::Period);
                    self.position += 1;
                }
                '+' => {
                    self.tokens.push(TokenKind::Plus);
                    self.position += 1;
                }
                '-' => {
                    self.tokens.push(TokenKind::Minus);
                    self.position += 1;
                }
                '*' => {
                    self.tokens.push(TokenKind::Asterisk);
                    self.position += 1;
                }
                '&' => {
                    self.tokens.push(TokenKind::Ampersand);
                    self.position += 1;
                }
                '/' => {
                    self.tokens.push(TokenKind::Divide);
                    self.position += 1;
                }
                '=' => {
                    self.tokens.push(TokenKind::Equal);
                    self.position += 1;
                }
                '<' => {
                    self.tokens.push(TokenKind::Less);
                    self.position += 1;
                }
                '>' => {
                    self.tokens.push(TokenKind::Greater);
                    self.position += 1;
                }
                '!' => {
                    self.tokens.push(TokenKind::Not);
                    self.position += 1;
                }
                '"' => {
                    self.position += 1;
                    let mut c = self.chars[self.position];
                    let start = self.position;
                    while c != '"' {
                        if c == '\0' {
                            panic!("Hit EOF during string literal");
                        }

                        self.position += 1;
                        c = self.char();
                    }
                    self.tokens.push(TokenKind::StringLiteral {
                        text: self.chars[start..self.position].iter().collect(),
                    });
                    self.position += 1;
                }
                _ => panic!(
                    "Error while parsing at {}, {}",
                    self.chars[self.position], self.position
                ),
            }
        }
    }

    fn handle_single_line_comment(&mut self) {
        while self.char() != '\n' {
            self.position += 1;
        }

        self.position += 1;
    }

    fn handle_multi_line_comment(&mut self) {
        // Track the number of open multi-line comments, so that these comments can be nested.
        let mut open_count = 1;
        self.position += 1;

        while open_count > 0 {
            if self.try_consume_string("/*") {
                open_count += 1;
            } else if self.try_consume_string("*/") {
                open_count -= 1;
            }

            self.position += 1;
        }
    }

    fn is_char_valid_in_identifier(c: char) -> bool {
        c.is_alphabetic() || c.is_numeric() || c == '_'
    }
}
