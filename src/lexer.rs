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
    Plus,
    Minus,
    Asterisk,
    ForwardSlash,
    Equals,
    Var,
    Val,
    Fun,
    Return,
    Extern,
    Int,
    String,
    IntLiteral { text: String },
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
        if self.position + str.len() >= self.chars.len() {
            return false;
        }

        for (i, c) in str.chars().enumerate() {
            if self.chars[self.position + i] != c {
                return false;
            }
        }

        self.position += str.len();

        true
    }

    pub fn lex(&mut self) {
        while self.position < self.chars.len() {
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
                '/' => {
                    self.tokens.push(TokenKind::ForwardSlash);
                    self.position += 1;
                }
                '=' => {
                    self.tokens.push(TokenKind::Equals);
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
                _ if self.try_consume_string("var") => {
                    self.tokens.push(TokenKind::Var);
                }
                _ if self.try_consume_string("val") => {
                    self.tokens.push(TokenKind::Val);
                }
                _ if self.try_consume_string("fun") => {
                    self.tokens.push(TokenKind::Fun);
                }
                _ if self.try_consume_string("return") => {
                    self.tokens.push(TokenKind::Return);
                }
                _ if self.try_consume_string("extern") => {
                    self.tokens.push(TokenKind::Extern);
                }
                _ if self.try_consume_string("Int") => {
                    self.tokens.push(TokenKind::Int);
                }
                _ if self.try_consume_string("String") => {
                    self.tokens.push(TokenKind::String);
                }
                c if c.is_alphabetic() => {
                    let mut c = self.chars[self.position];
                    let start = self.position;
                    while c.is_alphabetic() {
                        self.position += 1;
                        c = self.char();
                    }
                    self.tokens.push(TokenKind::Identifier {
                        text: self.chars[start..self.position].iter().collect(),
                    });
                }
                c if c.is_numeric() => {
                    let mut c = self.chars[self.position];
                    let start = self.position;
                    while c.is_numeric() {
                        self.position += 1;
                        c = self.char();
                    }
                    self.tokens.push(TokenKind::IntLiteral {
                        text: self.chars[start..self.position].iter().collect(),
                    });
                }
                c if c.is_whitespace() => {
                    self.position += 1;
                }
                _ => panic!(
                    "Error while parsing at {}, {}",
                    self.chars[self.position], self.position
                ),
            }
        }
    }
}
