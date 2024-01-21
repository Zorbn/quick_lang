#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Plus,
    Minus,
    Asterisk,
    ForwardSlash,
    Equals,
    IntLiteral {
        text: String,
    },
    Identifier {
        text: String,
    },
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

    pub fn lex(&mut self) {
        while self.position < self.chars.len() {
            match self.char() {
                '(' => {
                    self.tokens.push(TokenKind::LParen);
                    self.position += 1;
                },
                ')' => {
                    self.tokens.push(TokenKind::RParen);
                    self.position += 1;
                },
                '{' => {
                    self.tokens.push(TokenKind::LBrace);
                    self.position += 1;
                },
                '}' => {
                    self.tokens.push(TokenKind::RBrace);
                    self.position += 1;
                },
                ',' => {
                    self.tokens.push(TokenKind::Comma);
                    self.position += 1;
                },
                ';' => {
                    self.tokens.push(TokenKind::Semicolon);
                    self.position += 1;
                },
                '+' => {
                    self.tokens.push(TokenKind::Plus);
                    self.position += 1;
                },
                '-' => {
                    self.tokens.push(TokenKind::Minus);
                    self.position += 1;
                },
                '*' => {
                    self.tokens.push(TokenKind::Asterisk);
                    self.position += 1;
                },
                '/' => {
                    self.tokens.push(TokenKind::ForwardSlash);
                    self.position += 1;
                },
                '=' => {
                    self.tokens.push(TokenKind::Equals);
                    self.position += 1;
                },
                c if c.is_alphabetic() => {
                    let mut c = self.chars[self.position];
                    let start = self.position;
                    while c.is_alphabetic() {
                        self.position += 1;
                        c = self.char();
                    }
                    self.tokens.push(TokenKind::Identifier { text: self.chars[start..self.position].iter().collect() });
                },
                c if c.is_numeric() => {
                    let mut c = self.chars[self.position];
                    let start = self.position;
                    while c.is_numeric() {
                        self.position += 1;
                        c = self.char();
                    }
                    self.tokens.push(TokenKind::IntLiteral { text: self.chars[start..self.position].iter().collect() });
                },
                c if c.is_whitespace() => {
                    self.position += 1;
                },
                _ => panic!("Error while parsing at {}, {}", self.chars[self.position], self.position),
            }
        }
    }
}