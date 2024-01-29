#[derive(Clone, Copy, Debug)]
pub struct Position {
    pub index: usize,
    pub line: usize,
    pub column: usize,
}

impl Position {
    pub fn new() -> Self {
        Self { index: 0, line: 1, column: 1 }
    }

    pub fn newline(&mut self) {
        self.index += 1;
        self.line += 1;
        self.column = 1;
    }

    pub fn advance(&mut self) {
        self.advance_by(1);
    }

    pub fn advance_by(&mut self, count: usize) {
        self.index += count;
        self.column += count;
    }
}