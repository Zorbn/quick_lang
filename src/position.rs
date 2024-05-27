#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Position {
    pub index: usize,
    pub line: usize,
    pub column: usize,
    pub file_index: usize,
}

impl Position {
    pub fn new(file_index: usize) -> Self {
        Self {
            index: 0,
            line: 1,
            column: 1,
            file_index,
        }
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
