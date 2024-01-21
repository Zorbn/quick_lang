pub struct Emitter {
    pub string: String,
    indent_count: usize,
    is_on_newline: bool,
}

impl Emitter {
    pub fn new() -> Self {
        Self {
            string: String::new(),
            indent_count: 0,
            is_on_newline: true,
        }
    }

    pub fn emit(&mut self, str: &str) {
        if self.is_on_newline {
            for _ in 0..self.indent_count {
                self.string.push('\t');
            }

            self.is_on_newline = false;
        }

        self.string.push_str(str);
    }

    pub fn newline(&mut self) {
        self.string.push('\n');
        self.is_on_newline = true;
    }

    pub fn emitln(&mut self, str: &str) {
        self.emit(str);
        self.newline();
    }

    pub fn indent(&mut self) {
        self.indent_count += 1;
    }

    pub fn unindent(&mut self) {
        self.indent_count -= 1;
    }
}