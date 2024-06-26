use std::fs::File;
use std::io::{BufWriter, Write};

pub struct Emitter {
    pub string: String,
    line_string: String,
    indent_count: usize,
    is_on_newline: bool,
}

impl Emitter {
    pub fn new(indent_count: usize) -> Self {
        Self {
            string: String::new(),
            line_string: String::new(),
            indent_count,
            is_on_newline: true,
        }
    }

    fn flush_line(&mut self) {
        self.string.push_str(&self.line_string);
        self.line_string.clear();
    }

    pub fn append(&mut self, str: &str) {
        self.flush_line();
        self.string.push_str(str);
    }

    pub fn append_indented(&mut self, str: &str, indent_amount: usize) {
        self.flush_line();

        for line in str.lines() {
            for _ in 0..indent_amount {
                self.string.push('\t');
            }

            self.string.push_str(line);
            self.string.push('\n');
        }
    }

    fn indent_on_newlines(&mut self) {
        if self.is_on_newline {
            for _ in 0..self.indent_count {
                self.line_string.push('\t');
            }

            self.is_on_newline = false;
        }
    }

    pub fn emit(&mut self, str: &str) {
        self.indent_on_newlines();
        self.line_string.push_str(str);
    }

    pub fn emit_char(&mut self, c: char) {
        self.indent_on_newlines();
        self.line_string.push(c);
    }

    pub fn newline(&mut self) {
        self.line_string.push('\n');
        self.flush_line();

        self.is_on_newline = true;
    }

    pub fn emitln(&mut self, str: &str) {
        self.emit(str);
        self.newline();
    }

    pub fn indent_count(&self) -> usize {
        self.indent_count
    }

    pub fn indent(&mut self) {
        self.indent_count += 1;
    }

    pub fn unindent(&mut self) {
        self.indent_count -= 1;
    }

    pub fn emitln_before(&mut self, str: &str) {
        for _ in 0..self.indent_count {
            self.string.push('\t');
        }

        self.string.push_str(str);
    }

    pub fn write(&self, file: &mut BufWriter<File>) {
        write!(file, "{}", self.string).unwrap();
    }
}
