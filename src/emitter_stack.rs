use std::{fs::File, io::BufWriter};

use crate::emitter::Emitter;

pub struct EmitterSet {
    pub top: Emitter,
    pub body: Emitter,
    bottom: Vec<Emitter>,
}

pub struct EmitterStack {
    stack: Vec<EmitterSet>,
}

impl EmitterStack {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn push(&mut self, indent_amount: usize) {
        let indent_count = if let Some(last_set) = self.stack.last() {
            last_set.top.indent_count() + indent_amount
        } else {
            0
        };

        self.stack.push(EmitterSet {
            top: Emitter::new(indent_count),
            body: Emitter::new(indent_count),
            bottom: Vec::new(),
        });
    }

    pub fn top(&mut self) -> &mut EmitterSet {
        self.stack.last_mut().unwrap()
    }

    pub fn exiting_all_scopes(&mut self) {
        let Some((destination, sources)) = self.stack.split_last_mut() else {
            return;
        };

        for segment in destination.bottom.iter().rev() {
            destination.body.append(&segment.string);
        }

        for source in sources.iter().rev() {
            if source.bottom.is_empty() {
                continue;
            }

            let indent_amount = destination.body.indent_count() - source.body.indent_count();
            for segment in source.bottom.iter().rev() {
                destination
                    .body
                    .append_indented(&segment.string, indent_amount);
            }
        }
    }

    pub fn pop(&mut self, needs_bottom: bool) {
        let Some(source) = self.stack.pop() else {
            return;
        };

        let destination = self.stack.last_mut().unwrap();

        destination.body.append(&source.top.string);
        destination.body.append(&source.body.string);

        if needs_bottom {
            for segment in source.bottom.iter().rev() {
                destination.body.append(&segment.string);
            }
        }
    }

    pub fn pop_to_bottom(&mut self) {
        let Some(source) = self.stack.pop() else {
            return;
        };

        let mut destination = Emitter::new(0);

        destination.append(&source.top.string);
        destination.append(&source.body.string);

        for segment in source.bottom.iter().rev() {
            destination.append(&segment.string);
        }

        self.stack.last_mut().unwrap().bottom.push(destination);
    }

    pub fn write(&self, file: &mut BufWriter<File>) {
        if self.stack.len() != 1 {
            panic!(
                "invalid stack length while trying to get result of emitter stack: {}",
                self.stack.len()
            );
        }

        self.stack[0].body.write(file);
    }
}
