use crate::emitter::Emitter;

pub struct EmitterSet {
    pub top: Emitter,
    pub body: Emitter,
}

pub struct EmitterStack {
    stack: Vec<EmitterSet>,
}

impl EmitterStack {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn push(&mut self) {
        let indent_count = if let Some(last_set) = self.stack.last() {
            last_set.top.indent_count() + 1
        } else {
            0
        };

        self.stack.push(EmitterSet {
            top: Emitter::new(indent_count),
            body: Emitter::new(indent_count),
        });
    }

    pub fn top(&mut self) -> &mut EmitterSet {
        self.stack.last_mut().unwrap()
    }

    pub fn pop(&mut self) {
        let Some(source) = self.stack.pop() else {
            return;
        };

        let destination = self.stack.last_mut().unwrap();

        destination.body.emit(&source.top.string);
        destination.body.emit(&source.body.string);
    }

    pub fn string(&self) -> &str {
        if self.stack.len() != 1 {
            panic!(
                "Invalid stack length while trying to get result of emitter stack: {}",
                self.stack.len()
            );
        }

        &self.stack[0].body.string
    }
}
