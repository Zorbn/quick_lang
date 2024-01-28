use crate::emitter::Emitter;

pub struct EmitterSet {
    pub top: Emitter,
    pub body: Emitter,
    bottom: Emitter,
}

pub struct EmitterStack {
    stack: Vec<EmitterSet>,
}

#[derive(Clone, Copy, Debug)]
pub enum EmitterKind {
    Body,
    Bottom,
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
            bottom: Emitter::new(0),
        });
    }

    pub fn top(&mut self) -> &mut EmitterSet {
        self.stack.last_mut().unwrap()
    }

    pub fn exiting_all_scopes(&mut self) {
        for i in 0..self.stack.len() {
            if self.stack[i].bottom.string.is_empty() {
                continue;
            }

            let indent_amount = self.top().body.indent_count() - self.stack[i].body.indent_count();
            let string = self.stack[i].bottom.string.clone();
            self.top().body.append_indented(&string, indent_amount);
        }
    }

    pub fn pop(&mut self, needs_bottom: bool) {
        self.pop_to(EmitterKind::Body, needs_bottom);
    }

    pub fn pop_to_bottom(&mut self) {
        self.pop_to(EmitterKind::Bottom, true);
    }

    fn pop_to(&mut self, kind: EmitterKind, needs_bottom: bool) {
        let Some(source) = self.stack.pop() else {
            return;
        };

        let destination = self.stack.last_mut().unwrap();

        let emitter = match kind {
            EmitterKind::Body => &mut destination.body,
            EmitterKind::Bottom => &mut destination.bottom,
        };

        emitter.append(&source.top.string);
        emitter.append(&source.body.string);

        if needs_bottom {
            emitter.append(&source.bottom.string);
        }
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
