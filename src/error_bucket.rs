use std::{collections::HashSet, sync::Arc};

use crate::{file_data::FileData, position::Position};

#[derive(Clone)]
pub struct CompilerError {
    pub position: Position,
    pub message: Arc<str>,
}

#[derive(Clone)]
pub struct ErrorBucket {
    errors: Vec<CompilerError>,
    occupied_positions: HashSet<Position>,
}

impl ErrorBucket {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            occupied_positions: HashSet::new(),
        }
    }

    pub fn error(&mut self, position: Position, tag: &str, message: &str, files: &[FileData]) {
        let full_message = format!(
            "{} error at {}:{}:{}: {}",
            tag,
            files[position.file_index].path.to_str().unwrap_or("?"),
            position.line,
            position.column,
            message,
        );

        self.push(CompilerError {
            position,
            message: full_message.into(),
        });
    }

    pub fn usage_error(&mut self, position: Position, files: &[FileData]) {
        let full_message = format!(
            "^ From usage at {}:{}:{}",
            files[position.file_index].path.to_str().unwrap_or("?"),
            position.line,
            position.column,
        );

        self.push(CompilerError {
            position,
            message: full_message.into(),
        });
    }

    pub fn extend(&mut self, other_bucket: &ErrorBucket) {
        for error in &other_bucket.errors {
            self.push(error.clone());
        }
    }

    pub fn push(&mut self, error: CompilerError) {
        if !self.occupied_positions.insert(error.position) {
            return;
        }

        self.errors.push(error);
    }

    pub fn len(&self) -> usize {
        self.errors.len()
    }

    pub fn print(&self) {
        for error in &self.errors {
            println!("{}", error.message);
        }
    }
}
