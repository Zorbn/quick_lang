use std::{fs, io::Write};

use crate::{lexer::Lexer, parser::Parser, codegenerator::CodeGenerator};

mod lexer;
mod parser;
mod codegenerator;
mod emitter;

/*
 * Stuff not attempted yet:
 * Graceful error handling (keep lexing/parsing/etc even if you hit an error)
 * Report error messages that are helpful, have file positions.
 * Generics, etc.
 */

fn main() {
    let chars: Vec<char> = fs::read_to_string("test.quick").unwrap().chars().collect();

    println!("~~ Lexing ~~");

    let mut lexer = Lexer::new(chars);
    lexer.lex();

    // for token in &lexer.tokens {
    //     println!("{:?}", *token);
    // }

    println!("~~ Parsing ~~");

    let mut parser = Parser::new(lexer.tokens);
    let start_index = parser.parse();

    // for node in &parser.nodes {
    //     println!("{:?}", *node);
    // }
    //
    println!("~~ Generating ~~");

    let mut code_generator = CodeGenerator::new(parser.nodes);
    code_generator.gen(start_index);

    let mut output_file = fs::File::create("out/test.c").unwrap();
    output_file.write_all(code_generator.header_emitter.string.as_bytes()).unwrap();
    if !code_generator.header_emitter.string.is_empty() {
        output_file.write_all("\n".as_bytes()).unwrap();
    }
    output_file.write_all(code_generator.prototype_emitter.string.as_bytes()).unwrap();
    if !code_generator.prototype_emitter.string.is_empty() {
        output_file.write_all("\n".as_bytes()).unwrap();
    }
    output_file.write_all(code_generator.body_emitter.string.as_bytes()).unwrap();
}
