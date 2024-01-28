use std::{fs, io::Write};

use crate::{
    code_generator::CodeGenerator, lexer::Lexer, parser::Parser, type_checker::TypeChecker,
};

mod code_generator;
mod emitter;
mod emitter_stack;
mod environment;
mod lexer;
mod parser;
mod type_checker;
mod types;

/*
 * BIG TODOS:
 * Graceful error handling (keep lexing/parsing/etc even if you hit an error)
 * Report error messages that are helpful, have file positions.
 * Complete type checking.
 * Simple type inference.
 * If/While
 * For
 * Pointers.
 * Generics.
 * Variadic arguments.
 * Default parameters.
 * Multiple files and incremental compilation.
 *
 * Create special statements for alloc and free:
 * var a: Int* = alloc 5;
 * free a;
 *
 * note that these both must be STATEMENT, alloc 5 cannot be an expression because it needs to be codegen'ed to:
 * int* a = malloc(sizeof(int));
 * *a = 5;
 * which isn't an expression, and wouldn't be helpful as an expression anyway, because there are very few times
 * you want to allocate and then not immediately assign the resulting pointer to a variable.
 *
 * There should be const versions of pointer types that you can't free,
 * including strings, ie: const String = "hello"; String = ... some allocation ...
 * or maybe String should be the const version of a character pointer?
 *
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

    println!("~~ Checking ~~");

    let mut type_checker = TypeChecker::new(
        parser.nodes,
        parser.types,
        parser.function_declaration_indices,
        parser.struct_definition_indices,
        parser.array_type_kinds,
    );
    type_checker.check(start_index);

    let typed_nodes = type_checker
        .typed_nodes
        .iter()
        .map(|n| n.clone().unwrap())
        .collect();

    println!("~~ Generating ~~");

    let mut code_generator = CodeGenerator::new(typed_nodes, type_checker.types);
    code_generator.gen(start_index);

    let mut output_file = fs::File::create("out/test.c").unwrap();
    output_file
        .write_all(code_generator.header_emitter.string.as_bytes())
        .unwrap();
    if !code_generator.header_emitter.string.is_empty() {
        output_file.write_all("\n".as_bytes()).unwrap();
    }
    output_file
        .write_all(code_generator.prototype_emitter.string.as_bytes())
        .unwrap();
    if !code_generator.prototype_emitter.string.is_empty() {
        output_file.write_all("\n".as_bytes()).unwrap();
    }
    output_file
        .write_all(code_generator.body_emitters.string().as_bytes())
        .unwrap();

    println!("~~~ Calling system compiler ~~~");
    match std::process::Command::new("clang")
        .args(["out/test.c", "-o", "out/test.exe"])
        .output()
    {
        Err(_) => panic!("Couldn't compile using the system compiler!"),
        Ok(output) => {
            if !output.stderr.is_empty() {
                println!("System compiler error:\n");
                std::io::stdout().write_all(&output.stderr).unwrap();
            }
        }
    }
}
