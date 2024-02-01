use std::{
    env,
    ffi::OsStr,
    fs,
    io::{self, stdout, Write},
    path::Path,
    process::{Command, ExitCode},
};

use crate::{
    code_generator::CodeGenerator, lexer::Lexer, parser::Parser, type_checker::TypeChecker,
};

mod code_generator;
mod emitter;
mod emitter_stack;
mod environment;
mod lexer;
mod parser;
mod position;
mod type_checker;
mod types;

/*
 * BIG TODOS:
 * Complete type checking.
 * Incremental and parallel compilation.
 * Default parameters.
 * Variadic arguments.
 * Generics.
 *
 * SMALL TODOS:
 * for elem in array {}
 * Bitwise operations.
 * Make sure all non-void functions return, and that all functions return the correct value.
 * Enums.
 * Tagged unions? Still need to figure out what those should look like.
 * Some way to represent pointer to immutable data (eg. you can modify the pointer but not the thing it's pointing to).
 * Print the file name in error messages.
 *
 * NOTES:
 * After adding generics, add functions for alloc and free to the standard library.
 * fn alloc<T>(value: T) { calls malloc with sizeof T, assigns value to new memory, returns new memory }
 *
 * Of course, there will still need to be some equivalent of void* for direct malloc without having to first initialize the value
 * on the stack like the alloc<T> method would. eg, if you're creating an array that is too big for the stack you need to
 * allocate the memory on the heap then write into it directly.
 */

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        args.push(".".into());
    }

    let mut files = Vec::new();
    let path = Path::new(&args[1]);
    if is_path_source_file(path) {
        let chars = read_chars_at_path(path);
        files.push(chars);
    } else if path.is_dir() {
        collect_source_files(path, &mut files).unwrap();
    }

    if files.is_empty() {
        return ExitCode::SUCCESS;
    }

    let mut file_lexers = Vec::with_capacity(files.len());
    let mut had_lexing_error = false;

    for chars in files {
        let mut lexer = Lexer::new(chars);
        lexer.lex();
        had_lexing_error = had_lexing_error || lexer.had_error;
        file_lexers.push(lexer);
    }

    if had_lexing_error {
        return ExitCode::FAILURE;
    }

    let mut parser = Parser::new();

    let mut start_indices = Vec::with_capacity(file_lexers.len());
    for lexer in file_lexers {
        let start_index = parser.parse(lexer.tokens);
        start_indices.push(start_index);
    }

    if parser.had_error {
        return ExitCode::FAILURE;
    }

    let mut type_checker = TypeChecker::new(
        parser.nodes,
        parser.types,
        parser.function_declaration_indices,
        parser.struct_definition_indices,
        parser.array_type_kinds,
        parser.pointer_type_kinds,
    );
    for start_index in &start_indices {
        type_checker.check(*start_index);
    }

    if type_checker.had_error {
        return ExitCode::FAILURE;
    }

    let typed_nodes = type_checker
        .typed_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            match n.clone() {
                Some(n) => n,
                None => panic!("Mismatch between nodes and typed nodes, expected typed node for: {:?}", type_checker.nodes[i]),
            }
        })
        .collect();

    let mut code_generator = CodeGenerator::new(typed_nodes, type_checker.types);
    for start_index in &start_indices {
        code_generator.gen(*start_index);
    }

    let mut output_file = fs::File::create("bin/out.c").unwrap();

    if !code_generator.header_emitter.is_empty() {
        code_generator.header_emitter.write(&mut output_file);
        output_file.write_all("\n".as_bytes()).unwrap();
    }

    if !code_generator.prototype_emitter.is_empty() {
        code_generator.prototype_emitter.write(&mut output_file);
        output_file.write_all("\n".as_bytes()).unwrap();
    }

    code_generator.body_emitters.write(&mut output_file);

    let mut command_builder = Command::new("clang");

    let mut c_args_start = None;
    for (i, arg) in args.iter().enumerate() {
        if *arg == "-C" {
            c_args_start = Some(i + 1);
            break;
        }
    }

    if let Some(c_args_start) = c_args_start {
        command_builder.args(&args[c_args_start..]);
    }

    match command_builder
        .args(["bin/out.c", "-o", "bin/out.exe"])
        .output()
    {
        Err(_) => panic!("Couldn't compile using the system compiler!"),
        Ok(output) => {
            if !output.stderr.is_empty() {
                println!("System compiler error:\n");
                stdout().write_all(&output.stderr).unwrap();
                return ExitCode::FAILURE;
            }
        }
    }

    ExitCode::SUCCESS
}

fn collect_source_files(directory: &Path, files: &mut Vec<Vec<char>>) -> io::Result<()> {
    if !directory.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_source_files(&path, files)?;
        } else if is_path_source_file(&path) {
            let chars = read_chars_at_path(&path);
            files.push(chars);
        }
    }

    Ok(())
}

fn is_path_source_file(path: &Path) -> bool {
    path.extension().is_some() && path.extension().unwrap() == OsStr::new("quick")
}

fn read_chars_at_path(path: &Path) -> Vec<char> {
    fs::read_to_string(path).unwrap().chars().collect()
}
