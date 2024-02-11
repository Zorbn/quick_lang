use std::{
    env,
    ffi::OsStr,
    fs,
    io::{self, stdout, Write},
    path::Path,
    process::{Command, ExitCode},
    sync::Arc,
    time::Instant,
};

use file_data::FileData;

use crate::{
    code_generator::CodeGenerator,
    lexer::Lexer,
    parser::{NodeKind, Parser},
    type_checker::{TypeChecker, TypedNode},
};

mod code_generator;
mod const_value;
mod emitter;
mod emitter_stack;
mod environment;
mod file_data;
mod lexer;
mod parser;
mod position;
mod type_checker;
mod types;

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        args.push(".".into());
    }

    let mut c_args_start = None;
    let mut is_debug_mode = false;
    for (i, arg) in args.iter().enumerate().skip(2) {
        match arg.as_str() {
            "--debug" => {
                is_debug_mode = true;
            }
            "--cflags" => {
                c_args_start = Some(i + 1);
                break;
            }
            _ => {
                println!("Unexpected argument \"{}\"", args[i]);
                return ExitCode::FAILURE;
            }
        }
    }

    let mut files = Vec::new();
    let path = Path::new(&args[1]);
    if is_path_source_file(path) {
        let chars = read_chars_at_path(path);
        files.push(FileData {
            path: path.to_path_buf(),
            chars,
        });
    } else if path.is_dir() {
        collect_source_files(path, &mut files).unwrap();
    }
    let files = Arc::new(files);

    if files.is_empty() {
        return ExitCode::SUCCESS;
    }

    let frontend_start = Instant::now();

    let mut file_lexers = Vec::with_capacity(files.len());
    let mut had_lexing_error = false;

    for i in 0..files.len() {
        let mut lexer = Lexer::new(i, files.clone());
        lexer.lex();
        had_lexing_error = had_lexing_error || lexer.had_error;
        file_lexers.push(lexer);
    }

    if had_lexing_error {
        return ExitCode::FAILURE;
    }

    let mut parser = Parser::new(files.clone());

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
        parser.type_kinds,
        parser.array_type_kinds,
        parser.pointer_type_kinds,
        parser.function_type_kinds,
        parser.struct_type_kinds,
        parser.definition_indices,
        files.clone(),
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
        .map(|n| match n.clone() {
            Some(n) => n,
            None => TypedNode {
                node_kind: NodeKind::Error,
                node_type: None,
            },
        })
        .collect();

    let mut code_generator = CodeGenerator::new(
        typed_nodes,
        type_checker.type_kinds,
        type_checker.generic_usages,
        is_debug_mode,
    );
    for start_index in &start_indices {
        code_generator.gen(*start_index);
    }

    let mut output_file = fs::File::create("bin/out.c").unwrap();

    code_generator.header_emitter.write(&mut output_file);
    code_generator
        .type_prototype_emitter
        .write(&mut output_file);
    code_generator
        .function_prototype_emitter
        .write(&mut output_file);
    code_generator.body_emitters.write(&mut output_file);

    println!(
        "Frontend finished in: {:.2?}ms",
        frontend_start.elapsed().as_millis()
    );

    let mut command_builder = Command::new("clang");

    if let Some(c_args_start) = c_args_start {
        command_builder.args(&args[c_args_start..]);
    }

    let backend_start = Instant::now();

    match command_builder
        .args(["bin/out.c", "-o", "bin/out.exe"])
        .output()
    {
        Err(_) => panic!("couldn't compile using the system compiler!"),
        Ok(output) => {
            if !output.stderr.is_empty() {
                println!("System compiler error:\n");
                stdout().write_all(&output.stderr).unwrap();
                return ExitCode::FAILURE;
            }
        }
    }

    println!(
        "Backend finished in: {:.2?}ms",
        backend_start.elapsed().as_millis()
    );

    ExitCode::SUCCESS
}

fn collect_source_files(directory: &Path, files: &mut Vec<FileData>) -> io::Result<()> {
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
            files.push(FileData { path, chars });
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
