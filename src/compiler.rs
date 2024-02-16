use std::{ffi::OsStr, fs, io::{self, Write}, path::Path, process::{Command, ExitCode}, sync::Arc, time::Instant};

use crate::{code_generator::CodeGenerator, file_data::FileData, lexer::Lexer, parser::Parser, typer::Typer};

pub fn compile(project_path: &str, is_debug_mode: bool, c_flags: &[String]) -> ExitCode {
    let mut files = Vec::new();
    let path = Path::new(project_path);
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

    let mut parser_start_indices = Vec::with_capacity(file_lexers.len());
    for lexer in file_lexers {
        let start_index = parser.parse(lexer.tokens);
        parser_start_indices.push(start_index);
    }

    if parser.error_count > 0 {
        return ExitCode::FAILURE;
    }

    let mut typer = Typer::new(parser.nodes, parser.definition_indices, files.clone());

    for start_index in &parser_start_indices {
        typer.check(*start_index);
    }

    if typer.error_count > 0 {
        return ExitCode::FAILURE;
    }

    let mut code_generator = CodeGenerator::new(
        typer.typed_nodes,
        typer.type_kinds,
        typer.main_function_type_kind_id,
        typer.typed_definition_indices,
        parser.extern_function_names,
        is_debug_mode,
    );
    code_generator.gen();

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

    let backend_start = Instant::now();

    match Command::new("clang")
        .args(["bin/out.c", "-o", "bin/out.exe"])
        .args(c_flags)
        .output()
    {
        Err(_) => panic!("couldn't compile using the system compiler!"),
        Ok(output) => {
            if !output.stderr.is_empty() {
                println!("System compiler error:\n");
                io::stdout().write_all(&output.stderr).unwrap();
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