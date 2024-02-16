use std::{ffi::OsStr, fs, io, path::Path, process::ExitCode, sync::Arc, time::Instant};

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

    let lexers = if let Some(lexers) = lex(&files) {
        lexers
    } else {
        return ExitCode::FAILURE;
    };

    let parsers = if let Some(parsers) = parse(lexers, &files) {
        parsers
    } else {
        return ExitCode::FAILURE;
    };

    let typers = if let Some(typers) = check(parsers, &files) {
        typers
    } else {
        return ExitCode::FAILURE
    };

    gen(typers, &files, is_debug_mode);

    println!(
        "Frontend finished in: {:.2?}ms",
        frontend_start.elapsed().as_millis()
    );

//     let backend_start = Instant::now();
//
//     match Command::new("clang")
//         .args(["bin/out.c", "-o", "bin/out.exe"])
//         .args(c_flags)
//         .output()
//     {
//         Err(_) => panic!("couldn't compile using the system compiler!"),
//         Ok(output) => {
//             if !output.stderr.is_empty() {
//                 println!("System compiler error: {}\n", output.stderr);
//                 return ExitCode::FAILURE;
//             }
//         }
//     }
//
//     println!(
//         "Backend finished in: {:.2?}ms",
//         backend_start.elapsed().as_millis()
//     );

    ExitCode::SUCCESS
}

fn lex(files: &Arc<Vec<FileData>>) -> Option<Vec<Lexer>> {
    let mut lexers = Vec::with_capacity(files.len());
    let mut had_lexing_error = false;

    for i in 0..files.len() {
        let mut lexer = Lexer::new(i, files.clone());
        lexer.lex();
        had_lexing_error = had_lexing_error || lexer.had_error;
        lexers.push(lexer);
    }

    if had_lexing_error {
        None
    } else {
        Some(lexers)
    }
}

fn parse(lexers: Vec<Lexer>, files: &Arc<Vec<FileData>>) -> Option<Vec<Parser>> {
    let mut parsers = Vec::with_capacity(lexers.len());
    let mut had_parsing_error = false;

    for (i, lexer) in lexers.into_iter().enumerate() {
        let mut parser = Parser::new(i, files.clone());
        parser.parse(lexer.tokens);
        had_parsing_error = had_parsing_error || parser.error_count > 0;
        parsers.push(parser);
    }

    if had_parsing_error {
        None
    } else {
        Some(parsers)
    }
}

fn check(parsers: Vec<Parser>, files: &Arc<Vec<FileData>>) -> Option<Vec<Typer>> {
    let mut typers = Vec::with_capacity(parsers.len());
    let mut had_typing_error = false;

    let mut all_nodes = Vec::with_capacity(parsers.len());
    let mut all_start_indices = Vec::with_capacity(parsers.len());
    let mut all_definition_indices = Vec::with_capacity(parsers.len());

    for parser in parsers {
        all_nodes.push(parser.nodes);
        all_start_indices.push(parser.start_index);
        all_definition_indices.push(parser.definition_indices);
    }

    let all_nodes = Arc::new(all_nodes);
    let all_definition_indices = Arc::new(all_definition_indices);

    for (i, start_index) in all_start_indices.into_iter().enumerate() {
        let mut typer = Typer::new(all_nodes.clone(), all_definition_indices.clone(), files.clone(), i);
        typer.check(start_index);
        had_typing_error = had_typing_error || typer.error_count > 0;
        typers.push(typer);
    }

    if had_typing_error {
        None
    } else {
        Some(typers)
    }
}

fn gen(typers: Vec<Typer>, files: &Arc<Vec<FileData>>, is_debug_mode: bool) {
    for (i, typer) in typers.into_iter().enumerate() {
        let mut code_generator = CodeGenerator::new(
            typer.typed_nodes,
            typer.type_kinds,
            typer.main_function_type_kind_id,
            typer.typed_definition_indices,
            is_debug_mode,
        );
        code_generator.gen();

        let mut output_path = Path::new("bin/").join(&files[i].path);
        output_path.set_extension("c");
        let mut output_file = fs::File::create(output_path).unwrap();

        code_generator.header_emitter.write(&mut output_file);
        code_generator
            .type_prototype_emitter
            .write(&mut output_file);
        code_generator
            .function_prototype_emitter
            .write(&mut output_file);
        code_generator.body_emitters.write(&mut output_file);
    }
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