use std::{
    ffi::{OsStr, OsString},
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    sync::Arc,
    time::Instant,
};

use crate::{
    code_generator::CodeGenerator, file_data::FileData, lexer::Lexer, parser::Parser, typer::Typer,
};

pub fn compile(
    project_path: &str,
    core_path: &str,
    is_debug_mode: bool,
    use_msvc: bool,
    c_flags: &[String],
) -> ExitCode {
    let mut files = Vec::new();

    let core_path = Path::new(core_path);
    collect_source_files(core_path, core_path, &mut files).unwrap();

    let project_path = Path::new(project_path);
    if is_path_source_file(project_path) {
        let chars = read_chars_at_path(project_path);
        files.push(FileData {
            path: project_path
                .strip_prefix(project_path.parent().unwrap())
                .unwrap()
                .to_path_buf(),
            chars,
        });
    } else if project_path.is_dir() {
        collect_source_files(project_path, project_path, &mut files).unwrap();
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

    let paths_components = Arc::new(get_paths_components(&files));

    let typers = if let Some(typers) = check(parsers, &files, paths_components) {
        typers
    } else {
        return ExitCode::FAILURE;
    };

    let output_paths = get_output_paths(&files);

    gen(typers, &output_paths, is_debug_mode);

    println!(
        "Frontend finished in: {:.2?}ms",
        frontend_start.elapsed().as_millis()
    );

    let backend_start = Instant::now();

    let mut compiler_command =
        create_compiler_command(is_debug_mode, use_msvc, c_flags, &output_paths);

    match compiler_command.output() {
        Err(_) => panic!("couldn't compile using the system compiler!"),
        Ok(output) => {
            if !output.status.success() {
                println!("System compiler error:\n");
                io::stdout().write_all(&output.stdout).unwrap();
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

fn check(
    parsers: Vec<Parser>,
    files: &Arc<Vec<FileData>>,
    paths_components: Arc<Vec<Vec<OsString>>>,
) -> Option<Vec<Typer>> {
    let mut typers = Vec::with_capacity(parsers.len());
    let mut had_typing_error = false;

    let mut all_nodes = Vec::with_capacity(parsers.len());
    let mut all_start_indices = Vec::with_capacity(parsers.len());

    for parser in parsers {
        all_nodes.push(parser.nodes);
        all_start_indices.push(parser.start_index);
    }

    let all_nodes = Arc::new(all_nodes);

    for (i, start_index) in all_start_indices.iter().enumerate() {
        let mut typer = Typer::new(
            all_nodes.clone(),
            &all_start_indices,
            files.clone(),
            paths_components.clone(),
            i,
        );
        typer.check(*start_index);
        had_typing_error = had_typing_error || typer.error_count > 0;
        typers.push(typer);
    }

    if had_typing_error {
        None
    } else {
        Some(typers)
    }
}

fn gen(typers: Vec<Typer>, output_paths: &[PathBuf], is_debug_mode: bool) {
    for (typer, output_path) in typers.into_iter().zip(output_paths.iter()) {
        let mut code_generator = CodeGenerator::new(
            typer.typed_nodes,
            typer.type_kinds,
            typer.namespaces,
            typer.string_view_type_kind_id,
            typer.main_function_declaration,
            typer.typed_definitions,
            is_debug_mode,
        );
        code_generator.gen();

        fs::create_dir_all(output_path.parent().unwrap()).unwrap();
        let output_file = File::create(output_path).unwrap();
        let mut output_file = BufWriter::new(output_file);

        code_generator.header_emitter.write(&mut output_file);
        code_generator
            .type_prototype_emitter
            .write(&mut output_file);
        code_generator
            .function_prototype_emitter
            .write(&mut output_file);
        code_generator
            .global_variable_emitter
            .write(&mut output_file);
        code_generator.body_emitters.write(&mut output_file);

        output_file.flush().unwrap();
    }
}

fn get_output_paths(files: &Arc<Vec<FileData>>) -> Vec<PathBuf> {
    let mut output_paths = Vec::new();

    for file in files.iter() {
        let mut output_path = Path::new("build/").join(&file.path);
        output_path.set_extension("c");
        output_paths.push(output_path);
    }

    output_paths
}

fn get_paths_components(files: &Arc<Vec<FileData>>) -> Vec<Vec<OsString>> {
    let mut paths_components = Vec::new();

    for file in files.iter() {
        let mut components = Vec::new();
        let path = file.path.with_extension("");

        for component in path.components() {
            let component = component.as_os_str().to_os_string();
            components.push(component);
        }

        paths_components.push(components)
    }

    paths_components
}

fn collect_source_files(
    root_directory: &Path,
    directory: &Path,
    files: &mut Vec<FileData>,
) -> io::Result<()> {
    if !directory.is_dir() {
        return Ok(());
    }

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_source_files(root_directory, &path, files)?;
        } else if is_path_source_file(&path) {
            let chars = read_chars_at_path(&path);
            let path = path.strip_prefix(root_directory).unwrap().to_path_buf();

            files.push(FileData {
                path,
                chars,
            });
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

fn create_compiler_command(
    is_debug_mode: bool,
    use_msvc: bool,
    c_flags: &[String],
    output_paths: &[PathBuf],
) -> Command {
    if use_msvc {
        let mut compiler_command = Command::new("cl");

        if is_debug_mode {
            compiler_command.args(["-Z7", "-MTd", "/DEBUG"]);
        }

        compiler_command
            .args(["-Fe:", "build/Out.exe"])
            .args(output_paths)
            .args(c_flags);

        compiler_command
    } else {
        let mut compiler_command = Command::new("clang");

        if is_debug_mode {
            compiler_command.args(["-g", "-gcodeview"]);
        }

        compiler_command
            .args(["-o", "build/Out.exe"])
            .args(output_paths)
            .args(c_flags);

        compiler_command
    }
}
