use std::{env, fs, path::PathBuf, process::ExitCode};

mod code_generator;
mod compiler;
mod const_value;
mod emitter;
mod emitter_stack;
mod environment;
mod file_data;
mod lexer;
mod namespace;
mod parser;
mod position;
mod tester;
mod type_kinds;
mod typer;

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        args.push(".".into());
    }

    let mut c_flags_start = None;
    let mut is_test = false;
    let mut is_expected = false;
    let mut is_debug_mode = false;
    let mut is_unsafe_mode = false;
    let mut do_clean = false;
    let mut do_measure_time = false;
    let mut do_use_msvc = false;

    let Some(core_path) = get_core_path() else {
        println!("Couldn't determine default core path");
        return ExitCode::FAILURE;
    };
    let mut core_path = core_path.to_str();
    let mut is_looking_for_core_path = false;

    for (i, arg) in args.iter().enumerate().skip(2) {
        if is_looking_for_core_path {
            is_looking_for_core_path = false;
            core_path = Some(arg);
            continue;
        }

        match arg.as_str() {
            "--test" => {
                is_test = true;
            }
            "--test-expected" => {
                is_test = true;
                is_expected = true;
            }
            "--clean" => {
                do_clean = true;
            }
            "--time" => {
                do_measure_time = true;
            }
            "--debug" => {
                is_debug_mode = true;
            }
            "--unsafe" => {
                is_unsafe_mode = true;
            }
            "--msvc" => {
                do_use_msvc = true;
            }
            "--core-path" => {
                is_looking_for_core_path = true;
            }
            "--cflags" => {
                c_flags_start = Some(i + 1);
                break;
            }
            _ => {
                println!("Unexpected argument \"{}\"", args[i]);
                return ExitCode::FAILURE;
            }
        }
    }

    if is_looking_for_core_path {
        println!("Expected \"--core-path\" to be followed by a path");
        return ExitCode::FAILURE;
    }

    let c_flags = if let Some(c_flags_start) = c_flags_start {
        &args[c_flags_start..]
    } else {
        &[]
    };

    let Some(core_path) = core_path else {
        println!("No core path available");
        return ExitCode::FAILURE;
    };

    if is_test {
        tester::test(&args[0], &args[1], core_path, do_clean, is_expected)
            .expect("Unable to complete tests");

        return ExitCode::SUCCESS;
    }

    if do_clean {
        clean_project(&args[1]);

        return ExitCode::SUCCESS;
    }

    compiler::compile(
        &args[1],
        core_path,
        is_debug_mode,
        is_unsafe_mode,
        do_measure_time,
        do_use_msvc,
        c_flags,
    )
}

fn clean_project(project_path: &str) {
    let mut build_path = PathBuf::from(project_path);
    build_path.push("./build");

    if fs::remove_dir_all(build_path).is_ok() {
        println!("Cleaned build directory of project \"{}\"!", project_path);
    }
}

fn get_core_path() -> Option<PathBuf> {
    Some(env::current_exe().ok()?.parent()?.join("Core/"))
}
