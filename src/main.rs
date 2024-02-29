use std::{env, path::PathBuf, process::ExitCode};

mod code_generator;
mod compiler;
mod const_value;
mod emitter;
mod emitter_stack;
mod environment;
mod file_data;
mod lexer;
mod parser;
mod position;
mod type_kinds;
mod typer;

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        args.push(".".into());
    }

    let mut c_flags_start = None;
    let mut is_debug_mode = false;
    let mut use_msvc = false;

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
            "--debug" => {
                is_debug_mode = true;
            }
            "--msvc" => {
                use_msvc = true;
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

    compiler::compile(&args[1], core_path, is_debug_mode, use_msvc, c_flags)
}

fn get_core_path() -> Option<PathBuf> {
    Some(env::current_exe().ok()?.parent()?.join("Core/"))
}