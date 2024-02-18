use std::{env, process::ExitCode};

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
mod utils;

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().collect();

    if args.len() == 1 {
        args.push(".".into());
    }

    let mut c_flags_start = None;
    let mut is_debug_mode = false;
    let mut use_msvc = false;
    for (i, arg) in args.iter().enumerate().skip(2) {
        match arg.as_str() {
            "--debug" => {
                is_debug_mode = true;
            }
            "--msvc" => {
                use_msvc = true;
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

    let c_flags = if let Some(c_flags_start) = c_flags_start {
        &args[c_flags_start..]
    } else {
        &[]
    };

    compiler::compile(&args[1], is_debug_mode, use_msvc, c_flags)
}
