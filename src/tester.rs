use std::{fs, io, path::PathBuf, process::{Command, Output}};

pub fn test(
    exe_path: &str,
    test_directory: &str,
    core_path: &str,
    is_expected: bool,
) -> io::Result<()> {
    let canonical_core_path = PathBuf::from(core_path).canonicalize()?;
    let core_path = canonical_core_path.to_str().unwrap();

    for entry in fs::read_dir(test_directory)? {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let path = path.to_str().unwrap();

        test_project(exe_path, path, core_path, is_expected);
    }

    Ok(())
}

fn test_project(exe_path: &str, project_path: &str, core_path: &str, is_expected: bool) {
    let mut expected_output_path = PathBuf::from(project_path);
    expected_output_path.push("./output.txt");

    let mut output = String::new();

    let compiler_output = Command::new(exe_path)
        .current_dir(project_path)
        .args([".", "--core-path", core_path])
        .output()
        .unwrap();

    push_output(&mut output, &compiler_output);

    let mut output_exe_path = PathBuf::from(project_path);
    output_exe_path.push("./build/Out.exe");

    if let Ok(output_exe_output) = Command::new(output_exe_path)
        .current_dir(project_path)
        .output()
    {
        push_output(&mut output, &output_exe_output);
    };

    if is_expected {
        fs::write(expected_output_path, output).unwrap();
        println!(
            "Project \"{}\" had it's expected output updated!",
            project_path
        );

        return;
    }

    let Ok(expected_output) = fs::read_to_string(&expected_output_path) else {
        return;
    };

    if output != expected_output {
        println!(
            "Project \"{}\" had unexpected output:\n{}\n",
            project_path, output
        );
    } else {
        println!("Project \"{}\" succeeded!", project_path);
    }
}

fn push_output(string: &mut String, output: &Output) {
    let output_stdout = String::from_utf8_lossy(&output.stdout);
    string.push_str(&output_stdout);

    let output_stderr = String::from_utf8_lossy(&output.stderr);
    string.push_str(&output_stderr);
}
