//! The main compiler driver program.

use std::io;
use std::path::{Path, PathBuf};

use argh::FromArgs;

use garnet::backend::Backend;

/// Garnet compiler
#[derive(Debug, FromArgs)]
struct Opt {
    /// save intermediate generated code?
    #[argh(switch, short = 's')]
    save: bool,

    /// run resulting program immediately, handy for unit tests.  Does not produce an executable.
    #[argh(switch, short = 'r')]
    run: bool,

    /// output file name
    #[argh(option, short = 'o')]
    out: Option<PathBuf>,

    /// input files
    #[argh(positional)]
    file: PathBuf,

    /// code gen backend
    #[argh(option, short = 'b', default = "Backend::Rust")]
    backend: Backend,
}

/// Maybe should be part of the backend selection?
/// Not yet I suppose.  The compiler itself has its job
/// end at "codegen" for now, calling whatever else
/// can be done just by this driver program.
///
/// Takes the input file name and the desired name of the
/// exe file, and returns the name of the file containing
/// intermediate code.  Which feels sorta weird, but here
/// we are.  But if our downstream compiler produces multiple
/// output files (.S, .o, etc) then maybe this function
/// should do its own cleanup after all.
fn compile_rust(input_file: &Path, exe_name: &Path) -> io::Result<PathBuf> {
    let mut rust_file;
    // Output to file
    {
        let src = std::fs::read_to_string(&input_file)?;
        let output = garnet::compile(&input_file.to_str().unwrap(), &src, Backend::Rust);
        rust_file = input_file.to_owned();
        rust_file.set_extension("rs");
        std::fs::write(&rust_file, &output)?;
    }
    // Invoke rustc
    use std::process::{Command, Stdio};
    let res = Command::new("rustc")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .arg("-o")
        .arg(&exe_name)
        .arg(&rust_file)
        .output()
        .expect("Failed to execute rustc");
    if !res.status.success() {
        dbg!(&res);
        panic!("Generated Rust code that could not be compiled");
    }
    Ok(rust_file)
}

fn main() -> std::io::Result<()> {
    let opt: Opt = argh::from_env();

    let exe_name = if let Some(out) = &opt.out {
        out.clone()
    } else {
        let mut exe_name = opt.file.to_owned();
        exe_name.set_extension(std::env::consts::EXE_EXTENSION);
        exe_name
    };
    let output_file = match opt.backend {
        Backend::Rust => compile_rust(&opt.file, &exe_name)?,
        Backend::Null => todo!(),
    };

    use std::process::{Command, Stdio};
    // delete intermediate files if we want to
    if !opt.save {
        std::fs::remove_file(output_file).unwrap();
    }

    // Run output program if we want to
    if opt.run {
        let res = Command::new(&exe_name)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .output();
        std::fs::remove_file(&exe_name).unwrap();
        res.expect("Failed to run program");
    }
    Ok(())
}
