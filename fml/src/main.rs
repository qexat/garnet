//! The main compiler driver program.

use std::path::PathBuf;

use argh::FromArgs;

/// compiler
#[derive(Debug, FromArgs)]
struct Opt {
    /// save generated Rust code?
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
}

fn main() -> std::io::Result<()> {
    //let opt = parse_args();
    let opt: Opt = argh::from_env();

    let src = std::fs::read_to_string(&opt.file)?;
    let _output = fml::compile(&opt.file.to_str().unwrap(), &src);
    /*
    let mut rust_file;
    // Output to file
    {
        rust_file = opt.file.clone();
        rust_file.set_extension("rs");
        std::fs::write(&rust_file, &output)?;
    }
    // Invoke rustc
    let exe_file = if let Some(out) = opt.out {
        out
    } else {
        let mut exe_file = opt.file.clone();
        exe_file.set_extension(std::env::consts::EXE_EXTENSION);
        exe_file
    };
    use std::process::{Command, Stdio};
    let res = Command::new("rustc")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .arg("-o")
        .arg(&exe_file)
        .arg(&rust_file)
        .output()
        .expect("Failed to execute rustc");
    if !res.status.success() {
        dbg!(&res);
        panic!("Generated Rust code that could not be compiled");
    }

    // delete Rust files if we want to
    if !opt.save {
        std::fs::remove_file(rust_file).unwrap();
    }

    // Run output program if we want to
    if opt.run {
        let res = Command::new(&exe_file)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .output();
        std::fs::remove_file(&exe_file).unwrap();
        res.expect("Failed to run program");
    }
    */
    Ok(())
}
