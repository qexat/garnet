use std::path::PathBuf;

use argh::FromArgs;
use garnet;

/// Garnet compiler
#[derive(Debug, FromArgs)]
struct Opt {
    /// input files
    #[argh(positional)]
    files: Vec<PathBuf>,
}

fn main() -> std::io::Result<()> {
    //let opt = parse_args();
    let opt: Opt = argh::from_env();
    if opt.files.len() == 0 {
        println!("No files input, try --help?");
        return Ok(());
    }
    // Output to file
    for file in opt.files {
        let src = std::fs::read_to_string(&file)?;
        let output = garnet::compile(&src);
        let mut output_file = file.clone();
        output_file.set_extension("wasm");
        std::fs::write(&output_file, &output)?;
    }
    return Ok(());
}
