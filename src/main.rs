use std::path::PathBuf;

use garnet;
use pico_args;

struct Opt {
    files: Vec<PathBuf>,
}

fn parse_args() -> Opt {
    let args = pico_args::Arguments::from_env();
    let files = args.free().unwrap().iter().map(PathBuf::from).collect();
    Opt { files: files }
}

fn help() {
    println!("TODO: help text generator.");
}

fn main() -> std::io::Result<()> {
    let opt = parse_args();
    if opt.files.len() == 0 {
        help();
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
