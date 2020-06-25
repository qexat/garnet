use std::path::PathBuf;

use garnet;
use gumdrop::Options;

#[derive(Debug, Options)]
struct Opt {
    #[options(free, help = "Input files")]
    files: Vec<PathBuf>,
    #[options(help = "Print help")]
    help: bool,
}

/*
fn parse_args() -> Opt {
    let args = pico_args::Arguments::from_env();
    let files = args.free().unwrap().iter().map(PathBuf::from).collect();
    Opt { files: files }
}

fn help() {
    println!("TODO: help text generator.  structopt is great but heavy, gumdrop looks good but still relatively heavy.  Only 'cause it uses proc maros though, no other deps.  Walrus and logos already use proc macros anyway, so.");
}
*/

fn main() -> std::io::Result<()> {
    //let opt = parse_args();
    let opt = Opt::parse_args_default_or_exit();
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
