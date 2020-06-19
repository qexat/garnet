use std::path::PathBuf;

use structopt::StructOpt;

use garnet;

#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(name = "FILE", parse(from_os_str))]
    file: PathBuf,
}

fn main() -> std::io::Result<()> {
    let opt = Opt::from_args();
    let src = std::fs::read_to_string(&opt.file)?;
    let output = garnet::compile(&src);
    // Output to file
    let mut output_file = opt.file.clone();
    output_file.set_extension("wasm");
    std::fs::write(&output_file, &output)
}
