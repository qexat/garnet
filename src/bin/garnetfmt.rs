//! A code formatter.

use std::io::{self, Cursor};
use std::path::{PathBuf};

use argh::FromArgs;
use pretty_env_logger;

/// Garnet formatter
#[derive(Debug, FromArgs)]
struct Opt {
    #[argh(positional)]
    file: PathBuf,
}

fn main() -> io::Result<()> {
    pretty_env_logger::init();
    let opt: Opt = argh::from_env();

    use garnet::*;
    let src = std::fs::read_to_string(&opt.file)?;
    let ast = {
        let filename = &opt.file.to_string_lossy();
        let mut parser = parser::Parser::new(&filename, &src);
        parser.parse()
    };

    // something something bufferedwriter something something
    let mut formatted_src= Cursor::new(vec![]);
    format::unparse(&ast, &mut formatted_src)?;

    let mut output_file_name = PathBuf::from(&opt.file);
    output_file_name.set_extension("gt.tmp");
    std::fs::write(output_file_name, &formatted_src.into_inner())?;
    Ok(())
}
