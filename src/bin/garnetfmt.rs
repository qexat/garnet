//! A code formatter.

use std::io::{self, Cursor};
use std::path::PathBuf;

use argh::FromArgs;

/// Garnet formatter
#[derive(Debug, FromArgs)]
struct Opt {
    /// overwrite the original file with the new one?
    #[argh(switch, short = 'x')]
    overwrite: bool,
    /// don't output anything, just check if the new formatting parses
    #[argh(switch, short = 'c')]
    check: bool,
    /// input file name
    #[argh(positional)]
    file: PathBuf,
}

fn main() -> io::Result<()> {
    pretty_env_logger::init();
    let opt: Opt = argh::from_env();

    use garnet::*;
    let src = std::fs::read_to_string(&opt.file)?;
    let filename = &opt.file.to_string_lossy();
    let ast = {
        let mut parser = parser::Parser::new(filename, &src);
        parser.parse()
    };

    // something something bufferedwriter something something
    let mut formatted_src = Cursor::new(vec![]);
    format::unparse(&ast, &mut formatted_src)?;

    // reparse to make sure that we haven't totally fucked up
    // anything
    let formatted_data = &formatted_src.into_inner();
    let formatted_str = String::from_utf8_lossy(formatted_data);
    println!("{}", formatted_str);
    let formatted_ast = {
        let mut parser = parser::Parser::new(filename, &formatted_str);
        parser.parse()
    };
    if &ast != &formatted_ast {
        // we want more info here
        eprintln!("Error, reformatted AST parses differently from original");
        eprintln!("BEFORE:\n{}", src);
        eprintln!("AST: {:#?}", &ast);
        eprintln!("AFTER:\n{}", formatted_str);
        eprintln!("AST: {:#?}", &formatted_ast);
        panic!("reformat failed");
    }

    if !opt.check {
        let mut output_file_name = PathBuf::from(&opt.file);
        output_file_name.set_extension("gt.tmp");
        std::fs::write(&output_file_name, formatted_data)?;
        if opt.overwrite {
            std::fs::rename(&output_file_name, &opt.file)?;
        }
    }
    Ok(())
}
