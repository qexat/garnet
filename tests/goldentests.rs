use std::{fs::read_to_string, path::PathBuf, process::Command};

use lang_tester::LangTester;
use tempfile::TempDir;

static COMMENT_PREFIX: &str = "--";

fn main() {
    // We use rustc to compile files into a binary: we store those binary files into `tempdir`.
    // This may not be necessary for other languages.
    let tempdir = TempDir::new().unwrap();
    LangTester::new()
        .test_dir("tests/programs/")
        // Only use files named `*.rs` as test files.
        .test_file_filter(|p| p.extension().unwrap().to_str().unwrap() == "gt")
        // Extract the first sequence of commented line(s) as the tests.
        .test_extract(|p| {
            read_to_string(p)
                .unwrap()
                .lines()
                // Skip non-commented lines at the start of the file.
                .skip_while(|l| !l.starts_with(COMMENT_PREFIX))
                // Extract consecutive commented lines.
                .take_while(|l| l.starts_with(COMMENT_PREFIX))
                // Strip the initial "--" from commented lines.
                .map(|l| &l[COMMENT_PREFIX.len()..])
                .collect::<Vec<_>>()
                .join("\n")
        })
        // We have two test commands:
        //   * `Compiler`: runs garnetc.
        //   * `Run-time`: if garnetc does not error, and the `Compiler` tests succeed, then the
        //     output binary is run.
        .test_cmds(move |p| {
            // Test command 1: Compile `x.gt` into `tempdir/x`.
            let mut exe = PathBuf::new();
            exe.push(&tempdir);
            exe.push(p.file_stem().unwrap());
            let mut compile = Command::new("cargo");
            compile.args(&[
                "run",
                "--bin",
                "garnetc",
                "--",
                "-o",
                exe.to_str().unwrap(),
                p.to_str().unwrap(),
            ]);
            // Test command 2: run `tempdir/x`.
            let run = Command::new(exe);
            vec![("Compile", compile), ("Run", run)]
        })
        .run();
}
