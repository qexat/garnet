//! Test suite that compiles and runs reference programs from `tests/programs/`
//! and checks their output.
//!
//! Uses the `lang_tester` crate, which is a little wobbly in places,
//! but the best I can find.  `goldentests` is too magical and not flexible
//! enough, and `compiletests` is too Rust-specific and under-documented.

use std::{fs::read_to_string, path::PathBuf, process::Command};

use lang_tester::LangTester;
use tempfile::TempDir;

static COMMENT_PREFIX: &str = "--";

fn main() {
    // We use garnetc to compile files into a binary, then store those binary files into `tempdir`.
    let tempdir = TempDir::new().unwrap();
    LangTester::new()
        .test_dir("tests/programs/")
        // Only use files named `*.gt` as test files.
        .test_file_filter(|p| {
            p.extension()
                .map(std::ffi::OsStr::to_str)
                .unwrap_or(Some(""))
                .unwrap()
                == "gt"
        })
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
        // We have three test commands:
        //   * `Format`: Reformat the source and make sure that it parses identically
        //   * `Compile`: runs garnetc.
        //   * `Run`: if garnetc does not error, and the `Compile` tests succeed, then the
        //     output binary is run.
        .test_cmds(move |p| {
            // Test command 0: Reformat-check `x.gt` and make sure it succeeds
            // ie, the program after formatting parses the same way as before.
            // We have to have our own test command for those because it has to know
            // that it should fail on some things (like parse failures), but
            // succeed on others where actual compilation fails (like type errors).
            // Ah well.
            let mut fmt = Command::new("cargo");
            fmt.args(&[
                "run",
                "--bin",
                "garnetfmt",
                "--",
                "-c",
                p.to_str().unwrap()
            ]);
            
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
            vec![("Format", fmt), ("Compile", compile), ("Run", run)]
        })
        .run();
}
