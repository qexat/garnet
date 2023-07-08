//! Test suite that parses the programs in `tests/programs/`,
//! then runs garnetfmt on them and makes sure they parse identically.
//!
//! still uses `lang_tester` which is kinda odd but

use std::{fs, io, path::PathBuf};
use std::process::Command;

use lang_tester::LangTester;
use tempfile::TempDir;

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
        .test_extract(|p| String::from("Format:\n  status: success\n"))
        .test_cmds(move |p| {
            // Test command 1: reformat x.gt and make sure it succeeds,
            // which checks reparsing
            let mut format = Command::new("cargo");
            format.args(&[
                "run",
                "--bin",
                "garnetfmt",
                "--",
                "-c",
                p.to_str().unwrap(),
            ]);

            vec![("Format", format)]
        })
        .run();
}
