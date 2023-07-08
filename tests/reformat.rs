//! Test suite that parses the programs in `tests/programs/`,
//! then runs garnetfmt on them and makes sure they parse identically.
//!
//! Still uses `lang_tester` which is kinda odd but works.
//! It's slooooooow because it has lots of external invocations to
//! `garnetfmt`, but frankly `garnetfmt` should make sure that its output
//! is still valid anyway so in the end this is kinda the right abstraction
//! boundary.  Otherwise we'd just have to rewrite half of lang_tester and
//! half of garnetfmt in this file.

use std::process::Command;

use lang_tester::LangTester;

fn main() {
    // We use garnetc to compile files into a binary, then store those binary files into `tempdir`.
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
        .test_extract(|_p| String::from("Format:\n  status: success\n"))
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
