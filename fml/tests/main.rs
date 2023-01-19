//! Test suite that typechecks reference programs from `tests/programs/`
//! and checks their output.
//!
//! Uses the `lang_tester` crate, which is a little wobbly in places,
//! but the best I can find.

use std::{fs::read_to_string, process::Command};

use lang_tester::LangTester;

static COMMENT_PREFIX: &str = "--";

fn main() {
    // We just typecheck things and see if they pass
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
        // We have one test commands:
        //   * `check`: runs garnetc.
        .test_cmds(move |p| {
            // Test command 1: check `x.gt`
            let mut check = Command::new("cargo");
            check.args(&["run", "--", p.to_str().unwrap()]);
            vec![("Check", check)]
        })
        .run();
}
