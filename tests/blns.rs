//! Test Garnet's parser on the Big List Of Naughty Strings,
//! to at least ensure nothing absolutely insane happens.
//! List defined here: https://github.com/minimaxir/big-list-of-naughty-strings
//! Retrieved Sep 10, 2021

use logos::{Lexer, Logos};

use garnet::parser::{Parser, TokenKind};

const BLNS: &[u8] = include_bytes!("blns.txt");

/// Just make sure our lexer can actually consume these strings
#[test]
fn lex_blns() {
    let s = std::str::from_utf8(BLNS).unwrap();
    let lex = TokenKind::lexer(s);
    for t in lex {
        println!("t is {:?}", t);
    }
}

/*
/// Test that our parser produces errors instead of just exploding
/// TODO: Our parser always explodes on errors, see https://todo.sr.ht/~icefox/garnet/15
#[test]
fn parse_blns() {
    let s = std::str::from_utf8(BLNS).unwrap();
    let mut parser = Parser::new("blns.txt", s);
    let ast = parser.parse();
}
*/
