//! We're just going to do a simple LL recursive-descent parser.
//! It's simple, robust, fast, and pretty hard to screw up.

/*
Broad syntax thoughts:
Everything is an expression
Go for Lua-style preference of keywords rather than punctuation
But don't go ham sacrificing familiarity for How It Should Be

Keyword-delimited blocks instead of curly braces
and/or/not keywords for logical operators instead of ||, && etc
Keep | and & and ~ for binary operations
TODO: Make sure trailing commas are always allowed

I kinda want Erlang-style pattern matching in function args, but
the point of this language is to KISS.


decl =
  | function_decl
  | const_decl

const_decl = "const" ident ":" typename "=" expr
function_decl = "fn" ident "(" fn_args ")" [":" typename] {expr} "end"

value =
  | NUMBER
  | BOOL
  | UNIT
  | ident

expr =
  | let
  | if
  | loop
  | block
  | funcall
  | lambda

// Currently, type inference is not a thing
let = "let" ident ":" typename "=" expr
if = "if" expr "then" {expr} {"elif" expr "then" {expr}} ["else" {expr}] "end"
loop = "loop" {expr} "end"
block = "do" {expr} "end"
funcall = expr "(" [expr {"," expr}] ")"
// TODO: 'lambda' is long to type.  : or -> for return type?
lambda = "lambda" fn_args [":" typename] "=" {expr} "end"

fn_args = [ident ":" typename {"," ident ":" typename}]

typename =
  | "i32"
  | "bool"
  // Tuples with curly braces like Erlang seem less ambiguous than the more traditional parens...
  // I hope that will let us get away without semicolons.
  | "{" [typename {"," typename}] "}"
  // Fixed-size arrays
  | "[" typename ";" INTEGER} "]"
  // TODO: Generics?
  // | ID "[" typename {"," typename} "]"
  // slices can just then be slice[...]
  // TODO: Function syntax?  Possibilities below.
  // | "fn" "(" fn_args ")" [":" typename]
  // | "fn" fn_args ["->" typename]

// Things to add, roughly in order
// while/for loops
// arrays/slices
// enums
// structs (anonymous and otherwise?)
// assignments
// generics
// references

*/

use logos::Logos;

use crate::ast;
use crate::{Cx, TypeDef, TypeSym, VarSym};

#[derive(Logos, Debug, PartialEq)]
pub enum Token {
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_owned())]
    Ident(String),
    #[token("()")]
    Unit,
    #[regex("true|false", |lex| lex.slice().parse())]
    Bool(bool),
    #[regex("[-][0-9]+", |lex| lex.slice().parse())]
    Number(i32),

    #[token("const")]
    Const,
    #[token("fn")]
    Fn,

    #[token("let")]
    Let,
    #[token(":")]
    Colon,
    #[token("=")]
    Equals,
    #[token("end")]
    End,
    #[token("if")]
    If,
    #[token("elif")]
    Elif,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("loop")]
    Loop,
    #[token("do")]
    Do,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(",")]
    Comma,

    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Error,
}

use self::Token as T;

type Lexer<'a> = logos::Lexer<'a, Token>;

fn lex(s: &str) -> Lexer {
    Token::lexer(s)
}

fn error(token: Option<Token>, lex: &mut Lexer) -> ! {
    let msg = format!(
        "Parse error on {:?}, got token {:?} on str {}",
        lex.span(),
        token,
        lex.slice()
    );
    panic!(msg)
}

/// Consume a token that doesn't return anything
fn expect(tok: Token, lex: &mut Lexer) {
    match lex.next() {
        Some(t) if t == tok => (),
        other => error(other, lex),
    }
}

/// Consume an identifier and return its interned symbol.
/// Note this returns a VarSym, not a TypeSym...
fn expect_ident(cx: &mut Cx, lex: &mut Lexer) -> VarSym {
    match lex.next() {
        Some(T::Ident(s)) => cx.intern(s),
        other => error(other, lex),
    }
}

/// Exactly the same as above but gets a type name...
fn expect_type(cx: &mut Cx, lex: &mut Lexer) -> TypeSym {
    match lex.next() {
        Some(T::Ident(s)) => cx.intern_type(&TypeDef::Named(s)),
        other => error(other, lex),
    }
}

fn parse_decl(cx: &mut Cx, lex: &mut Lexer) -> ast::Decl {
    //-> ast::Decl {
    match lex.next() {
        Some(T::Const) => parse_const(cx, lex),
        Some(T::Fn) => parse_fn(lex),
        other => error(other, lex),
    }
}

fn parse_const(cx: &mut Cx, lex: &mut Lexer) -> ast::Decl {
    let name = expect_ident(cx, lex);
    expect(T::Colon, lex);
    let typename = parse_type(cx, lex);
    expect(T::Equals, lex);
    let init = parse_expr(cx, lex);
    ast::Decl::Const {
        name,
        typename,
        init,
    }
}
fn parse_fn(lex: &mut Lexer) -> ast::Decl {
    // TODO
    unimplemented!()
}

fn parse_expr(cx: &mut Cx, lex: &mut Lexer) -> ast::Expr {
    // TODO
    expect(T::Unit, lex);
    ast::Expr::int(3)
}

fn parse_type(cx: &mut Cx, lex: &mut Lexer) -> TypeSym {
    // TODO
    expect_type(cx, lex)
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use crate::parser::*;
    use crate::{Cx, TypeSym, VarSym};

    fn assert_decl(s: &str, res: ast::Decl) {
        let cx = &mut Cx::new();
        let l = &mut Token::lexer(s);
        let d = parse_decl(cx, l);
        assert_eq!(d, res);
    }

    #[test]
    fn test_const() {
        // TODO: How can we not hardcode VarSym and TypeSym here,
        // without it being a PITA?
        assert_decl(
            "const foo: bar = ()",
            ast::Decl::Const {
                name: VarSym(0),
                typename: TypeSym(0),
                init: ast::Expr::int(3),
            },
        );
    }
}
