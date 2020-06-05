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
    #[regex("true|false", |lex| lex.slice().parse())]
    Bool(bool),
    #[regex("-?[0-9]+", |lex| lex.slice().parse())]
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
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token(",")]
    Comma,

    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Error,
}

use self::Token as T;

pub struct Parser<'cx, 'input> {
    lex: logos::Lexer<'input, Token>,
    cx: &'cx Cx,
}

impl<'cx, 'input> Parser<'cx, 'input> {
    pub fn new(cx: &'cx Cx, s: &'input str) -> Self {
        let lex = Token::lexer(s);
        Parser { lex, cx }
    }

    /// Read all its input and returns an Ast.
    ///
    /// Currently just panics on error.
    pub fn parse(&mut self) -> ast::Ast {
        let mut decls = vec![];
        while let Some(d) = self.parse_decl() {
            decls.push(d);
        }
        ast::Ast { decls: decls }
    }

    fn error(&self, token: Option<Token>) -> ! {
        let msg = format!(
            "Parse error on {:?}, got token {:?} on str {}",
            self.lex.span(),
            token,
            self.lex.slice()
        );
        panic!(msg)
    }

    /// Consume a token that doesn't return anything
    fn expect(&mut self, tok: Token) {
        match self.lex.next() {
            Some(t) if t == tok => (),
            other => self.error(other),
        }
    }

    /// Consume an identifier and return its interned symbol.
    /// Note this returns a VarSym, not a TypeSym...
    fn expect_ident(&mut self) -> VarSym {
        match self.lex.next() {
            Some(T::Ident(s)) => self.cx.intern(s),
            other => self.error(other),
        }
    }

    /// Returns None on EOF.
    fn parse_decl(&mut self) -> Option<ast::Decl> {
        //-> ast::Decl {
        match self.lex.next() {
            Some(T::Const) => Some(self.parse_const()),
            Some(T::Fn) => Some(self.parse_fn()),
            None => None,
            other => self.error(other),
        }
    }

    fn parse_const(&mut self) -> ast::Decl {
        let name = self.expect_ident();
        self.expect(T::Colon);
        let typename = self.parse_type();
        self.expect(T::Equals);
        let init = self.parse_expr();
        ast::Decl::Const {
            name,
            typename,
            init,
        }
    }
    fn parse_fn(&mut self) -> ast::Decl {
        // TODO
        let name = self.expect_ident();
        unimplemented!()
    }

    fn parse_expr(&mut self) -> ast::Expr {
        // TODO
        match self.lex.next() {
            Some(T::Ident(_)) | Some(T::Bool(_)) | Some(T::Number(_)) => ast::Expr::int(3),
            other => self.error(other),
        }
    }

    fn parse_type(&mut self) -> TypeSym {
        match self.lex.next() {
            Some(T::Ident(s)) => match s.as_ref() {
                // TODO: This is a bit too hardwired tbh...
                "i32" => self.cx.intern_type(&TypeDef::SInt(4)),
                "bool" => self.cx.intern_type(&TypeDef::Bool),
                _ => self.error(Some(T::Ident(s))),
            },
            Some(T::LBrace) => {
                self.expect(T::RBrace);
                self.cx.intern_type(&TypeDef::Tuple(vec![]))
            }
            other => self.error(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use crate::parser::*;
    use crate::{Cx, TypeDef, TypeSym, VarSym};

    fn assert_decl(cx: &Cx, s: &str, res: ast::Decl) {
        let p = &mut Parser::new(cx, s);
        let d = p.parse_decl();
        assert_eq!(d, Some(res));
    }

    #[test]
    fn test_const() {
        let cx = &Cx::new();
        let foosym = cx.intern("foo");
        let i32_t = cx.intern_type(&TypeDef::SInt(4));
        // TODO: How can we not hardcode VarSym and TypeSym here,
        // without it being a PITA?
        assert_decl(
            cx,
            "const foo: i32 = -9",
            ast::Decl::Const {
                name: foosym,
                typename: i32_t,
                init: ast::Expr::int(3),
            },
        );
    }

    #[test]
    fn test_multiple_decls() {
        let s = r#"
const foo: i32 = -9
const bar: bool = 4
"#;
        let cx = &Cx::new();
        let p = &mut Parser::new(cx, s);
        let foosym = cx.intern("foo");
        let barsym = cx.intern("bar");
        let i32_t = cx.intern_type(&TypeDef::SInt(4));
        let bool_t = cx.intern_type(&TypeDef::Bool);
        let d = p.parse();
        assert_eq!(
            d,
            ast::Ast {
                decls: vec![
                    ast::Decl::Const {
                        name: foosym,
                        typename: i32_t,
                        init: ast::Expr::int(3),
                    },
                    ast::Decl::Const {
                        name: barsym,
                        typename: bool_t,
                        init: ast::Expr::int(3),
                    }
                ],
            }
        );
    }
}
