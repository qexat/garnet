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

#[derive(Logos, Debug, PartialEq, Clone)]
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
    // We save comment strings so we can use this same
    // parser as a reformatter or such.
    // TODO: How do we skip these in the parser?
    #[regex(r"--.*\n", |lex| lex.slice().to_owned())]
    Comment(String),

    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Error,
}

use self::Token as T;

type Tok = (Token, logos::Span);

pub struct Parser<'cx, 'input> {
    lex: std::iter::Peekable<logos::SpannedIter<'input, Token>>,
    cx: &'cx Cx,
    source: &'input str,
}

impl<'cx, 'input> Parser<'cx, 'input> {
    pub fn new(cx: &'cx Cx, source: &'input str) -> Self {
        let lex = Token::lexer(source).spanned().peekable();
        Parser { lex, cx, source }
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

    fn error(&self, token: Option<Tok>) -> ! {
        if let Some((ref tok, ref span)) = token {
            let msg = format!(
                "Parse error on {:?}, got token {:?} on str {}",
                span,
                tok,
                &self.source[span.clone()]
            );
            panic!(msg)
        } else {
            let msg = format!("FDSAFDSAFDSAFSDA TODO");
            panic!(msg)
        }
    }

    /// Consume a token, we don't care what it is.
    /// Presumably because we've already peeked at it.
    fn drop(&mut self) {
        self.lex.next();
    }

    /// Consume a token that doesn't return anything
    fn expect(&mut self, tok: Token) {
        match self.lex.next() {
            Some((t, _span)) if t == tok => (),
            other => self.error(other),
        }
    }

    /// Consume an identifier and return its interned symbol.
    /// Note this returns a VarSym, not a TypeSym...
    fn expect_ident(&mut self) -> VarSym {
        match self.lex.next() {
            Some((T::Ident(s), _span)) => self.cx.intern(s),
            other => self.error(other),
        }
    }

    /// Returns None on EOF.
    fn parse_decl(&mut self) -> Option<ast::Decl> {
        //-> ast::Decl {
        match self.lex.next() {
            Some((T::Const, _span)) => Some(self.parse_const()),
            Some((T::Fn, _span)) => Some(self.parse_fn()),
            None => None,
            other => self.error(other),
        }
    }

    fn parse_const(&mut self) -> ast::Decl {
        let name = self.expect_ident();
        self.expect(T::Colon);
        let typename = self.parse_type();
        self.expect(T::Equals);
        let init = self.parse_expr().unwrap();
        ast::Decl::Const {
            name,
            typename,
            init,
        }
    }
    fn parse_fn(&mut self) -> ast::Decl {
        // TODO
        let name = self.expect_ident();
        let signature = self.parse_fn_signature();
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Decl::Function {
            name,
            signature,
            body,
        }
    }

    // signature = fn_args [":" typename]
    fn parse_fn_signature(&mut self) -> ast::Signature {
        let params = self.parse_fn_args();
        let rettype = if let Some((T::Colon, _span)) = self.lex.peek() {
            self.expect(T::Colon);
            self.parse_type()
        } else {
            self.cx.intern_type(&TypeDef::Tuple(vec![]))
        };
        ast::Signature { params, rettype }
    }

    // fn_args = [ident ":" typename {"," ident ":" typename}]
    fn parse_fn_args(&mut self) -> Vec<(VarSym, TypeSym)> {
        let mut args = vec![];
        if let Some((T::Ident(_i), _span)) = self.lex.peek() {
            let name = self.expect_ident();
            self.expect(T::Colon);
            let tname = self.parse_type();
            args.push((name, tname));

            while let Some((T::Comma, _span)) = self.lex.peek() {
                self.expect(T::Comma);
                let name = self.expect_ident();
                self.expect(T::Colon);
                let tname = self.parse_type();
                args.push((name, tname));
            }
            // Consume trailing comma if it's there.
            if let Some((T::Comma, _span)) = self.lex.peek() {
                self.expect(T::Comma);
            }
        }
        args
    }

    fn parse_exprs(&mut self) -> Vec<ast::Expr> {
        let mut exprs = vec![];
        while let Some(e) = self.parse_expr() {
            exprs.push(e);
        }
        exprs
    }

    fn parse_expr(&mut self) -> Option<ast::Expr> {
        let token: Option<Tok> = self.lex.peek().cloned();
        match token {
            Some((T::Bool(b), _span)) => {
                self.drop();
                Some(ast::Expr::bool(b))
            }
            Some((T::Number(i), _span)) => {
                self.drop();
                Some(ast::Expr::int(i as i64))
            }
            Some((T::LBrace, _span)) => {
                self.drop();
                self.expect(T::RBrace);
                Some(ast::Expr::unit())
            }
            _other => None,
        }
    }

    fn parse_type(&mut self) -> TypeSym {
        match self.lex.next() {
            Some((T::Ident(s), span)) => match s.as_ref() {
                // TODO: This is a bit too hardwired tbh...
                "i32" => self.cx.intern_type(&TypeDef::SInt(4)),
                "bool" => self.cx.intern_type(&TypeDef::Bool),
                _ => self.error(Some((T::Ident(s), span))),
            },
            Some((T::LBrace, _span)) => {
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
                init: ast::Expr::int(-9),
            },
        );
    }

    #[test]
    fn test_fn() {
        let cx = &Cx::new();
        let foosym = cx.intern("foo");
        let xsym = cx.intern("x");
        let i32_t = cx.intern_type(&TypeDef::SInt(4));
        // TODO: How can we not hardcode VarSym and TypeSym here,
        // without it being a PITA?
        assert_decl(
            cx,
            "fn foo x:i32 : i32 -9 end",
            ast::Decl::Function {
                name: foosym,
                signature: ast::Signature {
                    params: vec![(xsym, i32_t)],
                    rettype: i32_t,
                },
                body: vec![ast::Expr::int(-9)],
            },
        );
    }

    #[test]
    fn test_multiple_decls() {
        let s = r#"
const foo: i32 = -9
const bar: bool = 4
const baz: {} = {}
"#;
        let cx = &Cx::new();
        let p = &mut Parser::new(cx, s);
        let foosym = cx.intern("foo");
        let barsym = cx.intern("bar");
        let bazsym = cx.intern("baz");
        let i32_t = cx.intern_type(&TypeDef::SInt(4));
        let bool_t = cx.intern_type(&TypeDef::Bool);
        let unit_t = cx.intern_type(&TypeDef::Tuple(vec![]));
        let d = p.parse();
        assert_eq!(
            d,
            ast::Ast {
                decls: vec![
                    ast::Decl::Const {
                        name: foosym,
                        typename: i32_t,
                        init: ast::Expr::int(-9),
                    },
                    ast::Decl::Const {
                        name: barsym,
                        typename: bool_t,
                        init: ast::Expr::int(4),
                    },
                    ast::Decl::Const {
                        name: bazsym,
                        typename: unit_t,
                        init: ast::Expr::unit(),
                    }
                ],
            }
        );
    }
}
