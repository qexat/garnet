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


Deliberate choices so that we don't go ham:
 * we use C/Lua-style function application
   `foo(bar, baz)` instead of Lisp style `(foo bar baz)` or ML/Haskell
   style `foo bar baz`.
 * I kinda want Erlang-style pattern matching in function args, but
   the point of this language is to KISS.
 * Parens for tuples would make life simpler in some ways, but I feel
   also make syntax and parsing more confusing, so let's go with Erlang
   curly braces.  If we use curly braces for structs too, then this also
   emphasizes the equivalence between structs and tuples.


decl =
  | function_decl
  | const_decl

const_decl = "const" ident ":" typename "=" expr
function_decl = "fn" ident fn_args [":" typename] {expr} "end"

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
if = "if" expr "then" {expr} {"else" "if" expr "then" {expr}} ["else" {expr}] "end"
loop = "loop" {expr} "end"
block = "do" {expr} "end"
funcall = expr "(" [expr {"," expr}] ")"
// TODO: 'lambda' is long to type.  : or -> for return type?
lambda = "lambda" fn_args [":" typename] "=" {expr} "end"

fn_args = "(" [ident ":" typename {"," ident ":" typename}] ")"

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
// Break and return
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

    // Decl stuff
    #[token("const")]
    Const,
    #[token("fn")]
    Fn,

    // Keywords
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
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("loop")]
    Loop,
    #[token("do")]
    Do,
    #[token("lambda")]
    Lambda,

    // Punctuation
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

    // Operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Mul,
    #[token("/")]
    Div,
    #[token("%")]
    Mod,
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("not")]
    Not,
    #[token("xor")]
    Xor,
    #[token("==")]
    Equal,
    #[token("/=")]
    NotEqual,
    #[token(">")]
    Gt,
    #[token("<")]
    Lt,
    #[token(">=")]
    Gte,
    #[token("<=")]
    Lte,

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
    // TODO: The only methods we actually call on this are next() and peek,
    // and it always returns a span which we then throw away.
    // ...actually we want to keep spans attached to AST nodes,
    // so we'll need those later.
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
    fn expect(&mut self, expected: Token) {
        match self.lex.next() {
            Some((t, _span)) if t == expected => (),
            Some((tok, span)) => {
                let msg = format!(
                    "Parse error on {:?}: got token {:?} from str {}.  Expected token: {:?}",
                    span,
                    tok,
                    &self.source[span.clone()],
                    expected
                );
                panic!(msg);
            }
            None => {
                let msg = format!(
                    "Parse error: Got end of input or malformed token.  Expected token: {:?}",
                    expected
                );
                panic!(msg);
            }
        }
    }

    fn peek_is(&mut self, expected: Token) -> bool {
        if let Some((got, _)) = self.lex.peek().cloned() {
            got == expected
        } else {
            false
        }
    }

    /// Consume an identifier and return its interned symbol.
    /// Note this returns a VarSym, not a TypeSym...
    fn expect_ident(&mut self) -> VarSym {
        match self.lex.next() {
            Some((T::Ident(s), _span)) => self.cx.intern(s),
            Some((tok, span)) => {
                let msg = format!(
                    "Parse error on {:?}: got token {:?} from str {}.  Expected identifier.",
                    span,
                    tok,
                    &self.source[span.clone()],
                );
                panic!(msg);
            }
            None => {
                let msg = format!(
                    "Parse error: Got end of input or malformed token.  Expected identifier",
                );
                panic!(msg);
            }
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

    /// signature = fn_args [":" typename]
    fn parse_fn_signature(&mut self) -> ast::Signature {
        let params = self.parse_fn_args();
        let rettype = if self.peek_is(T::Colon) {
            self.expect(T::Colon);
            self.parse_type()
        } else {
            self.cx.intern_type(&TypeDef::Tuple(vec![]))
        };
        ast::Signature { params, rettype }
    }

    /// fn_args = [ident ":" typename {"," ident ":" typename}]
    ///
    /// TODO: Trailing comma!
    fn parse_fn_args(&mut self) -> Vec<(VarSym, TypeSym)> {
        let mut args = vec![];
        self.expect(T::LParen);
        if let Some((T::Ident(_i), _span)) = self.lex.peek() {
            let name = self.expect_ident();
            self.expect(T::Colon);
            let tname = self.parse_type();
            args.push((name, tname));

            // If it leads to another arg, carry on.
            while self.peek_is(T::Comma) {
                self.expect(T::Comma);
                let name = self.expect_ident();
                self.expect(T::Colon);
                let tname = self.parse_type();
                args.push((name, tname));
            }
        }
        // Consume trailing comma if it's there.
        // This doesn't work right, fix.
        /*
        if let Some((T::Comma, _span)) = self.lex.peek() {
            self.expect(T::Comma);
        }
        */
        self.expect(T::RParen);
        args
    }

    fn parse_exprs(&mut self) -> Vec<ast::Expr> {
        let mut exprs = vec![];
        while let Some(e) = self.parse_expr() {
            exprs.push(e);
        }
        exprs
    }

    /// Returns None if there is no valid follow-on expression,
    /// which usually means the end of a block or such.
    fn parse_expr(&mut self) -> Option<ast::Expr> {
        if let Some((token, _span)) = self.lex.peek().cloned() {
            match token {
                T::Bool(b) => {
                    self.drop();
                    Some(ast::Expr::bool(b))
                }
                T::Number(i) => {
                    self.drop();
                    Some(ast::Expr::int(i as i64))
                }
                T::LBrace => {
                    self.drop();
                    self.expect(T::RBrace);
                    Some(ast::Expr::unit())
                }
                T::Ident(_) => {
                    let ident = self.expect_ident();
                    Some(ast::Expr::Var { name: ident })
                }
                T::Let => Some(self.parse_let()),
                T::If => Some(self.parse_if()),
                T::Loop => Some(self.parse_loop()),
                T::Do => Some(self.parse_block()),
                T::Lambda => Some(self.parse_lambda()),
                _x => None,
            }
        } else {
            None
        }
    }

    /// let = "let" ident ":" typename "=" expr
    fn parse_let(&mut self) -> ast::Expr {
        self.expect(T::Let);
        let varname = self.expect_ident();
        self.expect(T::Colon);
        let typename = self.parse_type();
        self.expect(T::Equals);
        let init = Box::new(self.parse_expr().expect("TODO: Better error message"));
        ast::Expr::Let {
            varname,
            typename,
            init,
        }
    }

    /// {"else" "if" expr "then" {expr}}
    ///
    /// TODO: `else if` instead?  I kinda like it more but it's
    /// a little more ambiguous.
    fn parse_elif(&mut self, accm: &mut Vec<ast::IfCase>) {
        while self.peek_is(T::Else) {
            self.expect(T::Else);
            if self.peek_is(T::If) {
                self.expect(T::If);
                let condition = self
                    .parse_expr()
                    .expect("TODO: Expect better error message");
                self.expect(T::Then);
                let body = self.parse_exprs();
                accm.push(ast::IfCase {
                    condition: Box::new(condition),
                    body,
                });
            } else {
                break;
            }
        }
    }

    /// if = "if" expr "then" {expr} {"else" "if" expr "then" {expr}} ["else" {expr}] "end"
    fn parse_if(&mut self) -> ast::Expr {
        self.expect(T::If);
        let condition = self
            .parse_expr()
            .expect("TODO: Expect better error message");
        self.expect(T::Then);
        let body = self.parse_exprs();
        let mut ifcases = vec![ast::IfCase {
            condition: Box::new(condition),
            body,
        }];
        // Parse 0 or more else if cases
        self.parse_elif(&mut ifcases);
        let next = self.lex.peek().cloned();
        let falseblock = match next {
            // No else block, we're done
            Some((T::End, _)) => {
                self.expect(T::End);
                vec![]
            }
            // We're in an else block but not an else-if block
            Some((_, _)) => {
                let elsepart = self.parse_exprs();
                self.expect(T::End);
                elsepart
            }
            other => self.error(other),
        };
        ast::Expr::If {
            cases: ifcases,
            falseblock,
        }
    }

    fn parse_loop(&mut self) -> ast::Expr {
        self.expect(T::Loop);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::Loop { body }
    }

    fn parse_block(&mut self) -> ast::Expr {
        self.expect(T::Do);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::Block { body }
    }

    fn parse_lambda(&mut self) -> ast::Expr {
        self.expect(T::Lambda);
        let signature = self.parse_fn_signature();
        self.expect(T::Equals);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::Lambda { signature, body }
    }

    /* TODO
    fn parse_funcall(&mut self) -> ast::Expr {
    }

    fn parse_term(&mut self) -> ast::Expr {
    */

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

// Binding power functions for the Pratt parser portion.
// Reference:
// https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html
//
fn prefix_binding_power(op: Token) -> ((), usize) {
    match op {
        T::Plus | T::Minus => ((), 110),
        T::Not => todo!(),
        x => panic!("{:?} is not a binary op", x),
    }
}

fn postfix_binding_power(op: Token) -> Option<(usize, ())> {
    match op {
        T::LParen => todo!(),
        x => panic!("{:?} is not a binary op", x),
    }
}

fn infix_binding_power(op: Token) -> Option<(usize, usize)> {
    match op {
        T::Mul | T::Div | T::Mod => Some((100, 101)),
        T::Plus | T::Minus => Some((90, 91)),
        T::And | T::Or | T::Xor | T::Equal | T::NotEqual | T::Gt | T::Lt | T::Gte | T::Lte => {
            todo!()
        }
        x => panic!("{:?} is not a binary op", x),
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
            "fn foo(x: i32): i32 -9 end",
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

    /// Take a list of strings and try parsing them with the given function.
    fn test_parse_with<T>(f: impl Fn(&mut Parser) -> T, strs: &[&str]) {
        for s in strs {
            let cx = &Cx::new();
            let mut p = Parser::new(cx, s);
            f(&mut p);
        }
    }

    #[test]
    fn parse_fn_args() {
        let valid_args = vec![
            "()",
            "(x: bool)",
            //"(x: bool,)",
            "(x: i32, y: bool)",
            //"(x: i32, y: bool,)",
        ];
        test_parse_with(|p| p.parse_fn_args(), &valid_args)
    }
    #[test]
    fn parse_fn_signature() {
        let valid_args = vec![
            "()",
            "(x: bool):i32",
            "(x: bool):{}",
            "(x: i32, y: bool)",
            "(x: i32, y: bool):bool",
        ];
        test_parse_with(|p| p.parse_fn_signature(), &valid_args)
    }
    #[test]
    fn parse_let() {
        let valid_args = vec!["let x: i32 = 5", "let y: bool = false", "let z: {} = z"];
        // The lifetimes and inference here gets WEIRD if you try to pass it Parser::parse_let.
        test_parse_with(|p| p.parse_let(), &valid_args);
        test_parse_with(|p| p.parse_expr(), &valid_args);
    }

    #[test]
    fn parse_if() {
        let valid_args = vec![
            "if x then y end",
            "if 10 then let x: bool = false 10 end",
            r#"if 10 then false
            else true
            end
            "#,
            r#"if 10 then false
            else if 20 then false
            else true
            end
            "#,
            r#"if 10 then false
            else if 20 then {} false
            else if 30 then {}
            else true
            end
            "#,
        ];
        test_parse_with(|p| p.parse_if(), &valid_args);
        test_parse_with(|p| p.parse_expr(), &valid_args);
    }
    #[test]
    #[should_panic]
    fn parse_if_aiee() {
        let valid_args = vec!["if true then 10 else else else 20 end"];
        test_parse_with(|p| p.parse_if(), &valid_args);
    }

    #[test]
    fn parse_loop() {
        let valid_args = vec!["loop 10 end", "loop 10 20 30 end", "loop end"];
        test_parse_with(|p| p.parse_loop(), &valid_args);
        test_parse_with(|p| p.parse_expr(), &valid_args);
    }

    #[test]
    fn parse_block() {
        let valid_args = vec!["do 10 end", "do 10 20 30 end", "do end"];
        test_parse_with(|p| p.parse_block(), &valid_args);
        test_parse_with(|p| p.parse_expr(), &valid_args);
    }

    #[test]
    fn parse_lambda() {
        let valid_args = vec![
            "lambda(x:i32):i32 = x end",
            "lambda(x:i32, i:bool) = x end",
            "lambda() = {} end",
            "lambda() = end",
        ];
        test_parse_with(|p| p.parse_lambda(), &valid_args);
        test_parse_with(|p| p.parse_expr(), &valid_args);
    }
}
