//! We're just going to do a simple LL recursive-descent parser.
//! It's simple, robust, fast, and pretty hard to screw up.
//!
//! This does get enhanced for parsing expressions into a Pratt
//! parser, as described  here:
//! <https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html>
//! This is extremely nice for parsing infix and postfix operators
//! with a precendence that's defined trivially by a simple look-up function.
//! I like it a lot.

/*
Broad syntax thoughts:
Everything is an expression
Go for Lua-style preference of keywords rather than punctuation
But don't go ham sacrificing familiarity for How It Should Be

Keyword-delimited blocks instead of curly braces
and/or/not keywords for logical operators instead of ||, && etc
Keep | and & and ~ for binary operations
Make sure trailing commas are always allowed


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
 * Tempting as it is, use `fn foo() ...` for functions, not `let foo = fn() ...`
   or other OCaml-y iterations on it.


decl =
  | function_decl
  | const_decl

const_decl = "const" ident ":" typename "=" expr
function_decl = "fn" ident fn_signature "=" {expr} "end"

value =
  | NUMBER
  | BOOL
  | UNIT
  | ident

constructor =
  // Tuple constructor
  | "{" [expr {"," expr} [","] "}"

expr =
  | let
  | if
  | loop
  | block
  | funcall
  | lambda
  | constructor
  | binop
  | prefixop
  | postfixop

// Currently, type inference is not a thing
let = "let" ident ":" typename "=" expr
if = "if" expr "then" {expr} {"else" "if" expr "then" {expr}} ["else" {expr}] "end"
loop = "loop" {expr} "end"
block = "do" {expr} "end"
funcall = expr "(" [expr {"," expr}] ")"
lambda = "fn" fn_signature "=" {expr} "end"

fn_args = "(" [ident ":" typename {"," ident ":" typename}] ")"
fn_signature = fn_args [":" typename]

typename =
  | "I32"
  | "Bool"
  | "fn" "(" [typename {"," typename} [","]] ")" [":" typename]
  // Tuples with curly braces like Erlang seem less ambiguous than the more traditional parens...
  // I hope that will let us get away without semicolons.
  | "{" [typename {"," typename}] [","] "}"
  // Fixed-size arrays
  // | "[" typename ";" INTEGER} "]"
  // TODO: Generics?
  // | ID "[" typename {"," typename} [","] "]"
  // slices can just then be slice[...]

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
    #[regex("[0-9][0-9_]*", |lex| lex.slice().parse())]
    Number(i32),

    // Decl stuff
    #[token("const")]
    Const,
    #[token("fn")]
    Fn,

    // Keywords
    #[token("let")]
    Let,
    #[token("mut")]
    Mut,
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
    #[token("elseif")]
    Elseif,
    #[token("else")]
    Else,
    #[token("loop")]
    Loop,
    #[token("do")]
    Do,

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
    #[token(".")]
    Period,

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
    #[token("^")]
    Carat,

    // We save comment strings so we can use this same
    // parser as a reformatter or such.
    // TODO: How do we skip these in the parser?
    #[regex(r"--.*\n", |lex| lex.slice().to_owned())]
    Comment(String),

    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Error,
}

fn bop_for(t: &Token) -> Option<ast::BOp> {
    match t {
        T::Plus => Some(ast::BOp::Add),
        T::Minus => Some(ast::BOp::Sub),
        T::Mul => Some(ast::BOp::Mul),
        T::Div => Some(ast::BOp::Div),
        T::Mod => Some(ast::BOp::Mod),

        T::Equal => Some(ast::BOp::Eq),
        T::NotEqual => Some(ast::BOp::Neq),
        T::Gt => Some(ast::BOp::Gt),
        T::Lt => Some(ast::BOp::Lt),
        T::Gte => Some(ast::BOp::Gte),
        T::Lte => Some(ast::BOp::Lte),

        T::And => Some(ast::BOp::And),
        T::Or => Some(ast::BOp::Or),
        T::Xor => Some(ast::BOp::Xor),
        _other => None,
    }
}

fn uop_for(t: &Token) -> Option<ast::UOp> {
    match t {
        //T::Plus => Some(ast::UOp::Plus),
        T::Minus => Some(ast::UOp::Neg),
        T::Not => Some(ast::UOp::Not),
        _other => None,
    }
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
            let msg = format!("Unexpected end of file!");
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

    /// Consumes a number and returns it.
    fn expect_int(&mut self) -> i32 {
        match self.lex.next() {
            Some((T::Number(s), _span)) => s,
            Some((tok, span)) => {
                let msg = format!(
                    "Parse error on {:?}: got token {:?} from str {}.  Expected number.",
                    span,
                    tok,
                    &self.source[span.clone()],
                );
                panic!(msg);
            }
            None => {
                let msg =
                    format!("Parse error: Got end of input or malformed token.  Expected number",);
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
        let init = self.parse_expr(0).unwrap();
        ast::Decl::Const {
            name,
            typename,
            init,
        }
    }
    fn parse_fn(&mut self) -> ast::Decl {
        let name = self.expect_ident();
        let signature = self.parse_fn_signature();
        self.expect(T::Equals);
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
            self.cx.unit()
        };
        ast::Signature { params, rettype }
    }

    /// sig = ident ":" typename
    /// fn_args = "(" [sig {"," sig} [","]] ")"
    fn parse_fn_args(&mut self) -> Vec<(VarSym, TypeSym)> {
        let mut args = vec![];
        self.expect(T::LParen);

        while let Some((T::Ident(_i), _span)) = self.lex.peek() {
            let name = self.expect_ident();
            self.expect(T::Colon);
            let tname = self.parse_type();
            args.push((name, tname));

            if self.peek_is(T::Comma) {
                self.expect(T::Comma);
            } else {
                break;
            }
        }
        self.expect(T::RParen);
        args
    }

    fn parse_fn_type(&mut self) -> TypeDef {
        let params = self.parse_fn_type_args();
        let rettype = if self.peek_is(T::Colon) {
            self.expect(T::Colon);
            self.parse_type()
        } else {
            self.cx.unit()
        };
        TypeDef::Lambda(params, rettype)
    }

    fn parse_fn_type_args(&mut self) -> Vec<TypeSym> {
        let mut args = vec![];
        self.expect(T::LParen);

        while !self.peek_is(T::RParen) {
            let tname = self.parse_type();
            args.push(tname);

            if self.peek_is(T::Comma) {
                self.expect(T::Comma);
            } else {
                break;
            }
        }
        self.expect(T::RParen);
        args
    }

    fn parse_tuple_type(&mut self) -> TypeDef {
        let mut body = vec![];
        while !self.peek_is(T::RBrace) {
            //while let Some(expr) = self.parse_type() {
            let t = self.parse_type();
            body.push(t);

            if self.peek_is(T::Comma) {
                self.expect(T::Comma);
            } else {
                break;
            }
        }
        self.expect(T::RBrace);
        TypeDef::Tuple(body)
    }

    fn parse_exprs(&mut self) -> Vec<ast::Expr> {
        let mut exprs = vec![];
        while let Some(e) = self.parse_expr(0) {
            exprs.push(e);
        }
        exprs
    }

    /// Returns None if there is no valid expression,
    /// which usually means the end of a block or such.
    ///
    /// This departs from pure recursive descent and uses a Pratt
    /// parser to parse math expressions and such.
    fn parse_expr(&mut self, min_bp: usize) -> Option<ast::Expr> {
        let (token, _span) = self.lex.peek().cloned()?;
        let mut lhs = match token {
            T::Bool(b) => {
                self.drop();
                ast::Expr::bool(b)
            }
            T::Number(_) => ast::Expr::int(self.expect_int() as i64),
            // Tuple literal
            T::LBrace => self.parse_constructor(),
            T::Ident(_) => {
                let ident = self.expect_ident();
                ast::Expr::Var { name: ident }
            }
            T::Let => self.parse_let(),
            T::If => self.parse_if(),
            T::Loop => self.parse_loop(),
            T::Do => self.parse_block(),
            T::Fn => self.parse_lambda(),
            // Parenthesized expr's
            T::LParen => {
                self.drop();
                let lhs = self.parse_expr(0)?;
                self.expect(T::RParen);
                lhs
            }

            // Unary prefix ops
            x if uop_for(&x).is_some() => {
                self.drop();
                let ((), r_bp) = prefix_binding_power(&x);
                let op = uop_for(&x).expect("Should never happen");
                let rhs = self.parse_expr(r_bp)?;
                ast::Expr::UniOp {
                    op: op,
                    rhs: Box::new(rhs),
                }
            }
            // Something else not a valid expr
            _x => return None,
        };
        // Parse a prefix, postfix or infix expression with a given
        // binding power or greater.
        loop {
            let op_token = match self.lex.peek().cloned() {
                Some((maybe_op, _span)) => maybe_op,
                // End of input
                _other => break,
            };
            // Is our token a postfix op?
            if let Some((l_bp, ())) = postfix_binding_power(&op_token) {
                if l_bp < min_bp {
                    break;
                }
                lhs = match op_token {
                    T::LParen => {
                        let params = self.parse_function_args();
                        ast::Expr::Funcall {
                            func: Box::new(lhs),
                            params,
                        }
                    }
                    T::Period => {
                        self.expect(T::Period);
                        let elt = self.expect_int();
                        assert!(elt > -1);
                        ast::Expr::TupleRef {
                            expr: Box::new(lhs),
                            elt: elt as usize,
                        }
                    }
                    T::Carat => {
                        self.expect(T::Carat);
                        ast::Expr::Deref {
                            expr: Box::new(lhs),
                        }
                    }
                    _ => return None,
                };
                continue;
            }
            // Is our token an infix op?
            if let Some((l_bp, r_bp)) = infix_binding_power(&op_token) {
                if l_bp < min_bp {
                    break;
                }
                self.drop();
                let rhs = self.parse_expr(r_bp).unwrap();
                lhs = if op_token == T::Equals {
                    ast::Expr::Assign {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    }
                } else {
                    let bop = bop_for(&op_token).unwrap();
                    //dbg!(&lhs, &op, self.lex.peek());
                    ast::Expr::BinOp {
                        op: bop,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    }
                };
                continue;
            }
            // None of the above, so we are done parsing the expr
            break;
        }
        Some(lhs)
    }

    fn parse_function_args(&mut self) -> Vec<ast::Expr> {
        let mut params = vec![];
        self.expect(T::LParen);
        // TODO: Refactor out this pattern somehow?
        // There's now three places it's used and it's only going to grow.
        while let Some(expr) = self.parse_expr(0) {
            params.push(expr);
            if !self.peek_is(T::Comma) {
                break;
            }
            self.expect(T::Comma);
        }
        self.expect(T::RParen);
        params
    }

    /// let = "let" ident ":" typename "=" expr
    fn parse_let(&mut self) -> ast::Expr {
        self.expect(T::Let);
        let mutable = if self.peek_is(T::Mut) {
            self.expect(T::Mut);
            true
        } else {
            false
        };
        let varname = self.expect_ident();
        self.expect(T::Colon);
        let typename = self.parse_type();
        self.expect(T::Equals);
        let init = Box::new(
            self.parse_expr(0)
                .expect("Expected expression after `let ... =`, did not get one"),
        );
        ast::Expr::Let {
            varname,
            typename,
            init,
            mutable,
        }
    }

    /// {"elseif" expr "then" {expr}}
    fn parse_elif(&mut self, accm: &mut Vec<ast::IfCase>) {
        while self.peek_is(T::Elseif) {
            self.expect(T::Elseif);
            let condition = self
                .parse_expr(0)
                .expect("TODO: be better; could not parse expr after elseif");
            self.expect(T::Then);
            let body = self.parse_exprs();
            accm.push(ast::IfCase {
                condition: Box::new(condition),
                body,
            });
        }
    }

    /// if = "if" expr "then" {expr} {"elseif" expr "then" {expr}} ["else" {expr}] "end"
    fn parse_if(&mut self) -> ast::Expr {
        self.expect(T::If);
        let condition = self
            .parse_expr(0)
            .expect("TODO: Expected expression after if, did not get one");
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
            Some((T::Else, _)) => {
                self.expect(T::Else);
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

    /// loop = "loop" {expr} "end"
    fn parse_loop(&mut self) -> ast::Expr {
        self.expect(T::Loop);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::Loop { body }
    }

    /// block = "block" {expr} "end"
    fn parse_block(&mut self) -> ast::Expr {
        self.expect(T::Do);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::Block { body }
    }

    /// lambda = "fn" "(" ...args... ")" [":" typename] = {exprs} "end"
    fn parse_lambda(&mut self) -> ast::Expr {
        self.expect(T::Fn);
        let signature = self.parse_fn_signature();
        self.expect(T::Equals);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::Lambda { signature, body }
    }

    /// let = "let" ident ":" typename "=" expr
    /// tuple constructor = "{" [expr {"," expr} [","] "}"
    fn parse_constructor(&mut self) -> ast::Expr {
        self.expect(T::LBrace);
        let mut body = vec![];
        while let Some(expr) = self.parse_expr(0) {
            body.push(expr);

            if self.peek_is(T::Comma) {
                self.expect(T::Comma);
            } else {
                break;
            }
        }
        self.expect(T::RBrace);
        ast::Expr::TupleCtor { body }
    }

    fn parse_type(&mut self) -> TypeSym {
        match self.lex.next() {
            Some((T::Ident(s), span)) => match s.as_ref() {
                // TODO: This is a bit too hardwired tbh...
                "I32" => self.cx.i32(),
                "Bool" => self.cx.bool(),
                _ => self.error(Some((T::Ident(s), span))),
            },
            Some((T::LBrace, _span)) => {
                let tuptype = self.parse_tuple_type();
                self.cx.intern_type(&tuptype)
            }
            Some((T::Fn, _span)) => {
                let fntype = self.parse_fn_type();
                self.cx.intern_type(&fntype)
            }
            other => self.error(other),
        }
    }
}

/// Specifies binding power of prefix operators.
///
/// Panics on invalid token, which should never happen
/// since we always know what kind of expression we're parsing
/// from the get-go with prefix operators.
fn prefix_binding_power(op: &Token) -> ((), usize) {
    match op {
        T::Plus | T::Minus | T::Not => ((), 110),
        x => unreachable!("{:?} is not a prefix op, should never happen!", x),
    }
}

/// Specifies binding power of postfix operators.
fn postfix_binding_power(op: &Token) -> Option<(usize, ())> {
    match op {
        // "(" opening function call args
        T::LParen => Some((120, ())),
        // "." for tuple/struct references.
        T::Period => Some((130, ())),
        // "^" for pointer derefs.  TODO: Check precedence?
        T::Carat => Some((105, ())),
        _x => None,
    }
}

/// Specifies binding power of infix operators.
/// The binding power on one side should always be trivially
/// greater than the other, so there's never ambiguity.
fn infix_binding_power(op: &Token) -> Option<(usize, usize)> {
    // Right associations are slightly more powerful so we always produce
    // a deterministic tree.
    match op {
        T::Mul | T::Div | T::Mod => Some((100, 101)),
        T::Plus | T::Minus => Some((90, 91)),
        T::Lt | T::Gt | T::Lte | T::Gte => Some((80, 81)),
        T::Equal | T::NotEqual => Some((70, 71)),
        T::And => Some((60, 61)),
        // Logical xor has same precedence as or, I guess?  It's sorta an odd duck.
        T::Or | T::Xor => Some((50, 51)),
        // Assignment
        T::Equals => Some((10, 11)),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{self, Expr};
    use crate::parser::*;
    use crate::{Cx, TypeDef};

    /// Take a list of strings and try parsing them with the given function.
    /// Is ok iff the parsing succeeds, does no checking that the produced
    /// AST is actually something that you want, or anything at all.
    fn test_parse_with<T>(f: impl Fn(&mut Parser) -> T, strs: &[&str]) {
        for s in strs {
            let cx = &Cx::new();
            let mut p = Parser::new(cx, s);
            f(&mut p);
            // Make sure we've parsed the whole string.
            assert_eq!(p.lex.peek(), None);
        }
    }

    /// Take a list of strings, parse them, make sure they match
    /// the given ast.  The function gets passed a cx so it can
    /// intern strings for identifiers.
    ///
    /// For now it's just for expr's, since that's most of the language.
    fn test_expr_is(s: &str, f: impl Fn(&Cx) -> Expr) {
        let cx = &Cx::new();
        let ast = f(cx);
        let mut p = Parser::new(cx, s);
        let parsed_expr = p.parse_expr(0).unwrap();
        assert_eq!(&ast, &parsed_expr);
        // Make sure we've parsed the whole string.
        assert_eq!(p.lex.peek(), None);
    }

    /// Same as test_expr_is but with decl's
    fn test_decl_is(s: &str, f: impl Fn(&Cx) -> ast::Decl) {
        let cx = &Cx::new();
        let ast = f(cx);
        let mut p = Parser::new(cx, s);
        let parsed_decl = p.parse_decl().unwrap();
        assert_eq!(&ast, &parsed_decl);
        // Make sure we've parsed the whole string.
        assert_eq!(p.lex.peek(), None);
    }

    #[test]
    fn test_const() {
        test_decl_is("const foo: I32 = -9", |cx| ast::Decl::Const {
            name: cx.intern("foo"),
            typename: cx.intern_type(&TypeDef::SInt(4)),
            init: Expr::UniOp {
                op: ast::UOp::Neg,
                rhs: Box::new(Expr::int(9)),
            },
        });
    }

    #[test]
    fn test_fn() {
        test_decl_is("fn foo(x: I32): I32 = 9 end", |cx| {
            let i32_t = cx.intern_type(&TypeDef::SInt(4));
            ast::Decl::Function {
                name: cx.intern("foo"),
                signature: ast::Signature {
                    params: vec![(cx.intern("x"), i32_t)],
                    rettype: i32_t,
                },
                body: vec![Expr::int(9)],
            }
        });
    }

    #[test]
    fn test_multiple_decls() {
        let s = r#"
const foo: I32 = -9
const bar: Bool = 4
const baz: {} = {}
"#;
        let cx = &Cx::new();
        let p = &mut Parser::new(cx, s);
        let foosym = cx.intern("foo");
        let barsym = cx.intern("bar");
        let bazsym = cx.intern("baz");
        let i32_t = cx.i32();
        let bool_t = cx.bool();
        let unit_t = cx.unit();
        let d = p.parse();
        assert_eq!(
            d,
            ast::Ast {
                decls: vec![
                    ast::Decl::Const {
                        name: foosym,
                        typename: i32_t,
                        init: Expr::UniOp {
                            op: ast::UOp::Neg,
                            rhs: Box::new(Expr::int(9)),
                        }
                    },
                    ast::Decl::Const {
                        name: barsym,
                        typename: bool_t,
                        init: Expr::int(4),
                    },
                    ast::Decl::Const {
                        name: bazsym,
                        typename: unit_t,
                        init: Expr::unit(),
                    }
                ],
            }
        );
    }
    #[test]
    fn parse_fn_args() {
        let valid_args = vec![
            "()",
            "(x: Bool)",
            "(x: Bool,)",
            "(x: I32, y: Bool)",
            "(x: I32, y: Bool,)",
        ];
        test_parse_with(|p| p.parse_fn_args(), &valid_args)
    }
    #[test]
    fn parse_fn_signature() {
        let valid_args = vec![
            "()",
            "(x: Bool):I32",
            "(x: Bool):{}",
            "(x: I32, y: Bool)",
            "(x: I32, y: Bool):Bool",
            "(x: I32, y: Bool,)",
            "(x: I32, y: Bool,):Bool",
            "(f: fn(I32):I32, x: I32):Bool",
        ];
        test_parse_with(|p| p.parse_fn_signature(), &valid_args)
    }
    #[test]
    fn parse_let() {
        let valid_args = vec!["let x: I32 = 5", "let y: Bool = false", "let z: {} = z"];
        // The lifetimes and inference here gets WEIRD if you try to pass it Parser::parse_let.
        test_parse_with(|p| p.parse_let(), &valid_args);
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_if() {
        let valid_args = vec![
            "if x then y end",
            "if 10 then let x: Bool = false 10 end",
            r#"if 10 then false
            else true
            end
            "#,
            r#"if 10 then false
            elseif 20 then false
            else true
            end
            "#,
            r#"if 10 then false
            elseif 20 then {} false
            elseif 30 then {}
            else true
            end
            "#,
        ];
        test_parse_with(|p| p.parse_if(), &valid_args);
        test_parse_with(|p| p.parse_expr(0), &valid_args);
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
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_block() {
        let valid_args = vec!["do 10 end", "do 10 20 30 end", "do end"];
        test_parse_with(|p| p.parse_block(), &valid_args);
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_lambda() {
        let valid_args = vec![
            "fn(x:I32):I32 = x end",
            "fn(x:I32, i:Bool) = x end",
            "fn(f:fn(I32):I32, x:I32) = f(x) end",
            "fn() = {} end",
            "fn() = end",
        ];
        test_parse_with(|p| p.parse_lambda(), &valid_args);
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_operators() {
        let valid_args = vec![
            "1 + 2",
            "1 + 2 + 3 + 4 + 5",
            "1 + 2 * 3",
            "1 + if true then 1 else 4 end",
            "1 + if true then 1 else 2 + 4 end",
            "3 * do 2 + 3 end",
            "do 2 + 3 end * 5",
            "-x",
            "- - -x",
            "if z then x + 3 else 5 / 9 end * 6",
            "if z then x + -3 else 5 / 9 end * 6",
            "x()",
        ];
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_fn_lambda() {
        let valid_args = vec!["fn apply_one(f: fn(I32):I32, x: I32): I32 = f(x) end"];
        test_parse_with(|p| p.parse_decl().unwrap(), &valid_args);
    }

    #[test]
    fn parse_funcall() {
        test_expr_is("y(1, 2, 3)", |cx| Expr::Funcall {
            func: Box::new(Expr::Var {
                name: cx.intern("y"),
            }),
            params: vec![Expr::int(1), Expr::int(2), Expr::int(3)],
        });

        test_expr_is("foo(0, bar(1 * 2), 3)", |cx| Expr::Funcall {
            func: Box::new(Expr::Var {
                name: cx.intern("foo"),
            }),
            params: vec![
                Expr::int(0),
                Expr::Funcall {
                    func: Box::new(Expr::Var {
                        name: cx.intern("bar"),
                    }),
                    params: vec![Expr::BinOp {
                        op: ast::BOp::Mul,
                        lhs: Box::new(Expr::int(1)),
                        rhs: Box::new(Expr::int(2)),
                    }],
                },
                Expr::int(3),
            ],
        });

        test_expr_is("(1)", |_cx| Expr::int(1));
        test_expr_is("(((1)))", |_cx| Expr::int(1));
    }

    #[test]
    fn verify_elseif() {
        use Expr;
        test_expr_is(
            r#"
            if x then
                1
            elseif y then
                2
            else
                3
            end
            "#,
            |cx| Expr::If {
                cases: vec![
                    ast::IfCase {
                        condition: Box::new(Expr::var(cx, "x")),
                        body: vec![Expr::int(1)],
                    },
                    ast::IfCase {
                        condition: Box::new(Expr::var(cx, "y")),
                        body: vec![Expr::int(2)],
                    },
                ],
                falseblock: vec![Expr::int(3)],
            },
        );

        test_expr_is(
            r#"
            if x then
                1
            else
                if y then
                    2
                else
                    3
                end
            end
            "#,
            |cx| Expr::If {
                cases: vec![ast::IfCase {
                    condition: Box::new(Expr::var(cx, "x")),
                    body: vec![Expr::int(1)],
                }],
                falseblock: vec![Expr::If {
                    cases: vec![ast::IfCase {
                        condition: Box::new(Expr::var(cx, "y")),
                        body: vec![Expr::int(2)],
                    }],
                    falseblock: vec![Expr::int(3)],
                }],
            },
        );
    }

    // Test op precedence works
    #[test]
    fn verify_precedence() {
        test_expr_is("1+2", |_cx| Expr::BinOp {
            op: ast::BOp::Add,
            lhs: Box::new(Expr::int(1)),
            rhs: Box::new(Expr::int(2)),
        });

        test_expr_is("1+2*3", |_cx| Expr::BinOp {
            op: ast::BOp::Add,
            lhs: Box::new(Expr::int(1)),
            rhs: Box::new(Expr::BinOp {
                op: ast::BOp::Mul,
                lhs: Box::new(Expr::int(2)),
                rhs: Box::new(Expr::int(3)),
            }),
        });

        test_expr_is("1*2+3", |_cx| Expr::BinOp {
            op: ast::BOp::Add,
            lhs: Box::new(Expr::BinOp {
                op: ast::BOp::Mul,
                lhs: Box::new(Expr::int(1)),
                rhs: Box::new(Expr::int(2)),
            }),
            rhs: Box::new(Expr::int(3)),
        });

        test_expr_is("x()", |cx| Expr::Funcall {
            func: Box::new(Expr::Var {
                name: cx.intern("x"),
            }),
            params: vec![],
        });
        test_expr_is("(x())", |cx| Expr::Funcall {
            func: Box::new(Expr::Var {
                name: cx.intern("x"),
            }),
            params: vec![],
        });

        test_expr_is("(1+2)*3", |_cx| Expr::BinOp {
            op: ast::BOp::Mul,
            lhs: Box::new(Expr::BinOp {
                op: ast::BOp::Add,
                lhs: Box::new(Expr::int(1)),
                rhs: Box::new(Expr::int(2)),
            }),
            rhs: Box::new(Expr::int(3)),
        });
    }

    #[test]
    fn parse_tuple_values() {
        test_expr_is("{}", |_cx| Expr::unit());
        test_expr_is("{1,2,3}", |_cx| Expr::TupleCtor {
            body: vec![Expr::int(1), Expr::int(2), Expr::int(3)],
        });
        test_expr_is("{1,2,{1,2,3},3}", |_cx| Expr::TupleCtor {
            body: vec![
                Expr::int(1),
                Expr::int(2),
                Expr::TupleCtor {
                    body: vec![Expr::int(1), Expr::int(2), Expr::int(3)],
                },
                Expr::int(3),
            ],
        });
    }

    #[test]
    fn parse_tuple_types() {
        let valid_args = &[
            "{}",
            "{I32}",
            "{I32,}",
            "{Bool, Bool, I32}",
            "{Bool, {}, I32}",
        ][..];
        test_parse_with(|p| p.parse_type(), &valid_args)
    }

    #[test]
    fn parse_deref() {
        let valid_args = &["x^", "10^", "z^.0", "z.0^", "(1+2*3)^"][..];
        test_parse_with(|p| p.parse_expr(0), &valid_args)
    }
}
