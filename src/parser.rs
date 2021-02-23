//! We're just going to do a simple LL recursive-descent parser.
//! It's simple, robust, fast, and pretty hard to screw up.
//!
//! This does get enhanced for parsing expressions into a Pratt
//! parser, as described  here:
//! <https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html>
//! This is extremely nice for parsing infix and postfix operators
//! with a precendence that's defined trivially by a simple look-up function.
//! I like it a lot.

use std::ops::Range;

use codespan_reporting as cs;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use logos::Logos;

use crate::ast;
use crate::{Cx, TypeDef, TypeSym, VarSym};

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum TokenKind {
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_owned())]
    Ident(String),
    #[regex("true|false", |lex| lex.slice().parse())]
    Bool(bool),
    #[regex("[0-9][0-9_]*", |lex| lex.slice().parse())]
    Integer(i32),

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
    #[token("&")]
    Ampersand,

    #[regex(r"---.*\n", |lex| lex.slice()[3..].to_owned())]
    DocComment(String),

    // We save comment strings so we can use this same
    // parser as a reformatter or such.
    // TODO: How do we skip these in the parser?
    #[regex(r"--.*\n", |lex| lex.slice().to_owned())]
    Comment(String),

    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Error,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Range<usize>,
}

impl Token {
    fn new(kind: TokenKind, span: Range<usize>) -> Self {
        Token { kind, span }
    }
}

fn bop_for(t: &TokenKind) -> Option<ast::BOp> {
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

fn uop_for(t: &TokenKind) -> Option<ast::UOp> {
    match t {
        //T::Plus => Some(ast::UOp::Plus),
        T::Minus => Some(ast::UOp::Neg),
        T::Not => Some(ast::UOp::Not),
        T::Carat => Some(ast::UOp::Deref),
        T::Ampersand => Some(ast::UOp::Ref),
        _other => None,
    }
}

use self::TokenKind as T;

// This is not dead code but sometimes cargo thinks some of it fields are, since their usage is
// cfg'd out in unit tests.
#[allow(dead_code)]
struct ErrorReporter {
    files: cs::files::SimpleFiles<String, String>,
    file_id: usize,
    config: cs::term::Config,
}

impl ErrorReporter {
    fn new(filename: &str, src: &str) -> Self {
        use codespan_reporting::files::SimpleFiles;
        let mut files = SimpleFiles::new();
        let file_id = files.add(filename.to_owned(), src.to_owned());

        Self {
            files,
            file_id,
            config: codespan_reporting::term::Config::default(),
        }
    }

    fn error(&self, _diag: &Diagnostic<usize>) -> ! {
        #[cfg(not(test))]
        {
            use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
            let writer = StandardStream::stderr(ColorChoice::Always);
            cs::term::emit(&mut writer.lock(), &self.config, &self.files, _diag)
                .expect("Could not print error message");
        }
        //std::process::exit(1)
        panic!()
    }
}

pub struct Parser<'cx, 'input> {
    lex: std::iter::Peekable<logos::SpannedIter<'input, TokenKind>>,
    cx: &'cx Cx,
    source: &'input str,
    err: ErrorReporter,
}

impl<'cx, 'input> Parser<'cx, 'input> {
    pub fn new(cx: &'cx Cx, source: &'input str) -> Self {
        let lex = TokenKind::lexer(source).spanned().peekable();
        let err = ErrorReporter::new("module", source);
        Parser {
            lex,
            cx,
            source,
            err,
        }
    }

    /// Read all its input and returns an Ast.
    ///
    /// Currently just panics on error.
    pub fn parse(&mut self) -> ast::Ast {
        let mut decls = vec![];
        while let Some(d) = self.parse_decl() {
            decls.push(d);
        }
        ast::Ast { decls }
    }

    /// Returns the next token, with span.
    fn next(&mut self) -> Option<Token> {
        let t = self.lex.next().map(|(tok, span)| Token::new(tok, span));
        // Skip comments
        match t {
            Some(Token {
                kind: T::Comment(_),
                ..
            }) => self.next(),
            _ => t,
        }
    }

    /// Peeks the next token, with span.
    fn peek(&mut self) -> Option<Token> {
        let t = self
            .lex
            .peek()
            .map(|(tok, span)| Token::new(tok.clone(), span.clone()));
        // Skip comments
        match t {
            Some(Token {
                kind: T::Comment(_),
                ..
            }) => {
                // This must be self.lex.next(), not just self.next().
                let _ = self.lex.next();
                self.peek()
            }
            _ => t,
        }
    }

    fn error(&self, token: Option<Token>) -> ! {
        if let Some(Token { span, .. }) = token {
            let diag = Diagnostic::error()
                .with_message("Parse error: got unexpected/unknown token")
                .with_labels(vec![Label::primary(self.err.file_id, span.clone())]);

            self.err.error(&diag);
        } else {
            panic!("Unexpected end of file!")
        }
    }

    /// Consume a token, we don't care what it is.
    /// Presumably because we've already peeked at it.
    fn drop(&mut self) {
        self.next();
    }

    /// Consume a token that doesn't return anything
    fn expect(&mut self, expected: TokenKind) {
        match self.next() {
            Some(t) if t.kind == expected => (),
            Some(t) => {
                let msg = format!(
                    "Parse error on got token {:?} from str {}.  Expected token: {:?}",
                    t.kind,
                    &self.source[t.span.clone()],
                    expected
                );
                let diag = Diagnostic::error()
                    .with_message(msg)
                    .with_labels(vec![Label::primary(self.err.file_id, t.span.clone())]);

                self.err.error(&diag);
            }
            None => {
                let msg = format!(
                    "Parse error: Got end of input or malformed token.  Expected token: {:?}",
                    expected
                );
                let diag = Diagnostic::error()
                    .with_message(msg)
                    .with_labels(vec![Label::primary(self.err.file_id, 0..0)]);

                self.err.error(&diag);
            }
        }
    }

    /// Returns whether the next token in the stream is what is expected
    fn peek_is(&mut self, expected: TokenKind) -> bool {
        if let Some(got) = self.peek() {
            got.kind == expected
        } else {
            false
        }
    }

    /// Consume an identifier and return its interned symbol.
    /// Note this returns a VarSym, not a TypeSym...
    fn expect_ident(&mut self) -> VarSym {
        match self.next() {
            Some(Token {
                kind: T::Ident(s), ..
            }) => self.cx.intern(s),
            Some(Token { kind, span }) => {
                let msg = format!("Parse error: got token {:?}.  Expected identifier.", kind,);
                let diag = Diagnostic::error()
                    .with_message(msg)
                    .with_labels(vec![Label::primary(self.err.file_id, span)]);

                self.err.error(&diag);
            }
            None => {
                let msg = format!(
                    "Parse error: Got end of input or malformed token.  Expected identifier",
                );
                let diag = Diagnostic::error()
                    .with_message(msg)
                    .with_labels(vec![Label::primary(self.err.file_id, 0..0)]);

                self.err.error(&diag);
            }
        }
    }

    /// Consumes a number and returns it.
    fn expect_int(&mut self) -> i32 {
        match self.next() {
            Some(Token {
                kind: T::Integer(s),
                ..
            }) => s,
            Some(Token { kind, span }) => {
                let msg = format!("Parse error: got token {:?}.  Expected identifier.", kind,);
                let diag = Diagnostic::error()
                    .with_message(msg)
                    .with_labels(vec![Label::primary(self.err.file_id, span)]);

                self.err.error(&diag);
            }
            None => {
                let msg = format!(
                    "Parse error: Got end of input or malformed token.  Expected identifier",
                );
                let diag = Diagnostic::error()
                    .with_message(msg)
                    .with_labels(vec![Label::primary(self.err.file_id, 0..0)]);

                self.err.error(&diag);
            }
        }
    }

    /// Returns None on EOF.
    fn parse_decl(&mut self) -> Option<ast::Decl> {
        fn parse_decl_inner(p: &mut Parser, doc_comments: Vec<String>) -> Option<ast::Decl> {
            match p.next() {
                Some(Token {
                    kind: T::DocComment(s),
                    ..
                }) => {
                    let mut dcs = doc_comments;
                    dcs.push(s);
                    parse_decl_inner(p, dcs)
                }
                Some(Token { kind: T::Const, .. }) => Some(p.parse_const(doc_comments)),
                Some(Token { kind: T::Fn, .. }) => Some(p.parse_fn(doc_comments)),
                None => None,
                other => p.error(other),
            }
        }
        parse_decl_inner(self, vec![])
    }

    fn parse_const(&mut self, doc_comment: Vec<String>) -> ast::Decl {
        let name = self.expect_ident();
        self.expect(T::Colon);
        let typename = self.parse_type();
        self.expect(T::Equals);
        let init = self.parse_expr(0).unwrap();
        ast::Decl::Const {
            name,
            typename,
            init,
            doc_comment,
        }
    }

    fn parse_fn(&mut self, doc_comment: Vec<String>) -> ast::Decl {
        let name = self.expect_ident();
        let signature = self.parse_fn_signature();
        self.expect(T::Equals);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Decl::Function {
            name,
            signature,
            body,
            doc_comment,
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
        let t = self.peek()?;
        let token = &t.kind;
        let mut lhs = match token {
            T::Bool(b) => {
                self.drop();
                ast::Expr::bool(*b)
            }
            T::Integer(_) => ast::Expr::int(self.expect_int() as i64),
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
                    op,
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
                    T::Ampersand => {
                        self.expect(T::Ampersand);
                        ast::Expr::Ref {
                            expr: Box::new(lhs),
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
        let next = self.peek();
        let falseblock = match next {
            // No else block, we're done
            Some(Token { kind: T::End, .. }) => {
                self.expect(T::End);
                vec![]
            }
            // We're in an else block but not an else-if block
            Some(Token { kind: T::Else, .. }) => {
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
        let t = self.next();
        match t {
            Some(Token {
                kind: T::Ident(s),
                span,
            }) => match s.as_ref() {
                // TODO: This is a bit too hardwired tbh...
                "I32" => self.cx.i32(),
                "Bool" => self.cx.bool(),
                _ => self.error(Some(Token {
                    kind: T::Ident(s),
                    span,
                })),
            },
            Some(Token {
                kind: T::LBrace, ..
            }) => {
                let tuptype = self.parse_tuple_type();
                self.cx.intern_type(&tuptype)
            }
            Some(Token { kind: T::Fn, .. }) => {
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
fn prefix_binding_power(op: &TokenKind) -> ((), usize) {
    match op {
        T::Plus | T::Minus | T::Not => ((), 110),
        x => unreachable!("{:?} is not a prefix op, should never happen!", x),
    }
}

/// Specifies binding power of postfix operators.
fn postfix_binding_power(op: &TokenKind) -> Option<(usize, ())> {
    match op {
        // "(" opening function call args
        T::LParen => Some((120, ())),
        // "." for tuple/struct references.
        T::Period => Some((130, ())),
        // "^" for pointer derefs.  TODO: Check precedence?
        T::Carat => Some((105, ())),
        T::Ampersand => Some((105, ())),
        _x => None,
    }
}

/// Specifies binding power of infix operators.
/// The binding power on one side should always be trivially
/// greater than the other, so there's never ambiguity.
fn infix_binding_power(op: &TokenKind) -> Option<(usize, usize)> {
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
            doc_comment: vec![],
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
                doc_comment: vec![],
            }
        });
    }

    #[test]
    fn test_multiple_decls() {
        let s = r#"
const foo: I32 = -9
const bar: Bool = 4
--- rawr!
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
                        },
                        doc_comment: vec![],
                    },
                    ast::Decl::Const {
                        name: barsym,
                        typename: bool_t,
                        init: Expr::int(4),
                        doc_comment: vec![],
                    },
                    ast::Decl::Const {
                        name: bazsym,
                        typename: unit_t,
                        init: Expr::unit(),
                        doc_comment: vec![String::from(" rawr!\n")],
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

    #[test]
    fn parse_ref() {
        let valid_args = &["x&", "10^&", "z&^.0", "z.0^&&", "(1+2*3)&"][..];
        test_parse_with(|p| p.parse_expr(0), &valid_args)
    }
}
