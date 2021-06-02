//! We're just going to do a simple LL recursive-descent parser.
//! It's simple, robust, fast, and pretty hard to screw up.
//!
//! This does get enhanced for parsing expressions into a Pratt
//! parser, as described  here:
//! <https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html>
//! This is extremely nice for parsing infix and postfix operators
//! with a precendence that's defined trivially by a simple look-up function.
//! I like it a lot.

use std::mem::Discriminant as Discr;
use std::ops::Range;

use codespan_reporting as cs;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use logos::{Lexer, Logos};

use crate::ast;
use crate::*;

fn bounds_check(val: i128, int_size: u8) -> Option<(i128, u8)> {
    let bound = 2_i128.pow(int_size as u32 * 8);
    dbg!(val, bound, bound / 2 - 1);
    if val > (bound / 2) - 1 || val <= -(bound / 2) {
        None
    } else {
        Some((val, int_size))
    }
}

/// Turn this into something parse() can parse,
/// so,
/// 123_456_I32 becomes 123456
fn extract_digits(s: &str) -> String {
    /// Is the char a digit or the separator '_'?
    fn is_digitish(c: &char) -> bool {
        c.is_digit(10) || (*c) == '_'
    }
    let digits = s
        .chars()
        .take_while(is_digitish)
        .filter(|c| c.is_digit(10))
        .collect();
    digits
}

fn make_i8(lex: &mut Lexer<TokenKind>) -> Option<(i128, u8)> {
    let digits = extract_digits(lex.slice());
    let m = digits.parse().ok()?;
    bounds_check(m, 1)
}

fn make_i16(lex: &mut Lexer<TokenKind>) -> Option<(i128, u8)> {
    let digits = extract_digits(lex.slice());
    let m = digits.parse().ok()?;
    bounds_check(m, 2)
}

fn make_i32(lex: &mut Lexer<TokenKind>) -> Option<(i128, u8)> {
    let digits = extract_digits(lex.slice());
    let m = digits.parse().ok()?;
    bounds_check(m, 4)
}

fn make_i64(lex: &mut Lexer<TokenKind>) -> Option<(i128, u8)> {
    let digits = extract_digits(lex.slice());
    let m = digits.parse().ok()?;
    bounds_check(m, 8)
}

fn make_i128(lex: &mut Lexer<TokenKind>) -> Option<(i128, u8)> {
    let digits = extract_digits(lex.slice());
    let m = digits.parse().ok()?;
    // No bounds check, since our internal type is i128 anyway.
    //bounds_check(m, 16)
    Some((m, 16))
}

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum TokenKind {
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_owned())]
    Ident(String),
    #[regex("true|false", |lex| lex.slice().parse())]
    Bool(bool),
    #[regex("[0-9][0-9_]*I8", make_i8)]
    #[regex("[0-9][0-9_]*I16", make_i16)]
    #[regex("[0-9][0-9_]*I32", make_i32)]
    #[regex("[0-9][0-9_]*I64", make_i64)]
    #[regex("[0-9][0-9_]*I128", make_i128)]
    IntegerSize((i128, u8)),
    #[regex("[0-9][0-9_]*", |lex| lex.slice().parse())]
    Integer(i128),

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
    #[token("return")]
    Return,
    #[token("type")]
    Type,
    #[token("struct")]
    Struct,

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
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token("=")]
    Equals,

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
    #[regex(r"--.*\n", |lex| lex.slice().to_owned())]
    Comment(String),

    #[error]
    #[regex(r"[ \t\n\f]+", logos::skip)]
    Error,
}

impl TokenKind {
    /// Shortcut for std::mem::discriminant()
    fn discr(&self) -> Discr<Self> {
        std::mem::discriminant(self)
    }
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
        panic!("Error somewhere in parser")
    }
}

pub struct Parser<'input> {
    lex: std::iter::Peekable<logos::SpannedIter<'input, TokenKind>>,
    source: &'input str,
    err: ErrorReporter,
}

impl<'input> Parser<'input> {
    pub fn new(source: &'input str) -> Self {
        let lex = TokenKind::lexer(source).spanned().peekable();
        let err = ErrorReporter::new("module", source);
        Parser { lex, source, err }
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
                .with_labels(vec![Label::primary(self.err.file_id, span)]);

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
                    .with_labels(vec![Label::primary(self.err.file_id, t.span)]);

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

    /// Returns whether the next token in the stream is what is expected.
    ///
    /// Basically, we can't (easily) pass in a strongly-typed enum discriminant
    /// and have only that checked, we have to pass the full enum.
    /// Ironically, this is exactly the sort of thing I want Garnet to be
    /// able to do nicely.
    ///
    /// ...double ironically, the above paragraph is incorrect as of 1.21,
    /// because std::mem::discriminant() is better than I thought.
    fn peek_is(&mut self, expected: Discr<TokenKind>) -> bool {
        if let Some(got) = self.peek() {
            std::mem::discriminant(&got.kind) == expected
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
            }) => crate::INT.intern(s),
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
    fn expect_int(&mut self) -> i128 {
        match self.next() {
            Some(Token {
                kind: T::Integer(s),
                ..
            }) => s,
            Some(Token {
                kind: T::IntegerSize((s, _)),
                ..
            }) => s,
            Some(Token { kind, span }) => {
                let msg = format!("Parse error: got token {:?}.  Expected integer.", kind,);
                let diag = Diagnostic::error()
                    .with_message(msg)
                    .with_labels(vec![Label::primary(self.err.file_id, span)]);

                self.err.error(&diag);
            }
            None => {
                let msg = format!(
                    "Parse error: Got end of input or malformed token.  Expected integer.",
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
                Some(Token { kind: T::Type, .. }) => Some(p.parse_typedef(doc_comments)),
                Some(Token {
                    kind: T::Struct, ..
                }) => Some(p.parse_structdef(doc_comments)),
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

    fn parse_typedef(&mut self, doc_comment: Vec<String>) -> ast::Decl {
        let name = self.expect_ident();
        self.expect(T::Equals);
        let typedecl = self.parse_type();
        ast::Decl::TypeDef {
            name,
            typedecl,
            doc_comment,
        }
    }

    fn parse_structdef(&mut self, doc_comment: Vec<String>) -> ast::Decl {
        let name = self.expect_ident();
        self.expect(T::LBrace);
        let fields = self.parse_struct_fields();
        self.expect(T::RBrace);
        ast::Decl::StructDef {
            name,
            fields,
            doc_comment,
        }
    }

    /// signature = fn_args [":" typename]
    fn parse_fn_signature(&mut self) -> ast::Signature {
        let params = self.parse_fn_args();
        let rettype = if self.peek_is(T::Colon.discr()) {
            self.expect(T::Colon);
            self.parse_type()
        } else {
            crate::INT.unit()
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

            if self.peek_is(T::Comma.discr()) {
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
        let rettype = if self.peek_is(T::Colon.discr()) {
            self.expect(T::Colon);
            self.parse_type()
        } else {
            crate::INT.unit()
        };
        TypeDef::Lambda(params, rettype)
    }

    fn parse_fn_type_args(&mut self) -> Vec<TypeSym> {
        let mut args = vec![];
        self.expect(T::LParen);

        while !self.peek_is(T::RParen.discr()) {
            let tname = self.parse_type();
            args.push(tname);

            if self.peek_is(T::Comma.discr()) {
                self.expect(T::Comma);
            } else {
                break;
            }
        }
        self.expect(T::RParen);
        args
    }

    fn parse_struct_fields(&mut self) -> Vec<(VarSym, TypeSym)> {
        let mut args = vec![];

        // TODO someday: Doc comments on struct fields
        while let Some((T::Ident(_i), _span)) = self.lex.peek() {
            let name = self.expect_ident();
            self.expect(T::Colon);
            let tname = self.parse_type();
            args.push((name, tname));

            if self.peek_is(T::Comma.discr()) {
                self.expect(T::Comma);
            } else {
                break;
            }
        }
        args
    }

    fn parse_tuple_type(&mut self) -> TypeDef {
        let mut body = vec![];
        while !self.peek_is(T::RBrace.discr()) {
            //while let Some(expr) = self.parse_type() {
            let t = self.parse_type();
            body.push(t);

            if self.peek_is(T::Comma.discr()) {
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
            // Hey and if we have a semicolon after an expr we can just eat it
            if self.peek_is(T::Semicolon.discr()) {
                self.drop();
            }
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
            T::Integer(_) => ast::Expr::int(self.expect_int() as i128),
            T::IntegerSize((_str, size)) => ast::Expr::sized_int(self.expect_int() as i128, *size),
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
            T::Return => self.parse_return(),
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
        while let Some((op_token, _span)) = self.lex.peek().cloned() {
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
                    T::Colon => {
                        self.expect(T::Colon);
                        let ident = self.expect_ident();
                        let ident_expr = ast::Expr::Var { name: ident };
                        let mut params = self.parse_function_args();
                        params.insert(0, lhs);
                        ast::Expr::Funcall {
                            func: Box::new(ident_expr),
                            params,
                        }
                    }
                    T::Period => {
                        self.expect(T::Period);
                        let next_token = self.next().unwrap_or_else(|| self.error(None));
                        // If the period is followed by an int, it's a
                        // tuple ref, otherwise it's a struct ref.
                        match next_token.kind.clone() {
                            T::Ident(i) => ast::Expr::StructRef {
                                expr: Box::new(lhs),
                                elt: INT.intern(i),
                            },
                            // Following Rust, we do not allow numbers
                            // with suffixes as tuple indices.
                            T::Integer(elt) => {
                                assert!(elt > -1);
                                ast::Expr::TupleRef {
                                    expr: Box::new(lhs),
                                    elt: elt as usize,
                                }
                            }
                            _ => self.error(Some(next_token)),
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
                // Not all things that parse like binary operations
                // actually produce the "BinOp" expression type.
                lhs = match op_token {
                    // x = y
                    T::Equals => ast::Expr::Assign {
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    _ => {
                        let bop = bop_for(&op_token).unwrap();
                        ast::Expr::BinOp {
                            op: bop,
                            lhs: Box::new(lhs),
                            rhs: Box::new(rhs),
                        }
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
            if !self.peek_is(T::Comma.discr()) {
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
        let mutable = if self.peek_is(T::Mut.discr()) {
            self.expect(T::Mut);
            true
        } else {
            false
        };
        let varname = self.expect_ident();
        self.expect(T::Colon);
        let typename = Some(self.parse_type());
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

    /// return = "return" expr
    fn parse_return(&mut self) -> ast::Expr {
        self.expect(T::Return);
        let retval = Box::new(
            self.parse_expr(0)
                .expect("Expected expression after `let ... =`, did not get one"),
        );
        ast::Expr::Return { retval }
    }

    /// {"elseif" expr "then" {expr}}
    fn parse_elif(&mut self, accm: &mut Vec<ast::IfCase>) {
        while self.peek_is(T::Elseif.discr()) {
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

    /// block = "do" {expr} "end"
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

    /// tuple constructor = "{" [expr {"," expr} [","] "}"
    fn parse_constructor(&mut self) -> ast::Expr {
        self.expect(T::LBrace);
        let mut body = vec![];
        while let Some(expr) = self.parse_expr(0) {
            body.push(expr);

            if self.peek_is(T::Comma.discr()) {
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
                span: _,
            }) => match s.as_ref() {
                // TODO: This is a bit too hardwired tbh...
                "I128" => crate::INT.i128(),
                "I64" => crate::INT.i64(),
                "I32" => crate::INT.i32(),
                "I16" => crate::INT.i16(),
                "I8" => crate::INT.i8(),
                "Bool" => crate::INT.bool(),
                s => crate::INT.named_type(s),
            },
            Some(Token {
                kind: T::LBrace, ..
            }) => {
                let tuptype = self.parse_tuple_type();
                crate::INT.intern_type(&tuptype)
            }
            Some(Token { kind: T::Fn, .. }) => {
                let fntype = self.parse_fn_type();
                crate::INT.intern_type(&fntype)
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
        // "." for tuple/struct references.
        T::Period => Some((130, ())),
        // "(" opening function call args
        T::LParen => Some((120, ())),
        // ":" universal function call syntax
        T::Colon => Some((115, ())),
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
    use crate::{TypeDef, INT};

    /// Take a list of strings and try parsing them with the given function.
    /// Is ok iff the parsing succeeds, does no checking that the produced
    /// AST is actually something that you want, or anything at all.
    fn test_parse_with<T>(f: impl Fn(&mut Parser) -> T, strs: &[&str]) {
        for s in strs {
            let mut p = Parser::new(s);
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
    fn test_expr_is(s: &str, f: impl Fn() -> Expr) {
        let ast = f();
        let mut p = Parser::new(s);
        let parsed_expr = p.parse_expr(0).unwrap();
        assert_eq!(&ast, &parsed_expr);
        // Make sure we've parsed the whole string.
        assert_eq!(p.lex.peek(), None);
    }

    /// Same as test_expr_is but with decl's
    fn test_decl_is(s: &str, f: impl Fn() -> ast::Decl) {
        let ast = f();
        let mut p = Parser::new(s);
        let parsed_decl = p.parse_decl().unwrap();
        assert_eq!(&ast, &parsed_decl);
        // Make sure we've parsed the whole string.
        assert_eq!(p.lex.peek(), None);
    }

    #[test]
    fn test_const() {
        test_decl_is("const foo: I32 = -9", || ast::Decl::Const {
            name: INT.intern("foo"),
            typename: INT.intern_type(&TypeDef::SInt(4)),
            init: Expr::UniOp {
                op: ast::UOp::Neg,
                rhs: Box::new(Expr::int(9)),
            },
            doc_comment: vec![],
        });
    }

    #[test]
    fn test_fn() {
        test_decl_is("fn foo(x: I32): I32 = 9 end", || {
            let i32_t = INT.intern_type(&TypeDef::SInt(4));
            ast::Decl::Function {
                name: INT.intern("foo"),
                signature: ast::Signature {
                    params: vec![(INT.intern("x"), i32_t)],
                    rettype: i32_t,
                },
                body: vec![Expr::int(9)],
                doc_comment: vec![],
            }
        });
    }

    #[test]
    fn test_typedef() {
        test_decl_is("type bop = I32", || ast::Decl::TypeDef {
            name: INT.intern("bop"),
            typedecl: INT.intern_type(&TypeDef::SInt(4)),
            doc_comment: vec![],
        });
    }

    #[test]
    fn test_multiple_decls() {
        let s = r#"
const foo: I32 = -9
const bar: Bool = 4
--- rawr!
const baz: {} = {}
type blar = I8
"#;
        let p = &mut Parser::new(s);
        let foosym = INT.intern("foo");
        let barsym = INT.intern("bar");
        let bazsym = INT.intern("baz");
        let blarsym = INT.intern("blar");
        let i32_t = INT.i32();
        let i8_t = INT.i8();
        let bool_t = INT.bool();
        let unit_t = INT.unit();
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
                    },
                    ast::Decl::TypeDef {
                        name: blarsym,
                        typedecl: i8_t,
                        doc_comment: vec![],
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
            "(x: I16, y: Bool)",
            "(x: I64, y: Bool):Bool",
            "(x: I8, y: Bool,)",
            "(x: I32, y: Bool,):Bool",
            "(f: fn(I32):I128, x: I32):Bool",
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
        test_expr_is("y(1, 2, 3)", || Expr::Funcall {
            func: Box::new(Expr::Var {
                name: INT.intern("y"),
            }),
            params: vec![Expr::int(1), Expr::int(2), Expr::int(3)],
        });

        test_expr_is("foo(0, bar(1 * 2), 3)", || Expr::Funcall {
            func: Box::new(Expr::Var {
                name: INT.intern("foo"),
            }),
            params: vec![
                Expr::int(0),
                Expr::Funcall {
                    func: Box::new(Expr::Var {
                        name: INT.intern("bar"),
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

        test_expr_is("(1)", || Expr::int(1));
        test_expr_is("(((1)))", || Expr::int(1));
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
            || Expr::If {
                cases: vec![
                    ast::IfCase {
                        condition: Box::new(Expr::var("x")),
                        body: vec![Expr::int(1)],
                    },
                    ast::IfCase {
                        condition: Box::new(Expr::var("y")),
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
            || Expr::If {
                cases: vec![ast::IfCase {
                    condition: Box::new(Expr::var("x")),
                    body: vec![Expr::int(1)],
                }],
                falseblock: vec![Expr::If {
                    cases: vec![ast::IfCase {
                        condition: Box::new(Expr::var("y")),
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
        test_expr_is("1+2", || Expr::BinOp {
            op: ast::BOp::Add,
            lhs: Box::new(Expr::int(1)),
            rhs: Box::new(Expr::int(2)),
        });

        test_expr_is("1+2*3", || Expr::BinOp {
            op: ast::BOp::Add,
            lhs: Box::new(Expr::int(1)),
            rhs: Box::new(Expr::BinOp {
                op: ast::BOp::Mul,
                lhs: Box::new(Expr::int(2)),
                rhs: Box::new(Expr::int(3)),
            }),
        });

        test_expr_is("1*2+3", || Expr::BinOp {
            op: ast::BOp::Add,
            lhs: Box::new(Expr::BinOp {
                op: ast::BOp::Mul,
                lhs: Box::new(Expr::int(1)),
                rhs: Box::new(Expr::int(2)),
            }),
            rhs: Box::new(Expr::int(3)),
        });

        test_expr_is("x()", || Expr::Funcall {
            func: Box::new(Expr::Var {
                name: INT.intern("x"),
            }),
            params: vec![],
        });
        test_expr_is("(x())", || Expr::Funcall {
            func: Box::new(Expr::Var {
                name: INT.intern("x"),
            }),
            params: vec![],
        });

        test_expr_is("(1+2)*3", || Expr::BinOp {
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
    fn lex_integer_values() {
        let tests = &[
            // Input string, expected integer, expected integer size
            ("999_888I32", 999_888, 4),
            ("43_I8", 43, 1),
            ("127_I8", 127, 1),
            ("22_I16", 22, 2),
            ("33_I32", 33, 4),
            ("91_I64", 91, 8),
            ("9_I128", 9, 16),
        ];
        for (s, expected_int, expected_bytes) in tests {
            let mut p = Parser::new(s);
            assert_eq!(
                p.next().unwrap().kind,
                TokenKind::IntegerSize((*expected_int, *expected_bytes))
            );
            // Make sure we don't lex the "i128" or whatever as the start of
            // another token
            assert!(p.next().is_none());
        }
    }

    /// Make sure out-of-bounds integer literals are errors.
    ///
    /// ...we actually don't have negative literals
    #[test]
    fn lex_integer_value_invalid() {
        let tests = &[
            // Input string, expected integer, expected integer size
            "999_I8",
            "256_I8",
            "128_I8",
            "65_536_I16",
            "32_768_I16",
            "999_999_I16",
        ];
        for s in tests {
            let mut p = Parser::new(s);
            assert_eq!(p.next().unwrap().kind, TokenKind::Error);
            assert!(p.next().is_none());
        }
    }

    #[test]
    fn parse_integer_values() {
        test_expr_is("43_I8", || Expr::sized_int(43, 1));
        /*
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
        */
    }

    #[test]
    fn parse_tuple_values() {
        test_expr_is("{}", || Expr::unit());
        test_expr_is("{1,2,3}", || Expr::TupleCtor {
            body: vec![Expr::int(1), Expr::int(2), Expr::int(3)],
        });
        test_expr_is("{1,2,{1,2,3},3}", || Expr::TupleCtor {
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
