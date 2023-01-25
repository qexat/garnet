use std::mem::Discriminant as Discr;
use std::ops::Range;

use codespan_reporting as cs;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use fnv::FnvHashMap;
use logos::{Lexer, Logos};

use crate::ast;
use crate::*;

fn eat_block_comment(lex: &mut Lexer<TokenKind>) -> String {
    let mut nest_depth = 1;
    while nest_depth != 0 {
        const DELIMITER_BYTES: usize = 2;
        let next_bit = &lex.remainder().get(..DELIMITER_BYTES);
        // Lexer::bump() works in bytes, not chars, so we have to track the
        // number of bytes we are stepping forward so we don't try to lex
        // the middle of a UTF-8 char.
        let bytes_to_advance = match next_bit {
            Some("/-") => {
                nest_depth += 1;
                DELIMITER_BYTES
            }
            Some("-/") => {
                nest_depth -= 1;
                DELIMITER_BYTES
            }
            Some(other) => other
                .chars()
                .next()
                .unwrap_or_else(|| panic!("Invalid UTF-8 in input file?  This should probably never happen otherwise."))
                .len_utf8(),
            None => panic!("Unclosed block comment?"),
        };
        lex.bump(bytes_to_advance);
    }
    String::from("")
}

#[allow(missing_docs)]
#[derive(Logos, Debug, PartialEq, Clone)]
pub enum TokenKind {
    #[regex("[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_owned())]
    Ident(String),
    #[regex("true|false", |lex| lex.slice().parse())]
    Bool(bool),
    #[regex("[0-9][0-9_]*", |lex| lex.slice().parse())]
    Integer(i32),

    // Decl stuff
    #[token("fn")]
    Fn,
    #[token("type")]
    Type,

    // Keywords
    #[token("let")]
    Let,
    #[token("end")]
    End,
    #[token("struct")]
    Struct,
    #[token("const")]
    Const,
    #[token("enum")]
    Enum,
    #[token("sum")]
    Sum,

    // Punctuation
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
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
    #[token("@")]
    At,
    #[token("$")]
    Dollar,

    // We save comment strings so we can use this same
    // parser as a reformatter or such.
    #[regex(r"--.*\n", |lex| lex.slice().to_owned())]
    #[regex(r"/-.*", eat_block_comment)]
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

/// The core parser struct.  It provides basic methods for
/// manipulating the input stream, and the `parse()` method to
/// try to drive the given input to completion.
pub struct Parser<'input> {
    lex: std::iter::Peekable<logos::SpannedIter<'input, TokenKind>>,
    source: &'input str,
    err: ErrorReporter,
}

impl<'input> Parser<'input> {
    pub fn new(filename: &str, source: &'input str) -> Self {
        let lex = TokenKind::lexer(source).spanned().peekable();
        let err = ErrorReporter::new(filename, source);
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
        match t {
            // Recurse to skip comments
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
        match t {
            // Skip comments
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

    /// Takes a string describing what it expected, and a token that is what
    /// it actually got.
    fn error(&self, expected: &str, token_or_eof: Option<Token>) -> ! {
        let diag = if let Some(got) = token_or_eof {
            let msg = format!(
                "Parse error on token {:?} from str {}.  Expected {}",
                got.kind,
                &self.source[got.span.clone()],
                expected
            );
            Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(self.err.file_id, got.span)])
        } else {
            let len = self.source.len();
            let msg = format!("Parse error on end of file.  Expected {}", expected);
            Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(self.err.file_id, len..len)])
        };
        self.err.error(&diag);
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
                    "Parse error on token {:?} from str {}.  Expected token: {:?}",
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

    /// Returns whether the next token in the stream is what is expected,
    /// and consume it if so.
    ///
    /// I ended up seeing a lot of `if self.peek_is(thing) { self.expect(thing)`
    /// so maybe this helps.
    fn peek_expect(&mut self, expected: Discr<TokenKind>) -> bool {
        if self.peek_is(expected) {
            self.drop();
            true
        } else {
            false
        }
    }

    /// Consume an identifier and return its interned symbol.
    /// Note this returns a String, not a TypeInfo...
    fn expect_ident(&mut self) -> String {
        match self.next() {
            Some(Token {
                kind: T::Ident(s), ..
            }) => s.to_string(),
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
        match self.next() {
            Some(Token { kind: T::Fn, .. }) => Some(self.parse_fn()),
            Some(Token { kind: T::Type, .. }) => Some(self.parse_typedef()),
            Some(Token { kind: T::Const, .. }) => Some(self.parse_const()),
            Some(other) => self.error("start of decl", Some(other)),
            None => None,
        }
    }

    /// typedef = "type" ident "=" type
    fn parse_typedef(&mut self) -> ast::Decl {
        let name = self.expect_ident();
        let mut params = vec![];
        if self.peek_expect(T::LParen.discr()) {
            while !self.peek_is(T::RParen.discr()) {
                self.expect(T::At);
                let name = self.expect_ident();
                params.push(name);
                if !self.peek_expect(T::Comma.discr()) {
                    break;
                }
            }
            self.expect(T::RParen);
        }

        self.expect(T::Equals);
        let ty = self.parse_type();
        ast::Decl::TypeDef { name, params, ty }
    }

    fn parse_const(&mut self) -> ast::Decl {
        let name = self.expect_ident();
        self.expect(T::Equals);
        let init = self
            .parse_expr(0)
            .expect("Expected expression after `const ... =`, did not get one");
        ast::Decl::ConstDef { name, init }
    }

    fn parse_fn(&mut self) -> ast::Decl {
        let name = self.expect_ident();
        let signature = ast::Signature {
            ..self.parse_fn_signature()
        };
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
        let rettype = self.parse_type();
        ast::Signature { params, rettype }
    }

    /// sig = ident ":" typename
    /// fn_args = "(" [sig {"," sig} [","]] ")"
    fn parse_fn_args(&mut self) -> Vec<(String, Type)> {
        let mut args = vec![];
        self.expect(T::LParen);

        while let Some((T::Ident(_i), _span)) = self.lex.peek() {
            let name = self.expect_ident();
            let tname = self.parse_type();
            args.push((name, tname));

            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        self.expect(T::RParen);
        args
    }

    fn parse_fn_type(&mut self) -> Type {
        let params = self.parse_type_list();
        let rettype = self.parse_type();
        Type::Func(params, Box::new(rettype))
    }

    /// type_list = "(" [type {"," type} [","] ")"
    fn parse_type_list(&mut self) -> Vec<Type> {
        let mut args = vec![];
        self.expect(T::LParen);

        while !self.peek_is(T::RParen.discr()) {
            let tname = self.parse_type();
            args.push(tname);

            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        self.expect(T::RParen);
        args
    }

    fn parse_tuple_type(&mut self) -> Type {
        let mut body = vec![];
        while !self.peek_is(T::RBrace.discr()) {
            let t = self.parse_type();
            body.push(t);
            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        self.expect(T::RBrace);
        Type::Named("Tuple".to_string(), body)
    }

    fn parse_exprs(&mut self) -> Vec<ast::ExprNode> {
        let mut exprs = vec![];
        let tok = self.peek();
        while let Some(e) = self.parse_expr(0) {
            // Hey and if we have a semicolon after an expr we can just eat it
            self.peek_expect(T::Semicolon.discr());
            exprs.push(e);
        }
        if exprs.is_empty() {
            self.error(
                "non-empty expression block, must have at least one value.",
                tok,
            );
        }
        exprs
    }

    /// Returns None if there is no valid expression,
    /// which usually means the end of a block or such.
    ///
    /// This departs from pure recursive descent and uses a Pratt
    /// parser to parse math expressions and such.
    fn parse_expr(&mut self, min_bp: usize) -> Option<ast::ExprNode> {
        let t = self.peek()?;
        let token = &t.kind;
        let mut lhs = match token {
            T::Integer(_) => ast::ExprNode::new(ast::Expr::int(self.expect_int())),
            T::Bool(b) => {
                self.drop();
                ast::ExprNode::new(ast::Expr::Lit {
                    val: ast::Literal::Bool(*b),
                })
            }
            T::Ident(_) => {
                let ident = self.expect_ident();
                ast::ExprNode::new(ast::Expr::Var { name: ident })
            }
            T::Struct => self.parse_struct_literal(),
            T::Let => self.parse_let(),
            T::Fn => self.parse_lambda(),
            // Parenthesized expr's
            T::LParen => {
                self.drop();
                let lhs = self.parse_expr(0)?;
                self.expect(T::RParen);
                lhs
            }
            // Tuple or struct literal
            T::LBrace => self.parse_constructor(),
            // Array literal
            T::LBracket => self.parse_array_constructor(),

            // Something else not a valid expr
            _x => return None,
        };
        // Parse a postfix or infix expression with a given
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
                        ast::ExprNode::new(ast::Expr::Funcall { func: lhs, params })
                    }
                    // If we see `foo {bar}`
                    // then parse it as `foo({bar})`
                    // Thanks Lua!!!
                    T::LBrace => {
                        let params = self.parse_constructor();
                        ast::ExprNode::new(ast::Expr::Funcall {
                            func: lhs,
                            params: vec![params],
                        })
                    }
                    T::Period => {
                        self.expect(T::Period);
                        let struct_field_name = self.expect_ident();
                        ast::ExprNode::new(ast::Expr::StructRef {
                            e: lhs,
                            name: struct_field_name,
                        })
                    }
                    T::Dollar => {
                        self.expect(T::Dollar);
                        ast::ExprNode::new(ast::Expr::TypeUnwrap { e: lhs })
                    }
                    _ => return None,
                };
                continue;
            }
            /*
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
            */
            // None of the above, so we are done parsing the expr
            break;
        }
        Some(lhs)
    }

    fn parse_function_args(&mut self) -> Vec<ast::ExprNode> {
        let mut params = vec![];
        self.expect(T::LParen);
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
    fn parse_let(&mut self) -> ast::ExprNode {
        self.expect(T::Let);
        let varname = self.expect_ident();
        let typename = if self.peek_expect(T::Equals.discr()) {
            None
        } else {
            let t = Some(self.parse_type());
            self.expect(T::Equals);
            t
        };
        let init = self
            .parse_expr(0)
            .expect("Expected expression after `let ... =`, did not get one");
        ast::ExprNode::new(ast::Expr::Let {
            varname,
            typename,
            init,
        })
    }

    /// lambda = "fn" "(" ...args... ")" [":" typename] = {exprs} "end"
    fn parse_lambda(&mut self) -> ast::ExprNode {
        self.expect(T::Fn);
        let signature = self.parse_fn_signature();
        self.expect(T::Equals);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::ExprNode::new(ast::Expr::Lambda { signature, body })
    }

    /// struct constructor = "{" "." ident "=" expr {"," ...} "}"
    fn parse_array_constructor(&mut self) -> ast::ExprNode {
        self.expect(T::LBracket);
        let mut body = vec![];
        while let Some(expr) = self.parse_expr(0) {
            body.push(expr);

            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        self.expect(T::RBracket);
        ast::ExprNode::new(ast::Expr::ArrayCtor { body })
    }

    /// If we see `.foo = bar` in our thing then it's a struct
    fn parse_constructor(&mut self) -> ast::ExprNode {
        self.expect(T::LBrace);
        if self.peek_is(T::Period.discr()) {
            self.parse_struct_literal()
        } else {
            self.parse_tuple_literal()
        }
    }

    /// struct constructor = "{" "." ident "=" expr {"," ...} "}"
    fn parse_struct_literal(&mut self) -> ast::ExprNode {
        let body = self.parse_struct_lit_fields();
        self.expect(T::RBrace);
        ast::ExprNode::new(ast::Expr::StructCtor { body })
    }

    /// tuple constructor = "{" [expr {"," expr} [","] "}"
    fn parse_tuple_literal(&mut self) -> ast::ExprNode {
        let mut body = vec![];
        while let Some(expr) = self.parse_expr(0) {
            body.push(expr);

            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        self.expect(T::RBrace);
        ast::ExprNode::new(ast::Expr::TupleCtor { body })
    }

    fn parse_struct_lit_fields(&mut self) -> FnvHashMap<String, ast::ExprNode> {
        let mut fields = FnvHashMap::default();

        loop {
            if self.peek_expect(T::Period.discr()) {
                let name = self.expect_ident();
                self.expect(T::Equals);
                let vl = self.parse_expr(0).unwrap();
                fields.insert(name, vl);
            } else {
                break;
            }

            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        fields
    }

    fn parse_type(&mut self) -> Type {
        let t = self.next().unwrap_or_else(|| self.error("type", None));
        let mut ty = match t.kind {
            T::Ident(ref s) => {
                let type_params = if self.peek_is(T::LParen.discr()) {
                    self.parse_type_list()
                } else {
                    vec![]
                };
                Type::Named(s.clone(), type_params)
            }
            T::At => {
                let s = self.expect_ident();
                Type::Generic(s)
            }
            T::LBrace => {
                let tuptype = self.parse_tuple_type();
                tuptype
            }
            T::Fn => {
                let fntype = self.parse_fn_type();
                fntype
            }
            T::Struct => self.parse_struct_type(),
            T::Enum => self.parse_enum_type(),
            T::Sum => self.parse_sum_type(),
            // Apparently all our types are prefix, which is weird.
            // Fix this someday.
            _ => self.error("type", Some(t)),
        };
        // Array?  (Or other postfix shit, this be gettin whack)
        while self.peek_is(T::LBracket.discr()) {
            self.expect(T::LBracket);
            let len = self.expect_int();
            assert!(len >= 0);
            self.expect(T::RBracket);
            ty = Type::Array(Box::new(ty), len as usize);
        }
        ty
    }

    /// isomorphic-ish with parse_type_list()
    fn parse_struct_type(&mut self) -> Type {
        let mut fields = FnvHashMap::default();

        loop {
            match self.lex.peek() {
                Some((T::Ident(_i), _span)) => {
                    let name = self.expect_ident();
                    self.expect(T::Colon);
                    let vl = self.parse_type();
                    fields.insert(name, vl);
                }
                _ => break,
            }

            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        self.expect(T::End);
        // Pull any @Foo types out of the structure's
        // declared types
        // To do this we have to recurse down into any
        // complex types THOSE may define (such as functions)
        // and pull out those types too.
        //
        // This feels kinda jank but appears to work
        let generic_names: Vec<_> = fields
            .iter()
            .map(|(_nm, ty)| ty.get_type_params())
            .flatten()
            .map(|ty| Type::Generic(ty))
            .collect();
        Type::Struct(fields, generic_names)
    }

    /// isomorphic with parse_type_list()
    fn parse_enum_type(&mut self) -> Type {
        let mut fields = vec![];
        while !self.peek_is(T::End.discr()) {
            let field = self.expect_ident();
            fields.push(field);

            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        self.expect(T::End);
        Type::Enum(fields)
    }

    /// isomorphic-ish with parse_type_list()
    fn parse_sum_type(&mut self) -> Type {
        let mut fields = FnvHashMap::default();
        while !self.peek_is(T::End.discr()) {
            let field = self.expect_ident();
            let ty = self.parse_type();
            fields.insert(field, ty);

            if self.peek_expect(T::Comma.discr()) {
            } else {
                break;
            }
        }
        self.expect(T::End);
        // Pull any @Foo types out of the structure's
        // declared types, again just like parse_struct_type above.
        let generic_names: Vec<_> = fields
            .iter()
            .map(|(_nm, ty)| ty.get_type_params())
            .flatten()
            .map(|ty| Type::Generic(ty))
            .collect();
        Type::Sum(fields, generic_names)
    }
}

/// Specifies binding power of postfix operators.
///
/// No idea if the precedences for these are sensible,
/// I think they only actually relate with infix operators,
/// not each other.
fn postfix_binding_power(op: &TokenKind) -> Option<(usize, ())> {
    match op {
        // "$" is our "type unwrap" operator, which is weird but ok
        T::Dollar => Some((130, ())),
        // "(" opening function call args
        T::LParen => Some((120, ())),
        // "{" opening function call for single struct arg
        T::LBrace => Some((119, ())),
        // "." separating struct refs a la foo.bar
        T::Period => Some((110, ())),
        // "[" opening array dereference
        T::LBracket => Some((100, ())),
        _x => None,
    }
}

/*
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
*/
