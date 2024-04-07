//! We're just going to do a simple LL recursive-descent parser, with Pratt parsing for infix
//! expressions.  It's simple, robust, fast, and pretty hard to screw up.
//!
//! Pratt parsing is best described here:
//! <https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html>
//! It is extremely nice for parsing infix and postfix operators
//! with a precendence that's defined trivially by a simple look-up function.
//! I like it a lot.

use std::collections::BTreeMap;
use std::mem::Discriminant as Discr;
use std::ops::Range;

use codespan_reporting as cs;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use logos::{Lexer, Logos};

use crate::ast;
use crate::types::*;
use crate::*;

/// Checks whether the given value can fit in `int_size` number
/// of bits.
fn bounds_check_signed(val: i128, int_size: u8) -> Option<(i128, u8)> {
    let bound = 2_i128.pow(int_size as u32 * 8);
    if val > (bound / 2) - 1 || val <= -(bound / 2) {
        None
    } else {
        Some((val, int_size))
    }
}

fn bounds_check_unsigned(val: i128, int_size: u8) -> Option<(i128, u8)> {
    let bound = 2_i128.pow(int_size as u32 * 8);
    if val > bound || val <= 0 {
        None
    } else {
        Some((val, int_size))
    }
}

/// Turn a valid number string into something Rust's `str::parse()` can parse,
/// so,
/// 123_456_I32 becomes 123456.
///
/// Ignores/chops off any trailing characters that are not digits or `_`, so
/// it's up to some other function to make sure that part is correct.
///
/// Only works for integers right now, but who needs floats anyway, amirite?
fn extract_digits(s: &str) -> String {
    /// Is the char a digit or the separator '_'?
    fn is_digitish(c: &char) -> bool {
        c.is_digit(10) || (*c) == '_'
    }
    s.chars()
        .take_while(is_digitish)
        .filter(|c| c.is_digit(10))
        .collect()
}

fn make_int(lex: &mut Lexer<TokenKind>, size: u8, signed: bool) -> Option<(i128, u8, bool)> {
    let digits = extract_digits(lex.slice());
    let m = digits.parse().ok()?;
    let (val, size) = if signed {
        bounds_check_signed(m, size)?
    } else {
        bounds_check_unsigned(m, size)?
    };
    Some((val, size, signed))
}

fn eat_block_comment(lex: &mut Lexer<TokenKind>) -> String {
    let mut nest_depth = 1;
    while nest_depth != 0 {
        const DELIMITER_BYTES: usize = 2;
        // bytes left in the file to lex
        let remaining_bytes = lex.remainder().len();
        if remaining_bytes < DELIMITER_BYTES {
            panic!("Unclosed block comment!")
        } else {
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
                // Not a comment-closing token, so we just step to the next char and keep going.
                // lex.bump() requires a position at the start of a valid character, so we can't just loop
                // through bytes to the start of the next char, we have to tell it where the start of the
                // next char is.

                // Invalid or truncated UTF-8 character, just keep stepping forward until we are past it.
                // The get(..DELIMITER_BYTES) call above will happily chop off the first 2 bytes of a 3+-byte character
                // and then say "oops this is invalid" and return None.
                Some(_) | None => lex.remainder()
                    .chars()
                    .next()
                    .unwrap_or_else(|| panic!("Invalid UTF-8 in input file?  This should probably never happen otherwise."))
                    .len_utf8(),

            };
            lex.bump(bytes_to_advance);
        }
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
    #[regex("[0-9][0-9_]*I8",   |lex| make_int(lex, 1, true))]
    #[regex("[0-9][0-9_]*I16",  |lex| make_int(lex, 2, true))]
    #[regex("[0-9][0-9_]*I32",  |lex| make_int(lex, 4, true))]
    #[regex("[0-9][0-9_]*I64",  |lex| make_int(lex, 8, true))]
    #[regex("[0-9][0-9_]*U8",   |lex| make_int(lex, 1, false))]
    #[regex("[0-9][0-9_]*U16",  |lex| make_int(lex, 2, false))]
    #[regex("[0-9][0-9_]*U32",  |lex| make_int(lex, 4, false))]
    #[regex("[0-9][0-9_]*U64",  |lex| make_int(lex, 8, false))]
    IntegerSize((i128, u8, bool)),
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
    #[regex("end[ \n]*")]
    End,
    #[token("if")]
    If,
    #[regex("then[ \n]*")]
    Then,
    #[regex("elseif[ \n]*")]
    Elseif,
    #[regex("else[ \n]*")]
    Else,
    #[regex("loop[ \n]*")]
    Loop,
    #[regex("while[ \n]*")]
    While,
    #[regex("do[ \n]*")]
    Do,
    #[token("return")]
    Return,
    #[token("break")]
    Break,
    #[token("type")]
    Type,
    #[regex("struct[ \n]*")]
    Struct,
    #[regex("enum[ \n]*")]
    Enum,
    #[regex("sum[ \n]*")]
    Sum,
    #[token("import")]
    Import,
    #[token("as")]
    As,

    // Punctuation
    #[regex("\\([ \n]*")]
    LParen,
    #[token(")")]
    RParen,
    #[regex("\\{[ \n]*")]
    LBrace,
    #[token("}")]
    RBrace,
    #[regex("\\[[ \n]*")]
    LBracket,
    #[token("]")]
    RBracket,
    #[regex(",[ \n]*")]
    Comma,
    #[token(".")]
    Period,
    #[token(":")]
    Colon,
    #[token(";")]
    #[token("\n")]
    Delimiter,
    #[token("=")]
    Equals,
    #[token("$")]
    Dollar,
    #[token("@")]
    At,
    #[token("|")]
    Bar,

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
    #[regex(r"/-.*", eat_block_comment)]
    Comment(String),

    #[error]
    #[regex(r"[ \t\f]+", logos::skip)]
    Error,
}

impl TokenKind {
    /// Shortcut for std::mem::discriminant()
    fn discr(&self) -> Discr<Self> {
        std::mem::discriminant(self)
    }
}

/// Token with contents and source debug info
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

/// Turns a token into an AST binop
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

/// Turns a token into an AST uni-op
fn uop_for(t: &TokenKind) -> Option<ast::UOp> {
    match t {
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
#[derive(Clone)]
struct ErrorReporter {
    files: cs::files::SimpleFiles<String, String>,
    file_id: usize,
    config: cs::term::Config,
}

impl ErrorReporter {
    /// takes the name of the file being parsed and its contents.
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

    /// Panic with a simple error message.
    fn error(&self, _diag: &Diagnostic<usize>) -> ! {
        // Disable fancy diagnostics in unit tests 'cause they get spammy
        // and the default test runner truncates 'em anyway.  Leave this
        // the way it is 'cause the lang_tester tests are set up to look
        // for specific strings on failure.
        //
        // TODO: It really would be nice to have the test harness capture things
        // properly.  Why isn't it?
        #[cfg(not(test))]
        {
            use codespan_reporting::term::termcolor::{ColorChoice, StandardStream};
            let writer = StandardStream::stdout(ColorChoice::Always);
            cs::term::emit(&mut writer.lock(), &self.config, &self.files, _diag)
                .expect("Could not print error message");
        }
        panic!("Error somewhere in parser")
    }
}

/// Helper to parse a delimited list of separated items, such as this pattern:
/// `"(" [sig {"," sig} [","]] ")"`
/// Assumes the starting token has already been consumed, put a block containing
/// your `sig` parser code in it, and it will run it in a loop and search for
/// the given `$tokenkind` such as `TokenKind::RParen` between each run of it.
/// You can break or return from the loop as normal.
///
/// This could be a function that takes a closure, but a macro is slightly easier
/// to work with 'cause you can just write a code block that breaks directly.
macro_rules! parse_delimited {
    ($parser: expr, $tokenkind: path, $body: block) => {{
        loop {
            $body
            if $parser.peek_expect($tokenkind.discr()) {
            } else {
                break;
            }
        }
    }};
}

/// The core parser struct.  It provides basic methods for
/// manipulating the input stream, and the `parse()` method to
/// try to drive the given input to completion.
#[derive(Clone)]
pub struct Parser<'input> {
    // Honestly, just cloning the lexer is easier than dealing with Peekable
    lex: logos::Lexer<'input, TokenKind>,
    source: &'input str,
    err: ErrorReporter,
}

impl<'input> Parser<'input> {
    pub fn new(filename: &str, source: &'input str) -> Self {
        let lex = TokenKind::lexer(source);
        let err = ErrorReporter::new(filename, source);
        Parser { lex, source, err }
    }

    /// Read all its input and returns an Ast.
    ///
    /// Currently just panics on error.
    pub fn parse(&mut self) -> ast::Ast {
        let mut decls = vec![];
        // Look for a doc comment as the first thing in the file.
        // If it's there, it's the package's doc comment.
        // To not do this, just put something else there, like
        // a newline
        let module_docstring = match self.peek() {
            Some(Token {
                kind: T::DocComment(s),
                ..
            }) => {
                self.drop();
                s
            }
            _ => String::new(),
        };
        self.eat_delimiters();
        while let Some(d) = self.parse_decl() {
            decls.push(d);
        }
        let filename = self.err.files.get(self.err.file_id).unwrap().name();
        // TODO: Handle paths and shit better
        let modulename = filename.replacen(".gt", "", 1).replace("/", ".");
        ast::Ast {
            decls,
            filename: filename.clone(),
            modulename,
            module_docstring,
        }
    }

    /// Returns the next token, with span.
    fn next(&mut self) -> Option<Token> {
        let t = self.lex.next().map(|tok| Token::new(tok, self.lex.span()));
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
        let mut peeked_lexer = self.lex.clone();
        // Get the next token without touching the actual lexer
        let t = peeked_lexer
            .next()
            .map(|tok| Token::new(tok.clone(), self.lex.span()));
        match t {
            // Skip comments
            Some(Token {
                kind: T::Comment(_),
                ..
            }) => {
                // Advance the actual lexer to skip past the comment token
                self.lex = peeked_lexer;
                self.peek()
            }
            _ => t,
        }
    }

    /// Panics with a semi-descriptive error.
    ///
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

    /// Consume a particular token that doesn't return anything
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
                    .with_labels(vec![Label::primary(self.err.file_id, self.lex.span())]);

                self.err.error(&diag);
            }
        }
    }

    /// Returns whether the next token in the stream is what is expected.
    /// Use `Token::discr()` so you don't have to create a dummy value for
    /// an Ident or something like that.
    fn peek_is(&mut self, expected: Discr<TokenKind>) -> bool {
        if let Some(got) = self.peek() {
            std::mem::discriminant(&got.kind) == expected
        } else {
            false
        }
    }

    /// gramble gramble
    fn peek_is_ident(&mut self) -> bool {
        self.peek_is(T::Ident("foo".into()).discr())
    }

    /// Returns whether the next token in the stream is what is expected,
    /// and consume it if so.
    ///
    /// I ended up seeing a lot of `if self.peek_is(thing) { self.expect(thing); ... }`
    fn peek_expect(&mut self, expected: Discr<TokenKind>) -> bool {
        if self.peek_is(expected) {
            self.drop();
            true
        } else {
            false
        }
    }

    /// Eat the next zero or more semicolons or newlines
    fn eat_delimiters(&mut self) {
        while self.peek_expect(T::Delimiter.discr()) {}
    }

    /// Consume an identifier and return its interned symbol.
    fn expect_ident(&mut self) -> Sym {
        match self.next() {
            Some(Token {
                kind: T::Ident(s), ..
            }) => Sym::new(s),
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
                    .with_labels(vec![Label::primary(self.err.file_id, self.lex.span())]);

                self.err.error(&diag);
            }
        }
    }

    /// Consumes a number and returns it.
    ///
    /// Returns an i128, which is safely bigger than any integer type we currently
    /// support.
    fn expect_int(&mut self) -> i128 {
        match self.next() {
            Some(Token {
                kind: T::Integer(s),
                ..
            }) => s,
            Some(Token {
                kind: T::IntegerSize((s, _, _)),
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
                    .with_labels(vec![Label::primary(self.err.file_id, self.lex.span())]);

                self.err.error(&diag);
            }
        }
    }

    /// Parse a top-level declaration. Returns None on EOF.
    fn parse_decl(&mut self) -> Option<ast::Decl> {
        // Helper to collect multiple lines of doc comments
        fn parse_decl_inner(p: &mut Parser, doc_comments: Vec<String>) -> Option<ast::Decl> {
            if let Some(tok) = p.next() {
                match &tok.kind {
                    T::DocComment(s) => {
                        let mut dcs = doc_comments;
                        dcs.push(s.clone());
                        parse_decl_inner(p, dcs)
                    }
                    T::Const => Some(p.parse_const(doc_comments)),
                    T::Fn => Some(p.parse_fn(doc_comments)),
                    T::Type => Some(p.parse_typedef(doc_comments)),
                    T::Import => Some(p.parse_import(doc_comments)),
                    T::Delimiter => parse_decl_inner(p, doc_comments),
                    _other => p.error("start of decl", Some(tok)),
                }
            } else {
                None
            }
        }
        parse_decl_inner(self, vec![])
    }

    fn parse_const(&mut self, doc_comment: Vec<String>) -> ast::Decl {
        let name = self.expect_ident();
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
        self.eat_delimiters();
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Decl::Function {
            name,
            signature,
            body,
            doc_comment,
        }
    }

    /// typedef = "type" ident "=" type
    fn parse_typedef(&mut self, doc_comment: Vec<String>) -> ast::Decl {
        let name = self.expect_ident();
        let mut params = vec![];
        if self.peek_expect(T::LParen.discr()) {
            parse_delimited!(self, T::Comma, {
                if self.peek_is(T::RParen.discr()) {
                    break;
                } else {
                    let name = self.expect_ident();
                    params.push(name);
                }
            });
            self.expect(T::RParen);
        }

        self.expect(T::Equals);
        let ty = self.parse_type();
        ast::Decl::TypeDef {
            name,
            params,
            typedecl: ty,
            doc_comment,
        }
    }

    fn parse_import(&mut self, doc_comment: Vec<String>) -> ast::Decl {
        if doc_comment.len() != 0 {
            panic!("We have a doc comment for an import statement, which seems valid but is... kinda weird.")
        }
        let name = self.expect_ident();
        let rename = if self.peek_expect(T::As.discr()) {
            Some(self.expect_ident())
        } else {
            None
        };
        ast::Decl::Import { name, rename }
    }

    /// signature = fn_args [typename]
    fn parse_fn_signature(&mut self) -> Signature {
        let (params, typeparams) = self.parse_fn_args();
        let rettype = self.try_parse_type().unwrap_or(Type::unit());
        let typeparams = Type::detype_names(&typeparams);
        Signature {
            params,
            rettype,
            typeparams,
        }
    }

    /// sig = ident typename
    /// fn_args = "(" ["|" types... "|"] [sig {"," sig} [","]] ["|" types...] ")"
    fn parse_fn_args(&mut self) -> (Vec<(Sym, Type)>, Vec<Type>) {
        let mut args = vec![];
        let mut typeparams = vec![];
        self.expect(T::LParen);
        if self.peek_expect(T::Bar.discr()) {
            parse_delimited!(self, T::Comma, {
                if !self.peek_is(T::Bar.discr()) {
                    let ty = self.parse_type();
                    typeparams.push(ty);
                } else {
                    break;
                }
            });
            self.peek_expect(T::Bar.discr());
        }

        parse_delimited!(self, T::Comma, {
            if self.peek_is(TokenKind::Ident("foo".into()).discr()) {
                let name = self.expect_ident();
                let tname = self.parse_type();
                args.push((name, tname));
            } else {
                break;
            }
        });
        self.expect(T::RParen);
        (args, typeparams)
    }

    /// type_list = "(" [type {"," type} [","] ")"
    fn try_parse_type_list(&mut self) -> Option<Vec<Type>> {
        let mut args = vec![];
        self.expect(T::LParen);

        parse_delimited!(self, T::Comma, {
            if !self.peek_is(T::RParen.discr()) {
                let tname = self.try_parse_type()?;
                args.push(tname);
            } else {
                break;
            }
        });
        self.expect(T::RParen);
        Some(args)
    }

    /// type_list_with_typeparams = "(" [types... "|"] [type {"," type} [","] ")"
    fn parse_type_list_with_typeparams(&mut self) -> (Vec<Type>, Vec<Type>) {
        let mut args = vec![];
        let mut typeparams = vec![];
        self.expect(T::LParen);
        if self.peek_expect(T::Bar.discr()) {
            parse_delimited!(self, T::Comma, {
                if self.peek_is(T::Bar.discr()) {
                    // Bit of a hack, but (||) is a valid sig
                    break;
                } else if !self.peek_is(T::RParen.discr()) {
                    let tname = self.parse_type();
                    typeparams.push(tname);
                } else {
                    break;
                }
            });
            self.expect(T::Bar);
        }

        parse_delimited!(self, T::Comma, {
            if !self.peek_is(T::RParen.discr()) {
                let tname = self.parse_type();
                args.push(tname);
            } else {
                break;
            }
        });
        self.expect(T::RParen);
        (args, typeparams)
    }

    fn parse_fn_type(&mut self) -> Type {
        let (params, typeparams) = self.parse_type_list_with_typeparams();
        let rettype = self.try_parse_type().unwrap_or(Type::unit());
        Type::function(&params, &rettype, &typeparams)
    }

    /// Parse the fields for a struct *type decl*
    fn parse_struct_fields(&mut self) -> (BTreeMap<Sym, Type>, Vec<Type>) {
        let mut fields = BTreeMap::new();
        let mut generics = vec![];
        if self.peek_expect(T::LParen.discr()) {
            trace!("Parsing type params for struct type");
            // parse type params
            parse_delimited!(self, T::Comma, {
                if !self.peek_expect(T::RParen.discr()) {
                    let ty = self.parse_type();
                    generics.push(ty);
                } else {
                    break;
                }
            });
            self.expect(T::RParen);
        }
        self.eat_delimiters();

        // TODO someday: Doc comments on struct fields
        parse_delimited!(self, T::Comma, {
            if self.peek_is_ident() {
                let name = self.expect_ident();
                self.expect(T::Colon);
                let tname = self.parse_type();
                fields.insert(name, tname);
            } else {
                break;
            }
            self.eat_delimiters();
        });
        (fields, generics)
    }

    /// Parse the fields for a struct *type literal*
    fn parse_struct_lit_fields(&mut self) -> BTreeMap<Sym, ast::Expr> {
        let mut fields = BTreeMap::default();

        parse_delimited!(self, T::Comma, {
            if self.peek_expect(T::Period.discr()) {
                let name = self.expect_ident();
                self.expect(T::Equals);
                let vl = self.parse_expr(0).unwrap();
                fields.insert(name, vl);
            } else {
                break;
            }
            self.eat_delimiters();
        });
        fields
    }

    fn parse_enum_fields(&mut self) -> Vec<(Sym, i32)> {
        let mut current_val = 0;
        let mut variants = vec![];

        // TODO someday: Doc comments on enum fields
        parse_delimited!(self, T::Comma, {
            if self.peek_is_ident() {
                let id = self.expect_ident();
                if self.peek_expect(T::Equals.discr()) {
                    current_val = self.expect_int() as i32;
                }
                variants.push((id, current_val));
                current_val += 1;
            } else {
                break;
            }
        });
        // Make sure we don't have any duplicates.
        let mut seen = std::collections::HashMap::new();
        for (name, vl) in variants.iter() {
            if let Some(other) = seen.get(vl) {
                eprintln!(
                    "Duplicate variant in enum: field {} and {} both have value {}",
                    &*name, &*other, *vl
                );
            }
            seen.insert(*vl, *name);
        }
        variants
    }

    fn try_parse_tuple_type(&mut self) -> Option<Type> {
        let mut body = vec![];
        parse_delimited!(self, T::Comma, {
            if !self.peek_is(T::RBrace.discr()) {
                let t = self.try_parse_type()?;
                body.push(t);
            }
        });
        self.expect(T::RBrace);
        Some(Type::tuple(body))
    }

    fn try_parse_struct_type(&mut self) -> Option<Type> {
        let (fields, type_params) = self.parse_struct_fields();
        self.expect(T::End);
        Some(Type::Struct(fields, type_params))
    }

    fn parse_enum_type(&mut self) -> Type {
        let variants = self.parse_enum_fields();
        self.expect(T::End);
        Type::Enum(variants)
    }

    /// isomorphic-ish with parse_type_list()
    fn parse_sum_type(&mut self) -> Type {
        let mut fields = BTreeMap::default();
        let mut generics = vec![];
        if self.peek_expect(T::LParen.discr()) {
            trace!("Parsing type params for sum type");
            // parse type params
            parse_delimited!(self, T::Comma, {
                if !self.peek_expect(T::RParen.discr()) {
                    let ty = self.parse_type();
                    generics.push(ty);
                } else {
                    break;
                }
            });
            self.expect(T::RParen);
            self.eat_delimiters();
        }
        parse_delimited!(self, T::Comma, {
            trace!("Parsing body for sum type");
            if !self.peek_is(T::End.discr()) {
                let field = self.expect_ident();
                let ty = self.parse_type();
                fields.insert(field, ty);
            }
        });
        self.expect(T::End);
        Type::Sum(fields, generics)
    }

    pub fn parse_exprs(&mut self) -> Vec<ast::Expr> {
        let mut exprs = vec![];
        let tok = self.peek();
        while let Some(e) = self.parse_expr(0) {
            // if we have delimiters after an expr we can just eat them
            // But we have no expressions that can *contain* delimiters at the end,
            // so if we see a delimiter it's the end of an expression.
            self.eat_delimiters();
            exprs.push(e);
        }
        // TODO: I think this was necessary for sanity's sake at some point
        // but I don't think it is anymore, figure it out.
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
    /// parser to parse math expressions and such.  It's a big chonky
    /// boi of a function, but really just tries a bunch of matches
    /// in sequence.
    ///
    /// The min_bp is the binding power used in the pratt parser; if
    /// you are calling this standalone the min_bp should be 0.
    pub fn parse_expr(&mut self, min_bp: usize) -> Option<ast::Expr> {
        let t = self.peek()?;
        let token = &t.kind;
        let mut lhs = match token {
            T::Bool(b) => {
                self.drop();
                ast::Expr::bool(*b)
            }
            T::Integer(_) => ast::Expr::int(self.expect_int() as i128),
            T::IntegerSize((_str, size, signed)) => {
                ast::Expr::sized_int(self.expect_int() as i128, *size, *signed)
            }
            // Tuple/struct literal
            T::LBrace => self.parse_constructor(),
            // Array literal
            T::LBracket => self.parse_array_constructor(),
            T::Struct => {
                // TODO: Bikeshed syntax more
                // { .foo = 1, .bar = 2 }
                // ???
                self.parse_struct_literal()
            }
            T::Ident(_) => {
                let ident = self.expect_ident();
                ast::Expr::Var { name: ident }
            }
            T::Let => self.parse_let(),
            T::If => self.parse_if(),
            T::Loop => self.parse_loop(),
            T::While => self.parse_while_loop(),
            T::Do => self.parse_block(),
            T::Fn => self.parse_lambda(),
            T::Return => self.parse_return(),
            T::Break => {
                self.expect(T::Break);
                ast::Expr::Break
            }
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
        // Parse a postfix or infix expression with a given
        // binding power or greater.
        while let Some(op_token) = self.peek().clone() {
            // we only really care about the token kind
            let op_token = op_token.kind;
            // Is the token a postfix op
            if let Some((l_bp, ())) = postfix_binding_power(&op_token) {
                if l_bp < min_bp {
                    break;
                }
                // TODO: Figure out some kind of turbofish for function calls???
                lhs = match op_token {
                    T::LParen => {
                        let (params, typeparams) = self.parse_function_args();
                        ast::Expr::Funcall {
                            func: Box::new(lhs),
                            params,
                            typeparams,
                        }
                    }
                    // If we see `foo {bar}`
                    // then parse it as `foo({bar})`
                    // Thanks Lua!!!
                    T::LBrace => {
                        let params = self.parse_constructor();
                        ast::Expr::Funcall {
                            func: Box::new(lhs),
                            params: vec![params],
                            typeparams: vec![],
                        }
                    }
                    T::LBracket => {
                        self.expect(T::LBracket);
                        let param = self.parse_expr(0)?;
                        self.expect(T::RBracket);
                        ast::Expr::ArrayRef {
                            expr: Box::new(lhs),
                            idx: Box::new(param),
                        }
                    }
                    T::Dollar => {
                        self.expect(T::Dollar);
                        ast::Expr::TypeUnwrap {
                            expr: Box::new(lhs),
                        }
                    }
                    T::Colon => {
                        self.expect(T::Colon);
                        let ident = self.expect_ident();
                        let ident_expr = ast::Expr::Var { name: ident };
                        let (mut params, typeparams) = self.parse_function_args();
                        params.insert(0, lhs);
                        ast::Expr::Funcall {
                            func: Box::new(ident_expr),
                            params,
                            typeparams,
                        }
                    }
                    T::Period => {
                        self.expect(T::Period);
                        // If the period is followed by an int, it's a
                        // tuple ref, otherwise it's a struct ref.
                        let tok = self.next();
                        match tok.as_ref().map(|t| &t.kind) {
                            Some(T::Ident(i)) => ast::Expr::StructRef {
                                expr: Box::new(lhs),
                                elt: Sym::new(i),
                            },
                            // Following Rust, we do not allow numbers
                            // with suffixes as tuple indices.
                            Some(T::Integer(elt)) => {
                                assert!(*elt > -1);
                                ast::Expr::TupleRef {
                                    expr: Box::new(lhs),
                                    elt: *elt as usize,
                                }
                            }
                            _other => self.error("ident or integer", tok),
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

    fn parse_function_args(&mut self) -> (Vec<ast::Expr>, Vec<Type>) {
        let mut params = vec![];
        let mut typeparams = vec![];
        self.expect(T::LParen);
        if self.peek_expect(T::Bar.discr()) {
            parse_delimited!(self, T::Comma, {
                if !self.peek_is(T::RParen.discr()) {
                    let ty = self.parse_type();
                    typeparams.push(ty);
                } else {
                    break;
                }
            });
            self.expect(T::Bar);
        }
        parse_delimited!(self, T::Comma, {
            if let Some(expr) = self.parse_expr(0) {
                params.push(expr);
            } else {
                break;
            }
        });
        self.expect(T::RParen);
        (params, typeparams)
    }

    /// let = "let" ident ":" typename "=" expr
    fn parse_let(&mut self) -> ast::Expr {
        self.expect(T::Let);
        let mutable = if self.peek_expect(T::Mut.discr()) {
            true
        } else {
            false
        };
        let varname = self.expect_ident();
        let typename = if self.peek_expect(T::Equals.discr()) {
            None
        } else {
            let t = Some(self.parse_type());
            self.expect(T::Equals);
            t
        };
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
        while self.peek_expect(T::Elseif.discr()) {
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
            other => self.error("else, elseif block or end", other),
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

    /// while = "while" expr "do" {expr} "end"
    fn parse_while_loop(&mut self) -> ast::Expr {
        self.expect(T::While);
        let cond = self
            .parse_expr(0)
            .expect("While loop condition was not an expression?");
        self.expect(T::Do);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::While {
            cond: Box::new(cond),
            body,
        }
    }

    /// block = "do" {expr} "end"
    fn parse_block(&mut self) -> ast::Expr {
        self.expect(T::Do);
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::Block { body }
    }

    /// lambda = "fn" "(" ...args... ")" [typename] = {exprs} "end"
    fn parse_lambda(&mut self) -> ast::Expr {
        self.expect(T::Fn);
        let signature = self.parse_fn_signature();
        self.expect(T::Equals);
        self.eat_delimiters();
        let body = self.parse_exprs();
        self.expect(T::End);
        ast::Expr::Lambda { signature, body }
    }

    /// tuple constructor = "{" [expr {"," expr} [","] "}"
    /// struct constructor = "{" ["." id "=" expr {"," "." id "=" expr}]  [","] "}"
    fn parse_constructor(&mut self) -> ast::Expr {
        self.expect(T::LBrace);
        if self.peek_is(T::Period.discr()) {
            self.parse_struct_literal()
        } else {
            self.parse_tuple_literal()
        }
    }

    /// struct constructor = "{" "." ident "=" expr {"," ...} "}"
    fn parse_struct_literal(&mut self) -> ast::Expr {
        let body = self.parse_struct_lit_fields();
        self.expect(T::RBrace);
        ast::Expr::StructCtor { body }
    }

    /// tuple constructor = "{" [expr {"," expr} [","] "}"
    fn parse_tuple_literal(&mut self) -> ast::Expr {
        let mut body = vec![];
        parse_delimited!(self, T::Comma, {
            if let Some(expr) = self.parse_expr(0) {
                body.push(expr);
            } else {
                break;
            }
        });
        self.expect(T::RBrace);
        ast::Expr::TupleCtor { body }
    }

    fn parse_array_constructor(&mut self) -> ast::Expr {
        self.expect(T::LBracket);
        let mut body = vec![];
        parse_delimited!(self, T::Comma, {
            if let Some(expr) = self.parse_expr(0) {
                body.push(expr);
            } else {
                break;
            }
        });
        self.expect(T::RBracket);
        ast::Expr::ArrayCtor { body }
    }

    /// Types compose prefix.
    /// So "array(3) of T" is "[3]T"
    fn parse_type(&mut self) -> Type {
        self.try_parse_type().unwrap_or_else(|| {
            let tok = self.peek();
            self.error("type", tok)
        })
    }

    /// If this can't parse a valid type, it will rewind
    /// back to where it started.  Magic!
    /// Also, you know, arbitrary lookahead/backtracking, but that's ok.
    fn try_parse_type(&mut self) -> Option<Type> {
        let old_lexer = self.lex.clone();
        let t = self.next()?;
        let x = match t.kind {
            T::Ident(ref s) => {
                if let Some(t) = Type::get_primitive_type(s) {
                    t
                } else {
                    let type_params = if self.peek_is(T::LParen.discr()) {
                        self.try_parse_type_list()?
                    } else {
                        vec![]
                    };
                    Type::Named(Sym::new(s), type_params)
                }
            }
            /*
                    T::At => {
                        let s = self.expect_ident();
                        Type::Generic(s)
                    }
            */
            T::LBrace => self.try_parse_tuple_type()?,
            T::Fn => self.parse_fn_type(),
            T::Struct => self.try_parse_struct_type()?,
            T::Enum => self.parse_enum_type(),
            T::Sum => self.parse_sum_type(),
            T::LBracket => {
                let len = self.expect_int();
                assert!(len >= 0);
                self.expect(T::RBracket);
                let inner = self.parse_type();
                Type::Array(Box::new(inner), len as usize)
            }
            T::Ampersand => {
                let next = self.try_parse_type()?;
                Type::Uniq(Box::new(next))
            }
            _ => {
                // Wind the parse stream back to wherever we started
                // TODO LATER: We should maybe think about making a better way
                // of doing this, but so far this is the only place it happens.
                self.lex = old_lexer;
                return None;
            }
        };
        Some(x)
    }
}

/// Specifies binding power of prefix operators.
///
/// Panics on invalid token, which should never happen
/// since we always know what kind of expression we're parsing
/// from the get-go with prefix operators.
///
/// Binding power of prefix operators < infix operators < postfix operators.
/// That way `-x + y[3]` parses to `-(x + (y[3]))`
fn prefix_binding_power(op: &TokenKind) -> ((), usize) {
    match op {
        T::Plus | T::Minus | T::Not => ((), 10),
        x => unreachable!("{:?} is not a prefix op, should never happen!", x),
    }
}

/// Specifies binding power of postfix operators.
/// All postfix operators can have the same binding power 'cause
/// you can't really intermix them, but we
/// need a real value for them 'cause they can intermix
/// with prefix and infix operators.
fn postfix_binding_power(op: &TokenKind) -> Option<(usize, ())> {
    match op {
        // "." for tuple/struct references.
        T::Period |
        // "$" type unwrap operator
        T::Dollar |
        // "(" opening function call args
        T::LParen |
        // "{" opening single-struct function call args
        T::LBrace |
        // ":" universal function call syntax
        T::Colon |
        // "[" array index
        T::LBracket |
        // "^" for pointer derefs.  
        T::Carat |
        // "&" for pointer refs.  
        T::Ampersand => Some((300, ())),
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
        T::Mul | T::Div | T::Mod => Some((200, 201)),
        T::Plus | T::Minus => Some((190, 191)),
        T::Lt | T::Gt | T::Lte | T::Gte => Some((180, 181)),
        T::Equal | T::NotEqual => Some((170, 171)),
        T::And => Some((160, 161)),
        // Logical xor has same precedence as or, I guess?  It's sorta an odd duck.
        T::Or | T::Xor => Some((150, 151)),
        // Assignment
        T::Equals => Some((110, 111)),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{self, Expr};
    use crate::parser::*;

    /// Take a list of strings and try parsing them with the given function.
    /// Is ok iff the parsing succeeds, does no checking that the produced
    /// AST is actually something that you want, or anything at all.
    fn test_parse_with<T>(f: impl Fn(&mut Parser) -> T, strs: &[&str]) {
        for s in strs {
            let mut p = Parser::new("unittest.gt", s);
            f(&mut p);
            // Make sure we've parsed the whole string.
            assert_eq!(p.peek(), None);
        }
    }

    /// Take a list of strings, parse them, make sure they match
    /// the given ast.  The function gets passed a cx so it can
    /// intern strings for identifiers.
    ///
    /// For now it's just for expr's, since that's most of the language.
    fn test_expr_is(s: &str, f: impl Fn() -> Expr) {
        let ast = f();
        let mut p = Parser::new("unittest.gt", s);
        let parsed_expr = p.parse_expr(0).unwrap();
        assert_eq!(&ast, &parsed_expr);
        // Make sure we've parsed the whole string.
        assert_eq!(p.peek(), None);
    }

    /// Same as test_expr_is but with decl's
    fn test_decl_is(s: &str, f: impl Fn() -> ast::Decl) {
        let ast = f();
        let mut p = Parser::new("unittest.gt", s);
        let parsed_decl = p.parse_decl().unwrap();
        assert_eq!(&ast, &parsed_decl);
        // Make sure we've parsed the whole string.
        assert_eq!(p.peek(), None);
    }

    /// And again with types
    fn test_type_is(s: &str, f: impl Fn() -> Type) {
        let ast = f();
        let mut p = Parser::new("unittest.gt", s);
        let parsed_type = p.parse_type();
        assert_eq!(&ast, &parsed_type);
        // Make sure we've parsed the whole string.
        assert_eq!(p.peek(), None);
    }

    #[test]
    fn test_const() {
        test_decl_is("const foo I32 = -9", || ast::Decl::Const {
            name: Sym::new("foo"),
            typename: Type::i32(),
            init: Expr::UniOp {
                op: ast::UOp::Neg,
                rhs: Box::new(Expr::int(9)),
            },
            doc_comment: vec![],
        });
    }

    #[test]
    fn test_fn() {
        test_decl_is("fn foo(x I32) I32 = 9 end", || {
            let i32_t = Type::i32();
            ast::Decl::Function {
                name: Sym::new("foo"),
                signature: Signature {
                    params: vec![(Sym::new("x"), i32_t.clone())],
                    rettype: i32_t,
                    typeparams: vec![],
                },
                body: vec![Expr::int(9)],
                doc_comment: vec![],
            }
        });
    }

    #[test]
    fn test_typedef() {
        test_decl_is("type bop = I32", || ast::Decl::TypeDef {
            name: Sym::new("bop"),
            typedecl: Type::i32(),
            doc_comment: vec![],
            params: vec![],
        });
    }

    #[test]
    fn test_typedef_generics() {
        test_decl_is("type bop(T) = T", || ast::Decl::TypeDef {
            name: Sym::new("bop"),
            typedecl: Type::Named(Sym::new("T"), vec![]),
            doc_comment: vec![],
            params: vec![Sym::new("T")],
        });
    }

    #[test]
    fn test_multiple_decls() {
        let s = r#"
const foo  I32 = -9
const bar  Bool = 4
--- rawr!
const baz  {} = {}
type blar = I8
"#;
        let p = &mut Parser::new("unittest.gt", s);
        let foosym = Sym::new("foo");
        let barsym = Sym::new("bar");
        let bazsym = Sym::new("baz");
        let blarsym = Sym::new("blar");
        let i32_t = Type::i32();
        let i8_t = Type::i8();
        let bool_t = Type::bool();
        let unit_t = Type::unit();
        let d = p.parse();
        assert_eq!(
            d,
            ast::Ast {
                filename: String::from("unittest.gt"),
                modulename: String::from("unittest"),
                module_docstring: String::new(),
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
                        params: vec![],
                    }
                ],
            }
        );
    }
    #[test]
    fn parse_fn_args() {
        let valid_args = vec![
            "()",
            "(x  Bool)",
            "(x  Bool,)",
            "(x  I32, y  Bool)",
            "(x X, y Y)",
            "(x  I32, y  Bool,)",
        ];
        test_parse_with(|p| p.parse_fn_args(), &valid_args)
    }
    #[test]
    fn parse_fn_signature() {
        let valid_args = vec![
            "() {}",
            "(x Bool) I32",
            "(x Bool) {}",
            "(x I16, y Bool) {}",
            "(x I64, y Bool) Bool",
            "(x I8, y Bool,) {}",
            "(x I32, y Bool,) Bool",
            "(f fn(I32) I64, x I32) Bool",
            "(f fn(|| I32) I64, x I32) Bool",
            "(f fn(||) I64, x I32) Bool",
            "(f fn(|T| I32) I64, x I32) Bool",
            // now without explicit return types
            "()",
            "(x Bool)",
            "(x Bool)",
            "(x I16, y Bool)",
            "(x I64, y Bool)",
            "(x I8, y Bool,)",
            "(x I32, y Bool,)",
            "(f fn(I32), x I32)",
            "(f fn(|| I32), x I32)",
            "(f fn(||), x I32)",
            "(f fn(|T| I32), x I32)",
        ];
        test_parse_with(|p| p.parse_fn_signature(), &valid_args)
    }
    #[test]
    fn parse_let() {
        let valid_args = vec!["let x I32 = 5", "let y Bool = false", "let z {} = z"];
        // The lifetimes and inference here gets WEIRD if you try to pass it Parser::parse_let.
        test_parse_with(|p| p.parse_let(), &valid_args);
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_let_type_inferred() {
        let valid_args = vec!["let x = 5", "let y = false", "let z = [1, 2, 3]"];
        // The lifetimes and inference here gets WEIRD if you try to pass it Parser::parse_let.
        test_parse_with(|p| p.parse_let(), &valid_args);
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_if() {
        let valid_args = vec![
            "if x then y end",
            "if 10 then let x  Bool = false 10 end",
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
        let valid_args = vec!["loop 10 end", "loop 10 20 30 end", "loop {} end"];
        test_parse_with(|p| p.parse_loop(), &valid_args);
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_block() {
        let valid_args = vec!["do 10; end", "do 10; 20; 30; end", "do {}; end"];
        test_parse_with(|p| p.parse_block(), &valid_args);
        test_parse_with(|p| p.parse_expr(0), &valid_args);
    }

    #[test]
    fn parse_lambda() {
        let valid_args = vec![
            "fn(x I32) I32 = x end",
            "fn(x I32, i Bool) I32 = x end",
            "fn(f fn(I32) I32, x I32) {} = f(x) end",
            "fn() {} = {} end",
            // for parse_expr there must be no leading newlines
            // but there can be trailing ones.
            r#"fn() {} = 
    {} 
end
"#,
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
    fn parse_fn_decls() {
        let valid_args = vec![
            "fn foo1(f I32) I32 = f end",
            "fn foo2(|T| f I32 ) I32 = f end",
            "fn foo3(|T|) {} = f end",
            "fn foo4(||) {} = f end",
            "fn foo5() {} = f end",
            "fn foo6(f T) T = f end",
            "fn foo7(|T1, T2, | f I32, g Bool, ) I32 = f end",
        ];
        test_parse_with(|p| p.parse_decl().unwrap(), &valid_args);
    }

    #[test]
    fn parse_fn_lambda() {
        let valid_args = vec!["fn apply_one(f fn(I32)I32, x I32) I32 = f(x) end"];
        test_parse_with(|p| p.parse_decl().unwrap(), &valid_args);
    }

    #[test]
    fn parse_funcall() {
        test_expr_is("y(1, 2, 3)", || Expr::Funcall {
            func: Box::new(Expr::Var {
                name: Sym::new("y"),
            }),
            params: vec![Expr::int(1), Expr::int(2), Expr::int(3)],
            typeparams: vec![],
        });

        test_expr_is("foo(0, bar(1 * 2), 3)", || Expr::Funcall {
            func: Box::new(Expr::Var {
                name: Sym::new("foo"),
            }),
            params: vec![
                Expr::int(0),
                Expr::Funcall {
                    func: Box::new(Expr::Var {
                        name: Sym::new("bar"),
                    }),
                    params: vec![Expr::BinOp {
                        op: ast::BOp::Mul,
                        lhs: Box::new(Expr::int(1)),
                        rhs: Box::new(Expr::int(2)),
                    }],
                    typeparams: vec![],
                },
                Expr::int(3),
            ],
            typeparams: vec![],
        });

        test_expr_is("(1)", || Expr::int(1));
        test_expr_is("(((1)))", || Expr::int(1));

        test_expr_is("y(|I32| 1)", || Expr::Funcall {
            func: Box::new(Expr::Var {
                name: Sym::new("y"),
            }),
            params: vec![Expr::int(1)],
            typeparams: vec![Type::i32()],
        });
    }

    #[test]
    fn verify_elseif() {
        use Expr;
        test_expr_is(
            r#"if x then
                1
            elseif y then
                2
            else
                3
            end"#,
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
            r#"if x then
                1
            else
                if y then
                    2
                else
                    3
                end
            end"#,
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
                name: Sym::new("x"),
            }),
            params: vec![],
            typeparams: vec![],
        });
        test_expr_is("(x())", || Expr::Funcall {
            func: Box::new(Expr::Var {
                name: Sym::new("x"),
            }),
            params: vec![],
            typeparams: vec![],
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
        ];
        for (s, expected_int, expected_bytes) in tests {
            let mut p = Parser::new("unittest.gt", s);
            assert_eq!(
                p.next().unwrap().kind,
                TokenKind::IntegerSize((*expected_int, *expected_bytes, true))
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
            let mut p = Parser::new("unittest.gt", s);
            assert_eq!(p.next().unwrap().kind, TokenKind::Error);
            assert!(p.next().is_none());
        }
    }

    #[test]
    fn parse_integer_values() {
        test_expr_is("43_I8", || Expr::sized_int(43, 1, true));
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
    fn parse_array_constructor() {
        test_expr_is("[4, 3, 2]", || Expr::ArrayCtor {
            body: vec![Expr::int(4), Expr::int(3), Expr::int(2)],
        });

        // Nested arrays now
        test_expr_is("[[4, 3, 2], [4, 3, 2], [4, 3, 2]]", || {
            let arr = Expr::ArrayCtor {
                body: vec![Expr::int(4), Expr::int(3), Expr::int(2)],
            };
            Expr::ArrayCtor {
                body: vec![arr.clone(), arr.clone(), arr.clone()],
            }
        });
    }

    #[test]
    fn parse_weird_nested_array_bug() {
        test_expr_is(
            "let x [3][3]I32 = [[4, 3, 2], [4, 3, 2], [4, 3, 2]]",
            || {
                let arr = Expr::ArrayCtor {
                    body: vec![Expr::int(4), Expr::int(3), Expr::int(2)],
                };
                let ty = Type::array(&Type::array(&Type::i32(), 3), 3);
                Expr::Let {
                    varname: Sym::new("x"),
                    typename: Some(ty),
                    mutable: false,
                    init: Box::new(Expr::ArrayCtor {
                        body: vec![arr.clone(), arr.clone(), arr.clone()],
                    }),
                }
            },
        );
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
    fn parse_array_types() {
        test_type_is("[4]I32", || Type::array(&Type::i32(), 4));
        test_type_is("[6][4]I32", || {
            Type::array(&Type::array(&Type::i32(), 4), 6)
        });
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

    #[test]
    fn parse_package_doc_comment() {
        let thing1 = r#"--- package with doc comment

--- doc comment for function
fn foo() {} = {} end
"#;

        let mut p = Parser::new("unittest.gt", thing1);
        let res = p.parse();
        assert_eq!(&res.module_docstring, " package with doc comment\n");

        let thing2 = r#"
            
--- package with no doc comment, this is the
--- doc comment for function
fn foo() {} = {} end
"#;

        let mut p = Parser::new("unittest.gt", thing2);
        let res = p.parse();
        assert_eq!(&res.module_docstring, "");
    }

    /// This tests a kinda horrible edge case in mixing line and block comments,
    /// but it's a rare enough one that I don't care about it right now.
    #[test]
    #[should_panic]
    fn parse_evil_nested_comment() {
        let thing1 = r#"

/- Block comments work fine
-/

/- Block comments work fine
/- And nested block comments work fine
-/
-/

-- Line comments work fine with a -/ in them
-- Line comments work fine with a /- in them
-- and no closing delimiter ever

/-
-- But if a line comment is commented out by a block comment and contains a 
-- surprising end delimiter like "-/" then the block comment is closed

"#;

        let mut p = Parser::new("unittest.gt", thing1);
        let _res = p.parse();
    }

    #[test]
    fn parse_big_chars_in_comments() {
        let thing1 = r#"

/- Block comments work fine with long unicode characters in them:
 霹靂
-/

/- Block comments work fine
/- And nested block comments work fine
-/
-/


"#;

        let mut p = Parser::new("unittest.gt", thing1);
        let _res = p.parse();
    }
}
