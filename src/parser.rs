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
