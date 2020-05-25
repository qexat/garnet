/*

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
// TODO: 'lambda' is long to type.
lambda = "lambda" fn_args "->" {expr} "end"

fn_args = [ident ":" typename {"," ident ":" typename}]

typename =
  | "i32"
  | "bool"
  // Tuples with curly braces like Erlang seem less ambiguous than the more traditional parens...
  // I hope that will let us get away without semicolons.
  | "{" [typename {"," typename}] "}"

// Things to add, roughly in order
// while/for loops
// arrays/slices
// enums
// structs (anonymous and otherwise?)
// generics
// references

*/
