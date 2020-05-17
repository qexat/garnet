//! Intermediate representation, basically what you get after
//! all the typechecking and other kinds of checking happen.
//! Slightly simplified over the AST -- basically the result of
//! a lowering pass.  This might give us a nice place to do
//! simple things like constant folding, dead code detection, etc.
//!
//! Is always a valid program.
