//! Intermediate representation, basically what you get after all the typechecking and other kinds
//! of checking happen.  Slightly simplified over the AST -- basically the result of a
//! lowering/desugaring pass.  This might give us a nice place to do other simple
//! lowering/optimization-like things like constant folding, dead code detection, etc.
//!
//! An IR is always assumed to be a valid program, since it has passed all the checking stuff.
//!
//! So... do we want it to be an expression tree like AST is, or SSA form, or a control-
//! flow graph, or what?  Ponder this, since different things are easy to do on different
//! representations. ...well, the FIRST thing we need to do is type checking and related
//! junk anyway, so, this should just be a slightly-lowered syntax tree.
