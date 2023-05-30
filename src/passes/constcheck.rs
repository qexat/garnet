//! Ok this is the skeleton of a Pass that will mark expressions/decls
//! as const, basically inferring them down the chain.
//! Shoooooould be pretty easy: some terminal expressions such as
//! assignment, loops, and such are non const, and if something contains
//! any non-const expressions it is not const.  We start off by assuming
//! nothing is const and annotating it downward from leaves to roots.
//!
//! We might need to do things like make sure we mark main() and
//! println functions as non-const somehow, we will see.
//!
//! Is this a Pass or a TckPass?  We need to be able to do name lookups
//! to see whether a function we are calling is const or not... hmmm.
