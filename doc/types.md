I've been reading about 1ML so let's try to formalize the type system in
some way or another.

# Design guidelines

 * Needs to be quick to compile -- so no undecidable things
 * I don't want this to be a research language, so keep Shiny New Ideas
   to a minimum
 * Needs to stay simple, so highly orthogonal ideas are good.

# Specific goals

Nominal types instead of structural.  Because I like it when types are
real, instead of just pretending they're real.

I want to be able to have modules and structs be basically
interchangable, 'cause...  modules are just compile-time singleton
structs.  Zig does something like this.

I'm not sure I want to have explicit compile-time evaluation, though it
might end up being there anyway.

I also want nice enums for numerical values.

I was thinking of NOT having arbitrary-sized numbers (ie, `I8 =
Number(-128,127)`), buuuuuuut....  might be a useful special case.

I do *not* want dependent types, because cool as they are, it seems like
they add a lot of complexity.  A limited form of dependent types for
variable-length arrays may be acceptable.

Don't be too scared of doing runtime checks for bounded numbers and
such, and trusting the backend to optimize them out.

I *really* want to be able to be polymorphic over owned types, shared
borrows, and unique borrows.  Unfortunately nobody seems to know how the
hell to do this.

