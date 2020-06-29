# ABI notes

Part of the goal of Garnet is to have a defined ABI so that a) separate
compilation is more possible, and b) other languages can interface with
it more easily.

This is, naturally, divvied up by platform, and currently there's only
one platform, so that makes life easy.

# WASM32

## Function params and returns

All function call values are passed on the stack.  Tuples are pushed
onto the stack in order of declaration, so `{I32, Bool}` would have an
`I32` pushed onto the stack, followed by a `Bool`, resulting in the
`Bool` on top of the stack.  Tuples can't be
returned on the stack though, for Reasons, TODO: dig up the extension
thread that talk about it.

Function values are passed as a single `I32` representing the function
offset in the table.  We do not have closures yet.

Thing is, we can't have actual pointers into the wasm stack because we
can't pick arbitrary elements out of it, and can't reach down past our
own activation record anyway (as far as I know).  So we need to make an
activation record stack to hold local values and stuff in memory, and
pass pointers into it.  Particularly, return values bigger than a single
atom will have to be passed by the calling function passing a pointer to
an appropriate slot in the activation record.  This is also something we
will need for particularly large structs and such anyway.

We will term this "spilling", since it's very analogous to spilling
values from registers onto a CPU stack.

We have to put the stack at the beginning of WASM memory and have it
grow upwards, since that's the only place we can guarentee it will be.
It must then be a fixed maximum size, and the heap must be above it and
also grow upwards.  This is going to get awful quick if we have threads,
but for now there's not much else we can do about it.

## Data layouts

Future thoughts: A struct and tuple declared similarly are guarenteed to
have the same representation.

## Modules and imports

TBD

# Other thoughts

## Pointers

Reading from a pointer that is uninitialized and/or is not pointing at
valid memory is guarenteed to either return something unknowable, or
produce a fault in the program.  (Let's define "fault" to mean an
uncontrolled error where the OS does something cruel and arbitrary.)

Writing to a pointer not pointing at valid memory is an Out Of Context
Problem.  It may nuke state that the rest of the program assumes is
consistent, and so have results that can't be predicted.

The compiler may or may not be able to detect that a read or write to a
pointer is invalid, but if it does, it MUST be an error.  Not a fucking
warning.

The thing is, sometimes reading or writing to a pointer has side effects
*beyond* locally observable memory.  Memory-mapped I/O is the main
thing, though you could also consider memory shared by threads to be in
kinda the same boat  if you wanted to try hard enough.  (I don't.)  So
we must have a *separate* pointer type that does not have quite these
same invariants, that points at memory-mapped I/O addresses.  Normal
pointers are not allowed to point to mmap I/O then.  Otherwise it's
like... I dunno, a double-Out Of Context Problem.  Think of something to
call it besides "volatile", please.  Just "IO pointer" is probably fine.

So we now have enough combinations to be awful, so think harder about
combinators.  We have const/mut, normal/io, and pointer/reference.  Hmm.
