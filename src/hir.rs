//! Intermediate representation, basically what you get after immediate parsing and where
//! all the typechecking and other kinds of checking happen.  Slightly simplified over the AST --
//! basically the result of a lowering/desugaring pass.  This might give us a nice place to do
//! other simple optimization-like things like constant folding, dead code detection, etc.
//!
//! Currently it's not really lowered by much!  If's and maybe loops or something.
//! It's mostly a layer of indirection for further stuff to happen to, so we can change
//! the parser without changing the typecheckin and such.

use std::sync::RwLock;

use crate::ast::{self};
pub use crate::ast::{BOp, IfCase, Literal, Signature, UOp};
use crate::*;

/// Expression ID.  Used for associating individual expressions to types, and whatever
/// other metadata we feel like.  That way we can keep that data in tables instead of
/// having to mutate the HIR tree.
/// The only guarantees the number offers is it will be unique.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Eid(usize);

/// Ugly horrible global for storing index values for expression ID's
/// TODO: Someday make this an atomic or something other than a rwlock
static NEXT_EID: Lazy<RwLock<Eid>> = Lazy::new(|| RwLock::new(Eid(0)));

impl Eid {
    fn new() -> Self {
        let mut current = NEXT_EID.write().unwrap();
        let ret = *current;
        let next_eid = Eid(current.0 + 1);
        *current = next_eid;
        ret
    }
}

/// An expression with a type annotation.
/// Currently will be () for something that hasn't been
/// typechecked, or a `TypeSym` with the appropriate type
/// after type checking has been done.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr {
    /// expression
    pub e: Expr,
    /// Expression ID
    pub id: Eid,
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit {
        val: Literal,
    },
    EnumLit {
        val: VarSym,
        ty: TypeSym,
    },
    Var {
        name: VarSym,
    },
    BinOp {
        op: BOp,
        lhs: Box<TypedExpr>,
        rhs: Box<TypedExpr>,
    },
    UniOp {
        op: UOp,
        rhs: Box<TypedExpr>,
    },
    Block {
        body: Vec<TypedExpr>,
    },
    Let {
        varname: VarSym,
        typename: TypeSym,
        init: Box<TypedExpr>,
        mutable: bool,
    },
    If {
        cases: Vec<(TypedExpr, Vec<TypedExpr>)>,
    },
    Loop {
        body: Vec<TypedExpr>,
    },
    Lambda {
        signature: Signature,
        body: Vec<TypedExpr>,
    },
    Funcall {
        func: Box<TypedExpr>,
        params: Vec<TypedExpr>,
    },
    Break,
    Return {
        retval: Box<TypedExpr>,
    },
    StructCtor {
        body: Vec<(VarSym, TypedExpr)>,
    },
    StructRef {
        expr: Box<TypedExpr>,
        elt: VarSym,
    },
    Assign {
        lhs: Box<TypedExpr>,
        rhs: Box<TypedExpr>,
    },
    Deref {
        expr: Box<TypedExpr>,
    },
    Ref {
        expr: Box<TypedExpr>,
    },
}

impl Expr {
    /// Shortcut function for making literal bools
    pub const fn bool(b: bool) -> Self {
        Self::Lit {
            val: Literal::Bool(b),
        }
    }

    /// Shortcut function for making literal integers
    pub const fn int(i: i128) -> Self {
        Self::Lit {
            val: Literal::Integer(i),
        }
    }

    /// Shortcut function for making literal unit
    pub const fn unit() -> Self {
        Self::StructCtor { body: vec![] }
    }
}

/// A top-level declaration in the source file.
/// Like TypedExpr, contains a type annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Function {
        name: VarSym,
        signature: Signature,
        body: Vec<TypedExpr>,
    },
    Const {
        name: VarSym,
        typename: TypeSym,
        init: TypedExpr,
    },
    TypeDef {
        name: VarSym,
        typedecl: TypeSym,
    },
    /// Our first compiler intrinsic!  \o/
    ///
    /// This is a function that is generated along with
    /// each typedef, which takes the args of that typedef
    /// and produces that type.  We have no way to write such
    /// a function by hand, so here we are.  In all other ways
    /// though we treat it exactly like a function though.
    Constructor {
        name: VarSym,
        signature: Signature,
    },
}

/// A compilable chunk of IR.
///
/// Currently, basically a compilation unit.
///
/// The T is a type annotation of some kind, but we really don't care what.
#[derive(Debug, Clone, Default)]
pub struct Ir {
    pub decls: Vec<Decl>,
}

/// Transforms AST into IR
///
/// The function `f` is a function that should generate whatever value we need
/// for our type info attached to the HIR node.  To start with it's a unit, 'cause
/// we have no type information.
pub fn lower(ast: &ast::Ast) -> Ir {
    lower_decls(&ast.decls)
}

fn lower_lit(lit: &ast::Literal) -> Literal {
    lit.clone()
}

fn lower_bop(bop: &ast::BOp) -> BOp {
    *bop
}

fn lower_uop(uop: &ast::UOp) -> UOp {
    *uop
}

fn lower_signature(sig: &ast::Signature) -> Signature {
    sig.clone()
}

/// This is the biggie currently
fn lower_expr(expr: &ast::Expr) -> TypedExpr {
    use ast::Expr as E;
    use Expr::*;
    let new_exp = match expr {
        E::Lit { val } => Lit {
            val: lower_lit(val),
        },
        E::Var { name } => Var { name: *name },
        E::BinOp { op, lhs, rhs } => {
            let nop = lower_bop(op);
            let nlhs = lower_expr(lhs);
            let nrhs = lower_expr(rhs);
            BinOp {
                op: nop,
                lhs: Box::new(nlhs),
                rhs: Box::new(nrhs),
            }
        }
        E::UniOp { op, rhs } => {
            let nop = lower_uop(op);
            let nrhs = lower_expr(rhs);
            UniOp {
                op: nop,
                rhs: Box::new(nrhs),
            }
        }
        E::Block { body } => {
            let nbody = body.iter().map(|e| lower_expr(e)).collect();
            Block { body: nbody }
        }
        E::Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            let ninit = Box::new(lower_expr(init));
            Let {
                varname: *varname,
                typename: *typename,
                init: ninit,
                mutable: *mutable,
            }
        }
        E::If { cases, falseblock } => {
            assert!(!cases.is_empty(), "Should never happen");
            let mut cases: Vec<_> = cases
                .iter()
                .map(|case| (lower_expr(&*case.condition), lower_exprs(&case.body)))
                .collect();
            // Add the "else" case, which we can just make `if true then...`
            // No idea if this is a good idea, but it makes life easier right
            // this instant, so.  Hasn't bit me yet, so it's not a *bad* idea.
            let else_case = E::Lit {
                val: ast::Literal::Bool(true),
            };
            let nelse_case = lower_expr(&else_case);
            // Empty false block becomes a false block that returns unit
            let false_exprs = if falseblock.len() == 0 {
                lower_exprs(&vec![ast::Expr::TupleCtor { body: vec![] }])
            } else {
                lower_exprs(falseblock)
            };
            cases.push((nelse_case, false_exprs));
            If { cases }
        }
        E::Loop { body } => {
            let nbody = lower_exprs(body);
            Loop { body: nbody }
        }
        E::Lambda { signature, body } => {
            let nsig = lower_signature(signature);
            let nbody = lower_exprs(body);
            Lambda {
                signature: nsig,
                body: nbody,
            }
        }
        E::Funcall { func, params } => {
            let nfunc = Box::new(lower_expr(func));
            let nparams = lower_exprs(params);
            Funcall {
                func: nfunc,
                params: nparams,
            }
        }
        E::Break => Break,
        E::Return { retval: e } => Return {
            retval: Box::new(lower_expr(e)),
        },
        // Turn {foo, bar} into {_0: foo, _1: bar}
        E::TupleCtor { body } => {
            let body_pairs = body
                .into_iter()
                .enumerate()
                .map(|(i, vl)| (INT.intern(format!("_{}", i)), lower_expr(vl)))
                .collect();
            Expr::StructCtor { body: body_pairs }
        }
        E::StructCtor {
            body,
            types: _types,
        } => {
            let lowered_body = body
                .iter()
                .map(|(nm, expr)| (*nm, lower_expr(expr)))
                .collect();
            Expr::StructCtor { body: lowered_body }
        }
        // Turn tuple.0 into struct._0
        E::TupleRef { expr, elt } => Expr::StructRef {
            expr: Box::new(lower_expr(expr)),
            elt: INT.intern(format!("_{}", elt)),
        },
        E::StructRef { expr, elt } => Expr::StructRef {
            expr: Box::new(lower_expr(expr)),
            elt: *elt,
        },
        E::Deref { expr } => Expr::Deref {
            expr: Box::new(lower_expr(expr)),
        },
        E::Ref { expr } => Expr::Ref {
            expr: Box::new(lower_expr(expr)),
        },
        E::Assign { lhs, rhs } => Expr::Assign {
            lhs: Box::new(lower_expr(lhs)),
            rhs: Box::new(lower_expr(rhs)),
        },
    };
    TypedExpr {
        e: new_exp,
        id: Eid::new(),
    }
}

/// handy shortcut to lower Vec<ast::Expr>
fn lower_exprs(exprs: &[ast::Expr]) -> Vec<TypedExpr> {
    exprs.iter().map(|e| lower_expr(e)).collect()
}

/// Lower an AST decl to IR.
///
/// Is there a more elegant way of doing this than passing an accumulator?
/// Returning a vec is lame.  Return an iterator?  Sounds like work.
fn lower_decl(accm: &mut Vec<Decl>, decl: &ast::Decl) {
    use ast::Decl as D;
    match decl {
        D::Function {
            name,
            signature,
            body,
            ..
        } => accm.push(Decl::Function {
            name: *name,
            signature: lower_signature(signature),
            body: lower_exprs(body),
        }),
        D::Const {
            name,
            typename,
            init,
            ..
        } => accm.push(Decl::Const {
            name: *name,
            typename: *typename,
            init: lower_expr(init),
        }),
        // this needs to generate the typedef AND the type constructor
        // declaration.  Plus the type deconstructor.
        D::TypeDef {
            name: _name,
            typedecl: _typedecl,
            ..
        } => {
            todo!("named types");
            /*

            accm.push(Decl::TypeDef {
                name: *name,
                typedecl: *typedecl,
            });
            let paramname = INT.intern("input");
            let rtype = INT.intern_type(&TypeDef::Named(*name));
            accm.push(Decl::Constructor {
                name: *name,
                signature: hir::Signature {
                    params: vec![(paramname, *typedecl)],
                    rettype: rtype,
                },
            });
            // If we have an enum, generate a const struct value containing its
            // values.
            let def = &*INT.fetch_type(*typedecl);
            match def {
                TypeDef::Enum { variants } => {

                    let mut struct_fields = BTreeMap::new();
                    let mut constructor_fields = vec![];
                    for (var, _val) in variants {
                        //struct_fields.insert(*var, *typedecl);
                        let named_type = INT.intern_type(&TypeDef::Named(*name));
                        struct_fields.insert(*var, named_type);
                        let e = Expr::EnumLit {
                            ty: named_type,
                            val: *var,
                        };
                        let t = f(&e);
                        let te = TypedExpr {
                            e,
                            t,
                            s: ISymtbl::default(),
                        };
                        constructor_fields.push((*var, te));
                    }
                    // Now we create a const struct containing all the fields
                    let struct_name = "Rawr";
                    let e = Expr::StructCtor {
                        body: constructor_fields,
                        types: struct_fields.clone(),
                    };
                    let struct_type = TypeDef::Struct {
                        fields: struct_fields,
                        typefields: BTreeSet::default(),
                    };
                    let t = f(&e);
                    accm.push(Decl::Const {
                        name: INT.intern(struct_name),
                        typename: INT.intern_type(&struct_type),
                        init: TypedExpr {
                            e,
                            t,
                            s: ISymtbl::default(),
                        },
                    });
                }
                _ => (),
            }
            */
        }
    }
}

fn lower_decls(decls: &[ast::Decl]) -> Ir {
    let mut accm = Vec::with_capacity(decls.len() * 2);
    for d in decls.iter() {
        lower_decl(&mut accm, d)
    }
    Ir { decls: accm }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr as A;
    use crate::hir::Expr as I;
    use crate::testutil::*;

    /// HACK to get around the Eid's not matching :|
    fn test_eq(e1: &TypedExpr, e2: &TypedExpr) {
        let mut e1 = e1.clone();
        e1.id = e1.id;
        assert_eq!(&e1, e2);
    }

    /*
    /// Shortcut to take an Expr and wrap it
    /// in a TypedExpr with a unit type.
    ///
    /// TODO: Better name?
    #[cfg(test)]
    pub(crate) fn plz(e: Expr) -> Box<TypedExpr> {
        Box::new(TypedExpr { e, id: Eid::new() })
    }

        /// Does `return;` turn into `return ();`?
        /// Not doing that to make parsing simpler, since we don't
        /// actually have semicolons.
        #[test]
        fn test_return_none() {
            let input = A::Return { retval: None };
            let output = *plz(I::Return {
                retval: plz(I::unit()),
            });
            let res = lower_expr(&input);
            assert_eq!(&res, &output);
        }
        */
    /* TODO: Unfuck these tests, Eid doesn't compare correctly :|

    /// Do we turn chained if's properly into nested ones?
    /// Single if.
    #[test]
    fn test_if_lowering_single() {
        let input = A::If {
            cases: vec![ast::IfCase {
                condition: Box::new(A::bool(false)),
                body: vec![A::int(1)],
            }],
            falseblock: vec![A::int(2)],
        };
        let output = *plz(I::If {
            cases: vec![
                (*plz(I::bool(false)), vec![*plz(I::int(1))]),
                (*plz(I::bool(true)), vec![*plz(I::int(2))]),
            ],
        });
        let res = lower_expr(&mut rly, &input);
        test_eq(&res, &output);
    }

    /// Do we turn chained if's properly into nested ones?
    /// Chained if/else's
    #[test]
    fn test_if_lowering_chained() {
        let input = A::If {
            cases: vec![
                ast::IfCase {
                    condition: Box::new(A::bool(false)),
                    body: vec![A::int(1)],
                },
                ast::IfCase {
                    condition: Box::new(A::bool(true)),
                    body: vec![A::int(2)],
                },
            ],
            falseblock: vec![A::int(3)],
        };
        let output = *plz(I::If {
            cases: vec![
                (*plz(I::bool(false)), vec![*plz(I::int(1))]),
                (*plz(I::bool(true)), vec![*plz(I::int(2))]),
                (*plz(I::bool(true)), vec![*plz(I::int(3))]),
            ],
        });
        let res = lower_expr(&mut rly, &input);
        test_eq(&res, &output);
    }
    */

    /// Do we panic if we get an impossible if with no cases?
    #[test]
    #[should_panic]
    fn test_if_nothing() {
        let input = A::If {
            cases: vec![],
            falseblock: vec![],
        };
        let _ = lower_expr(&input);
    }
}
