//! High-level intermediate representation, basically what you get after immediate parsing and
//! where all the typechecking and other kinds of checking happen.  Slightly simplified over the
//! AST -- basically the result of a lowering/desugaring pass.  This might give us a nice place to
//! do other simple optimization-like things like constant folding, dead code detection, etc.
//!
//! Currently it's not really lowered by much!  If's and maybe loops or something.
//! It's mostly a layer of indirection for further stuff to happen to, so we can change
//! the parser without changing the typechecking and codegen.

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

/// Ugly horrible global for storing index values for expression ID's,
/// so each expression gets a unique ID.
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

/// An expression node with a unique ID attached.
/// Expressions are assigned types by the typechecker
/// via an `Eid`->type map.
#[derive(Debug, Clone)]
pub struct ExprNode {
    /// expression
    pub e: Box<Expr>,
    /// Expression ID
    pub id: Eid,
}

impl PartialEq for ExprNode {
    /// ExprNode's must ignore the Eid for equality purposes,
    /// otherwise they will just *never* be equal.
    ///
    /// If you need to compare ExprNode including Eid's, then
    /// write a new function.  (And then you might as well *only*
    /// compare the Eid's unless you are purposefully testing for
    /// ill-constructed ones.)
    fn eq(&self, other: &Self) -> bool {
        self.e == other.e
    }
}

impl ExprNode {
    fn new(e: Expr) -> Self {
        ExprNode {
            e: Box::new(e),
            id: Eid::new(),
        }
    }
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit {
        val: Literal,
    },
    EnumLit {
        val: Sym,
        ty: Type,
    },
    Var {
        name: Sym,
    },
    BinOp {
        op: BOp,
        lhs: ExprNode,
        rhs: ExprNode,
    },
    UniOp {
        op: UOp,
        rhs: ExprNode,
    },
    Block {
        body: Vec<ExprNode>,
    },
    Let {
        varname: Sym,
        typename: Option<Type>,
        init: ExprNode,
        mutable: bool,
    },
    If {
        cases: Vec<(ExprNode, Vec<ExprNode>)>,
    },
    Loop {
        body: Vec<ExprNode>,
    },
    Lambda {
        signature: Signature,
        body: Vec<ExprNode>,
    },
    Funcall {
        func: ExprNode,
        params: Vec<ExprNode>,
        generic_types: Vec<Type>,
    },
    Break,
    Return {
        retval: ExprNode,
    },
    StructCtor {
        body: Vec<(Sym, ExprNode)>,
    },
    StructRef {
        expr: ExprNode,
        elt: Sym,
    },
    Assign {
        lhs: ExprNode,
        rhs: ExprNode,
    },
    Deref {
        expr: ExprNode,
    },
    Ref {
        expr: ExprNode,
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
/// Like ExprNode, contains a type annotation.
#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Function {
        name: Sym,
        signature: Signature,
        body: Vec<ExprNode>,
    },
    Const {
        name: Sym,
        typename: Type,
        init: ExprNode,
    },
    TypeDef {
        name: Sym,
        params: Vec<Sym>,
        typedecl: Type,
    },
    /// Our first compiler intrinsic!  \o/
    ///
    /// This is a function that is generated along with
    /// each typedef, which takes the args of that typedef
    /// and produces that type.  We have no way to write such
    /// a function by hand, so here we are.  In all other ways
    /// though we treat it exactly like a function though.
    ///
    /// TODO: These should just be functions generated by lowering,
    /// and the TypeCtor, etc. exprs should be the builtins that
    /// these generated functions contain.
    Constructor { name: Sym, signature: Signature },
}

/// A compilable chunk of IR.
///
/// Currently, basically a compilation unit.
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
fn lower_expr(expr: &ast::Expr) -> ExprNode {
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
                lhs: nlhs,
                rhs: nrhs,
            }
        }
        E::UniOp { op, rhs } => {
            let nop = lower_uop(op);
            let nrhs = lower_expr(rhs);
            UniOp { op: nop, rhs: nrhs }
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
            let ninit = lower_expr(init);
            Let {
                varname: *varname,
                typename: Some(typename.clone()),
                init: ninit,
                mutable: *mutable,
            }
        }
        E::If { cases, falseblock } => {
            // One of the actual transformations, this makes all if/else statements
            // into essentially a switch: `if ... else if ... else if ... else if true ... end`
            // This is more consistent and easier to handle for typechecking.
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
        E::Funcall {
            func,
            params,
            generic_types,
        } => {
            let nfunc = lower_expr(func);
            let nparams = lower_exprs(params);
            Funcall {
                func: nfunc,
                params: nparams,
                generic_types: generic_types.to_vec(),
            }
        }
        E::Break => Break,
        E::Return { retval: e } => Return {
            retval: lower_expr(e),
        },
        // Turn {foo, bar} into {_0: foo, _1: bar}
        E::TupleCtor { body } => {
            let body_pairs = body
                .into_iter()
                .enumerate()
                .map(|(i, vl)| (Sym::new(format!("_{}", i)), lower_expr(vl)))
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
            expr: lower_expr(expr),
            elt: Sym::new(format!("_{}", elt)),
        },
        E::StructRef { expr, elt } => Expr::StructRef {
            expr: lower_expr(expr),
            elt: *elt,
        },
        E::Deref { expr } => Expr::Deref {
            expr: lower_expr(expr),
        },
        E::Ref { expr } => Expr::Ref {
            expr: lower_expr(expr),
        },
        E::Assign { lhs, rhs } => Expr::Assign {
            lhs: lower_expr(lhs),
            rhs: lower_expr(rhs),
        },
    };
    ExprNode::new(new_exp)
}

/// handy shortcut to lower Vec<ast::Expr>
fn lower_exprs(exprs: &[ast::Expr]) -> Vec<ExprNode> {
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
            typename: typename.clone(),
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
            let paramname = Sym::new("input");
            let rtype = Sym::new_type(&TypeDef::Named(*name));
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
                        let named_type = Sym::new_type(&TypeDef::Named(*name));
                        struct_fields.insert(*var, named_type);
                        let e = Expr::EnumLit {
                            ty: named_type,
                            val: *var,
                        };
                        let t = f(&e);
                        let te = ExprNode {
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
                        name: Sym::new(struct_name),
                        typename: Sym::new_type(&struct_type),
                        init: ExprNode {
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
    use crate::hir::ExprNode as EN;
    use crate::testutil::*;

    /// Does `return;` turn into `return ();`?
    /// Not doing that to make parsing simpler, since we don't
    /// actually have semicolons.
    #[test]
    fn test_return_none() {
        let input = A::Return {
            retval: Box::new(A::unit()),
        };
        let output = EN::new(I::Return {
            retval: EN::new(I::unit()),
        });
        let res = lower_expr(&input);
        assert_eq!(&res, &output);
    }

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
        let output = EN::new(I::If {
            cases: vec![
                (EN::new(I::bool(false)), vec![EN::new(I::int(1))]),
                (EN::new(I::bool(true)), vec![EN::new(I::int(2))]),
            ],
        });
        let res = lower_expr(&input);
        assert_eq!(&res, &output);
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
        let output = EN::new(I::If {
            cases: vec![
                (EN::new(I::bool(false)), vec![EN::new(I::int(1))]),
                (EN::new(I::bool(true)), vec![EN::new(I::int(2))]),
                (EN::new(I::bool(true)), vec![EN::new(I::int(3))]),
            ],
        });
        let res = lower_expr(&input);
        assert_eq!(&res, &output);
    }

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
