//! A pass that takes Named type parameters and turns them into Generics
//! if they are not declared.
//! We could do this in the typechecker but I think it will be simpler as a preprocessing pass.

use crate::passes::*;

// This is slightly bonkers and a big dependency inversion but it also does
// *exactly* what we want.  :thonk:
use crate::typeck::Symtbl;

fn generic_infer_type(symtbl: &Symtbl, ty: Type) -> Type {
    match ty {
        Type::Named(nm, ref _tys) => {
            if let Some(t) = symtbl.get_type(nm) {
                // TODO: Do we have to recurse down the type params?
                // or will type_map() do that for us?
                // t should just be a Generic(nm), so we just replace this type with it.
                t
            } else {
                // Not declared, just carry on
                ty
            }
        }
        _ => ty,
    }
}

/*
/// signature_map() doesn't do the correct thing here 'cause we have to
/// shuffle around generic param
fn generic_infer_sig(symtbl: &Symtbl, sig: Signature) -> Signature {
    sig
}
*/

/// Basically, if we see a Named type mentioned, we check to see if it's
/// in the symtbl.  If it is, we replace it with a Generic.  Otherwise
/// we leave it as is and let typeck decide whether or not it exists.
fn generic_infer_expr(_symtbl: &Symtbl, expr: ExprNode) -> ExprNode {
    expr
}

pub(super) fn generic_infer(ir: Ir) -> Ir {
    let symtbl = Symtbl::default();
    let generic_infer_expr = &mut |e| generic_infer_expr(&symtbl, e);
    let generic_infer_ty = &mut |ty| generic_infer_type(&symtbl, ty);
    // Fuck decl_map, it's not made for this.  So we'll just
    // do it by hand.
    let mut new_decls = vec![];
    for decl in ir.decls {
        let d = match decl {
            D::Function {
                name,
                signature,
                body,
            } => {
                let _guard = symtbl.push_scope();
                for typeparam in &signature.typeparams {
                    match typeparam {
                        Type::Named(sym, _) if *sym == Sym::new("Tuple") => (),
                        Type::Named(nm, _v) => {
                            symtbl.add_type(*nm, &Type::Generic(*nm));
                        }
                        _ => (),
                    }
                }
                let new_sig = signature_map(signature, generic_infer_ty);
                D::Function {
                    name,
                    signature: new_sig,
                    body: exprs_map(body, &mut id, generic_infer_expr, generic_infer_ty),
                }
            }
            D::Const { name, typ, init } => {
                let _guard = symtbl.push_scope();
                dbg!("dealing with const", name);
                for name in typ.get_type_params() {
                    dbg!(name);
                    symtbl.add_type(name, &Type::Generic(name));
                }
                D::Const {
                    name,
                    typ: type_map(typ, generic_infer_ty),
                    init: expr_map(init, &mut id, generic_infer_expr, generic_infer_ty),
                }
            }
            D::TypeDef {
                name,
                params,
                typedecl,
            } => {
                let _guard = symtbl.push_scope();
                for name in &params {
                    symtbl.add_type(*name, &Type::Generic(*name));
                }
                D::TypeDef {
                    name,
                    params,
                    typedecl: type_map(typedecl, generic_infer_ty),
                }
            }
            D::Import { .. } => decl,
        };
        new_decls.push(d);
    }
    Ir {
        decls: new_decls,
        ..ir
    }
}

#[cfg(test)]
mod tests {
    use crate::hir::{Decl as D, Ir};
    use crate::*;

    #[test]
    fn heckin_heck() {
        let typedef = D::TypeDef {
            name: Sym::new("thing"),
            params: vec![Sym::new("T")],
            typedecl: Type::Named(Sym::new("T"), vec![]),
        };
        let ir = Ir {
            decls: vec![typedef],
            filename: String::from("test"),
            modulename: String::from("test"),
        };
        let res = super::generic_infer(ir);

        let wanted_typedef = D::TypeDef {
            name: Sym::new("thing"),
            params: vec![Sym::new("T")],
            typedecl: Type::generic("T"),
        };

        assert_eq!(res.decls[0], wanted_typedef);
    }

    #[test]
    fn heckin_heck2() {
        let structbody = [
            (Sym::new("a"), Type::i32()),
            (Sym::new("b"), Type::Named(Sym::new("T"), vec![])),
        ];
        let typedef = D::TypeDef {
            name: Sym::new("thing"),
            params: vec![Sym::new("T")],
            typedecl: Type::Struct(BTreeMap::from(structbody), vec![]),
        };
        let ir = Ir {
            decls: vec![typedef],
            filename: String::from("test"),
            modulename: String::from("test"),
        };
        let res = super::generic_infer(ir);

        let wanted_structbody = [
            (Sym::new("a"), Type::i32()),
            (Sym::new("b"), Type::generic("T")),
        ];
        let wanted_typedef = D::TypeDef {
            name: Sym::new("thing"),
            params: vec![Sym::new("T")],
            typedecl: Type::Struct(BTreeMap::from(wanted_structbody), vec![]),
        };

        assert_eq!(res.decls[0], wanted_typedef);
    }

    /// Todo: maybe check out nested functions?
    #[test]
    fn heckin_heck3() {
        let sig1 = hir::Signature {
            params: vec![
                (Sym::new("a"), Type::named0("T")),
                (Sym::new("b"), Type::named0("Something")),
            ],
            rettype: Type::Named(Sym::new("T"), vec![]),
            typeparams: vec![Type::Named(Sym::new("T"), vec![])],
        };
        let f1 = D::Function {
            name: Sym::new("foo"),
            signature: sig1,
            body: vec![],
        };
        let ir = Ir {
            decls: vec![f1],
            filename: String::from("test"),
            modulename: String::from("test"),
        };
        let res = super::generic_infer(ir);

        let wanted_sig = hir::Signature {
            params: vec![
                (Sym::new("a"), Type::generic("T")),
                (Sym::new("b"), Type::named0("Something")),
            ],
            rettype: Type::generic("T"),
            typeparams: vec![Type::generic("T")],
        };
        let wanted_f = D::Function {
            name: Sym::new("foo"),
            signature: wanted_sig,
            body: vec![],
        };

        assert_eq!(res.decls[0], wanted_f);
    }
}
