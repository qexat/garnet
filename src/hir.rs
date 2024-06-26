//! High-level intermediate representation, basically what you get after immediate parsing and
//! where all the typechecking and other kinds of checking happen.  Slightly simplified over the
//! AST -- basically the result of a lowering/desugaring pass.  This might give us a nice place to
//! do other simple optimization-like things like constant folding, dead code detection, etc.
//!
//! Currently it's not really lowered by much!  If's and maybe loops or something.
//! It's mostly a layer of indirection for further stuff to happen to, so we can change
//! the parser without changing the typechecking and codegen.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ast;
pub use crate::ast::{BOp, IfCase, Literal, Signature, UOp};
use crate::*;

/// Expression ID.  Used for associating individual expressions to types, and whatever
/// other metadata we feel like.  That way we can keep that data in tables instead of
/// having to mutate the HIR tree.
/// The only guarantees the number offers is it will be unique.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Eid(usize);

/// Slightly less ugly horrible atomic global for storing index values for
/// expression ID's, so each expression gets a unique ID.
static NEXT_EID: AtomicUsize = AtomicUsize::new(0);

impl Eid {
    fn new() -> Self {
        // We may be able to use Ordering::Relaxed here since we don't
        // care what the value actually is, just that it increments
        // atomically.  Eh, it's fine.
        let current = NEXT_EID.fetch_add(1, Ordering::AcqRel);
        // Check for overflow, since fetch_add can't do it.
        // It returns the previous value, so if the next value
        // is < than it, we've overflowed.
        if (current.wrapping_add(1)) < current {
            panic!("Integer overflow incrementing Eid; is your program absolutely enormous?");
        }
        Eid(current)
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
    /// Whether or not the expression is constant,
    /// ie can be entirely evaluated at compile time.
    pub is_const: bool,
}

impl PartialEq for ExprNode {
    /// ExprNode's must ignore the Eid for equality purposes,
    /// otherwise they will just *never* be equal.
    ///
    /// If you need to compare ExprNode including Eid's, then
    /// write a new function.  (And then you might as well *only*
    /// compare the Eid's unless you are purposefully testing for
    /// ill-constructed nodes.)
    fn eq(&self, other: &Self) -> bool {
        self.e == other.e
    }
}

impl ExprNode {
    pub fn write(&self, indent: usize, f: &mut dyn fmt::Write) -> fmt::Result {
        // TODO: We can make this take a callback that gets called for each
        // expr node, which prints out types for each node.  Will make the
        // formatting even more horrible though.
        self.e.write(indent, f)?;
        Ok(())
    }
}

impl fmt::Display for Decl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::hir::Decl as D;
        match self {
            D::Function {
                name,
                signature,
                body,
            } => {
                writeln!(f, "fn {}{} =", name, signature.to_name())?;
                for e in body {
                    e.write(1, f)?;
                }
                writeln!(f, "\nend")?;
            }

            D::Const {
                name,
                typ: typename,
                init,
            } => {
                write!(f, "const {} {} = ", name, typename.get_name())?;
                init.write(1, f)?;
                writeln!(f)?;
            }
            D::TypeDef {
                name,
                typedecl,
                params,
            } => {
                let param_str: Vec<String> = params.iter().map(|t| t.to_string()).collect();
                let param_str = param_str.join(", ");
                writeln!(f, "type {}({}) = {}", name, param_str, typedecl.get_name())?;
            }
            D::Import { name, localname } => {
                writeln!(f, "import {} as {}", name, localname)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write(0, f)
    }
}

impl ExprNode {
    pub fn new(e: Expr) -> Self {
        ExprNode {
            e: Box::new(e),
            id: Eid::new(),
            is_const: false,
        }
    }

    /// Call the function on the contents
    /// and turn the result into a new ExprNode *without*
    /// changing our Eid.
    ///
    /// Consumes self to preserve the invariant that no
    /// Eid's are ever duplicated.
    pub fn map(self, f: &mut dyn FnMut(Expr) -> Expr) -> Self {
        ExprNode {
            e: Box::new(f(*self.e)),
            ..self
        }
    }

    /*
    /// Apply a function with extra state to the given node.
    /// Unneeded so far, see passes::expr_fold()
    pub fn fold<S>(self, state: S, f: &dyn Fn(Expr, S) -> (Expr, S)) -> (Self, S) {
        let (new_e, new_s) = f(*self.e, state);
        (
            ExprNode {
                e: Box::new(new_e),
                id: self.id,
            },
            new_s,
        )
    }
    */
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit {
        val: Literal,
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
        type_params: Vec<Type>,
    },
    Break,
    Return {
        retval: ExprNode,
    },
    TupleCtor {
        body: Vec<ExprNode>,
    },
    StructCtor {
        body: Vec<(Sym, ExprNode)>,
    },
    ArrayCtor {
        body: Vec<ExprNode>,
    },
    EnumCtor {
        name: Sym,
        variant: Sym,
        value: i32,
    },
    // Create a new sum type.
    // Like EnumCtor, this is generated for you by the
    // compiler.
    SumCtor {
        name: Sym,
        variant: Sym,
        body: ExprNode,
    },
    // Wrap a new type.
    // Like EnumCtor, this is generated by the compiler.
    TypeCtor {
        name: Sym,
        type_params: Vec<Type>,
        body: ExprNode,
    },
    // Opposite of TypeCtor
    TypeUnwrap {
        expr: ExprNode,
    },
    TupleRef {
        expr: ExprNode,
        elt: usize,
    },
    StructRef {
        expr: ExprNode,
        elt: Sym,
    },
    ArrayRef {
        expr: ExprNode,
        idx: ExprNode,
    },
    Assign {
        lhs: ExprNode,
        rhs: ExprNode,
    },
    Typecast {
        e: ExprNode,
        to: Type,
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
        Self::TupleCtor { body: vec![] }
    }

    /// A fairly janky pretty-printer for IR, that is at least better
    /// than using Debug on the raw structures.
    pub fn write(&self, indent: usize, f: &mut dyn fmt::Write) -> fmt::Result {
        use Expr::*;
        for _ in 0..(indent * 2) {
            write!(f, " ")?;
        }
        match self {
            Lit { val } => {
                write!(f, "{}", val)?;
            }
            Var { name } => write!(f, "{}", name)?,
            BinOp { op, lhs, rhs } => {
                write!(f, "({:?} ", op)?;
                lhs.write(indent, f)?;
                rhs.write(indent, f)?;
                writeln!(f, ")")?;
            }
            UniOp { op, rhs } => {
                write!(f, "({:?} ", op)?;
                rhs.write(indent, f)?;
                writeln!(f, ")")?;
            }
            Block { body } => {
                writeln!(f, "(block")?;
                for b in body {
                    b.write(indent + 1, f)?;
                }
                writeln!(f, ")")?;
            }
            Let {
                varname,
                typename,
                init,
                mutable,
            } => {
                let m = if *mutable { " mut" } else { "" };
                let type_str = typename
                    .as_ref()
                    .map(|inner| inner.get_name())
                    .unwrap_or(Cow::from(""));
                write!(f, "(let {}{} {} = ", &*varname.val(), m, type_str)?;
                init.write(0, f)?;
                writeln!(f, ")")?;
            }
            If { cases } => {
                writeln!(f, "(if")?;
                for (case, arm) in cases {
                    case.write(indent + 1, f)?;
                    for expr in arm {
                        expr.write(indent + 2, f)?;
                    }
                }
                writeln!(f, ")")?;
            }
            Loop { body } => {
                writeln!(f, "(loop")?;
                for b in body {
                    b.write(indent + 1, f)?;
                }
                writeln!(f, ")")?;
            }
            Lambda { signature, body } => {
                writeln!(f, "(lambda {}", signature.to_name())?;
                for b in body {
                    b.write(indent + 1, f)?;
                }
                writeln!(f, ")")?;
            }
            Funcall {
                func,
                params,
                type_params,
            } => {
                write!(f, "(funcall ")?;
                func.write(0, f)?;
                write!(f, " |")?;
                for ty in type_params {
                    write!(f, "{}, ", ty.get_name())?;
                }
                write!(f, "|")?;
                for b in params {
                    b.write(indent + 1, f)?;
                    write!(f, " ")?;
                }
                write!(f, ")")?;
            }
            Break => {
                writeln!(f, "(break)")?;
            }
            Return { retval } => {
                write!(f, "(return ")?;
                retval.write(0, f)?;
                writeln!(f)?;
            }
            EnumCtor {
                name,
                variant,
                value,
            } => {
                write!(f, "(enumctor {} {} {})", name, &*variant.val(), value)?;
            }
            TupleCtor { body } => {
                write!(f, "(tuple ")?;
                for b in body {
                    b.write(0, f)?;
                    write!(f, " ")?;
                }
                write!(f, ")")?;
            }
            StructCtor { body } => {
                writeln!(f, "(struct")?;
                for (nm, expr) in body {
                    write!(f, "{} = ", &*nm.val())?;
                    expr.write(0, f)?;
                    writeln!(f)?;
                }
                writeln!(f, ")")?;
            }
            ArrayCtor { body } => {
                writeln!(f, "(array")?;
                for b in body {
                    b.write(indent + 1, f)?;
                }
                write!(f, ")")?;
            }
            SumCtor {
                name,
                variant,
                body,
            } => {
                write!(f, "(sum {} {} ", name, &*variant.val())?;
                body.write(0, f)?;
                write!(f, ")")?;
            }
            TypeCtor {
                name,
                type_params,
                body,
            } => {
                // TODO: type_params
                write!(f, "(typector |")?;
                for ty in type_params {
                    write!(f, "{} ", ty.get_name())?;
                }
                write!(f, "| {} ", name)?;
                body.write(0, f)?;
                write!(f, ")")?;
            }
            TypeUnwrap { expr } => {
                write!(f, "(typeunwrap ")?;
                expr.write(0, f)?;
                write!(f, ")")?;
            }
            TupleRef { expr, elt } => {
                write!(f, "(tupleref ")?;
                expr.write(0, f)?;
                write!(f, " {})", elt)?;
            }
            StructRef { expr, elt } => {
                write!(f, "(structref ")?;
                expr.write(0, f)?;
                write!(f, " {})", elt)?;
            }
            ArrayRef { expr: e, idx } => {
                write!(f, "(arrayref ")?;
                e.write(0, f)?;
                write!(f, " ")?;
                idx.write(0, f)?;
                write!(f, ")")?;
            }
            Assign { lhs, rhs } => {
                write!(f, "(assign ")?;
                lhs.write(0, f)?;
                write!(f, " ")?;
                rhs.write(0, f)?;
                write!(f, ")")?;
            }
            Typecast { e, to } => {
                write!(f, "(cast {} ", to.get_name())?;
                e.write(0, f)?;
                write!(f, ")")?;
            }
            Deref { expr } => {
                write!(f, "(ref ")?;
                expr.write(0, f)?;
                write!(f, ")")?;
            }
            Ref { expr } => {
                write!(f, "(ref ")?;
                expr.write(0, f)?;
                write!(f, ")")?;
            }
        }
        Ok(())
    }
}

impl ExprNode {
    /// Shortcut function for making literal bools
    pub fn bool(b: bool) -> Self {
        Self::new(Expr::bool(b))
    }

    /// Shortcut function for making literal integers
    pub fn int(i: i128) -> Self {
        Self::new(Expr::int(i))
    }

    /// Shortcut function for making literal unit
    pub fn unit() -> Self {
        Self::new(Expr::unit())
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
        typ: Type,
        init: ExprNode,
    },
    TypeDef {
        name: Sym,
        params: Vec<Sym>,
        typedecl: Type,
    },
    Import {
        name: Sym,
        localname: Sym,
    },
}

/// A compilable chunk of IR.
///
/// Currently, basically a compilation unit.
#[derive(Debug, Clone, Default)]
pub struct Ir {
    pub decls: Vec<Decl>,
    pub filename: String,
    pub modulename: String,
}

impl fmt::Display for Ir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for decl in &self.decls {
            write!(f, "{}", decl)?;
        }
        Ok(())
    }
}

/// Transforms AST into HIR
pub fn lower(ast: &ast::Ast) -> Ir {
    let mut accm = Vec::with_capacity(ast.decls.len() * 2);
    for d in ast.decls.iter() {
        lower_decl(&mut accm, d)
    }

    Ir {
        decls: accm,
        filename: ast.filename.clone(),
        modulename: ast.modulename.clone(),
    }
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
pub fn lower_expr(expr: &ast::Expr) -> ExprNode {
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
            let nbody = body.iter().map(lower_expr).collect();
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
                typename: typename.clone(),
                init: ninit,
                mutable: *mutable,
            }
        }
        E::If { cases, falseblock } => {
            // One of the actual transformations, this makes all if/else statements
            // into essentially a cond: `if ... else if ... else if ... else if true ... end`
            // This is more consistent and easier to handle for typechecking.
            assert!(!cases.is_empty(), "Should never happen");
            let mut cases: Vec<_> = cases
                .iter()
                .map(|case| (lower_expr(&case.condition), lower_exprs(&case.body)))
                .collect();
            // Add the "else" case, which we can just make `else if true then...`
            // No idea if this is a good idea, but it makes life easier right
            // this instant, so.  Hasn't bit me yet, so it's not a *bad* idea.
            let nelse_case = ExprNode::bool(true);
            // Empty false block becomes a false block that returns unit
            let false_exprs = if falseblock.is_empty() {
                lower_exprs(&[ast::Expr::TupleCtor { body: vec![] }])
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
        E::While { cond, body } => {
            // While loops just get turned into a Loop containing
            // `if not cond then break end`
            let inverted_cond = E::UniOp {
                op: UOp::Not,
                rhs: cond.clone(),
            };
            let test = lower_expr(&inverted_cond);
            let brk = vec![ExprNode::new(Break)];
            // As per above, we need to always have an "else" end case
            let else_case = ExprNode::bool(true);
            let else_exprs = lower_exprs(&[ast::Expr::TupleCtor { body: vec![] }]);
            let if_expr = ExprNode::new(If {
                cases: vec![(test, brk), (else_case, else_exprs)],
            });
            let mut nbody = lower_exprs(body);
            nbody.insert(0, if_expr);
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
            typeparams,
        } => {
            let nfunc = lower_expr(func);
            let nparams = lower_exprs(params);
            Funcall {
                func: nfunc,
                params: nparams,
                type_params: typeparams.clone(),
            }
        }
        E::Break => Break,
        E::Return { retval: e } => Return {
            retval: lower_expr(e),
        },
        E::TupleCtor { body } => {
            /*
            // Turn {foo, bar} into {_0: foo, _1: bar}
                let body_pairs = body
                    .into_iter()
                    .enumerate()
                    .map(|(i, vl)| (Sym::new(format!("_{}", i)), lower_expr(vl)))
                    .collect();
                Expr::StructCtor { body: body_pairs }
                */
            let body = lower_exprs(body);
            Expr::TupleCtor { body }
        }
        E::StructCtor { body } => {
            let lowered_body = body
                .iter()
                .map(|(nm, expr)| (*nm, lower_expr(expr)))
                .collect();
            Expr::StructCtor { body: lowered_body }
        }
        E::ArrayRef { expr, idx } => Expr::ArrayRef {
            expr: lower_expr(expr),
            idx: lower_expr(idx),
        },
        E::TupleRef { expr, elt } => Expr::TupleRef {
            expr: lower_expr(expr),
            elt: *elt,
        },
        E::StructRef { expr, elt } => Expr::StructRef {
            expr: lower_expr(expr),
            elt: *elt,
        },
        E::TypeUnwrap { expr } => Expr::TypeUnwrap {
            expr: lower_expr(expr),
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
        E::ArrayCtor { body } => Expr::ArrayCtor {
            body: lower_exprs(body.as_slice()),
        },
    };
    ExprNode::new(new_exp)
}

/// handy shortcut to lower Vec<ast::Expr>
pub fn lower_exprs(exprs: &[ast::Expr]) -> Vec<ExprNode> {
    exprs.iter().map(lower_expr).collect()
}

fn lower_typedef(accm: &mut Vec<Decl>, name: Sym, ty: &Type, params: &[Sym]) {
    use Decl::*;
    match ty {
        // For `type Foo = enum Foo, Bar, Baz end`
        // synthesize
        // const Foo = {
        //     .Foo = <magic unprintable thing>
        //     .Bar = <magic unprintable thing>
        //     .Baz = <magic unprintable thing>
        // }
        Type::Enum(ts) => {
            let struct_body: Vec<_> = ts
                .iter()
                .map(|(enumname, enumval)| {
                    // TODO: Enum vals?
                    let e = ExprNode::new(Expr::EnumCtor {
                        name,
                        variant: *enumname,
                        value: *enumval,
                    });
                    let e2 = ExprNode::new(Expr::TypeCtor {
                        name,
                        type_params: vec![],
                        body: e,
                    });
                    (*enumname, e2)
                })
                .collect();
            let init_val = ExprNode::new(Expr::StructCtor { body: struct_body });
            let struct_signature = ts
                .iter()
                //.map(|(enumname, _enumval)| (*enumname, ty.clone()))
                .map(|(enumname, _enumval)| (*enumname, Type::Named(name, vec![])))
                .collect();
            // Enums cannot have type parameters, so this works.
            let init_type = Type::Struct(struct_signature, vec![]);
            let new_constdef = Const {
                name,
                typ: init_type,
                init: init_val,
            };
            accm.push(new_constdef);
        }

        // For `type Foo = sum X {}, Y Thing end`
        // synthesize
        // const Foo = {
        //     .X = fn({}) Foo = <magic unprintable thing> end
        //     .Y = fn(Thing) Foo = <magic unprintable thing> end
        // }
        //
        // Maybe also something like this???
        // type X = {}
        // type Y = Thing
        // TODO: What do we do with the generics...  Right now they just
        // get stuffed into the constructor functions verbatim.
        Type::Sum(body, type_params) => {
            trace!("Lowering sum type {}", name);
            let new_type_params = Type::detype_names(type_params);

            let struct_body: Vec<_> = body
                .iter()
                .map(|(variant_name, variant_type)| {
                    let paramname = Sym::new("x");
                    let signature = ast::Signature {
                        params: vec![(paramname, variant_type.clone())],
                        rettype: Type::Named(name, type_params.clone()),
                        typeparams: new_type_params.clone(),
                    };
                    // Just return the value passed to it wrapped
                    // in a constructor of some kind...?
                    let body = vec![ExprNode::new(Expr::SumCtor {
                        name,
                        variant: *variant_name,
                        body: ExprNode::new(Expr::Var { name: paramname }),
                    })];
                    let e = ExprNode::new(Expr::Lambda { signature, body });
                    //println!("{} is {:#?}", variant_name, e);
                    (*variant_name, e)
                })
                .collect();
            let init_val = ExprNode::new(Expr::StructCtor { body: struct_body });
            let struct_typebody = body
                .iter()
                .map(|(variant_name, variant_type)| {
                    // The function inside the struct has the signature
                    // `fn(variant_type) name`
                    (
                        *variant_name,
                        Type::Func(
                            vec![variant_type.clone()],
                            Box::new(Type::Named(name, type_params.clone())),
                            type_params.clone(),
                        ),
                    )
                })
                .collect();
            let struct_type = Type::Struct(struct_typebody, type_params.clone());
            let new_constdef = Const {
                name: name.to_owned(),
                typ: struct_type,
                init: init_val,
            };
            //println!("Lowered to {:#?}", &new_constdef);
            accm.push(new_constdef);
        }
        // For other types, we create a constructor function to build them.
        other => {
            let s = Sym::new("x");
            trace!("Lowering params {:?}", params);
            let type_params: Vec<_> = params.iter().map(|s| Type::named0(*s)).collect();
            let signature = ast::Signature {
                params: vec![(s, other.clone())],
                rettype: Type::Named(name.to_owned(), type_params.clone()),
                typeparams: params.to_owned(),
            };
            // The generated function just returns the value passed to it wrapped
            // in a type constructor
            let body = vec![ExprNode::new(Expr::TypeCtor {
                name,
                type_params,
                body: ExprNode::new(Expr::Var { name: s }),
            })];
            //println!("{} is {:#?}", variant_name, e);
            let new_fundecl = Function {
                name: name.to_owned(),
                signature,
                body,
            };
            accm.push(new_fundecl);
        }
    }
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
            typ: typename.clone(),
            init: lower_expr(init),
        }),
        // this needs to generate the typedef AND the type constructor
        // declaration.  Plus the type deconstructor.
        D::TypeDef {
            name,
            typedecl,
            doc_comment: _,
            params,
        } => {
            lower_typedef(accm, *name, typedecl, params);
            accm.push(Decl::TypeDef {
                name: *name,
                params: params.clone(),
                typedecl: typedecl.clone(),
            });
        }
        D::Import { name, rename } => accm.push(Decl::Import {
            name: *name,
            localname: rename.unwrap_or(*name),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr as A;
    use crate::hir::Expr as I;
    use crate::hir::ExprNode as EN;

    /// Does `return;` turn into `return ();`?
    /// Not doing that to make parsing simpler, since we don't
    /// actually have semicolons.
    #[test]
    fn test_return_none() {
        let input = A::Return {
            retval: Box::new(A::unit()),
        };
        let output = EN::new(I::Return { retval: EN::unit() });
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
                (EN::bool(false), vec![EN::int(1)]),
                (EN::bool(true), vec![EN::int(2)]),
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
                (EN::bool(false), vec![EN::int(1)]),
                (EN::bool(true), vec![EN::int(2)]),
                (EN::bool(true), vec![EN::int(3)]),
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
