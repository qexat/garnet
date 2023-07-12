//! Typechecking and other semantic checking.
//! Operates on the HIR.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use log::*;

use crate::*;

use crate::hir;
use crate::{Sym, Type};

/// A identifier to uniquely refer to our `TypeInfo`'s
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TypeId(usize);

/// Potentially-incomplete information about the type of an expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeInfo {
    /// No information about the type of this expr
    Unknown,
    /// This type is the same as another expr's type
    Ref(TypeId),
    Prim(PrimType),
    Enum(Vec<(Sym, i32)>),
    /// This expression never returns so it can "be" any type
    Never,
    /// N-ary type constructor.
    /// It could be Int()
    /// or List(Int)
    /// or List(T)
    Named(Sym, Vec<TypeId>),
    /// This type is definitely a function
    Func(Vec<TypeId>, TypeId, Vec<TypeId>),
    /// This is definitely some kind of struct
    Struct(BTreeMap<Sym, TypeId>),
    /// Definitely a sum type
    Sum(BTreeMap<Sym, TypeId>),
    /// Definitely an array, one we might or might not know the length of
    Array(TypeId, Option<usize>),
    /// This is some generic type that has a name like `@A`
    /// AKA a type parameter.
    TypeParam(Sym),
    /// Unique borrow.  Probably needs a lifetime param of some kind added someday?
    Uniq(TypeId),
}

impl TypeInfo {
    /// Conjure forth a printable name for a type term, suitable for printing
    /// in debug or error messages.
    fn get_name(&self, tck: &Tck) -> Cow<'static, str> {
        use TypeInfo::*;
        match self {
            Unknown => Cow::Borrowed("{unknown}"),
            Never => Cow::Borrowed("{never}"),
            Prim(p) => p.get_name(),
            Ref(id) => Cow::Owned(format!("Ref({})", tck.get(id).get_name(tck))),
            Enum(v) => {
                let mut accm = String::from("enum ");
                // TODO: Do heckin' something with the enum's values???
                for (name, _vl) in v {
                    accm += &format!("{}, ", name);
                }
                accm += "end";
                accm.into()
            }
            Named(s, g) => {
                let name = (*s.val()).to_owned();
                let generics: Vec<_> = g.iter().map(|t| tck.get(t).get_name(tck)).collect();
                let joined_generics = generics.join(", ");
                format!("{}({})", name, joined_generics).into()
            }
            Func(params, rettype, typeparams) => {
                let paramstr: Vec<_> = params.iter().map(|id| tck.get(id).get_name(tck)).collect();
                let paramstr = paramstr.join(", ");
                let rettypestr = tck.get(rettype).get_name(tck);
                let tparamstr: Vec<_> = typeparams
                    .iter()
                    .map(|id| tck.get(id).get_name(tck))
                    .collect();
                let tparamstr = tparamstr.join(", ");
                format!("fn(|{}| {}) {}", tparamstr, paramstr, rettypestr).into()
            }
            Struct(body) => {
                let mut accm = String::from("struct\n");
                for (name, id) in body {
                    let typename = tck.get(id).get_name(tck);
                    accm += &format!("{}: {},\n", name, typename);
                }
                accm += "end\n";
                accm.into()
            }
            Sum(body) => {
                let mut accm = String::from("sum\n");
                for (name, id) in body {
                    let typename = tck.get(id).get_name(tck);
                    accm += &format!("{} {},\n", name, typename);
                }
                accm += "end\n";
                accm.into()
            }
            Array(id, len) => {
                // Output something intelligible if we don't know the
                // length of the array.
                let len_str = len
                    .map(|l| format!("{}", l))
                    .unwrap_or_else(|| "unknown".to_string());
                Cow::Owned(format!("{}[{}]", tck.get(id).get_name(tck), len_str))
            }
            TypeParam(sym) => Cow::Owned(format!("@{}", &*sym.val())),
            Uniq(ty) => {
                let typename = tck.get(ty).get_name(tck);
                format!("&{}", typename).into()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownVar(Sym),
    AlreadyDefined(Sym),
    TypeMismatch {
        expr_name: Cow<'static, str>,
        got: Type,
        expected: Type,
    },
    TypeListMismatch, // TODO
    AmbiguousType {
        expr_name: Cow<'static, str>,
    },
    Mutability {
        expr_name: Cow<'static, str>,
    },
    ReconstructFail {
        tid: TypeId,
    },
    /*
    TODO: Oh heck make these used
    UnknownType(Sym),
    InvalidReturn,
    Return {
        fname: Sym,
        got: Type,
        expected: Type,
    },
    BopType {
        bop: hir::BOp,
        got1: Type,
        got2: Type,
        expected: Type,
    },
    UopType {
        op: hir::UOp,
        got: Type,
        expected: Type,
    },
    LetType {
        name: Sym,
        got: Type,
        expected: Type,
    },
    IfType {
        expected: Type,
        got: Type,
    },
    Cond {
        got: Type,
    },
    Param {
        got: Type,
        expected: Type,
    },
    Call {
        got: Type,
    },
    TupleRef {
        got: Type,
    },
    StructRef {
        fieldname: Sym,
        got: Type,
    },
    StructField {
        expected: Vec<Sym>,
        got: Vec<Sym>,
    },
    EnumVariant {
        expected: Vec<Sym>,
        got: Sym,
    },
    */
    Placeholder(String),
}

impl From<String> for TypeError {
    fn from(s: String) -> Self {
        Self::Placeholder(s)
    }
}

impl std::error::Error for TypeError {}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

impl TypeError {
    /// TODO: This should just be turned into the Display impl
    pub fn format(&self) -> String {
        match self {
            TypeError::UnknownVar(sym) => format!("Unknown var: {}", sym.val()),
            TypeError::AlreadyDefined(sym) => format!(
                "Type, var, const, or function already defined: {}",
                sym.val()
            ),
            TypeError::TypeMismatch {
                expr_name,
                expected,
                got,
            } => format!(
                "Type mismatch in '{}' expresssion, expected {} but got {}",
                expr_name,
                expected.get_name(),
                got.get_name()
            ),
            TypeError::TypeListMismatch => {
                "Type list mismatch, make better error message".to_string()
            }
            TypeError::AmbiguousType { expr_name } => {
                format!("Ambiguous/unknown type for expression '{}'", expr_name)
            }
            TypeError::Mutability { expr_name } => {
                format!("Mutability mismatch in '{}' expresssion", expr_name)
            }
            TypeError::ReconstructFail { tid } => {
                format!("Reconstruction failed, type ID ${} is unbound", tid.0)
            } /*
            TypeError::UnknownType(sym) => format!("Unknown type: {}", *sym.val()),
            TypeError::InvalidReturn => {
            "return expression happened somewhere that isn't in a function!".to_string()
            }
            TypeError::Return {
            fname,
            got,
            expected,
            } => format!(
            "Function {} returns {} but should return {}",
             *fname.val(),
             *got.val().get_name(),
             *expected.val().get_name(),
            ),
            TypeError::BopType {
            bop,
            got1,
            got2,
            expected,
            } => format!(
            "Invalid types for BOp {:?}: expected {}, got {} + {}",
            bop,
             *expected.val().get_name(),
             *got1.val().get_name(),
             *got2.val().get_name()
            ),
            TypeError::UopType { op, got, expected } => format!(
            "Invalid types for UOp {:?}: expected {}, got {}",
            op,
             *expected.val().get_name(),
             *got.val().get_name()
            ),
            TypeError::LetType {
            name,
            got,
            expected,
            } => format!(
            "initializer for variable {}: expected {} ({:?}), got {} ({:?})",
             *name.val(),
             *expected.val().get_name(),
             *expected,
             *got.val().get_name(),
             *got,
            ),
            TypeError::IfType { expected, got } => format!(
            "If block return type is {}, but we thought it should be something like {}",
             *expected.val().get_name(),
             *got.val().get_name(),
            ),
            TypeError::Cond { got } => format!(
            "If expr condition is {}, not bool",
             *got.val().get_name(),
            ),
            TypeError::Param { got, expected } => format!(
            "Function wanted type {} in param but got type {}",
             *expected.val().get_name(),
             *got.val().get_name()
            ),
            TypeError::Call { got } => format!(
            "Tried to call function but it is not a function, it is a {}",
             *got.val().get_name()
            ),
            TypeError::TupleRef { got } => format!(
            "Tried to reference tuple but didn't get a tuple, got {}",
             *got.val().get_name()
            ),
            TypeError::StructRef { fieldname, got } => format!(
            "Tried to reference field {} of struct, but struct is {}",
             *fieldname.val(),
             *got.val().get_name(),
            ),
            TypeError::StructField { expected, got } => format!(
            "Invalid field in struct constructor: expected {:?}, but got {:?}",
            expected, got
            ),
            TypeError::EnumVariant { expected, got } => {
            let expected_names: Vec<String> = expected
            .into_iter()
            .map(|nm| (&**nm.val()).clone())
            .collect();
            format!(
            "Unknown enum variant '{}', valid ones are {:?}",
             *got.val(),
            expected_names,
            )
            }
             */
            TypeError::Placeholder(s) => format!("Placeholder error: {}", s),
        }
    }
}

/// Type checking engine
#[derive(Default)]
pub struct Tck {
    /// Used to generate unique IDs
    id_counter: usize,
    /// Binding from type vars to what we know about the type
    vars: BTreeMap<TypeId, TypeInfo>,
    /// What we know about the type of each node in the AST.
    types: BTreeMap<hir::Eid, TypeId>,
}

impl Tck {
    fn get(&self, id: &TypeId) -> &TypeInfo {
        self.vars.get(id).unwrap_or_else(|| {
            panic!(
                "TypeId {:?} does not exist, should probably never happen",
                id
            )
        })
    }
    /// Save the type variable associated with the given expr.
    /// Panics if it is redefining the expr's type.
    fn set_expr_type(&mut self, expr: &hir::ExprNode, ty: TypeId) {
        let res = self.types.insert(expr.id, ty);
        assert!(
            res.is_none(),
            "Redefining known type, this is generally a bad thing.  It was: {:?}",
            self.reconstruct(res.unwrap())
        );
    }

    /// Replace the given expr's type with a new one.
    /// Used for HIR transforms that change one node type into another,
    /// but generally not used for actual typechecking since unification
    /// always progresses.
    ///
    /// Panics if the expression's type is not set.
    pub fn replace_expr_type(&mut self, expr: &hir::ExprNode, ty: &Type) {
        let typeid = self.insert_known(ty);
        let res = self.types.insert(expr.id, typeid);
        assert!(res.is_some());
    }

    /// Returns the `TypeId` associated with the HIR node.
    /// Panics if the expression's type is not set.
    pub fn get_expr_type(&self, expr: &hir::ExprNode) -> TypeId {
        *self.types.get(&expr.id).unwrap_or_else(|| {
            panic!(
                "Could not get type of expr with ID {:?}!\nExpr was {:?}",
                expr.id, expr
            )
        })
    }

    /// Create a new type var with whatever we have about its type
    pub fn insert(&mut self, info: TypeInfo) -> TypeId {
        // Generate a new ID for our type term
        self.id_counter += 1;
        let id = TypeId(self.id_counter);
        assert!(self.vars.get(&id).is_none(), "Can't happen");
        self.vars.insert(id, info);
        id
    }

    /// Create a new type var out of a known type, such as if we
    /// declare a var's type.
    pub fn insert_known(&mut self, t: &Type) -> TypeId {
        // Recursively insert all subtypes.
        let tinfo = match t {
            Type::Prim(ty) => TypeInfo::Prim(*ty),
            Type::Never => TypeInfo::Never,
            Type::Enum(vals) => TypeInfo::Enum(vals.clone()),
            Type::Named(s, args) => {
                let new_args = args.iter().map(|t| self.insert_known(t)).collect();
                TypeInfo::Named(*s, new_args)
            }
            Type::Func(args, rettype, typeparams) => {
                let new_args = args.iter().map(|t| self.insert_known(t)).collect();
                let new_rettype = self.insert_known(rettype);
                let new_typeparams = typeparams.iter().map(|t| self.insert_known(t)).collect();
                TypeInfo::Func(new_args, new_rettype, new_typeparams)
            }
            Type::Generic(s) => TypeInfo::TypeParam(*s),
            Type::Array(ty, len) => {
                let new_body = self.insert_known(ty);
                TypeInfo::Array(new_body, Some(*len))
            }
            // TODO: Generics?
            Type::Struct(body, _names) => {
                let new_body = body
                    .iter()
                    .map(|(nm, t)| (*nm, self.insert_known(t)))
                    .collect();
                TypeInfo::Struct(new_body)
            }
            // TODO: Generics?
            Type::Sum(body, _names) => {
                let new_body = body
                    .iter()
                    .map(|(nm, t)| (*nm, self.insert_known(t)))
                    .collect();
                TypeInfo::Sum(new_body)
            }
            Type::Uniq(ty) => {
                let new_body = self.insert_known(&*ty);
                TypeInfo::Uniq(new_body)
            }
        };
        self.insert(tinfo)
    }

    /// Looks up a type var which must resolve to a struct (sooner or later)
    /// and if it's a struct we know about then looks up the type of the
    /// named field in it.
    ///
    /// Panics on invalid field name or not a struct type
    pub fn get_struct_field_type(
        &mut self,
        symtbl: &Symtbl,
        struct_type: TypeId,
        field_name: Sym,
    ) -> TypeId {
        use TypeInfo::*;
        match self.get(&struct_type).clone() {
            Ref(t) => self.get_struct_field_type(symtbl, t, field_name),
            Struct(body) => body.get(&field_name).cloned().unwrap_or_else(|| {
                panic!(
                    "Struct has no field {}, valid fields are: {:#?}",
                    field_name, body
                )
            }),
            other => {
                panic!(
                    "Tried to get struct field {} from non-struct type {:?}",
                    field_name, other
                )
            }
        }
    }

    /// Same as `get_struct_field_type()` but takes a tuple type and an integer.
    pub fn get_tuple_field_type(
        &mut self,
        symtbl: &Symtbl,
        tuple_type: TypeId,
        n: usize,
    ) -> TypeId {
        use TypeInfo::*;
        match self.get(&tuple_type).clone() {
            Ref(t) => self.get_tuple_field_type(symtbl, t, n),
            Named(nm, tys) if &*nm.val() == "Tuple" => tys.get(n).cloned().unwrap_or_else(|| {
                panic!("Tuple has no field {}, valid fields are: {:#?}", n, tys)
            }),
            other => {
                panic!(
                    "Tried to get tuple field {} from non-tuple type {:?}",
                    n, other
                )
            }
        }
    }

    /// Returns the type that the bin op operates on.
    pub fn bop_input_type(&mut self, bop: hir::BOp) -> TypeId {
        use hir::BOp::*;
        match bop {
            And | Or | Xor => self.insert_known(&Type::bool()),
            Eq | Neq => self.insert(TypeInfo::Unknown),
            // Don't know what it is but it's gotta be a number
            Gt | Lt | Gte | Lte => self.insert_known(&Type::iunknown()),
            Add | Sub | Mul | Div | Mod => self.insert_known(&Type::iunknown()),
        }
    }

    /// What the resultant type of the binop is.
    ///
    /// Needs to know what the type of the expression given to it is,
    /// but also assumes that the LHS and RHS have the same input type.
    /// Ensuring that is left as an exercise to the user.
    pub fn bop_output_type(&mut self, bop: hir::BOp, input: TypeId) -> TypeId {
        use hir::BOp::*;
        match bop {
            // Output of math ops is same type as input
            Add | Sub | Mul | Div | Mod => self.insert(TypeInfo::Ref(input)),
            // Output of logic ops is always a bool
            Eq | Neq | Gt | Lt | Gte | Lte | And | Or | Xor => self.insert_known(&Type::bool()),
        }
    }

    /// Returns the type that the unary op operates on.
    pub fn uop_input_type(&mut self, uop: hir::UOp) -> TypeId {
        use hir::UOp::*;
        match uop {
            Neg => self.insert_known(&Type::iunknown()),
            Not => self.insert_known(&Type::bool()),
            Ref => todo!(),
            Deref => todo!(),
        }
    }

    /// What the resultant type of the uop is
    pub fn uop_output_type(&mut self, uop: hir::UOp, input: TypeId) -> TypeId {
        use hir::UOp::*;
        match uop {
            Neg => self.insert(TypeInfo::Ref(input)),
            Not => self.insert_known(&Type::bool()),
            Ref => todo!(),
            Deref => todo!(),
        }
    }

    /// Unify two lists of types.  They must be the same length.
    fn unify_lists(
        &mut self,
        symtbl: &Symtbl,
        a: &[TypeId],
        b: &[TypeId],
    ) -> Result<(), TypeError> {
        if a.len() == b.len() {
            for (arg1, arg2) in a.iter().zip(b.iter()) {
                self.unify(symtbl, *arg1, *arg2)?;
            }
            Ok(())
        } else {
            Err(TypeError::TypeListMismatch)
        }
    }

    /// Make the types of two type terms equivalent (or produce an error if
    /// there is a conflict between them)
    pub fn unify(&mut self, symtbl: &Symtbl, a: TypeId, b: TypeId) -> Result<(), TypeError> {
        //trace!("> Unifying {:?} with {:?}", self.get(&a), self.get(&b));
        // If a == b then it's a little weird but shoooooould be fine
        // as long as we don't get any mutual recursion or self-recursion
        // involved
        // Per MBones:
        // Yes it makes sense. The unifier is tasked with solving literally whatever equations you throw at it, and this is an important edge case to check for (to avoid accidentally making cyclic datastructures). (The correct action from the unifier is to succeed with no updates, since it's already equal to itself)
        if a == b {
            return Ok(());
        }
        use TypeInfo::*;
        match (self.get(&a).clone(), self.get(&b).clone()) {
            // Primitives just match directly
            (Prim(p1), Prim(p2)) if p1 == p2 => Ok(()),
            // Unknown integers unified with known integers become known
            // integers.
            (Prim(PrimType::UnknownInt), Prim(PrimType::SInt(_))) => {
                self.vars.insert(a, TypeInfo::Ref(b));
                Ok(())
            }
            (Prim(PrimType::SInt(_)), Prim(PrimType::UnknownInt)) => {
                self.vars.insert(b, TypeInfo::Ref(a));
                Ok(())
            }
            // Never types can turn into anything 'cause they never happen
            // I'm like.... 97% sure this is correct.
            (Never, _) => Ok(()),
            (_, Never) => Ok(()),
            // Follow any references
            (Ref(a), _) => self.unify(symtbl, a, b),
            (_, Ref(b)) => self.unify(symtbl, a, b),

            // When we don't know anything about either term, assume that
            // they match and make the one we know nothing about reference the
            // one we may know something about
            (Unknown, _) => {
                self.vars.insert(a, TypeInfo::Ref(b));
                Ok(())
            }
            (_, Unknown) => {
                self.vars.insert(b, TypeInfo::Ref(a));
                Ok(())
            }

            // For type constructors, if their names are the same we try
            // to unify their args
            (Named(n1, args1), Named(n2, args2)) => {
                if n1 == n2 {
                    self.unify_lists(symtbl, &args1, &args2)
                } else {
                    panic!(
                        "Mismatch between types {}({:?}) and {}({:?})",
                        n1, args1, n2, args2
                    )
                }
            }
            // Enums are the same if their contents are the same.
            // No need to recurse, since they're terminal types.
            (Enum(a), Enum(b)) if a == b => Ok(()),
            // When unifying complex types, we must check their sub-types. This
            // can be trivially implemented for tuples, sum types, etc.
            (Func(a_i, a_o, a_params), Func(b_i, b_o, b_params)) => {
                self.unify_lists(symtbl, &a_i, &b_i)?;
                self.unify(symtbl, a_o, b_o)?;
                self.unify_lists(symtbl, &a_params, &b_params)?;

                Ok(())
            }
            // same basic idea
            (Struct(body1), Struct(body2)) => {
                for (nm, t1) in body1.iter() {
                    let t2 = body2[nm];
                    self.unify(symtbl, *t1, t2)?;
                }
                // Now we just do it again the other way around
                // which is a dumb but effective way of making sure
                // struct2 doesn't have any fields that struct1 doesn't.
                for (nm, t2) in body2.iter() {
                    let t1 = body1[nm];
                    self.unify(symtbl, t1, *t2)?;
                }
                Ok(())
            }
            // Same as struct types
            (Sum(body1), Sum(body2)) => {
                for (nm, t1) in body1.iter() {
                    let t2 = body2[nm];
                    self.unify(symtbl, *t1, t2)?;
                }
                for (nm, t2) in body2.iter() {
                    let t1 = body1[nm];
                    self.unify(symtbl, t1, *t2)?;
                }
                Ok(())
            }
            // Either both array lengths are the same or both are
            // unknown.  The latter might need to be an error, we will see.
            (Array(body1, len1), Array(body2, len2)) if len1 == len2 => {
                self.unify(symtbl, body1, body2)
            }
            (Array(body1, None), Array(body2, Some(_len2))) => {
                self.vars.insert(a, TypeInfo::Ref(b));
                self.unify(symtbl, body1, body2)
                //todo!("propegate array lengths")
            }
            (Array(body1, Some(_len1)), Array(body2, None)) => {
                self.vars.insert(b, TypeInfo::Ref(a));
                self.unify(symtbl, body1, body2)
                // todo!("propegate array lengths")
            }
            (Array(body1, len1), Array(body2, len2)) if len1 == len2 => {
                self.unify(symtbl, body1, body2)
            }
            // For declared type parameters like @T they match if their names match.
            // TODO: And if they have been declared?  Not sure we can ever get to
            // here if they haven't.
            (TypeParam(s1), TypeParam(s2)) if s1 == s2 => Ok(()),
            (Uniq(s1), Uniq(s2)) => self.unify(symtbl, s1, s2),
            // If no previous attempts to unify were successful, raise an error
            (a, b) => {
                self.print_types();
                Err(format!(
                    "Conflict between {} and {}",
                    a.get_name(self),
                    b.get_name(self)
                )
                .into())
            }
        }
    }

    /// Attempt to reconstruct a concrete type from the given type var ID. This
    /// may fail if we don't yet have enough information to figure out what the
    /// type is.
    pub fn reconstruct(&self, id: TypeId) -> Result<Type, TypeError> {
        use TypeInfo::*;
        match &self.get(&id) {
            Unknown => Err(TypeError::ReconstructFail { tid: id }),
            Never => Ok(Type::Never),
            Prim(ty) => Ok(Type::Prim(*ty)),
            Ref(id) => self.reconstruct(*id),
            Enum(ts) => Ok(Type::Enum(ts.clone())),
            Named(s, args) => {
                let arg_types: Result<Vec<_>, _> =
                    args.iter().map(|x| self.reconstruct(*x)).collect();
                Ok(Type::Named(*s, arg_types?))
            }
            Func(args, rettype, typeparams) => {
                let real_args: Result<Vec<Type>, TypeError> =
                    args.iter().map(|arg| self.reconstruct(*arg)).collect();
                let type_param_types: Result<Vec<Type>, _> =
                    typeparams.iter().map(|x| self.reconstruct(*x)).collect();

                Ok(Type::Func(
                    real_args?,
                    Box::new(self.reconstruct(*rettype)?),
                    type_param_types?,
                ))
            }
            TypeParam(name) => Ok(Type::Generic(*name)),
            Struct(body) => {
                let real_body: Result<BTreeMap<_, _>, TypeError> = body
                    .iter()
                    .map(|(nm, t)| {
                        let new_t = self.reconstruct(*t)?;
                        Ok((*nm, new_t))
                    })
                    .collect();
                // TODO: The empty params here feels suspicious, verify.
                let params = vec![];
                Ok(Type::Struct(real_body?, params))
            }
            Array(ty, len) => {
                let real_body = self.reconstruct(*ty)?;
                Ok(Type::Array(Box::new(real_body), len.unwrap()))
            }
            Sum(body) => {
                let real_body: Result<BTreeMap<_, _>, TypeError> = body
                    .iter()
                    .map(|(nm, t)| {
                        let new_t = self.reconstruct(*t)?;
                        Ok((*nm, new_t))
                    })
                    .collect();
                let params = vec![];
                Ok(Type::Sum(real_body?, params))
            }
            Uniq(ty) => {
                let inner_type = self.reconstruct(*ty)?;
                Ok(Type::Uniq(Box::new(inner_type)))
            }
        }
    }

    fn print_types(&self) {
        let mut vars_report: Vec<_> = self.vars.iter().collect();
        vars_report.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
        for (k, v) in vars_report.iter() {
            trace!("  ${} => {:?}", k.0, v);
        }
    }

    /// Kinda the opposite of reconstruction; takes a concrete type
    /// and generates a new type with unknown's (type variables) for the generic types (type
    /// parameters)
    ///
    /// The named_types is a *local* binding of generic type names to type variables.
    /// We use this to make multiple mentions of the same type name, such as
    /// `id :: T -> T`, all refer to the same type variable.
    /// Feels Weird but it works.
    ///
    /// This has to actually be an empty hashtable on the first instantitaion
    /// instead of the symtbl, since the symtbl is full of type parameter names from the
    /// enclosing function and those are what we explicitly want to get away from.
    fn instantiate(&mut self, t: &Type, known_types: Option<BTreeMap<Sym, TypeId>>) -> TypeId {
        fn helper(tck: &mut Tck, named_types: &mut BTreeMap<Sym, TypeId>, t: &Type) -> TypeId {
            let typeinfo = match t {
                Type::Prim(val) => TypeInfo::Prim(*val),
                Type::Never => TypeInfo::Never,
                Type::Enum(vals) => TypeInfo::Enum(vals.clone()),
                Type::Named(s, args) => {
                    let inst_args: Vec<_> =
                        args.iter().map(|t| helper(tck, named_types, t)).collect();
                    TypeInfo::Named(*s, inst_args)
                }
                Type::Generic(s) => {
                    // If we know this is is a particular generic, match wiht it
                    if let Some(ty) = named_types.get(s) {
                        TypeInfo::Ref(*ty)
                    } else {
                        panic!("Referred to unknown generic named {}", s);
                    }
                }
                Type::Func(args, rettype, typeparams) => {
                    let inst_args: Vec<_> =
                        args.iter().map(|t| helper(tck, named_types, t)).collect();
                    let inst_ret = helper(tck, named_types, rettype);
                    let inst_type_params: Vec<_> = typeparams
                        .iter()
                        .map(|t| helper(tck, named_types, t))
                        .collect();
                    TypeInfo::Func(inst_args, inst_ret, inst_type_params)
                }
                Type::Struct(body, _names) => {
                    let inst_body = body
                        .iter()
                        .map(|(nm, ty)| (*nm, helper(tck, named_types, ty)))
                        .collect();
                    TypeInfo::Struct(inst_body)
                }
                Type::Array(ty, len) => {
                    let inst_ty = helper(tck, named_types, ty);
                    TypeInfo::Array(inst_ty, Some(*len))
                }
                Type::Sum(body, _names) => {
                    let inst_body = body
                        .iter()
                        .map(|(nm, ty)| (*nm, helper(tck, named_types, ty)))
                        .collect();
                    TypeInfo::Sum(inst_body)
                }
                Type::Uniq(ty) => {
                    let inst_ty = helper(tck, named_types, ty);
                    TypeInfo::Uniq(inst_ty)
                }
            };
            tck.insert(typeinfo)
        }
        // We have to pluck any unknowns out of the toplevel type and create
        // new type vars for them.
        // We don't worry about binding those vars or such, that is what unification
        // will do later.
        // We do have to take any of those unknowns that are actually
        // known and preserve that knowledge though.
        let known_types = &mut known_types.unwrap_or_default();
        let type_params = t.get_type_params();
        // Probably a cleaner way to do this but oh well.
        // We to through the type params, and if any of them
        // are unknown we put a new TypeInfo::Unknown in for them.
        for param in type_params {
            known_types
                .entry(param)
                .or_insert_with(|| self.insert(TypeInfo::Unknown));
        }
        helper(self, known_types, t)
    }
}

#[derive(Clone, Default)]
struct ScopeFrame {
    /// Values are (type, mutability)
    symbols: BTreeMap<Sym, (TypeId, bool)>,
    types: BTreeMap<Sym, Type>,
}

/// Basic symbol table that maps names to type ID's
/// and manages scope.
/// Looks ugly, works well.
#[derive(Clone)]
pub struct Symtbl {
    frames: Rc<RefCell<Vec<ScopeFrame>>>,
}

impl Default for Symtbl {
    /// We start with an empty toplevel scope existing,
    /// then add some builtin's to it.
    fn default() -> Self {
        Self {
            frames: Rc::new(RefCell::new(vec![ScopeFrame::default()])),
        }
    }
}

pub struct ScopeGuard {
    scope: Symtbl,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        self.scope
            .frames
            .borrow_mut()
            .pop()
            .expect("Scope stack underflow");
    }
}

impl Symtbl {
    fn add_builtins(&self, tck: &mut Tck) {
        for builtin in &*builtins::BUILTINS {
            let ty = tck.insert_known(&builtin.sig);
            self.add_var(builtin.name, ty, false);
        }
    }
    fn push_scope(&self) -> ScopeGuard {
        self.frames.borrow_mut().push(ScopeFrame::default());
        ScopeGuard {
            scope: self.clone(),
        }
    }

    fn add_var(&self, var: Sym, ty: TypeId, mutable: bool) {
        self.frames
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .symbols
            .insert(var, (ty, mutable));
    }

    /// Checks whether the var exists in the currently alive scopes
    fn get_var_binding(&self, var: Sym) -> Option<(TypeId, bool)> {
        for scope in self.frames.borrow().iter().rev() {
            let v = scope.symbols.get(&var);
            if v.is_some() {
                return v.cloned();
            }
        }
        None
    }

    fn add_type(&self, name: Sym, ty: &Type) {
        self.frames
            .borrow_mut()
            .last_mut()
            .expect("Scope stack underflow")
            .types
            .insert(name, ty.to_owned());
    }

    fn get_type(&self, ty: Sym) -> Option<Type> {
        for scope in self.frames.borrow().iter().rev() {
            let v = scope.types.get(&ty);
            if v.is_some() {
                return v.cloned();
            }
        }
        None
    }
}

fn infer_lit(lit: &ast::Literal) -> TypeInfo {
    match lit {
        ast::Literal::Integer(_) => TypeInfo::Prim(PrimType::UnknownInt),
        ast::Literal::SizedInteger { bytes, .. } => TypeInfo::Prim(PrimType::SInt(*bytes)),
        ast::Literal::Bool(_) => TypeInfo::Prim(PrimType::Bool),
        //ast::Literal::EnumLit(nm, _) => TypeInfo::Named(*nm, vec![]),
    }
}

fn is_mutable_lvalue(symtbl: &Symtbl, expr: &hir::ExprNode) -> bool {
    use hir::Expr::*;
    match &*expr.e {
        Var { name } => {
            let (_ty, mutable) = symtbl.get_var_binding(*name).unwrap();
            mutable
        }
        StructRef { expr, .. } => is_mutable_lvalue(symtbl, expr),
        _ => false,
    }
}

/// TODO: This doesn't necessarily handle a func body as much
/// as a block with its own scope, which is what we actually want.
fn typecheck_func_body(
    name: Option<Sym>,
    tck: &mut Tck,
    symtbl: &Symtbl,
    signature: &hir::Signature,
    body: &[hir::ExprNode],
) -> Result<TypeId, TypeError> {
    /*
    trace!(
        "Typechecking function {:?} with signature {:?}",
        name, signature
    );
    */
    // Insert info about the function signature
    let mut params = vec![];
    for (_paramname, paramtype) in &signature.params {
        let p = tck.insert_known(paramtype);
        params.push(p);
    }
    let rettype = tck.insert_known(&signature.rettype);
    let tparams = signature
        .typeparams
        .iter()
        .map(|p| tck.insert_known(p))
        .collect();
    let f = tck.insert(TypeInfo::Func(params, rettype, tparams));

    // If we have a name (ie, are not a lambda), bind the function's type to its name
    // A gensym might make this easier/nicer someday, but this works for now.
    //
    // Note we do this *before* pushing the scope and checking its body,
    // so this will add the function's name to the outer scope.
    if let Some(n) = name {
        symtbl.add_var(n, f, false);
    }

    // Add params to function's scope
    let _guard = symtbl.push_scope();
    for (paramname, paramtype) in &signature.params {
        let p = tck.insert_known(paramtype);
        symtbl.add_var(*paramname, p, false);
    }

    // Typecheck body
    let last_expr_type = typecheck_exprs(tck, symtbl, rettype, body)?;
    /*
    trace!(
        "Unifying last expr...?  Is type {:?}, we want {:?}",
        last_expr_type, rettype
    );
    */
    tck.unify(symtbl, last_expr_type, rettype)?;

    for expr in body {
        let t = tck.get_expr_type(expr);
        tck.reconstruct(t).unwrap();
    }

    /*
    trace!(
        "Typechecked function {}, types are",
        name.unwrap_or(Sym::new("(lambda)"))
    );
    tck.print_types();
    */
    Ok(f)
}

/// Typechecks a list of expressions and returns the last return type,
/// or unit if the list is empty.  Does NOT push a new scope.
fn typecheck_exprs(
    tck: &mut Tck,
    symtbl: &Symtbl,
    func_rettype: TypeId,
    exprs: &[hir::ExprNode],
) -> Result<TypeId, TypeError> {
    for expr in exprs {
        typecheck_expr(tck, symtbl, func_rettype, expr)?;
    }
    let last_exprtype = exprs
        .last()
        .map(|last_expr| tck.get_expr_type(last_expr))
        // If we have an empty body, our rettype is unit
        .unwrap_or_else(|| tck.insert_known(&Type::unit()));
    Ok(last_exprtype)
}

fn typecheck_expr(
    tck: &mut Tck,
    symtbl: &Symtbl,
    func_rettype: TypeId,
    expr: &hir::ExprNode,
) -> Result<TypeId, TypeError> {
    use hir::Expr::*;
    let rettype: Result<_, TypeError> = match &*expr.e {
        Lit { val } => {
            let lit_type = infer_lit(val);
            let typeid = tck.insert(lit_type);
            tck.set_expr_type(expr, typeid);
            Ok(typeid)
        }
        Var { name } => {
            let (ty, _mutable) = symtbl
                .get_var_binding(*name)
                .unwrap_or_else(|| panic!("unbound var: {:?}", name));
            tck.set_expr_type(expr, ty);
            Ok(ty)
        }
        BinOp { op, lhs, rhs } => {
            let t1 = typecheck_expr(tck, symtbl, func_rettype, lhs)?;
            let t2 = typecheck_expr(tck, symtbl, func_rettype, rhs)?;
            let t3 = tck.bop_input_type(*op);
            tck.unify(symtbl, t1, t2)?;
            tck.unify(symtbl, t2, t3)?;
            // Figuring out the type of a binop output is kinda a pain.
            // The point is that if we know a type of an input we know its output.
            // But this is fundamentally at odds with unification because we don't
            // evaluate types greedily, we may not know what the types of the
            // inputs are until later in the program and that's fine!
            //
            // BUT we need to know that type to handle this differently for predicates and num ops,
            // because for predicates the output type is always known and is often NOT the same as
            // the input type, but for numerical ops the output type may NOT be known (may be Ref,
            // Unknown, etc if we haven't checked enough of the program yet), but once it IS known
            // it has to be the same as its inputs!
            //
            // This actually appears to work out just fine though, we make the
            // output type of the numerical ops a Ref to the input types, which
            // we know must be the same anyway.  Soooooo once we FIND the input
            // types, the output type snaps to it automatically.  Whew!
            let expected_output = tck.bop_output_type(*op, t3);
            tck.set_expr_type(expr, expected_output);
            Ok(expected_output)
        }
        UniOp { op, rhs } => {
            let expr_in = typecheck_expr(tck, symtbl, func_rettype, rhs)?;
            let expected_in = tck.uop_input_type(*op);
            tck.unify(symtbl, expr_in, expected_in)?;
            // Similar problem as BOp
            let expected_output = tck.uop_output_type(*op, expected_in);
            //tck.unify(symtbl, expr_in, expected_output)?;
            tck.set_expr_type(expr, expected_output);
            Ok(expected_output)
        }
        Block { body } => {
            let rettype = typecheck_exprs(tck, symtbl, func_rettype, body)?;
            tck.set_expr_type(expr, rettype);
            Ok(rettype)
        }
        Loop { body } => {
            let rettype = typecheck_exprs(tck, symtbl, func_rettype, body)?;
            tck.set_expr_type(expr, rettype);
            Ok(rettype)
        }
        Funcall {
            func,
            params,
            type_params,
        } => {
            // Oh, defined generics are "easy".
            // Each time I call a function I create new type
            // vars for its generic args.
            // Apparently that is the "instantiation".

            let func_type = typecheck_expr(tck, symtbl, func_rettype, func)?;
            // We know this will work because we require full function signatures
            // on our functions.
            let actual_func_type = tck.reconstruct(func_type)?;
            let actual_func_type_params = match &actual_func_type {
                Type::Func(_args, _rettype, typeparams) => {
                    //trace!("Calling function {:?} is {:?}", func, actual_func_type);
                    // So when we call a function we need to know what its
                    // type params are.  Then we bind those type parameters
                    // to things.
                    typeparams
                }
                other => panic!(
                    "Tried to call something not a function, it is a {:?}",
                    other
                ),
            };

            // Synthesize what we know about the function
            // from the call.
            let params_list: Result<Vec<_>, _> = params
                .iter()
                .map(|param| typecheck_expr(tck, symtbl, func_rettype, param))
                .collect();
            let params_list = params_list?;
            // Also synthesize what we know about the generics passed, if any,
            // or if there aren't any then figure them out from the params that
            // are passed.
            let type_param_vars = match (type_params.len(), actual_func_type_params.len()) {
                // all is good, no inference needed
                (0, 0) => vec![],
                // Infer type params from function call
                //
                // TODO: Should this be Ref(expected) rather than
                // Unknown?  For now let's just infer it from nothing
                // and see what happens.
                (0, _expected) => actual_func_type_params
                    .iter()
                    .map(|_| tck.insert(TypeInfo::Unknown))
                    .collect(),
                // Map the given type params to the ones the function expects.
                (given, expected) if given == expected => {
                    type_params.iter().map(|t| tck.insert_known(t)).collect()
                }
                // Wrong number of type params given.
                (given, expected) => {
                    let errmsg = format!(
                        "Tried to call func with {} type parameters but it expects {} of them!",
                        given, expected
                    );
                    return Err(errmsg.into());
                }
            };
            // We don't know what the expected return type of the function call
            // is yet; we make a type var that will get resolved when the enclosing
            // expression is.
            let rettype_var = tck.insert(TypeInfo::Unknown);
            // So this is now the inferred type that the function has been
            // called with.
            let funcall_var = tck.insert(TypeInfo::Func(
                params_list,
                rettype_var,
                type_param_vars.clone(),
            ));

            // Now I guess this is where we make a copy of the function
            // with new generic types.
            // Is this "instantiation"???
            // Yes it is.  Differentiate "type parameters", which are the
            // types a function takes as input (our `Generic` or `TypeParam`
            // things I suppose), from "type variables" which are the TypeId
            // we have to solve for.
            //
            // So we go through the generics the function declares and create
            // new type vars for each of them.
            //
            // We also create type variables for any type paramss we've been
            // given values to.
            //
            // Ok we *also* need to see if we can infer type params on function
            // calls that haven't been passed them explicitly.
            // like, if we have id(thing @T | @T) @T and we call id(false | Bool)
            // that is fine, if we call id(false) then we need to infer the
            // Bool there. We require either all or no type params for these
            // functions, so it's actually possible.
            let input_type_params = type_param_vars
                .iter()
                .zip(actual_func_type_params.iter())
                .filter_map(
                    |(given_type_param, fnexpr_type_param)| match fnexpr_type_param {
                        Type::Generic(nm) => Some((*nm, *given_type_param)),
                        _ => None,
                    },
                )
                .collect();
            let heck = tck.instantiate(&actual_func_type, Some(input_type_params));
            tck.unify(symtbl, heck, funcall_var)?;

            tck.set_expr_type(expr, rettype_var);
            Ok(rettype_var)
        }
        Let {
            varname,
            typename,
            init,
            mutable,
        } => {
            typecheck_expr(tck, symtbl, func_rettype, init)?;
            let init_expr_type = tck.get_expr_type(init);
            // Does our let decl have a type attached to it?
            let var_type = if let Some(t) = typename {
                tck.insert_known(t)
            } else {
                tck.insert(TypeInfo::Unknown)
            };
            tck.unify(symtbl, init_expr_type, var_type)?;

            // A `let` expr returns unit, not the type of `init`
            let unit_type = tck.insert(TypeInfo::Named(Sym::new("Tuple"), vec![]));
            tck.set_expr_type(expr, unit_type);

            symtbl.add_var(*varname, var_type, *mutable);
            Ok(var_type)
        }
        If { cases } => {
            // We know from the parser/lowering that we have at least one case,
            // if there's no declared "else" then the last case is `...else if true then ...` where
            // the body is unit, etc.
            // So we can just treat all the cases consistently.
            let rettype = tck.insert(TypeInfo::Unknown);
            let booltype = tck.insert(TypeInfo::Prim(PrimType::Bool));
            for (case, body) in cases {
                let case_type = typecheck_expr(tck, symtbl, func_rettype, case)?;
                tck.unify(symtbl, case_type, booltype)?;

                let _guard = symtbl.push_scope();
                let body_type = typecheck_exprs(tck, symtbl, func_rettype, body)?;
                tck.unify(symtbl, body_type, rettype)?;
            }
            tck.set_expr_type(expr, rettype);
            Ok(rettype)
        }
        EnumCtor {
            name,
            variant,
            value,
        } => {
            let enumtype = symtbl.get_type(*name).expect("Unknown enum type!");
            // Make sure type actually is an enum
            // Enums are terminal types so I guess we don't need to do
            // any inference or unification or such here?
            match &enumtype {
                Type::Enum(body) => {
                    // make sure variant actually exists
                    if !body.iter().any(|(ky, vl)| (*ky, *vl) == (*variant, *value)) {
                        return Err(format!(
                            "Enum {} variant {} does not match typedef",
                            name, variant
                        )
                        .into());
                    }
                }
                other => {
                    return Err(format!(
                        "Expected an enum of type {}, instead got {:?}",
                        name, other
                    )
                    .into())
                }
            }
            let typeid = tck.insert_known(&enumtype);

            tck.set_expr_type(expr, typeid);
            Ok(typeid)
        }
        TupleCtor { body } => {
            let body_types: Result<Vec<_>, _> = body
                .iter()
                .map(|expr| typecheck_expr(tck, symtbl, func_rettype, expr))
                .collect();
            let body_types = body_types?;
            let tuple_type = TypeInfo::Named(Sym::new("Tuple"), body_types);
            let typeid = tck.insert(tuple_type);
            tck.set_expr_type(expr, typeid);
            Ok(typeid)
        }
        TupleRef { expr: e, elt } => {
            typecheck_expr(tck, symtbl, func_rettype, e)?;
            let tuple_type = tck.get_expr_type(e);
            let field_type = tck.get_tuple_field_type(symtbl, tuple_type, *elt);
            trace!(
                "Heckin tuple ref...  Type of {:?}.{} is {:?}",
                e,
                elt,
                field_type
            );
            tck.set_expr_type(expr, field_type);

            Ok(field_type)
        }
        StructCtor { body } => {
            let body_types: Result<BTreeMap<_, _>, _> = body
                .iter()
                .map(|(name, expr)| {
                    // ? in map doesn't work too well...
                    trace!("Checking field {} expr {:?}", name, expr);
                    typecheck_expr(tck, symtbl, func_rettype, expr).map(|t| (*name, t))
                })
                .collect();
            trace!("Typechecking struct ctor: {:?}", body_types);
            let body_types = body_types?;
            let struct_type = TypeInfo::Struct(body_types);
            let typeid = tck.insert(struct_type);
            tck.set_expr_type(expr, typeid);
            Ok(typeid)
        }
        StructRef { expr: e, elt } => {
            let struct_type = typecheck_expr(tck, symtbl, func_rettype, e)?;
            //println!("Heckin struct...  Type of {:?} is {:?}", expr, struct_type);
            // struct_type is the type of the struct... but the
            // type of this structref expr is the type of the *field in the struct*.
            let struct_field_type = tck.get_struct_field_type(symtbl, struct_type, *elt);
            trace!(
                "Heckin struct ref...  Type of {:?}.{} is {:?}",
                expr,
                elt,
                struct_field_type
            );
            tck.set_expr_type(expr, struct_field_type);

            // TODO: Wait does this still need to be there?
            match tck.reconstruct(struct_type)? {
                Type::Struct(body, _elts) => Ok(tck.insert_known(&body[elt])),
                Type::Named(s, _args) => {
                    let hopefully_a_struct = symtbl.get_type(s).unwrap();
                    match hopefully_a_struct {
                        Type::Struct(body, _elts) => Ok(tck.insert_known(&body[elt])),
                        _other => Err("Yeah I know this is wrong bite me".to_string().into()),
                    }
                }
                other => Err(format!(
                    "Tried to get field named {} but it is an {:?}, not a struct",
                    elt, other
                )
                .into()),
            }
        }
        Assign { lhs, rhs } => {
            let rhs_type = typecheck_expr(tck, symtbl, func_rettype, rhs)?;
            // TODO: Check for invalid lvalues???  Or does the parser make
            // that impossible?  I forget.
            let lhs_type = typecheck_expr(tck, symtbl, func_rettype, lhs)?;
            if !is_mutable_lvalue(symtbl, lhs) {
                let _s = "assignment";
                /*
                return Err(TypeError::Mutability {
                    expr_name: s.into(),
                });
                */
                return Err("Mutability mismatch in 'assignment'".to_string().into());
            }
            tck.unify(symtbl, lhs_type, rhs_type)?;
            let unit = tck.insert_known(&Type::unit());
            tck.set_expr_type(expr, unit);
            Ok(unit)
        }
        Break => {
            // TODO: I guess the return type of `break` is unit?
            // Or Never?
            // Someday we might want to break with a return value, but not yet.
            let unit = tck.insert_known(&Type::unit());
            tck.set_expr_type(expr, unit);
            Ok(unit)
        }
        Lambda { signature, body } => {
            let t = typecheck_func_body(None, tck, symtbl, signature, body)?;
            tck.set_expr_type(expr, t);
            Ok(t)
        }
        Return { retval } => {
            // Does the type of the expression match the function's return type?
            let t = typecheck_expr(tck, symtbl, func_rettype, retval)?;
            tck.unify(symtbl, t, func_rettype)?;
            // The return type of `return` is Never because it will never happen
            let unit_typeid = tck.insert_known(&Type::Never);
            tck.set_expr_type(expr, unit_typeid);
            Ok(unit_typeid)
        }
        TypeCtor {
            name,
            type_params,
            body,
        } => {
            let named_type = symtbl.get_type(*name).expect("Unknown type constructor");
            trace!("Got type named {}: is {:?}", name, named_type);
            // Ok if we have declared type params we gotta instantiate them
            // to match the type's generics.
            //let type_param_names = named_type.get_generic_args();
            let type_param_names = named_type.get_type_params();
            assert_eq!(
                type_params.len(),
                type_param_names.len(),
                "Type '{}' expected params {:?} but got params {:?}",
                name,
                type_param_names,
                type_params
            );
            let tid = tck.instantiate(&named_type, None);
            trace!("Instantiated {:?} into {:?}", named_type, tid);

            let body_type = typecheck_expr(tck, symtbl, func_rettype, body)?;
            trace!("Expected type is {:?}, body type is {:?}", tid, body_type);
            tck.unify(symtbl, tid, body_type)?;
            trace!("Done unifying type ctor");
            // The type the expression returns
            let constructed_type = tck.insert_known(&Type::Named(*name, type_params.clone()));
            tck.set_expr_type(expr, constructed_type);
            Ok(constructed_type)
        }
        TypeUnwrap { expr: inner_expr } => {
            /// Takes a typeid that should be either a Ref or a Named type, and
            /// tries to follow it until it hits a Named type, and then looks it
            /// up in the symtbl.  (Once; if the named type contains another named
            /// type it will not recurse.)
            ///
            /// So it's a little like a weird partial `reconstruct()`, except it
            /// also tries to look up names in the symbol table.
            ///
            /// TODO: This is a kinda weird level of abstraction, maybe the wrong one
            /// since it returns some odd disconnected data.  It'd just be utility code,
            /// but it needs to recurse.  Writing it iteratively is even more cursed.
            fn try_resolve_named_type(
                tck: &Tck,
                symtbl: &Symtbl,
                t: TypeId,
            ) -> Result<(Sym, Vec<TypeId>), String> {
                let tinfo = &tck.get(&t);
                match tinfo {
                    // Lookup name in symbol table
                    TypeInfo::Named(nm, params) => Ok((*nm, params.clone())),
                    // Follow type ref
                    TypeInfo::Ref(id) => try_resolve_named_type(tck, symtbl, *id),
                    other => Err(format!(
                        "Can't resolve {:?} because it's not a named type!",
                        other
                    )),
                }
            }

            let body_type = typecheck_expr(tck, symtbl, func_rettype, inner_expr)?;
            let (nm, params) = try_resolve_named_type(tck, symtbl, body_type)?;
            let inner_type = symtbl
                .get_type(nm)
                .ok_or(format!("Named type {} is not known!", nm))?;
            trace!("Inner type is {:?}", inner_type);

            // The inner_type  is a concrete Type, not a TypeInfo that may have
            // unknowns, so we instantiate it to sub out any of its
            // type params with new unknowns.
            //
            // But then we have to bind those type params to
            // what we *know* about the type already...
            let type_param_names = inner_type.get_type_params();
            let known_type_params = type_param_names
                .iter()
                .cloned()
                .zip(params.iter().cloned())
                .collect();
            let inst_t = tck.instantiate(&inner_type, Some(known_type_params));

            tck.set_expr_type(expr, inst_t);
            return Ok(inst_t);
        }
        SumCtor {
            name,
            variant,
            body,
        } => {
            let named_type = symtbl
                .get_type(*name)
                .expect("Unknown sum type constructor");

            // This might be wrong, we can probably do it the other way around
            // like we do with TypeUnwrap: start by checking the inner expr type and make
            // sure it matches what we expect.  Generics might require that.
            match named_type {
                Type::Sum(sum_body, generics) => {
                    let variant_type = &sum_body[variant];
                    let variant_typeid = tck.insert_known(variant_type);
                    let body_type = typecheck_expr(tck, symtbl, func_rettype, body)?;
                    tck.unify(symtbl, variant_typeid, body_type)?;

                    // The expr is the type we expect, our return type is the
                    // sum type we conjure up
                    // TODO: Might be easier to have our compiler generate
                    // the TypeCtor for it?
                    let rettype = tck.insert_known(&Type::Named(*name, generics));
                    tck.set_expr_type(expr, rettype);
                    Ok(rettype)
                }
                _ => unreachable!("This code is compiler generated, should never happen!"),
            }
        }
        ArrayCtor { body } => {
            let len = body.len();
            // So if the body has len 0 we can't know what type it is.
            // So we create a new unknown and then try unifying it with
            // all the expressions in the body.
            let body_type = tck.insert(TypeInfo::Unknown);
            for expr in body {
                let expr_type = typecheck_expr(tck, symtbl, func_rettype, expr)?;
                tck.unify(symtbl, body_type, expr_type)?;
            }
            let arr_type = tck.insert(TypeInfo::Array(body_type, Some(len)));
            tck.set_expr_type(expr, arr_type);
            Ok(arr_type)
        }
        ArrayRef {
            expr: arr_expr,
            idx,
        } => {
            let expr_type = typecheck_expr(tck, symtbl, func_rettype, arr_expr)?;
            let idx_type = typecheck_expr(tck, symtbl, func_rettype, idx)?;
            let elt_type = tck.insert(TypeInfo::Unknown);
            let expected_expr_type = tck.insert(TypeInfo::Array(elt_type, None));
            let expected_idx_type = tck.insert(TypeInfo::Prim(PrimType::UnknownInt));
            tck.unify(symtbl, expr_type, expected_expr_type)?;
            tck.unify(symtbl, idx_type, expected_idx_type)?;
            tck.set_expr_type(expr, elt_type);
            Ok(elt_type)
        }
        Typecast { e: _, to: _ } => todo!(),
        Ref { expr: inner_expr } => {
            // We can take a reference to basically anything
            let expr_type = typecheck_expr(tck, symtbl, func_rettype, &*inner_expr)?;
            // NOT a TypeInfo::Ref !!!
            let ref_type = tck.insert(TypeInfo::Uniq(expr_type));
            tck.set_expr_type(expr, ref_type);
            Ok(ref_type)
        }
        Deref { expr: inner_expr } => {
            let expr_type = typecheck_expr(tck, symtbl, func_rettype, &*inner_expr)?;
            let inner_type = tck.insert(TypeInfo::Unknown);
            let desired_type = tck.insert(TypeInfo::Uniq(inner_type));
            tck.unify(symtbl, expr_type, desired_type)?;
            tck.set_expr_type(expr, inner_type);
            Ok(inner_type)
        }
    };
    if let Err(e) = rettype {
        panic!("Error typechecking expression {:?}: {}", expr, e);
    }
    rettype
}

fn predeclare_decls(tck: &mut Tck, symtbl: &mut Symtbl, decls: &[hir::Decl]) {
    use hir::Decl::*;
    for d in decls {
        match d {
            Function {
                name,
                signature,
                body: _,
            } => {
                // TODO: Kinda duplicated, not a huge fan.
                let mut params = vec![];
                for (_paramname, paramtype) in &signature.params {
                    let p = tck.insert_known(paramtype);
                    params.push(p);
                }
                let rettype = tck.insert_known(&signature.rettype);
                let typeparams = signature
                    .typeparams
                    .iter()
                    .map(|t| tck.insert_known(t))
                    .collect();
                let f = tck.insert(TypeInfo::Func(params, rettype, typeparams));
                symtbl.add_var(*name, f, false);
            }
            TypeDef {
                name,
                params,
                typedecl,
            } => {
                // Make sure that there are no unbound generics in the typedef
                // that aren't mentioned in the params.
                let generic_names: BTreeSet<Sym> = typedecl.get_type_params().into_iter().collect();
                let param_names: BTreeSet<Sym> = params.iter().cloned().collect();
                let difference: Vec<_> = generic_names.symmetric_difference(&param_names).collect();
                if !difference.is_empty() {
                    // gramble gramble strings
                    let differences: Vec<_> = difference
                        .into_iter()
                        .map(|sym| (*sym.val()).clone())
                        .collect();
                    let differences = differences.join(", ");
                    panic!("Error in typedef {}: Type params do not match generics mentioned in body.  Unmatched types: {}", name, differences);
                }

                // Remember that we know about a type with this name
                symtbl.add_type(*name, typedecl)
            }
            Const {
                name,
                typ: typename,
                init: _,
            } => {
                // We don't try typechecking the body yet.
                let ty = tck.insert_known(typename);
                symtbl.add_var(*name, ty, false)
            }
            Import { .. } => todo!(),
        }
    }
}

/// From example code:
/// "In reality, the most common approach will be to walk your AST, assigning type
/// terms to each of your nodes with whatever information you have available. You
/// will also need to call `engine.unify(x, y)` when you know two nodes have the
/// same type, such as in the statement `x = y;`."
pub fn typecheck(ast: &hir::Ir) -> Result<Tck, TypeError> {
    let mut t = Tck::default();
    let tck = &mut t;
    let symtbl = &mut Symtbl::default();
    symtbl.add_builtins(tck);
    predeclare_decls(tck, symtbl, &ast.decls);
    for decl in &ast.decls {
        use hir::Decl::*;

        match decl {
            Function {
                name,
                signature,
                body,
            } => {
                let t = typecheck_func_body(Some(*name), tck, symtbl, signature, body);
                t.unwrap_or_else(|e| {
                    error!("Error, type context is:");
                    tck.print_types();
                    panic!("Error while typechecking function {}:\n{}", name, e)
                });
            }
            TypeDef {
                name,
                params,
                typedecl,
            } => {
                // TODO: Handle recursive types properly?  Somehow.
                // Make sure that there are no unbound generics in the typedef
                // that aren't mentioned in the params.
                let generic_names: BTreeSet<Sym> = typedecl.get_type_params().into_iter().collect();
                let param_names: BTreeSet<Sym> = params.iter().cloned().collect();
                let difference: Vec<_> = generic_names.symmetric_difference(&param_names).collect();
                if !difference.is_empty() {
                    let differences: Vec<_> = difference
                        .into_iter()
                        .map(|sym| (*sym.val()).clone())
                        .collect();
                    panic!("Error in typedef {}: Type params do not match generics mentioned in body.  Unmatched types: {:?}", name, differences);
                }

                // Remember that we know about a type with this name
                symtbl.add_type(*name, typedecl)
            }
            Const {
                name: _,
                typ: typename,
                init,
            } => {
                // The init expression is typechecked in its own
                // scope, since it may theoretically be a `let` or
                // something that introduces new names inside it.
                let desired_type = tck.insert_known(typename);
                let init_type = {
                    let _guard = symtbl.push_scope();

                    typecheck_expr(tck, symtbl, desired_type, init).unwrap()
                };
                tck.unify(symtbl, desired_type, init_type)
                    .expect("Error typechecking const decl");
                //println!("Typechecked const {}, type is {:?}", name, init_type);
            }
            Import { .. } => todo!(),
        }
    }
    // Print out toplevel symbols
    for (name, id) in &symtbl.frames.borrow().last().unwrap().symbols {
        info!(
            "value {} type is {:?}",
            name,
            tck.reconstruct(id.0).map(|t| t.get_name())
        );
    }
    trace!("Type variables:");
    for (id, info) in &tck.vars {
        trace!("  ${} = {info:?}", id.0);
    }
    Ok(t)
}
