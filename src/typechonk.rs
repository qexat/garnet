
use crate::hir;
use crate::scope;
use crate::{TypeDef, TypeSym, VarSym, INT};

#[derive(Debug, Clone)]
pub enum TypeError {
    UnknownType(VarSym),
    UnknownVar(VarSym)
}

impl TypeError {
    fn format(&self) -> String {
        format!("{:?}", self)
    }
}

impl std::error::Error for TypeError {}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format())
    }
}

/// Inferred type.
///
/// A type that may be unknown or incomplete
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IType {
    /// Signed integer with the given number of bytes
    SInt(u8),
    UnknownInt,
    /// Boolean, obv's
    Bool,
    /// Never is a real type, I guess!
    Never,
    /// The type of a lambda is its signature
    Lambda(Vec<Self>, Box<Self>),
    /// A struct
    Struct(Vec<(VarSym, Self)>),
    /// Tuple.  The types inside it may or may not be fully known I guess
    Tuple(Vec<Self>),
    /// A type name that we don't know anything else about.
    Named(VarSym),
}

impl IType {
    fn unit() -> Self {
        IType::Tuple(vec![])
    }

    fn i16() -> Self {
        IType::SInt(2)
    }

    fn i32() -> Self {
        IType::SInt(4)
    }

    fn i64() -> Self {
        IType::SInt(8)
    }

    fn bool() -> Self {
        IType::Bool
    }

    fn from_signature(params: &[(VarSym, TypeSym)], rettype: TypeSym) -> Self {
        let type_params = params.iter().map(|(_name, t)| IType::from(*t)).collect();
        let function_type = IType::Lambda(type_params, Box::new(IType::from(rettype)));
        function_type
    }
}

impl From<&TypeDef> for IType {
    fn from(t: &TypeDef) -> Self {
        use TypeDef::*;
        use IType as I;
        match t {
            SInt(i) => I::SInt(*i),
            UnknownInt => I::UnknownInt,
            Bool => I::Bool,
            Tuple(vals) => {
                I::Tuple(vals.iter().map(|x| INT.fetch_type(*x)).map(|x|From::from(&*x)).collect())
            },
            Never => I::Never,
            Lambda(args, rettype) => todo!(),
            Named(vsym) => I::Named(*vsym),
            Struct {
                name,
                fields,
                typefields,
            } => todo!()
        }
    }
}


impl From<TypeSym> for IType {
    fn from(t: TypeSym) -> Self {
        Self::from(&*INT.fetch_type(t))
    }
}

/// Known type.
///
/// A type def that can only represent things
/// we know are valid and complete.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KType {
    /// Signed integer with the given number of bytes
    SInt(u8),
    /// Boolean, obv's
    Bool,
    /// Tuple.  The types inside it may or may not be fully known I guess
    Tuple(Vec<Self>),
    /// Never is a real type, I guess!
    Never,
    /// The type of a lambda is its signature
    Lambda(Vec<Self>, Box<Self>),
    /// A struct
    Struct(VarSym, Vec<(VarSym, Self)>),
}

/// A variable binding
#[derive(Debug, Clone)]
pub struct VarBinding {
    name: VarSym,
    type_: IType,
    mutable: bool,
}

///
#[derive(Default)]
struct Symtbl {
    vars: scope::Symbols<VarSym, VarBinding>,
    /// Bindings for typedefs
    types: scope::Symbols<VarSym, IType>,
}

impl Symtbl {
    /// Create new symbol table with some built-in functions.
    ///
    /// Also see the prelude defined in `backend/rust.rs`
    pub fn new() -> Self {
        let mut x = Self::default();
        // We add a built-in function for printing, currently.
        {
            let name = INT.intern("__println");
            let typesym = IType::Lambda(vec![IType::i32()], Box::new(IType::unit()));
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_bool");
            let typesym = IType::Lambda(vec![IType::bool()], Box::new(IType::unit()));
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_i64");
            let typesym = IType::Lambda(vec![IType::i64()], Box::new(IType::unit()));
            x.add_var(name, typesym, false);
        }
        {
            let name = INT.intern("__println_i16");
            let typesym = IType::Lambda(vec![IType::i16()], Box::new(IType::unit()));
            x.add_var(name, typesym, false);
        }
        x
    }

    fn add_type(&mut self, name: VarSym, typedef: IType) {
        self.types.add(name, typedef);
    }

    fn get_typedef(&mut self, name: VarSym) -> Option<IType> {
        self.types.get(name)
    }

    /// Looks up a typedef and if it is `Named` try to keep looking
    /// it up until we find the actual concrete type.  Returns None
    /// if it can't.
    ///
    /// TODO: This is weird, currently necessary for structs though.
    fn follow_typedef(&mut self, name: VarSym) -> Option<IType> {
        match self.types.get(name) {
            Some(IType::Named(vsym)) => self.follow_typedef(vsym),
            Some(other) => Some(other),
            None => None,
        }
    }

    /// Returns Ok if the type exists, or a TypeError of the appropriate
    /// kind if it does not.
    fn type_exists(&mut self, tsym: IType) -> Result<(), TypeError> {
        match tsym {
            IType::Named(name) => {
                if self.types.get(name).is_some() {
                    Ok(())
                } else {
                    Err(TypeError::UnknownType(name))
                }
            }
            // Primitive type
            _ => Ok(()),
        }
    }

    /// Add a variable to the top level of the scope.
    /// Shadows the old var if it already exists in that scope.
    fn add_var(&mut self, name: VarSym, typedef: IType, mutable: bool) {
        let binding = VarBinding {
            name,
            type_: typedef,
            mutable,
        };
        self.vars.add(name, binding);
    }

    /// Get the type of the given variable, or an error
    fn get_var_type(&self, name: VarSym) -> Result<IType, TypeError> {
        Ok(self.get_binding(name)?.type_)
    }

    /// Get the binding of the given variable, or an error
    fn get_binding(&self, name: VarSym) -> Result<VarBinding, TypeError> {
        if let Some(binding) = self.vars.get(name) {
            return Ok(binding);
        }
        Err(TypeError::UnknownVar(name))
    }

    fn binding_exists(&self, name: VarSym) -> bool {
        self.get_binding(name).is_ok()
    }

    fn push_scope(&mut self) -> scope::ScopeGuard<VarSym, VarBinding> {
        self.vars.push_scope()
    }

    fn push_type_scope(&mut self) -> scope::ScopeGuard<VarSym, IType> {
        self.types.push_scope()
    }
}

/// Scan through all decl's and add any bindings to the symbol table,
/// so we don't need to do anything with forward references.
fn predeclare_decl(symtbl: &mut Symtbl, decl: &hir::Decl<()>) {
    match decl {
        hir::Decl::Function {
            name, signature, ..
        } => {
            if symtbl.binding_exists(*name) {
                panic!("Tried to redeclare function {}!", INT.fetch(*name));
            }
            // Add function to global scope
            let type_params = signature.params.iter().map(|(_name, t)| IType::from(*t)).collect();
            let function_type = IType::Lambda(type_params, Box::new(IType::from(signature.rettype)));
            symtbl.add_var(*name, function_type, false);
        }
        hir::Decl::Const { name, typename, .. } => {
            if symtbl.binding_exists(*name) {
                panic!("Tried to redeclare const {}!", INT.fetch(*name));
            }
            symtbl.add_var(*name, IType::from(*typename), false);
        }
        hir::Decl::TypeDef { name, typedecl } => {
            // Gotta make sure there's no duplicate declarations
            // This kinda has to happen here rather than in typeck()
            if symtbl.get_typedef(*name).is_some() {
                panic!("Tried to redeclare type {}!", INT.fetch(*name));
            }
            symtbl.add_type(*name, IType::from(*typedecl));
        }
        hir::Decl::StructDef {
            name,
            fields,
            typefields,
        } => {
            let typ = TypeDef::Struct {
                name: *name,
                fields: fields.clone(),
                typefields: typefields.clone(),
            };
            if symtbl.get_typedef(*name).is_some() {
                panic!("Tried to redeclare struct {}!", INT.fetch(*name));
            }
            let typesym = INT.intern_type(&typ);
            //println!("Adding struct of type {}", INT.fetch(*name));
            symtbl.add_type(*name, IType::from(typesym))
        }
        hir::Decl::Constructor { name, signature } => {
            {
                if symtbl.get_var_type(*name).is_ok() {
                    panic!(
                        "Aieeee, redeclaration of function/type constructor named {}",
                        INT.fetch(*name)
                    );
                }
                let function_type = IType::from_signature(&signature.params, signature.rettype);
                symtbl.add_var(*name, function_type, false);
            }

            // Also we need to add a deconstructor function.  This is kinda a placeholder, but,
            // should work for now.
            {
                let deconstruct_name = INT.intern(format!("{}_unwrap", INT.fetch(*name)));
                if symtbl.get_var_type(deconstruct_name).is_ok() {
                    panic!(
                        "Aieeee, redeclaration of function/type destructure named {}",
                        INT.fetch(deconstruct_name)
                    );
                }
                let type_params = vec![IType::from(signature.rettype)];
                let rettype = IType::from(signature.params[0].1);
                let function_type = IType::Lambda(type_params, Box::new(rettype));
                symtbl.add_var(deconstruct_name, function_type, false);
            }
        }
    }
}


/// Try to walk through the IR and return one with types that have some kind of
/// guesstimated info attached to them.
pub fn typecheck1(ir: hir::Ir<()>) -> Result<hir::Ir<IType>, TypeError> {
    let symtbl = &mut Symtbl::new();
    ir.decls.iter().for_each(|d| predeclare_decl(symtbl, d));
    let checked_decls = ir
        .decls
        .into_iter()
        .map(|decl| typecheck_decl(symtbl, decl))
        .collect::<Result<Vec<hir::Decl<IType>>, TypeError>>()?;
    Ok(hir::Ir {
        decls: checked_decls,
    })
}

fn typecheck_decl(symtbl: &mut Symtbl, d: hir::Decl<()>) -> Result<hir::Decl<IType>, TypeError> {
    match d {
    match decl {
        hir::Decl::Function {
            name,
            signature,
            body,
        } => {
            // Push scope, typecheck and add params to symbol table
            let _g = symtbl.push_scope();
            for (pname, ptype) in signature.params.iter() {
                symtbl.add_var(*pname, *ptype, false);
            }

            // This is squirrelly; basically, we want to return unit
            // if the function has no body, otherwise return the
            // type of the last expression.
            //
            // If there's a return expr, we just return the Never type
            // for it and it all shakes out to work.
            let typechecked_exprs = typecheck_exprs(symtbl, body, Some(signature.rettype))?;
            // Ok, so we *also* need to walk through all the expressions
            // and look for any "return" exprs (or later `?`/`try` exprs
            // also) and see make sure the return types match.
            let last_expr_type = last_type_of(&typechecked_exprs);
            if let Some(t) = infer_type(last_expr_type, signature.rettype) {
                let inferred_exprs = reify_last_types(last_expr_type, t, typechecked_exprs);
                Ok(hir::Decl::Function {
                    name,
                    signature,
                    body: inferred_exprs,
                })
            } else {
                //if !type_matches(signature.rettype, last_expr_type) {
                Err(TypeError::Return {
                    fname: name,
                    got: last_expr_type,
                    expected: signature.rettype,
                })
            }
        }
        hir::Decl::Const {
            name,
            typename,
            init,
        } => {
            // Make sure the const's type exists
            symtbl.type_exists(typename)?;
            Ok(hir::Decl::Const {
                name,
                typename,
                init: typecheck_expr(symtbl, init, None)?,
            })
        }
        // Ok, we are declaring a new type.  We need to make sure that the typedecl
        // it's using is real.  We've already checked to make sure it's not a duplicate.
        hir::Decl::TypeDef { name, typedecl } => {
            // Make sure the body of the typedef is a real type.
            symtbl.type_exists(typedecl)?;
            Ok(hir::Decl::TypeDef { name, typedecl })
        }
        hir::Decl::StructDef {
            name,
            fields,
            typefields,
        } => {
            let typedecl = INT.intern_type(&TypeDef::Struct {
                name: name,
                fields: fields.clone(),
                typefields: BTreeSet::new(),
            });
            symtbl.type_exists(typedecl)?;
            Ok(hir::Decl::StructDef {
                name,
                fields: fields.clone(),
                typefields: typefields.clone(),
            })
        }
        // Don't need to do anything here since we generate these in the lowering
        // step and have already verified no names clash.
        hir::Decl::Constructor { name, signature } => {
            Ok(hir::Decl::Constructor { name, signature })
        }
    }

    }
}
