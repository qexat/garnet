//! Garnet compiler guts.

pub mod ast;
pub mod parser;

use std::collections::HashMap;

/// A concrete type that has been fully inferred
#[derive(Debug)]
pub enum Type {
    Num,
    Bool,
    Func(Vec<Type>, Box<Type>),
}

/// A identifier to uniquely refer to our type terms
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct TypeId(usize);

/// Information about a type term
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum TypeInfo {
    // No information about the type of this type term
    Unknown,
    // This type term is the same as another type term
    Ref(TypeId),
    // This type term is definitely a number
    Num,
    // This type term is definitely a bool
    Bool,
    // This type term is definitely a function
    Func(Vec<TypeId>, TypeId),
}

impl TypeInfo {
    fn get_primitive_type(s: &str) -> Option<TypeInfo> {
        match s {
            "I32" => Some(TypeInfo::Num),
            "Bool" => Some(TypeInfo::Bool),
            //"Never" => Some(TypeInfo::Never),
            _ => None,
        }
    }
}

/// Type checking engine
#[derive(Default)]
struct Tck {
    /// Used to generate unique IDs
    id_counter: usize,
    vars: HashMap<TypeId, TypeInfo>,
}

impl Tck {
    /// Create a new type term with whatever we have about its type
    pub fn insert(&mut self, info: TypeInfo) -> TypeId {
        // Generate a new ID for our type term
        self.id_counter += 1;
        let id = TypeId(self.id_counter);
        self.vars.insert(id, info);
        id
    }

    /// Make the types of two type terms equivalent (or produce an error if
    /// there is a conflict between them)
    pub fn unify(&mut self, a: TypeId, b: TypeId) -> Result<(), String> {
        use TypeInfo::*;
        match (self.vars[&a].clone(), self.vars[&b].clone()) {
            // Follow any references
            (Ref(a), _) => self.unify(a, b),
            (_, Ref(b)) => self.unify(a, b),

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

            // Primitives are trivial to unify
            (Num, Num) => Ok(()),
            (Bool, Bool) => Ok(()),

            // When unifying complex types, we must check their sub-types. This
            // can be trivially implemented for tuples, sum types, etc.
            (Func(a_i, a_o), Func(b_i, b_o)) => {
                for (arg_a, arg_b) in a_i.iter().zip(b_i) {
                    self.unify(*arg_a, arg_b)?;
                }
                self.unify(a_o, b_o)
            }

            // If no previous attempts to unify were successful, raise an error
            (a, b) => Err(format!("Conflict between {:?} and {:?}", a, b)),
        }
    }

    /// Attempt to reconstruct a concrete type from the given type term ID. This
    /// may fail if we don't yet have enough information to figure out what the
    /// type is.
    pub fn reconstruct(&self, id: TypeId) -> Result<Type, String> {
        use TypeInfo::*;
        match &self.vars[&id] {
            Unknown => Err(format!("Cannot infer")),
            Ref(id) => self.reconstruct(*id),
            Num => Ok(Type::Num),
            Bool => Ok(Type::Bool),
            Func(i, o) => {
                let is: Result<Vec<Type>, String> =
                    i.iter().copied().map(|arg| self.reconstruct(arg)).collect();
                Ok(Type::Func(is?, Box::new(self.reconstruct(*o)?)))
            }
        }
    }
}

/// Basic symbol table that maps names to type ID's
#[derive(Default)]
struct Symtbl {
    symbols: Vec<HashMap<String, TypeId>>,
}

fn infer_lit(lit: &ast::Literal) -> TypeInfo {
    match lit {
        ast::Literal::Integer(_) => TypeInfo::Num,
    }
}

fn typecheck_expr(tck: &mut Tck, expr: &ast::Expr) {
    use ast::Expr::*;
    match expr {
        Lit { val } => {
            let lit_type = infer_lit(val);
            let typeid = tck.insert(lit_type);
            todo!("save typeid");
        }
        Var { name } => todo!(),
        Let {
            varname,
            typename,
            init,
        } => todo!(),
        Lambda { signature, body } => todo!(),
        Funcall { func, params } => todo!(),
    }
}

// # Example usage
// In reality, the most common approach will be to walk your AST, assigning type
// terms to each of your nodes with whatever information you have available. You
// will also need to call `engine.unify(x, y)` when you know two nodes have the
// same type, such as in the statement `x = y;`.
pub fn typecheck(ast: &ast::Ast) {
    let mut tck = Tck::default();
    let mut symtbl = Symtbl::default();
    symtbl.symbols.push(HashMap::new());
    for decl in &ast.decls {
        use ast::Decl::*;

        match decl {
            Function {
                name,
                signature,
                body,
            } => {
                // Insert info about the function signature
                let mut params = vec![];
                for (_paramname, paramtype) in &signature.params {
                    let p = tck.insert(paramtype.clone());
                    params.push(p);
                }
                let rettype = tck.insert(signature.rettype.clone());
                let f = tck.insert(TypeInfo::Func(params, rettype));
                symtbl
                    .symbols
                    .last_mut()
                    .unwrap()
                    .insert(name.to_string(), f);

                // Typecheck body
                for expr in body {
                    typecheck_expr(&mut tck, expr);
                }
            }
        }
    }
    // Print out toplevel symbols
    for (name, id) in symtbl.symbols.last().unwrap() {
        println!("fn {} type is {:?}", name, tck.reconstruct(*id));
    }
}
/*
pub fn typecheck2() {
    let mut tck = Tck::default();

    // A function with an unknown input
    let i = tck.insert(TypeInfo::Unknown);
    let o = tck.insert(TypeInfo::Num);
    let f0 = tck.insert(TypeInfo::Func(vec![i, o.clone()], o));

    // A function with an unknown output
    let i = tck.insert(TypeInfo::Bool);
    let o = tck.insert(TypeInfo::Unknown);
    let f1 = tck.insert(TypeInfo::Func(vec![i, o.clone()], o));

    // Unify them together...
    tck.unify(f0, f1).unwrap();

    // An instance of the aforementioned function
    let thing = tck.insert(TypeInfo::Ref(f1));

    // ...and compute the resulting type
    println!("Final type = {:?}", tck.reconstruct(thing));
}
*/

pub fn compile(filename: &str, src: &str) -> Vec<u8> {
    //typecheck2();
    let mut parser = parser::Parser::new(filename, src);
    let ast = parser.parse();
    typecheck(&ast);
    //let res = format!("AST:\n{:#?}", ast);
    //res.as_bytes().to_owned()
    vec![]
}
