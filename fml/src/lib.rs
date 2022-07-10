//! Garnet compiler guts.
//#![deny(missing_docs)]
/*
 * type t =
    Type of int
  | Type_Constructor of int * t list
  | Type_Variable of t option ref

exception Not_Equal of t * t
exception Infinite_Type of t option ref * t

let rec occurs v_a = function
  | Type _ -> false
  | Type_Constructor (_, ts_b) -> List.exists (occurs v_a) ts_b
  | Type_Variable v_b ->
    match !v_b with
    | None -> v_a == v_b
    | Some t -> occurs v_a t

let rec deref t =
  match t with
  | Type_Variable v_t ->
    begin match !v_t with
    | None -> t
    | Some t2 -> deref t2
    end
  | _ -> t

let rec unify a b =
  let a = deref a in
  let b = deref b in
  match a, b with
    | Type t_a, Type t_b ->
      if t_a != t_b then
        raise (Not_Equal (a, b))
    | Type_Constructor (c_a, ts_a), Type_Constructor (c_b, ts_b) ->
      if c_a != c_b then
        raise (Not_Equal (a, b))
      else
        List.iter2 unify ts_a ts_b
    | Type _, Type_Constructor _ -> raise (Not_Equal (a, b))
    | Type_Constructor _, Type _ -> raise (Not_Equal (a, b))
    | Type_Variable v_a, Type_Variable v_b ->
      if not (v_a == v_b) then
        v_a := Some b;
    | Type_Variable v_a, _ ->
      if occurs v_a b then
        raise (Infinite_Type (v_a, b))
      else
        v_a := Some b
    | _, Type_Variable v_b -> unify b a


let fn_tycon = 0
let list_tycon = 1
let int_type = 2

(* forall a b. (a -> b) -> List a -> List b *)
let t_map =
  let a = Type_Variable (ref None) in
  let b = Type_Variable (ref None) in
  Type_Constructor (0,
   [Type_Constructor (0, [a; b]);
    Type_Constructor (0,
     [Type_Constructor (1, [a]);
      Type_Constructor (1, [b])])])

let t_double = Type_Constructor (0, [Type 2; Type 2])

let apply f x =
  let a = Type_Variable (ref None) in
  let b = Type_Variable (ref None) in
  (* Check that f is a function. *)
  unify f (Type_Constructor (0, [a; b]));
  (* Unify x with the argument of f. *)
  unify a x;
  (* Return the result type *)
  x

(* List Int -> List Int *)
(* Type_Constructor (0, [Type 2; Type 2]) *)
let example = apply t_map t_double
 */

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

#[derive(Default)]
struct Engine {
    id_counter: usize, // Used to generate unique IDs
    vars: HashMap<TypeId, TypeInfo>,
}

impl Engine {
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

#[derive(Default)]
struct Symtbl {
    symbols: Vec<HashMap<String, TypeId>>,
}

// # Example usage
// In reality, the most common approach will be to walk your AST, assigning type
// terms to each of your nodes with whatever information you have available. You
// will also need to call `engine.unify(x, y)` when you know two nodes have the
// same type, such as in the statement `x = y;`.
pub fn typecheck(ast: &ast::Ast) {
    let mut engine = Engine::default();
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
                let mut params = vec![];
                for (_paramname, paramtype) in &signature.params {
                    let p = engine.insert(paramtype.clone());
                    params.push(p);
                }
                let rettype = engine.insert(signature.rettype.clone());
                let f = engine.insert(TypeInfo::Func(params, rettype));
                symtbl
                    .symbols
                    .last_mut()
                    .unwrap()
                    .insert(name.to_string(), f);
            }
        }
    }
    for (name, id) in symtbl.symbols.last().unwrap() {
        println!("fn {} type is {:?}", name, engine.reconstruct(*id));
    }
}
pub fn typecheck2() {
    let mut engine = Engine::default();

    // A function with an unknown input
    let i = engine.insert(TypeInfo::Unknown);
    let o = engine.insert(TypeInfo::Num);
    let f0 = engine.insert(TypeInfo::Func(vec![i, o.clone()], o));

    // A function with an unknown output
    let i = engine.insert(TypeInfo::Bool);
    let o = engine.insert(TypeInfo::Unknown);
    let f1 = engine.insert(TypeInfo::Func(vec![i, o.clone()], o));

    // Unify them together...
    engine.unify(f0, f1).unwrap();

    // An instance of the aforementioned function
    let thing = engine.insert(TypeInfo::Ref(f1));

    // ...and compute the resulting type
    println!("Final type = {:?}", engine.reconstruct(thing));
}

pub fn compile(filename: &str, src: &str) -> Vec<u8> {
    typecheck2();
    let mut parser = parser::Parser::new(filename, src);
    let ast = parser.parse();
    typecheck(&ast);
    //let res = format!("AST:\n{:#?}", ast);
    //res.as_bytes().to_owned()
    vec![]
}
