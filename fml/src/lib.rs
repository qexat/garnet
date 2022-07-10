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

/// A complete-ish description of a type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeDef {
    Type(i32),
    TypeConstr(i32, Vec<TypeDef>),
    TypeVar(Option<Box<TypeDef>>),
}

fn not_equal(a: &TypeDef, b: &TypeDef) {
    panic!("Not equal: {:?} {:?}", a, b)
}

fn infinite_type(a: &Option<Box<TypeDef>>, b: &TypeDef) {
    panic!("infinite type: {:?} {:?}", a, b)
}

fn occurs(v_a: &Option<Box<TypeDef>>, thing: &TypeDef) -> bool {
    match thing {
        TypeDef::Type(_) => false,
        TypeDef::TypeConstr(_, ts_b) => ts_b.iter().any(|v| occurs(v_a, v)),
        TypeDef::TypeVar(v_b) => match v_b {
            None => v_a == v_b,
            Some(t) => occurs(v_a, t),
        },
    }
}

fn deref(t: &TypeDef) -> TypeDef {
    match t {
        TypeDef::TypeVar(Some(t2)) => deref(t2),
        TypeDef::TypeVar(None) => t.clone(),
        _ => t.clone(),
    }
}

fn unify(a: &TypeDef, b: &TypeDef) {
    let a = deref(a);
    let b = deref(b);
    use TypeDef::*;
    match (&a, &b) {
        (Type(t_a), Type(t_b)) if t_a != t_b => not_equal(&a, &b),
        (TypeConstr(c_a, ts_a), TypeConstr(c_b, ts_b)) => {
            if c_a != c_b {
                not_equal(&a, &b)
            } else {
                for (t1, t2) in ts_a.iter().zip(ts_b.iter()) {
                    unify(t1, t2)
                }
            }
        }
        (TypeVar(v_a), TypeVar(v_b)) => {
            if v_a != v_b {
                // v_a := Some(b)
                // This makes borrowing sticky.
                // See https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=174ca95a8b938168764846e97d5e9a2c
                todo!("v_a := Some(b)")
            }
        }
        (TypeVar(v_a), _) => {
            if occurs(v_a, &b) {
                infinite_type(v_a, &b)
            } else {
                todo!("v_a := Some(b)")
            }
        }
        (_, TypeVar(v_b)) => unify(&b, &a),
        (_, _) => not_equal(&a, &b),
    }
}

fn apply1(f: &TypeDef, x: &TypeDef) -> TypeDef {
    let a = todo!("TypeVar (ref None)");
    let b = todo!("TypeVar (ref None)");
    // Check that f is a function
    unify(f, &TypeDef::TypeConstr(0, vec![a, b]));
    // Unify x with the argument of f
    unify(&a, x);
    x.clone()
}

//pub fn typecheck(ast: &ast::Ast) {
pub fn typecheck() {
    let fn_tycon = 0;
    let list_tycon = 1;
    let int_tycon = 2;
    let t_double = TypeDef::TypeConstr(0, vec![TypeDef::Type(2), TypeDef::Type(2)]);
    // forall a b. (a -> b) -> List a -> List b
    let t_map = {
        let a = todo!("TypeVar (ref None)");
        let b = todo!("TypeVar (ref None)");
        use TypeDef::TypeConstr as TC;
        TC(
            0,
            vec![
                TC(0, vec![a, b]),
                TC(0, vec![TC(1, vec![a]), TC(1, vec![b])]),
            ],
        )
    };
    let example = apply1(&t_map, &t_double);
    println!("Example is {:?}", example);
}

pub fn compile(filename: &str, src: &str) -> Vec<u8> {
    /*
    let ast = {
        let mut parser = parser::Parser::new(filename, src);
        parser.parse()
    };
    */
    typecheck();
    //let res = format!("AST:\n{:#?}", ast);
    //res.as_bytes().to_owned()
    vec![]
}
