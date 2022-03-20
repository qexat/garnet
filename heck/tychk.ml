(* Bidirectional typechecking for higher-rank polymorphism in ocaml,
   without polymorphic subtyping.
   From MBones's https://gist.github.com/mb64/87ac275c327ea923a8d587df7863d8c7

   Compile with:
  $ ocamlfind ocamlc -package angstrom,stdio -linkpkg tychk.ml -o tychk
Example use:
  $ ./tychk <<EOF
  > let f = (fun x -> x) : forall a. a -> a
  > in f f
  > EOF
  input : forall a. a -> a
*)

module AST = struct
  type ty =
    | TNamed of string
    | TFun of ty * ty
    | TForall of string * ty
  type exp =
    | Var of string
    | App of exp * exp
    | Annot of exp * ty
    | Lam of string * exp
    | Let of string * exp * exp
end

module Infer = struct
  type lvl = int

  type ty =
    | TNamed of string
    | TFun of ty * ty
    | TForall of string * ty
    (* a type var bound in the context. "Complete and Easy" uses α, β for these *)
    | TBoundVar of lvl
    | THole of hole ref
  and hole =
    (* scope: when you fill in the hole, all its bound vars need to have
        lvl < scope
    *)
    | Empty of { scope: lvl }
    | Filled of ty
  (* An important invariant about the scope of holes: it only ever includes
     types in the context, never types bound from 'TForall'.

     It's possible to not have this invariant, but it makes everything a *lot*
     harder.
   *)

  (* the type_names are just for pretty-printing *)
  type ctx = { type_names: string list; lvl: lvl; env: (string * ty) list }

  let initial_ctx: ctx = { type_names = []; lvl = 0; env = [] }

  exception TypeError of string

  let print_ty (ctx: ctx) ty =
    let parens p s = if p then "(" ^ s ^ ")" else s in
    let rec helper p = function
      | TNamed n -> n
      | TFun(a, b) -> parens p (helper true a ^ " -> " ^ helper false b)
      | TForall(n, a) -> parens p ("forall " ^ n ^ ". " ^ helper false a)
      | TBoundVar lvl -> List.nth ctx.type_names (ctx.lvl - lvl - 1)
      | THole hole -> match !hole with
          | Empty { scope = lvl } -> Printf.sprintf "?[at lvl %d]" lvl
          | Filled a -> helper p a
    in helper false ty

  let rec ast_ty_to_ty (ast_ty: AST.ty) = match ast_ty with
    | TNamed n -> TNamed n
    | TFun(a, b) -> TFun (ast_ty_to_ty a, ast_ty_to_ty b)
    | TForall(n, a) -> TForall(n, ast_ty_to_ty a)

  let add_ty_to_ctx (name: string) (ctx: ctx): ctx =
    let rec freshen_name n =
      if List.mem n ctx.type_names then freshen_name (n ^ "'") else n in
    { type_names = freshen_name name :: ctx.type_names
    ; lvl = ctx.lvl + 1
    ; env = ctx.env }

  let add_var_to_ctx (name: string) (ty: ty) (ctx: ctx): ctx =
    { type_names = ctx.type_names
    ; lvl = ctx.lvl
    ; env = (name, ty) :: ctx.env }

  let lookup_var (name: string) (ctx: ctx) =
    match List.assoc_opt name ctx.env with
    | Some ty -> ty
    | None -> raise (TypeError ("variable " ^ name ^ " not in scope"))

  let subst name replacement ty =
    let rec helper = function
      | TNamed n ->
          if n = name then replacement else TNamed n
      | TFun (a, b) -> TFun (helper a, helper b)
      | TForall (n, a) ->
          if n = name then TForall (n, a) else TForall (n, helper a)
      | TBoundVar l -> TBoundVar l
      | THole hole ->
          match !hole with
            | Empty _ -> THole hole
            | Filled a -> helper a
    in helper ty

  (* fill in a TForall(name, ty) with a fresh hole *)
  let instantiate (ctx: ctx) name ty =
    let newHole = ref (Empty { scope = ctx.lvl }) in
    subst name (THole newHole) ty

  (* when filling in a hole, a few things need to be checked:
     - occurs check: check that you aren't making recursive types
     - scope check: check that you aren't using bound vars outside its scope
  *)
  let unify_hole_prechecks (ctx: ctx) (hole: hole ref) (scope: lvl) ty =
    let rec helper = function
      | TNamed _ -> ()
      | TFun (a, b) -> helper a; helper b
      | TForall (n, a) -> helper a
      | TBoundVar l ->
          if l >= scope
          then raise (TypeError ("type variable " ^ print_ty ctx (TBoundVar l) ^ " escaping its scope"))
      | THole h ->
          if h = hole
          then raise (TypeError "occurs check: can't make infinite type")
          else match !h with
            | Empty { scope = l } -> if l > scope then h := Empty { scope = scope }
            | Filled a -> helper a
    in helper ty

  let rec unify (ctx: ctx) a b =
    let raise_error () = raise (TypeError
      ("mismatch between " ^ print_ty ctx a ^ " and " ^ print_ty ctx b)) in
    match a, b with
    | THole hole_a, _ -> unify_hole_ty ctx hole_a b
    | _, THole hole_b -> unify_hole_ty ctx hole_b a
    | TNamed name_a, TNamed name_b ->
        if name_a <> name_b then raise_error ()
    | TFun (a1, a2), TFun (b1, b2) -> unify ctx a1 b1; unify ctx a2 b2
    | TForall (name_a, ty_a), TForall (name_b, ty_b) ->
        (* alpha equivalence: forall a. a -> a is equal to forall b. b -> b *)
        let new_ctx = add_ty_to_ctx name_a ctx in
        let ty_a' = subst name_a (TBoundVar ctx.lvl) ty_a in
        let ty_b' = subst name_b (TBoundVar ctx.lvl) ty_b in
        unify new_ctx ty_a' ty_b'
    | TBoundVar lvl_a, TBoundVar lvl_b ->
        if lvl_a <> lvl_b then raise_error ()
    | _ -> raise_error ()

  and unify_hole_ty (ctx: ctx) hole ty = match !hole with
    | Filled a -> unify ctx a ty
    | Empty { scope = scope } ->
        if ty = THole hole
        then ()
        else (unify_hole_prechecks ctx hole scope ty; hole := Filled ty)

  (* The mutually-recursive typechecking functions *)

  (* check a term has a type *)
  let rec check (ctx: ctx) (term: AST.exp) (ty: ty) = match term, ty with
    | _, THole { contents = Filled a } -> check ctx term a
    | _, TForall(n, a) ->
        check (add_ty_to_ctx n ctx) term (subst n (TBoundVar ctx.lvl) a)
    | Lam(var, body), TFun(a, b) ->
        check (add_var_to_ctx var a ctx) body b
    | Let(var, exp, body), a ->
        let exp_ty = infer ctx exp in
        check (add_var_to_ctx var exp_ty ctx) body a
    | _ ->
        let inferred_ty = infer ctx term in
        (* this is where it could optionally use the fancy <: intead of unification *)
        unify ctx inferred_ty ty

  (* infer the type of a term *)
  and infer (ctx: ctx) (term: AST.exp) = match term with
    | Var var -> lookup_var var ctx
    | Annot(e, ast_ty) ->
        let ty = ast_ty_to_ty ast_ty in
        (check ctx e ty; ty)
    | App(f, arg) ->
        let f_ty = infer ctx f in
        apply ctx f_ty arg
    | Lam(var, body) ->
        let arg_ty = THole (ref (Empty { scope = ctx.lvl })) in
        let res_ty = infer (add_var_to_ctx var arg_ty ctx) body in
        TFun(arg_ty, res_ty)
    | Let(var, exp, body) ->
        let exp_ty = infer ctx exp in
        infer (add_var_to_ctx var exp_ty ctx) body

  (* helper function for checking App(f, arg) *)
  (* it has a weird symbol in "Complete and Easy" *)
  and apply (ctx: ctx) (f_ty: ty) (arg: AST.exp) = match f_ty with
    | TFun(a, b) -> (check ctx arg a; b)
    | TForall(n, a) -> apply ctx (instantiate ctx n a) arg
    | THole { contents = Filled a } -> apply ctx a arg
    | THole ({ contents = Empty { scope } } as hole) ->
        let a = THole (ref (Empty { scope })) in
        let b = THole (ref (Empty { scope })) in
        hole := Filled (TFun(a, b));
        check ctx arg a;
        b
    | _ -> raise (TypeError (print_ty ctx f_ty ^ " is not a function type"))

end

module Parser = struct
  open AST
  open Angstrom (* parser combinators library *)

  let keywords = ["forall"; "let"; "in"; "fun"]

  let whitespace = take_while (String.contains " \n\t")
  let lexeme a = a <* whitespace
  let ident = lexeme (
    let is_ident_char c =
      c = '_' || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') in
    let* i = take_while is_ident_char in
    if String.length i > 0 then return i else fail "expected ident")

  let str s = lexeme (string s) *> return ()
  let name =
    let* i = ident in
    if List.mem i keywords then fail (i ^ " is a keyword") else return i
  let keyword k =
    let* i = ident in
    if i = k then return () else fail ("expected " ^ k)
  let parens p = str "(" *> p <* str ")"

  let ty = fix (fun ty ->
    let simple_ty = parens ty <|> lift (fun n -> TNamed n) name in
    let forall_ty =
      let+ () = keyword "forall"
      and+ names = many1 name
      and+ () = str "."
      and+ a = ty in
      List.fold_right (fun n a -> TForall(n, a)) names a in
    let fun_ty =
      let+ arg_ty = simple_ty
      and+ () = str "->"
      and+ res_ty = ty in
      TFun(arg_ty, res_ty) in
    forall_ty <|> fun_ty <|> simple_ty <?> "type")

  let exp = fix (fun exp ->
    let atomic_exp = parens exp <|> lift (fun n -> Var n) name in
    let make_app (f::args) =
      List.fold_left (fun f arg -> App(f,arg)) f args in
    let simple_exp = lift make_app (many1 atomic_exp) in
    let annot_exp =
      let+ e = simple_exp
      and+ annot = option (fun e -> e)
        (lift (fun t e -> Annot(e,t)) (str ":" *> ty)) in
      annot e in
    let let_exp =
      let+ () = keyword "let"
      and+ n = name
      and+ () = str "="
      and+ e = exp
      and+ () = keyword "in"
      and+ body = exp in
      Let(n, e, body) in
    let fun_exp =
      let+ () = keyword "fun"
      and+ args = many1 name
      and+ () = str "->"
      and+ body = exp in
      List.fold_right (fun arg body -> Lam(arg, body)) args body in
    let_exp <|> fun_exp <|> annot_exp <?> "expression")

  let parse (s: string) =
    match parse_string ~consume:All (whitespace *> exp) s with
    | Ok e -> e
    | Error msg -> failwith msg
end

let main () =
  let stdin = Stdio.In_channel.(input_all stdin) in
  let exp = Parser.parse stdin in
  let () = print_endline "parsed" in
  let open Infer in
  let ctx = initial_ctx in
  let ty = infer ctx exp in
  print_endline ("input : " ^ print_ty ctx ty)

let () = main ()
