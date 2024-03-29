open Core
open Antireduce
module S = Yojson.Safe
module SU = Yojson.Safe.Util

let () =
  let j = S.from_channel In_channel.stdin in
  let exploration_timeout = SU.to_number @@ SU.member "exploration_timeout" j in
  let eval_timeout = SU.to_number @@ SU.member "eval_timeout" j in
  let attempts = SU.to_int @@ SU.member "attempts" j in
  let domain = SU.to_string @@ SU.member "domain" j in
  let parse = Domains.dsl_insensitive_parser_of_domain domain in
  let representations_dir = SU.to_string @@ SU.member "representations_dir" j in
  let max_diff = SU.to_float @@ SU.member "max_diff" j in
  let frontier =
    List.map ~f:SU.to_string @@ SU.to_list @@ SU.member "frontier" j
  in
  let frontier, _, _ =
    Frontier.load_representations_from parse representations_dir frontier
  in
  let dsl =
    let weighted_dsl =
      Dsl.t_of_yojson @@ S.from_file @@ SU.to_string @@ SU.member "dsl_file" j
    in
    let uniform () =
      Dsl.of_primitives weighted_dsl.state_type
      @@ Dsl.to_primitives weighted_dsl
    in
    if List.is_empty frontier then uniform ()
    else
      Factorization.inside_outside (uniform ())
        (Domains.request_of_domain domain)
        frontier
      |> fst |> Dsl.rescale ~max_diff
  in
  let next_dsl_file = SU.to_string @@ SU.member "next_dsl_file" j in
  let max_novel_representations =
    SU.to_int @@ SU.member "max_novel_representations" j
  in
  let program_size_limit = SU.to_int @@ SU.member "program_size_limit" j in
  S.to_file next_dsl_file @@ Dsl.yojson_of_t dsl ;
  let n_new, n_replaced, replacements, n_enumerated, max_ll =
    Domains.explore domain ~exploration_timeout ~max_novel_representations
      ~program_size_limit ~eval_timeout ~attempts ~dsl ~representations_dir j
  in
  S.to_channel Out_channel.stdout
  @@ `Assoc
       [ ("new", yojson_of_int n_new)
       ; ("replaced", yojson_of_int n_replaced)
       ; ( "replacements"
         , yojson_of_list (yojson_of_list yojson_of_string)
           @@ List.map replacements ~f:(fun (prev, cur) -> [prev; cur]) )
       ; ("n_enumerated", yojson_of_int n_enumerated)
       ; ("max_description_length", yojson_of_float (-.max_ll)) ]
