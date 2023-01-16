open Core
open Antireduce
open Frontier
open Dsl
open Program
module S = Yojson.Safe
module SU = Yojson.Safe.Util

let () =
  let j = S.from_channel In_channel.stdin in
  let domain = SU.to_string @@ SU.member "domain" j in
  let dsl' =
    SU.member "invented_primitives" j
    |> list_of_yojson (list_of_yojson string_of_yojson)
    |> List.fold
         ~init:
           ( dsl_of_yojson @@ S.from_file @@ SU.to_string
           @@ SU.member "dsl_file" j )
         ~f:(fun dsl' inv ->
           let parse =
             Domains.stitch_invention_parser_of_domain domain dsl' j
           in
           let invented_primitive =
             match inv with
             | [name; body] ->
                 invention name @@ parse body
             | _ ->
                 failwith
                   "incorporate_stitch: improperly formatted invented \
                    primitives: expected list of lists of [name, body]"
           in
           dedup_dsl_of_primitives dsl'.state_type
             (invented_primitive :: primitives_of_dsl dsl') )
  in
  let replacements =
    let parse = Domains.dsl_sensitive_parser_of_domain domain dsl' j in
    let path_of =
      Fn.compose
        (repr_path @@ SU.to_string @@ SU.member "representations_dir" j)
        parse
    in
    let paths, file_contents, cur_programs =
      SU.member "replacements" j
      |> list_of_yojson (list_of_yojson string_of_yojson)
      |> List.map ~f:(function
           | [prev; cur] ->
               let path = path_of prev in
               (path, S.from_file path, parse cur)
           | _ ->
               failwith
                 "incorporate_stitch: improperly formatted replacements: \
                  expected list of lists of [original_program, \
                  rewritten_program]" )
      |> List.unzip3
    in
    overwrite_representations cur_programs paths file_contents
  in
  S.to_file (SU.to_string @@ SU.member "next_dsl_file" j) (yojson_of_dsl dsl') ;
  S.to_channel Out_channel.stdout
  @@ `Assoc
       [ ("new_dsl_mass", yojson_of_int dsl'.mass)
       ; ( "replacements"
         , yojson_of_list (yojson_of_list yojson_of_string)
           @@ List.map replacements ~f:(fun (prev, cur) -> [prev; cur]) ) ]
