open Core
open Antireduce
module S = Yojson.Safe
module SU = Yojson.Safe.Util

let () =
  let j = S.from_channel In_channel.stdin in
  let frontier =
    List.map ~f:SU.to_string @@ SU.to_list @@ SU.member "frontier" j
  in
  let domain = SU.to_string @@ SU.member "domain" j in
  let dsl =
    Dsl.t_of_yojson @@ S.from_file @@ SU.to_string @@ SU.member "dsl_file" j
  in
  let next_dsl_file = SU.to_string @@ SU.member "next_dsl_file" j in
  let iterations = SU.to_int @@ SU.member "iterations" j in
  let beam_size = SU.to_int @@ SU.member "beam_size" j in
  let n_invention_sizes =
    SU.to_option SU.to_int @@ SU.member "n_invention_sizes" j
  in
  let n_exactly_scored = SU.to_int @@ SU.member "n_exactly_scored" j in
  let primitive_size_penalty =
    SU.to_float @@ SU.member "primitive_size_penalty" j
  in
  let dsl_size_penalty = SU.to_float @@ SU.member "dsl_size_penalty" j in
  let n_beta_inversions = SU.to_int @@ SU.member "n_beta_inversions" j in
  let parse = Domains.dsl_insensitive_parser_of_domain domain j in
  let representations_dir = SU.to_string @@ SU.member "representations_dir" j in
  let frontier, paths, file_contents =
    Frontier.load_representations_from parse representations_dir frontier
  in
  let invention_name_prefix =
    SU.to_string @@ SU.member "invention_name_prefix" j
  in
  let verbose = SU.to_int @@ SU.member "verbosity" j in
  Compression.compression_verbosity := verbose ;
  let dsl', frontier' =
    Compression.compress ~invention_name_prefix ~inlining:true ~iterations
      ~n_beta_inversions ~n_invention_sizes ~n_exactly_scored
      ~primitive_size_penalty ~dsl_size_penalty ~beam_size
      ~request:(Domains.request_of_domain domain)
      ~dsl ~frontier
  in
  if List.length dsl'.library > List.length dsl.library then (
    S.to_file next_dsl_file @@ Dsl.yojson_of_t dsl' ;
    let replacements =
      Frontier.overwrite_representations frontier' paths file_contents
    in
    S.to_channel Out_channel.stdout
      (`Assoc
        [ ("success", yojson_of_bool true)
        ; ("next_dsl_mass", yojson_of_int dsl'.mass)
        ; ( "replacements"
          , yojson_of_list (yojson_of_list yojson_of_string)
            @@ List.map replacements ~f:(fun (prev, cur) -> [prev; cur]) ) ] ) )
  else
    S.to_channel Out_channel.stdout @@ `Assoc [("success", yojson_of_bool false)]
