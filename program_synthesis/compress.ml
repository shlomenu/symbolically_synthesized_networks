open Core
open Antireduce
open Dsl
open Compression
open Domains
module S = Yojson.Safe
module SU = Yojson.Safe.Util

let (_ : unit) =
  let j = S.from_channel In_channel.stdin in
  let frontier =
    Array.of_list @@ List.map ~f:SU.to_string @@ SU.to_list
    @@ SU.member "frontier" j
  in
  let domain = SU.to_string @@ SU.member "domain" j in
  let dsl =
    dsl_of_yojson @@ S.from_file @@ SU.to_string @@ SU.member "dsl_file" j
  in
  let next_dsl_file = SU.to_string @@ SU.member "next_dsl_file" j in
  let iterations = SU.to_int @@ SU.member "iterations" j in
  let beam_size = SU.to_int @@ SU.member "beam_size" j in
  let top_i = SU.to_int @@ SU.member "top_i" j in
  let dsl_size_penalty = SU.to_number @@ SU.member "dsl_size_penalty" j in
  let primitive_size_penalty =
    SU.to_number @@ SU.member "primitive_size_penalty" j
  in
  let n_beta_inversions = SU.to_int @@ SU.member "n_beta_inversions" j in
  let parse = Domains.parser_of_domain domain j in
  let representations_dir = SU.to_string @@ SU.member "representations_dir" j in
  let frontier, paths, file_contents =
    load_representations_from parse representations_dir frontier
  in
  let verbose = SU.to_int @@ SU.member "verbosity" j in
  compression_verbosity := verbose ;
  let dsl', frontier' =
    compress ~inlining:true ~iterations ~primitive_size_penalty
      ~dsl_size_penalty ~n_beta_inversions ~top_i ~beam_size
      ~request:(request_of_domain domain) ~dsl ~frontier
  in
  if List.length dsl'.library > List.length dsl.library then (
    S.to_file next_dsl_file @@ yojson_of_dsl dsl' ;
    let prev_files, cur_files =
      overwrite_representations frontier' paths file_contents
    in
    S.to_channel Out_channel.stdout
      (`Assoc
        [ ("rewritten", `Bool true)
        ; ("prev_files", yojson_of_list yojson_of_string prev_files)
        ; ("cur_files", yojson_of_list yojson_of_string cur_files) ] ) )
  else S.to_channel Out_channel.stdout @@ `Assoc [("rewritten", `Bool false)]
