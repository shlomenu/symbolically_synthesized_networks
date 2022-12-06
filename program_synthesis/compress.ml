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
  let n_cores = SU.to_int @@ SU.member "n_cores" j in
  let parse = Domains.parser_of_domain domain j in
  let executed_programs_dir =
    SU.to_string @@ SU.member "executed_programs_dir" j
  in
  let programs, paths, file_contents =
    load_frontier_from parse executed_programs_dir frontier
  in
  let verbose = SU.to_int @@ SU.member "verbosity" j in
  compression_verbosity := verbose ;
  let dsl', programs' =
    compress ~inlining:true ~iterations ~n_cores ~primitive_size_penalty
      ~dsl_size_penalty ~n_beta_inversions ~top_i ~beam_size
      (request_of_domain domain) dsl programs
  in
  if List.length dsl'.library > List.length dsl.library then (
    S.to_file next_dsl_file @@ yojson_of_dsl dsl' ;
    overwrite_frontier programs' paths file_contents ;
    S.to_channel Out_channel.stdout (`Assoc [("rewritten", `Bool true)]) )
  else S.to_channel Out_channel.stdout @@ `Assoc [("rewritten", `Bool false)]
