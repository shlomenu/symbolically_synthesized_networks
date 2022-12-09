open Core
open Antireduce
open Domains
module S = Yojson.Safe
module SU = Yojson.Safe.Util

let () =
  let j = S.from_channel In_channel.stdin in
  let exploration_timeout = SU.to_number @@ SU.member "exploration_timeout" j in
  let eval_timeout = SU.to_number @@ SU.member "eval_timeout" j in
  let attempts = SU.to_int @@ SU.member "attempts" j in
  let domain = SU.to_string @@ SU.member "domain" j in
  let dsl =
    Dsl.dsl_of_yojson @@ S.from_file @@ SU.to_string @@ SU.member "dsl_file" j
  in
  let representations_dir = SU.to_string @@ SU.member "representations_dir" j in
  let program_size = SU.to_int @@ SU.member "program_size" j in
  explore domain ~exploration_timeout ~eval_timeout ~attempts ~dsl
    ~representations_dir ~size:program_size j
