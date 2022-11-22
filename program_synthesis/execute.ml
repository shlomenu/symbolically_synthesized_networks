open Core
open Antireduce
open Domains
module S = Yojson.Safe
module SU = Yojson.Safe.Util

let () =
  let j = S.from_channel In_channel.stdin in
  let timeout = SU.to_number @@ SU.member "eval_timeout" j in
  let attempts = SU.to_int @@ SU.member "attempts" j in
  let domain = SU.to_string @@ SU.member "domain" j in
  let dsl =
    Dsl.dsl_of_yojson @@ S.from_file @@ SU.to_string @@ SU.member "dsl_file" j
  in
  execute_and_save domain ~timeout ~attempts ~dsl j
