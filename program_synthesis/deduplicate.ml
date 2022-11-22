open Core
open Antireduce
open Transforms
open Domains
module S = Yojson.Safe
module SU = Yojson.Safe.Util

let (_ : unit) =
  let j = S.from_channel In_channel.stdin in
  let domain = SU.to_string @@ SU.member "domain" j in
  let executed_programs_dir =
    SU.to_string @@ SU.member "executed_programs_dir" j
  in
  let to_discard, kept = find_duplicates domain executed_programs_dir in
  List.iter to_discard ~f:(List.iter ~f:(fun c -> Caml.Sys.remove c.path)) ;
  S.to_channel Out_channel.stdout
  @@ `Assoc
       [ ("kept", `List (List.map kept ~f:(fun c -> `String c.filename)))
       ; ( "discarded"
         , `List
             (List.map to_discard ~f:(fun l ->
                  `List (List.map l ~f:(fun c -> `String c.filename)) ) ) ) ]
