open Core
open Antireduce
open Domains

let (_ : unit) =
  let j = S.from_channel In_channel.stdin in
  let domain = SU.to_string @@ SU.member "domain" j in
  let executed_programs_dir =
    SU.to_string @@ SU.member "executed_programs_dir" j
  in
  let redundant, best = find_duplicates domain executed_programs_dir j in
  S.to_channel Out_channel.stdout
  @@ `Assoc
       [ ("best", `List (List.map best ~f:(fun c -> `String c.filename)))
       ; ( "redundant"
         , `List
             (List.map redundant ~f:(fun l ->
                  `List (List.map l ~f:(fun c -> `String c.filename)) ) ) ) ]
