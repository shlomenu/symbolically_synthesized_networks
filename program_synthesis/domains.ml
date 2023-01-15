open Core
open Antireduce
open Dsl
module Graph = Antireduce_graphs

let explore = function
  | "graph" ->
      Graph.explore
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let dsl_insensitive_parser_of_domain = function
  | "graph" ->
      Graph.parse_program_exn ~primitives:Graph.all_primitives'
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let dsl_sensitive_parser_of_domain = function
  | "graph" ->
      fun dsl ->
        Graph.parse_program_exn ~primitives:(fun _ ->
            Hashtbl.of_alist_exn (module String)
            @@ List.map dsl.library ~f:(fun ent ->
                   (ent.stitch_name, primitive_of_entry ent) ) )
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let request_of_domain = function
  | "graph" ->
      Graph.graph_transform
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized_domain: %s" name_of_domain