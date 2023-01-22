open Core
open Antireduce
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
            @@ List.map ~f:(fun ent ->
                   (Dsl_entry.stitch_name ent, Dsl_entry.to_primitive ent) )
            @@ Dsl.library dsl )
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let stitch_invention_parser_of_domain = function
  | "graph" ->
      fun dsl ->
        Graph.parse_stitch_invention_exn ~primitives:(fun _ ->
            Hashtbl.of_alist_exn (module String)
            @@ List.map ~f:(fun ent ->
                   (Dsl_entry.stitch_name ent, Dsl_entry.to_primitive ent) )
            @@ Dsl.library dsl )
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let request_of_domain = function
  | "graph" ->
      Graph.graph_transform
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized_domain: %s" name_of_domain