open Core
open Antireduce
module Graphs = Antireduce_graphs

let explore = function
  | "graph" ->
      Graphs.Ext.explore
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let dsl_insensitive_parser_of_domain = function
  | "graph" ->
      Graphs.Ext.parse_program_exn ~primitives:Graphs.Eval.all_primitives
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let dsl_sensitive_parser_of_domain = function
  | "graph" ->
      fun dsl ->
        Graphs.Ext.parse_program_exn
          ~primitives:
            ( Dsl.library dsl
            |> List.map ~f:(fun ent ->
                   (Dsl_entry.stitch_name ent, Dsl_entry.to_primitive ent) )
            |> Hashtbl.of_alist_exn (module String) )
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let stitch_invention_parser_of_domain = function
  | "graph" ->
      fun dsl ->
        Graphs.Ext.parse_stitch_invention_exn
          ~primitives:
            ( Dsl.library dsl
            |> List.map ~f:(fun ent ->
                   (Dsl_entry.stitch_name ent, Dsl_entry.to_primitive ent) )
            |> Hashtbl.of_alist_exn (module String) )
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let request_of_domain = function
  | "graph" ->
      Graphs.Eval.graph_transform
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized_domain: %s" name_of_domain