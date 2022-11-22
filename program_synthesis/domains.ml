open Core
module Graph = Antireduce_graphs

let execute_and_save = function
  | "graph" ->
      Graph.execute_and_save
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let find_duplicates = function
  | "graph" ->
      Graph.find_duplicates
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let parser_of_domain = function
  | "graph" ->
      Graph.parse_program_exn
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized domain: %s" name_of_domain

let request_of_domain = function
  | "graph" ->
      Graph.graph_transform
  | name_of_domain ->
      failwith @@ Format.sprintf "unrecognized_domain: %s" name_of_domain