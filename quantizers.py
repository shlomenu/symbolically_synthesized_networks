import os
import json
from typing import List
from hashlib import blake2b

import dgl
import pygraphviz as pgv
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from restart_manager import RestartManager
from utilities import (EXPLORE_BINARY_LOC, COLORS, explore,
                       compress, invoke_binary_with_json)

DSL_DIR = "dsls"
REPRESENTATIONS_DIR = "representations"
VISUALIZATION_DIR = "visualizations"


def node_tensor_tuple_of_json(graph_json):
    edges_1, edges_2 = [], []
    for [node_id, neighbors] in graph_json["forward_edges"]:
        for neighbor in neighbors:
            if neighbor is not None:
                edges_1.append(node_id)
                edges_2.append(neighbor)
    for [node_id, neighbors] in graph_json["backward_edges"]:
        for neighbor in neighbors:
            if neighbor is not None:
                edges_1.append(node_id)
                edges_2.append(neighbor)
    edges_1 = th.tensor(edges_1)
    edges_2 = th.tensor(edges_2)
    return (edges_1, edges_2)


def node_colors_of_json(graph_json, device):
    colors = th.zeros(len(graph_json["nodes"]),
                      graph_json["max_color"],
                      device=device)
    for [node_id, color] in graph_json["nodes"]:
        colors[node_id, color] = 1
    return colors


def dgl_graph_and_feats_of_json(graph_json, device):
    """
    Assumes that json represents graph with no isolated nodes
    and containing at least one edge.
    """
    graph = dgl.graph(node_tensor_tuple_of_json(graph_json), device=device)
    colors = node_colors_of_json(graph_json, device)
    return graph, colors


def dgl_heterograph_of_json(graph_json, device):
    graph, colors = dgl_graph_and_feats_of_json(graph_json, device)
    graph.ndata["color"] = colors
    return graph


def dgl_homograph_of_json(graph_json, device):
    graph, colors = dgl_graph_and_feats_of_json(graph_json, device)
    return dgl.add_self_loop(dgl.to_homogeneous(graph)), colors


def pygraphviz_graph_of_json(graph_json, caption):
    graph = pgv.AGraph(directed=True, labelloc="t", label=caption)
    for [node_id, color] in graph_json["nodes"]:
        graph.add_node(node_id, color=COLORS[color])
    forward_edges = set()
    for [node_id, neighbors] in graph_json["forward_edges"]:
        for neighbor in neighbors:
            if neighbor is not None:
                forward_edges.add((node_id, neighbor))
    backward_edges = set()
    for [node_id, neighbors] in graph_json["backward_edges"]:
        for neighbor in neighbors:
            if neighbor is not None:
                backward_edges.add((node_id, neighbor))
    for (n_1, n_2) in forward_edges:
        if (n_2, n_1) in backward_edges:
            graph.add_edge((n_1, n_2), arrowhead="vee")
        else:
            raise Exception("corrupted task json")
    graph.layout()
    return graph


class GraphQuantizer(nn.Module):

    def __init__(self,
                 E,
                 C,
                 dsl_name,
                 heads=2,
                 depth=1,
                 idleness_limit=8,
                 max_color=10):
        super().__init__()
        self.max_color = max_color
        self.name_of_domain = "graph"
        self.representations_save_path = os.path.join(self.name_of_domain,
                                                      REPRESENTATIONS_DIR)
        self.dsl_save_path = os.path.join(self.name_of_domain, DSL_DIR)
        self.visualization_save_path = os.path.join(self.name_of_domain,
                                                    VISUALIZATION_DIR)

        self.dsl_name = dsl_name
        self.representations = sorted(
            os.listdir(self.representations_save_path))
        self.restart_manager = RestartManager(
            idleness_limit, len(self.representations))
        self.attn_up = dgl.nn.GATv2Conv(in_feats=self.max_color,
                                        out_feats=C,
                                        num_heads=1,
                                        attn_drop=0.5)
        self.attns = nn.ModuleList([
            dgl.nn.GATv2Conv(in_feats=C,
                             out_feats=C,
                             num_heads=heads,
                             attn_drop=0.5) for _ in range(depth)
        ])
        self.ffs = nn.ModuleList([
            nn.Sequential(*[
                nn.LayerNorm(C),
                nn.Linear(C, 2 * C),
                nn.GELU(),
                nn.Linear(2 * C, C),
                Rearrange("n h c -> n (h c)", h=heads, c=C),
                nn.Linear(heads * C, C)
            ]) for _ in range(depth)
        ])
        self.attn_seq = dgl.nn.GATv2Conv(in_feats=C,
                                         out_feats=C,
                                         num_heads=E,
                                         attn_drop=0.5)
        self.ff_seq = nn.Sequential(*[
            nn.LayerNorm(C),
            nn.Linear(C, 2 * C),
            nn.GELU(),
            nn.Linear(2 * C, C),
            Rearrange("n e c -> n (e c)", e=E, c=C)
        ])
        self.pool = dgl.nn.GlobalAttentionPooling(nn.Linear(E * C, 1))
        self.to_out = nn.Sequential(*[nn.Unflatten(1, (E, C)), nn.Tanh()])

    def __len__(self):
        return len(self.representations)

    def forward(self, selections):
        graphs, feats, restarts, filenames = self._fetch(
            selections.tolist(), selections.device)
        feats = rearrange(self.attn_up(graphs, feats), "n ... -> n (...)")
        for attn, ff in zip(self.attns, self.ffs):
            feats = ff(attn(graphs, feats))
        return self.to_out(
            self.pool(graphs,
                      self.ff_seq(self.attn_seq(graphs,
                                                feats)))), restarts, filenames

    def _fetch(self, selections: List[int], device):
        restarts = self.restart_manager.find_restarts(selections)
        graphs, feats, filenames = [], [], []
        for selection in selections:
            filename = self.representations[selection]
            with open(os.path.join(self.representations_save_path, filename)) as f:
                repr = json.load(f)
            graph, feat = dgl_homograph_of_json(repr["output"], device)
            graphs.append(graph)
            feats.append(feat)
            filenames.append(filename)
        return dgl.batch(graphs), th.cat(feats, dim=0), restarts, filenames

    def explore(self,
                exploration_timeout,
                program_size,
                eval_timeout=.1,
                attempts=1):
        resp = explore(self.name_of_domain,
                       os.path.join(self.dsl_save_path,
                                    f"{self.dsl_name}.json"),
                       self.representations_save_path,
                       exploration_timeout,
                       program_size,
                       eval_timeout=eval_timeout,
                       attempts=attempts,
                       max_color=self.max_color)
        self._load_representations(resp["prev_files"], resp["cur_files"])
        return resp["new"], resp["replaced"]

    def compress(self,
                 frontier,
                 next_dsl_name,
                 load_new_dsl=True,
                 iterations=1,
                 beam_size=3,
                 top_i=3,
                 dsl_size_penalty=0.,
                 primitive_size_penalty=1.,
                 n_beta_inversions=2,
                 verbosity=0):
        resp = compress(frontier,
                        self.name_of_domain,
                        os.path.join(self.dsl_save_path,
                                     f"{self.dsl_name}.json"),
                        os.path.join(self.dsl_save_path,
                                     f"{next_dsl_name}.json"),
                        self.representations_save_path,
                        iterations=iterations,
                        beam_size=beam_size,
                        top_i=top_i,
                        dsl_size_penalty=dsl_size_penalty,
                        primitive_size_penalty=primitive_size_penalty,
                        n_beta_inversions=n_beta_inversions,
                        verbosity=verbosity,
                        max_color=self.max_color)
        if load_new_dsl:
            if resp["rewritten"]:
                self.dsl_name = next_dsl_name
                self._load_representations(
                    resp["prev_files"], resp["cur_files"])
        return resp["rewritten"]

    def _load_representations(self, prev_files, cur_files):
        self.repl = {prev_file: cur_file for prev_file,
                     cur_file in zip(prev_files, cur_files)}
        self.representations = [(self.repl[file] if file in self.repl else file)
                                for file in self.representations]
        reprs = set(self.representations)
        self.representations.extend(
            set(os.listdir(self.representations_save_path)) - reprs)
        for code in range(len(self)):
            self.restart_manager.add_code(code)

    def clear_visualizations(self):
        for filename in os.listdir(os.path.join(self.visualization_save_path)):
            if filename.endswith(".svg"):
                os.remove(os.path.join(self.visualization_save_path, filename))

    def visualize(self, to_visualize=None):
        if to_visualize is None:
            to_visualize = os.listdir(self.representations_save_path)
        named_graphs = {}
        for filename in to_visualize:
            if filename.endswith(".json"):
                with open(os.path.join(self.representations_save_path,
                                       filename)) as f:
                    repr = json.load(f)
                print(f"loading.. {filename}")
                named_graphs[filename[:-5]
                             ] = pygraphviz_graph_of_json(repr["output"], filename[:-5])
        for name, graph in named_graphs.items():
            graph.draw(
                os.path.join(self.visualization_save_path, name + ".svg"))


class NullQuantizer:

    loss = F.mse_loss

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
