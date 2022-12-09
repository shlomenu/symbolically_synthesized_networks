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
from utilities import (EXECUTE_BINARY_LOC, COLORS, compress,
                       invoke_binary_with_json, deduplicate)

DSL_DIR = "dsls"
EXECUTED_PROGRAMS_DIR = "executed_programs"
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
        self.executed_program_save_path = os.path.join(self.name_of_domain,
                                                       EXECUTED_PROGRAMS_DIR)
        self.dsl_save_path = os.path.join(self.name_of_domain, DSL_DIR)
        self.visualization_save_path = os.path.join(self.name_of_domain,
                                                    VISUALIZATION_DIR)
        self.timed_out = set()
        self.executed_programs = {}
        self.load_dsl(dsl_name)
        self.restart_manager = RestartManager(
            idleness_limit, len(self.entries), contextual=True)
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

    def forward(self, commands, always_cache=False):
        graphs, feats, restarts, hashes, programs = self.execute_commands(
            commands.tolist(), commands.device, always_cache)
        feats = rearrange(self.attn_up(graphs, feats), "n ... -> n (...)")
        for attn, ff in zip(self.attns, self.ffs):
            feats = ff(attn(graphs, feats))
        return self.to_out(
            self.pool(graphs,
                      self.ff_seq(self.attn_seq(graphs,
                                                feats)))), restarts, hashes, programs

    @property
    def dsl_size(self):
        return len(self.entries)

    def load_dsl(self, dsl_name):
        self.dsl_name = dsl_name
        with open(os.path.join(self.dsl_save_path,
                               f"{self.dsl_name}.json")) as f:
            dsl = json.load(f)
        entries = tuple(ent["name"] for ent in dsl["library"])
        if hasattr(self, "entries"):
            for code in range(len(entries)):
                self.restart_manager.add_code(code)
            ent_to_ind = {v: k for k, v in enumerate(entries)}
            ind_to_ind = {k: ent_to_ind[v]
                          for k, v in enumerate(self.entries)}
            executed_programs = {}
            for frozen_commands, hash in self.executed_programs.items():
                executed_programs[tuple(ind_to_ind[code] for code in frozen_commands)] = \
                    hash
            self.executed_programs = executed_programs
        self.entries = entries

    def execute_commands(self, commands: List[List[int]], device, always_cache):
        graphs, feats, restarts, hashes, programs = [], [], set(), [], []
        for cmds in commands:
            graph, feat, hash, program = self._execute_commands(
                cmds, device, always_cache)
            restarts.update(self.restart_manager.find_restarts(cmds))
            graphs.append(graph)
            feats.append(feat)
            hashes.append(hash)
            programs.append(program)
        return dgl.batch(graphs), th.cat(feats,
                                         dim=0), restarts, hashes, programs

    def _execute_commands(self, commands: List[int], device, always_cache):
        frozen_commands = tuple(commands)
        if frozen_commands in self.executed_programs:
            new = False
            hash = self.executed_programs[frozen_commands]
            with open(
                    os.path.join(self.executed_program_save_path,
                                 f"{hash}.json")) as f:
                resp = json.load(f)
        else:
            new = True
            resp = invoke_binary_with_json(
                EXECUTE_BINARY_LOC, {
                    "attempts":
                    1,
                    "eval_timeout":
                    .1,
                    "max_color":
                    self.max_color,
                    "commands":
                    commands,
                    "domain":
                    self.name_of_domain,
                    "dsl_file":
                    os.path.join(self.name_of_domain, DSL_DIR,
                                 f"{self.dsl_name}.json")
                })
            if self.training or always_cache:
                hashf = blake2b(digest_size=15)
                hashf.update(bytes(resp["original"], encoding="utf-8"))
                hash = hashf.hexdigest()
                self.executed_programs[frozen_commands] = hash
                with open(
                        os.path.join(self.executed_program_save_path,
                                     f"{hash}.json"), "w") as f:
                    f.write(json.dumps(resp))
            else:
                hash = None
        if not resp["translated"]:
            print(
                f"WARNING: commands_to_program failed to produce program: {'new' if new else 'reused'}: {hash}"
            )
        elif resp["timed_out"]:
            print(
                f"WARNING: program execution timed out: {'new' if new else 'reused'}: {hash}"
            )
            self.timed_out.add(frozen_commands)
        graph, feat = dgl_homograph_of_json(resp["output"], device)
        return graph, feat, hash, resp["original"]

    def deduplicate(self):
        resp = deduplicate(self.name_of_domain,
                           self.executed_program_save_path,
                           max_color=self.max_color)
        best, redundant = resp["best"], resp["redundant"]
        reduction = {}
        for b, rs in zip(best, redundant):
            reduction[b] = b
            for r in rs:
                reduction[r] = b
        executed_programs = {}
        for frozen_commands, hash in self.executed_programs.items():
            filename = f"{hash}.json"
            if filename in reduction:
                executed_programs[frozen_commands] = reduction[filename][:-5]
            else:
                raise Exception("executed program was not accounted for by")
        for rs in redundant:
            for r in rs:
                os.remove(os.path.join(self.executed_program_save_path, r))
        self.executed_programs = executed_programs
        return reduction

    def visualize(self, to_visualize=None):
        if to_visualize is None:
            to_visualize = os.listdir(self.executed_program_save_path)
        named_graphs = {}
        for fn in to_visualize:
            if fn.endswith(".json"):
                with open(os.path.join(self.executed_program_save_path,
                                       fn)) as f:
                    ep = json.load(f)
                print(f"loading.. {fn}")
                named_graphs[fn[:-5]] = \
                    pygraphviz_graph_of_json(ep["output"], ep["beta_reduced"])
        for name, graph in named_graphs.items():
            graph.draw(
                os.path.join(self.visualization_save_path, name + ".png"))

    def clear_visualizations(self):
        for filename in os.listdir(os.path.join(self.visualization_save_path)):
            if filename.endswith(".png"):
                os.remove(os.path.join(self.visualization_save_path, filename))

    def compress(self,
                 frontier,
                 next_dsl_name,
                 iterations=1,
                 beam_size=3,
                 top_i=3,
                 dsl_size_penalty=0.,
                 primitive_size_penalty=1.,
                 n_beta_inversions=2,
                 verbosity=0):
        next_dsl_file = os.path.join(self.dsl_save_path,
                                     f"{next_dsl_name}.json")
        resp = compress(frontier,
                        self.name_of_domain,
                        os.path.join(self.dsl_save_path,
                                     f"{self.dsl_name}.json"),
                        next_dsl_file,
                        self.executed_program_save_path,
                        iterations=iterations,
                        beam_size=beam_size,
                        top_i=top_i,
                        dsl_size_penalty=dsl_size_penalty,
                        primitive_size_penalty=primitive_size_penalty,
                        n_beta_inversions=n_beta_inversions,
                        verbosity=verbosity,
                        max_color=self.max_color)
        return resp["rewritten"]


class NullQuantizer:

    loss = F.mse_loss

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
