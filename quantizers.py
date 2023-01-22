from collections import defaultdict
import os
import json
from typing import List

import dgl
import pygraphviz as pgv
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from vit_pytorch.simple_vit import Transformer

from restart_manager import RestartManager
from utilities import COLORS, explore, dreamcoder_compress, incorporate_stitch, stitch_compress, stitch_rewrite
from networks import QuantizationEmbedding

DSL_DIR = "dsls"
REPRESENTATIONS_DIR = "representations"
VISUALIZATION_DIR = "visualizations"


def node_tensor_tuple_of_json(graph_json):
    edges_1, edges_2 = [], []
    for [[node_1, _color], node_2] in graph_json["forward_edges"]:
        edges_1.append(node_1)
        edges_2.append(node_2)
    for [[node_1, _color], node_2] in graph_json["backward_edges"]:
        edges_1.append(node_1)
        edges_2.append(node_2)
    return th.tensor(edges_1), th.tensor(edges_2)


def node_colors_of_json(graph_json, device):
    colors = th.zeros(len(graph_json["nodes"]),
                      graph_json["max_color"],
                      device=device)
    for [node, color] in graph_json["nodes"]:
        colors[node, color] = 1
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
    for [[node_1, _color], node_2] in graph_json["forward_edges"]:
        forward_edges.add((node_1, node_2))
    backward_edges = set()
    for [[node_1, _color], node_2] in graph_json["backward_edges"]:
        backward_edges.add((node_1, node_2))
    for (a, b) in forward_edges:
        if (b, a) in backward_edges:
            graph.add_edge((a, b), arrowhead="vee")
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
        with open(os.path.join(self.dsl_save_path, self.dsl_name + ".json")) as f:
            dsl = json.load(f)
            self.dsl_mass = dsl["mass"]
        self.representations = sorted(
            os.listdir(self.representations_save_path))
        self.restart_manager = RestartManager(
            idleness_limit, len(self.representations))
        self.masses, self.programs = self._fetch_meta()
        self.attn_up = dgl.nn.GATv2Conv(in_feats=self.max_color,  # type: ignore
                                        out_feats=C,
                                        num_heads=1,
                                        attn_drop=0.5)
        self.attns = nn.ModuleList([
            dgl.nn.GATv2Conv(in_feats=C,  # type: ignore
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
        self.attn_seq = dgl.nn.GATv2Conv(in_feats=C,  # type: ignore
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
        self.pool = dgl.nn.GlobalAttentionPooling(  # type: ignore
            nn.Linear(E * C, 1))
        self.to_out = nn.Sequential(*[nn.Unflatten(1, (E, C)), nn.Tanh()])

    def __len__(self):
        return len(self.representations)

    def forward(self, _v, selections):
        graphs, feats, restarts, filenames = self._fetch(
            selections.flatten().tolist(), selections.device)
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
                frontier,
                next_dsl_name,
                exploration_timeout,
                *,
                eval_timeout,
                eval_attempts):
        next_dsl_name = self._form_dsl_name(next_dsl_name)
        if next_dsl_name is None:
            parts = self.dsl_name.split("_")
            next_dsl_name = "_".join(
                ("_".join(parts[:-1]), str(int(parts[-1]) + 1)))
        result = explore(self.name_of_domain,
                         frontier,
                         os.path.join(self.dsl_save_path,
                                      f"{self.dsl_name}.json"),
                         os.path.join(self.dsl_save_path,
                                      f"{next_dsl_name}.json"),
                         self.representations_save_path,
                         exploration_timeout,
                         eval_timeout=eval_timeout,
                         eval_attempts=eval_attempts,
                         max_color=self.max_color)
        self._load_representations(dict(result["replacements"]))
        del result["replacements"]
        result.update(
            total=len(self),
            exploration_timeout=exploration_timeout,
            min_mass=self.min_mass,
            max_mass=self.max_mass,
            avg_mass=self.avg_mass,
            prev_dsl_name=self.dsl_name,
            next_dsl_name=next_dsl_name,
            eval_timeout=eval_timeout,
            eval_attempts=eval_attempts)
        self.dsl_name = next_dsl_name
        return result

    def dreamcoder_compress(self,
                            frontier,
                            next_dsl_name,
                            *,
                            iterations,
                            beam_size,
                            n_beta_inversions,
                            n_invention_sizes,
                            n_exactly_scored,
                            primitive_size_penalty,
                            dsl_size_penalty,
                            invention_name_prefix,
                            verbosity=0):
        next_dsl_name = self._form_dsl_name(next_dsl_name)
        result = dreamcoder_compress(frontier,
                                     self.name_of_domain,
                                     os.path.join(self.dsl_save_path,
                                                  f"{self.dsl_name}.json"),
                                     os.path.join(self.dsl_save_path,
                                                  f"{next_dsl_name}.json"),
                                     self.representations_save_path,
                                     iterations=iterations,
                                     beam_size=beam_size,
                                     n_beta_inversions=n_beta_inversions,
                                     n_invention_sizes=n_invention_sizes,
                                     n_exactly_scored=n_exactly_scored,
                                     primitive_size_penalty=primitive_size_penalty,
                                     dsl_size_penalty=dsl_size_penalty,
                                     invention_name_prefix=invention_name_prefix,
                                     verbosity=verbosity,
                                     max_color=self.max_color)
        result.update(
            iterations=iterations,
            beam_size=beam_size,
            n_beta_inversions=n_beta_inversions,
            n_invention_sizes=n_invention_sizes,
            n_exactly_scored=n_exactly_scored,
            primitives_size_penalty=primitive_size_penalty,
            dsl_size_penalty=dsl_size_penalty,
            invention_name_prefix=invention_name_prefix)
        if result["success"]:
            result.update(
                prev_dsl_name=self.dsl_name,
                next_dsl_name=next_dsl_name)
            self._load_representations(dict(result["replacements"]))
            del result["replacements"]
            self.dsl_name = next_dsl_name
            self.dsl_mass = result["next_dsl_mass"]
        return result

    def stitch_compress(self,
                        frontier,
                        next_dsl_name,
                        *,
                        iterations,
                        n_beta_inversions,
                        threads,
                        verbose,
                        **stitch_kwargs):
        next_dsl_name = self._form_dsl_name(next_dsl_name)
        frontier_programs = []
        for filename in frontier:
            with open(os.path.join(self.representations_save_path, filename)) as f:
                contents = json.load(f)
                frontier_programs.append(contents["stitch_program"])
        res = stitch_compress(
            frontier_programs,
            iterations=iterations,
            n_beta_inversions=n_beta_inversions,
            threads=threads,
            verbose=verbose,
            **stitch_kwargs)
        result = {
            "iterations": iterations,
            "n_beta_inversions": n_beta_inversions
        }
        result.update(**stitch_kwargs)
        invented_primitives = [[name, body] for name, body in sorted(
            ((a.name, a.body) for a in res.abstractions), key=lambda a: int(a[0].split("_")[-1]))]
        if invented_primitives:
            result.update(success=True)
            replacements = [[prev, cur] for prev, cur in zip(
                res.json["original"], res.json["rewritten"]) if prev != cur]
            non_frontier = list(set(self.representations) - set(frontier))
            if non_frontier:
                non_frontier_rewritten = stitch_rewrite(
                    non_frontier, res.abstractions)
                replacements.extend(([prev, cur] for prev, cur in zip(
                    non_frontier, non_frontier_rewritten)))
            resp = incorporate_stitch(
                replacements, invented_primitives, self.name_of_domain,
                os.path.join(self.dsl_save_path, f"{self.dsl_name}.json"),
                os.path.join(self.dsl_save_path, f"{next_dsl_name}.json"),
                self.representations_save_path)
            self._load_representations(dict(resp["replacements"]))
            del resp["replacements"]
            result.update(resp)
            result.update(
                prev_dsl_name=self.dsl_name,
                next_dsl_name=next_dsl_name)
            self.dsl_name = next_dsl_name
            self.dsl_mass = resp["next_dsl_mass"]
        else:
            result.update(success=False)
        return result

    def _form_dsl_name(self, name):
        parts = self.dsl_name.split("_")
        prev_root, prev_number = "_".join(parts[:-1]), int(parts[-1])
        next_number = str(prev_number + 1)
        if name is None:
            return "_".join(["_".join(prev_root), next_number])
        else:
            return "_".join([name, next_number])

    def _load_representations(self, replacements):
        self.replacements = replacements
        self.representations = [(self.replacements[file] if file in self.replacements else file)
                                for file in self.representations]
        self.representations.extend(
            set(os.listdir(self.representations_save_path)) - set(self.representations))
        for code in range(len(self)):
            self.restart_manager.add_code(code)
        self.masses, self.programs = self._fetch_meta()
        self.min_mass = min(self.masses.values())
        self.max_mass = max(self.masses.values())
        self.avg_mass = sum(self.masses.values()) / len(self.masses)

    def _fetch_meta(self):
        masses, programs = {}, {}
        for filename in self.representations:
            with open(os.path.join(self.representations_save_path, filename)) as f:
                contents = json.load(f)
                masses[filename] = contents["mass"]
                programs[filename] = contents["program"]
        return masses, programs

    def mass_of_representations(self, representations):
        return sum(self.masses[r] for r in representations) / len(representations)

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
                    contents = json.load(f)
                print(f"loading.. {filename}")
                named_graphs[filename[:-5]
                             ] = pygraphviz_graph_of_json(contents["output"], filename[:-5])
        for name, graph in named_graphs.items():
            graph.draw(
                os.path.join(self.visualization_save_path, name + ".svg"))


class BottleneckQuantizer(nn.Module):

    def __init__(self,
                 E,
                 C,
                 dsl_name,
                 idleness_limit=8,
                 n_representations=5,
                 depth=2,
                 heads=2,
                 temperature=10000,
                 coordinates_only=False):
        super().__init__()
        self.n_representations = n_representations
        self.coordinates_only = coordinates_only
        self.restart_manager = RestartManager(
            idleness_limit, self.n_representations)
        self.net = nn.Sequential(*[
            QuantizationEmbedding(
                E, C, self.n_representations, temperature=temperature, coordinates_only=coordinates_only),
            Transformer(C, depth, heads, int(.5 * C), int(2 * C))])

    def __len__(self):
        return self.n_representations

    def forward(self, latents, selections):
        restarts = self.restart_manager.find_restarts(
            selections.flatten().tolist())
        return self.net((latents, selections)), restarts, None


class NullQuantizer:

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
