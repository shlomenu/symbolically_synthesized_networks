import os
import json
from typing import List

import dgl
import pygraphviz as pgv
import torch as th
import torch.nn as nn
from einops import rearrange
from vit_pytorch.simple_vit import Transformer

from restart_manager import RestartManager
from utilities import explore, dreamcoder_compress, incorporate_stitch, stitch_compress, stitch_rewrite

DSL_DIR = "dsls"
REPRESENTATIONS_DIR = "representations"
VISUALIZATION_DIR = "visualizations"


def graph_data_of_json(graph_json):
    edges_1, edges_2, nodes, efeats, n_elt = [], [], set(), [], -1
    for [[node_1, port], [edge, node_2]] in graph_json["edges"]:
        edges_1.append(node_1)
        edges_2.append(node_2)
        efeats.append([edge, port + 1])
        nodes.add(node_1)
        nodes.add(node_2)
        n_elt = max((node_1, edge, node_2, n_elt))
    creation_ordered = sorted(nodes)
    unit_spaced_to_creation_ordered = dict(
        zip(range(len(nodes)), creation_ordered))
    nfeats = [[unit_spaced_to_creation_ordered[n], 0]
              for n in range(len(nodes))]
    creation_ordered_to_unit_spaced = dict(
        zip(creation_ordered, range(len(nodes))))
    edges_1 = th.tensor([creation_ordered_to_unit_spaced[node]
                        for node in edges_1])
    edges_2 = th.tensor([creation_ordered_to_unit_spaced[node]
                        for node in edges_2])
    return (edges_1, edges_2), nfeats, efeats, n_elt


def sincos_embedding_2d(n_elt, max_conn, dim, device, temperature=10000):
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    h, w = n_elt, max_conn + 1
    x, y = th.meshgrid(th.arange(h), th.arange(w), indexing="xy")
    omega = th.arange(dim // 4) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)
    y = rearrange(y.flatten()[:, None] * omega[None, :],
                  "(h w) qc -> h w qc", h=h, w=w, qc=(dim // 4))
    x = rearrange(x.flatten()[:, None] * omega[None, :],
                  "(h w) qc -> h w qc", h=h, w=w, qc=(dim // 4))
    return th.cat(
        (x.sin(), x.cos(), y.sin(), y.cos()), dim=2).to(device=device, dtype=th.float)


def graph_of_json(graph_json, emb_matrix, device):
    """
    Assumes that json represents graph with no isolated nodes
    and containing at least one edge.
    """
    data, nfeats, efeats, n_elt = graph_data_of_json(graph_json)
    if emb_matrix is None or n_elt <= emb_matrix.size(0):
        emb_matrix = sincos_embedding_2d(n_elt, emb_matrix.size(
            1) - 1, emb_matrix.size(2), device=emb_matrix.device)
    graph = dgl.graph(data, device=device)
    graph.ndata["emb"] = emb_matrix[nfeats]
    graph.edata["emb"] = emb_matrix[efeats]
    return emb_matrix, graph


def pygraphviz_graph_of_json(graph_json, caption):
    graph = pgv.AGraph(directed=True, labelloc="t", label=caption)
    nodes, edges, n_elt = set(), set(), -1
    for [[node_1, _port], [edge, node_2]] in graph_json["edges"]:
        edges.add((node_1, node_2))
        nodes.add(node_1)
        nodes.add(node_2)
        n_elt = max([node_1, edge, node_2])
    for node in nodes:
        graph.add_node(
            node, color=f"0.722 {node / n_elt:1.3f} 0.810")
    for (a, b) in edges:
        graph.add_edge((a, b), arrowhead="vee")
    graph.layout()
    return graph


class GraphQuantizer(nn.Module):

    def __init__(self,
                 output_dim,
                 graph_dim,
                 dsl_name,
                 depth=3,
                 idleness_limit=8,
                 max_conn=10):
        super().__init__()
        self.max_conn = max_conn
        self.name_of_domain = "graph"
        self.previous_abstractions = 0
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
        self.graph = graph_dim
        self.emb_matrix = None
        self.convs = nn.ModuleList([
            dgl.nn.GINEConv(  # type: ignore
                nn.Linear(graph_dim, graph_dim)) for _ in range(depth)
        ])
        self.pool = dgl.nn.AvgPooling()  # type:ignore
        self.to_out = nn.Sequential(
            *[nn.Linear(graph_dim, output_dim), nn.Tanh()])

    def __len__(self):
        return len(self.representations)

    def forward(self, _v, selections):
        graphs, restarts, filenames = self._fetch(
            selections.flatten().tolist(), selections.device)
        nfeats = graphs.ndata["emb"]
        for conv in self.convs:
            nfeats = conv(graphs, nfeats, graphs.edata["emb"])
        return self.to_out(self.pool(graphs, nfeats)), restarts, filenames

    def _fetch(self, selections: List[int], device):
        restarts = self.restart_manager.find_restarts(selections)
        graphs, filenames = [], []
        for selection in selections:
            filename = self.representations[selection]
            with open(os.path.join(self.representations_save_path, filename)) as f:
                repr = json.load(f)
            self.emb_matrix, graph = graph_of_json(
                repr["output"], self.emb_matrix, device)
            graphs.append(graph)
            filenames.append(filename)
        return dgl.batch(graphs), restarts, filenames

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
                         max_conn=self.max_conn)
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
                                     verbosity=verbosity)
        result.update(
            iterations=iterations,
            beam_size=beam_size,
            n_beta_inversions=n_beta_inversions,
            n_invention_sizes=n_invention_sizes,
            n_exactly_scored=n_exactly_scored,
            primitives_size_penalty=primitive_size_penalty,
            dsl_size_penalty=dsl_size_penalty,
            invention_name_prefix=invention_name_prefix)
        if result["n_added"] > 0:
            self.previous_abstractions += result["n_added"]
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
            self.previous_abstractions,
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
        result.update(n_added=len(invented_primitives))
        if result["n_added"] > 0:
            self.previous_abstractions += result["n_added"]
            replacements = [[prev, cur] for prev, cur in zip(
                res.json["original"], res.json["rewritten"]) if prev != cur]
            non_frontier = list(set(self.representations) - set(frontier))
            if non_frontier:
                non_frontier_programs = []
                for filename in non_frontier:
                    with open(os.path.join(self.representations_save_path, filename)) as f:
                        contents = json.load(f)
                        non_frontier_programs.append(
                            contents["stitch_program"])
                replacements.extend(([prev, cur] for prev, cur in zip(
                    non_frontier_programs, stitch_rewrite(
                        non_frontier_programs, res.abstractions))))
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
        else:
            to_visualize = list(set(to_visualize))
        visualized = 0
        for filename in to_visualize:
            if filename.endswith(".json"):
                with open(os.path.join(self.representations_save_path,
                                       filename)) as f:
                    contents = json.load(f)
                name = filename[:-5]
                graph = pygraphviz_graph_of_json(contents["output"], name)
                graph.draw(
                    os.path.join(self.visualization_save_path, name + ".svg"))
                visualized += 1
        print(f"produced {visualized} visualizations.")


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
