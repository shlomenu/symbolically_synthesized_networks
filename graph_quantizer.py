import os
import json
from typing import List

import dgl
import pygraphviz as pgv
import torch as th
import torch.nn as nn
from einops import rearrange

from quantizer import Quantizer

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


class GraphQuantizer(Quantizer):

    def __init__(self,
                 dsl_name,
                 codebook_dim,
                 beta,
                 *,
                 depth,
                 graph_dim,
                 max_conn):
        super().__init__("graph", dsl_name, codebook_dim, beta, max_conn=max_conn)
        self.graph_dim = graph_dim
        self.codebook_dim = codebook_dim
        self.emb_matrix = None
        self.convs = nn.ModuleList([
            dgl.nn.GINEConv(  # type: ignore
                nn.Linear(self.graph_dim, self.graph_dim)) for _ in range(depth)
        ])
        self.pool = dgl.nn.AvgPooling()  # type:ignore
        self.to_out = nn.Sequential(
            *[nn.Linear(self.graph_dim, self.codebook_dim), nn.Tanh()])

    def reconstruct(self, selections):
        graphs, restarts, filenames = self._fetch(
            selections.flatten().tolist(), selections.device)
        nfeats = graphs.ndata["emb"]
        for conv in self.convs:
            nfeats = conv(graphs, nfeats, graphs.edata["emb"])
        return self.to_out(self.pool(graphs, nfeats)), restarts, filenames

    def _fetch(self, selections: List[int], device):
        restarts = self.in_restart_manager.find_restarts(selections)
        graphs, filenames = [], []
        for selection in selections:
            filename = self.representations[selection]
            with open(os.path.join(self.representations_path, filename)) as f:
                repr = json.load(f)
            self.emb_matrix, graph = graph_of_json(
                repr["output"], self.emb_matrix, device)
            graphs.append(graph)
            filenames.append(filename)
        return dgl.batch(graphs), restarts, filenames

    def visualize(self, to_visualize=None):
        if to_visualize is None:
            to_visualize = os.listdir(self.representations_path)
        else:
            to_visualize = list(set(to_visualize))
        visualized = 0
        for filename in to_visualize:
            if filename.endswith(".json"):
                with open(os.path.join(self.representations_path,
                                       filename)) as f:
                    contents = json.load(f)
                name = filename[:-5]
                graph = pygraphviz_graph_of_json(contents["output"], name)
                graph.draw(
                    os.path.join(self.visualizations_path, name + ".svg"))
                visualized += 1
        print(f"produced {visualized} visualizations.")
