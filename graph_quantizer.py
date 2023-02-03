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
    if emb_matrix is None or n_elt + 1 > emb_matrix.size(0):
        emb_matrix = sincos_embedding_2d(n_elt + 1, emb_matrix.size(
            1) - 1, emb_matrix.size(2), device=emb_matrix.device)
    graph = dgl.to_homogeneous(dgl.graph(data, device=device))
    nfeats_ind = th.empty(len(nfeats), 1, emb_matrix.size(
        2), device=emb_matrix.device, dtype=th.long)
    for i, [creation_order_id, port] in enumerate(nfeats):
        nfeats_ind[i, port, :] = creation_order_id
    efeats_ind_order = th.empty(len(efeats), emb_matrix.size(1), emb_matrix.size(
        2), device=emb_matrix.device, dtype=th.long)
    efeats_ind_port = th.empty(len(efeats), 1, emb_matrix.size(
        2), device=emb_matrix.device, dtype=th.long)
    for i, [creation_order_id, port] in enumerate(efeats):
        for p in range(emb_matrix.size(1)):
            efeats_ind_order[i, p, :] = creation_order_id
        efeats_ind_port[i, 0, :] = port
    nfeats = th.gather(emb_matrix, 0, nfeats_ind).squeeze()
    efeats = th.gather(
        th.gather(emb_matrix, 0, efeats_ind_order), 1, efeats_ind_port).squeeze()
    if len(efeats.shape) == 1:
        efeats = efeats.unsqueeze(0)
    return (emb_matrix, graph, nfeats, efeats)


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
                 dropout_rate,
                 depth,
                 max_conn,
                 output_dim,
                 device):
        super().__init__("graph", dsl_name, codebook_dim, beta, max_conn=max_conn)
        self.codebook_dim = codebook_dim
        self.output_dim = output_dim
        self.max_conn = max_conn
        self.emb_matrix = sincos_embedding_2d(
            10, self.max_conn, self.codebook_dim, device=device)
        self.node_dropout = nn.Dropout(dropout_rate)
        self.edge_dropout = nn.Dropout(dropout_rate)
        self.convs = nn.ModuleList([
            dgl.nn.GINEConv(  # type: ignore
                nn.Linear(self.codebook_dim, self.codebook_dim)) for _ in range(depth)
        ])
        self.pool = dgl.nn.AvgPooling()  # type:ignore
        self.linear_out = nn.Linear(self.codebook_dim, output_dim)

    def structural_prediction(self, latent, selections):
        graphs, nfeats, efeats, restarts, filenames = self._fetch(
            latent, selections.flatten().tolist(), selections.device)
        nfeats, efeats = self.node_dropout(nfeats), self.edge_dropout(efeats)
        for conv in self.convs:
            nfeats = conv(graphs, nfeats, efeats)
        return self.linear_out(self.pool(graphs, nfeats)), restarts, filenames

    def _fetch(self, features, selections: List[int], device):
        restarts = self.restart_manager.find_restarts(selections)
        graphs, nfeats, efeats, filenames = [], [], [], []
        for i, selection in enumerate(selections):
            filename = self.representations[selection]
            with open(os.path.join(self.representations_path, filename)) as f:
                repr = json.load(f)
            self.emb_matrix, graph, nfeat, efeat = graph_of_json(
                repr["output"], self.emb_matrix, device)
            graphs.append(graph)
            nfeats.append(nfeat + features[i])
            efeats.append(efeat + features[i])
            filenames.append(filename)
        return (
            dgl.batch(graphs),
            th.cat(nfeats, dim=0),
            th.cat(efeats, dim=0),
            restarts, filenames)

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
