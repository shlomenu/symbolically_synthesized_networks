import os
import json
import math

import dgl
import pygraphviz as pgv
import torch as th
import torch.nn as nn
from einops import rearrange

from quantizer import Quantizer


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
        self.create_sincos_embedding_2d(10, device=device)
        self.edge_dropout = nn.Dropout(dropout_rate)
        self.convs = nn.ModuleList([
            dgl.nn.GINEConv(  # type: ignore
                nn.Linear(self.codebook_dim, self.codebook_dim)) for _ in range(depth)
        ])
        self.node_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(depth)])
        self.edge_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(depth)]
        )
        self.pool = dgl.nn.AvgPooling()  # type:ignore
        self.linear_out = nn.Linear(self.codebook_dim, output_dim)

    def create_sincos_embedding_2d(self, size, device, temperature=10000):
        assert (self.codebook_dim %
                4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
        h, w = size, self.max_conn + 1
        x, y = th.meshgrid(th.arange(h), th.arange(w), indexing="xy")
        omega = th.arange(self.codebook_dim // 4) / \
            (self.codebook_dim // 4 - 1)
        omega = 1. / (temperature ** omega)
        y = rearrange(y.flatten()[:, None] * omega[None, :],
                      "(h w) qc -> h w qc", h=h, w=w, qc=(self.codebook_dim // 4))
        x = rearrange(x.flatten()[:, None] * omega[None, :],
                      "(h w) qc -> h w qc", h=h, w=w, qc=(self.codebook_dim // 4))
        self.emb_matrix = th.cat(
            (x.sin(), x.cos(), y.sin(), y.cos()), dim=2).to(device=device, dtype=th.float)

    def graph_of_json(self, graph_json, choice_emb):
        device = self.emb_matrix.device
        seen_edges, edge_start, edge_end, node_order_ids, edge_order_ids, ports = \
            set(), [], [], set(), [], []
        for [[node_1_order_id, port], [edge_order_id, node_2_order_id]] in graph_json["edges"]:
            edge_canonical = (node_1_order_id, edge_order_id, node_2_order_id) \
                if node_1_order_id < node_2_order_id else (
                    node_2_order_id, node_1_order_id)
            if edge_canonical not in seen_edges:
                edge_start.append(node_1_order_id)
                edge_end.append(node_2_order_id)
                edge_order_ids.append(edge_order_id)
                ports.append(port + 1)
                node_order_ids.add(node_1_order_id)
                node_order_ids.add(node_2_order_id)
        node_order_ids = sorted(node_order_ids)
        n_nodes, n_edges = len(node_order_ids), len(edge_order_ids)
        n_elt = n_nodes + n_edges
        if n_elt > self.emb_matrix.size(0):
            self.create_sincos_embedding_2d(n_elt, device=device)
        nfeats_posemb_index = th.empty(
            n_nodes, 1, self.codebook_dim, device=device, dtype=th.long)
        efeats_posemb_order_index = th.empty(
            n_edges, self.max_conn + 1, self.codebook_dim, device=device, dtype=th.long)
        efeats_posemb_port_index = th.empty(
            n_edges, 1, self.codebook_dim, device=device, dtype=th.long)
        for i, order_id in enumerate(node_order_ids):
            nfeats_posemb_index[i, 0, :] = order_id
        for i, (order_id, port) in enumerate(zip(edge_order_ids, ports)):
            for p in range(self.max_conn + 1):
                efeats_posemb_order_index[i, p, :] = order_id
            efeats_posemb_port_index[i, 0, :] = port
        assert choice_emb.size(0) == self.codebook_dim
        increment = self.codebook_dim // n_elt
        remainder = self.codebook_dim - increment * n_elt
        increments = list(range(increment + remainder,
                          self.codebook_dim + 1, increment))
        choice_matrix = th.cat(
            [choice_emb[start:stop].tile(
                (math.ceil(self.codebook_dim / increment),))[:self.codebook_dim].unsqueeze(0)
                for start, stop in zip([0] + increments[:-1], increments)],
            dim=0)
        nfeats_choice_index = th.empty(
            n_nodes, self.codebook_dim, device=device, dtype=th.long)
        efeats_choice_index = th.empty(
            n_edges, self.codebook_dim, device=device, dtype=th.long)
        for i, order_id in enumerate(node_order_ids):
            nfeats_choice_index[i, :] = order_id
        for i, order_id in enumerate(edge_order_ids):
            efeats_choice_index[i, :] = order_id
        unit_spaced = dict(
            zip(node_order_ids, range(len(node_order_ids))))
        return (
            dgl.to_homogeneous(dgl.graph((
                th.tensor([unit_spaced[order_id]
                          for order_id in edge_start + edge_end]),
                th.tensor([unit_spaced[order_id]
                          for order_id in edge_end + edge_start])), device=device)),
            th.gather(
                self.emb_matrix, 0, nfeats_posemb_index).squeeze(dim=1) +
            th.gather(choice_matrix, 0, nfeats_choice_index),
            (th.gather(
                th.gather(self.emb_matrix, 0, efeats_posemb_order_index), 1, efeats_posemb_port_index).squeeze(dim=1) +
             th.gather(choice_matrix, 0, efeats_choice_index)).tile((2, 1)))

    @staticmethod
    def pygraphviz_graph_of_json(graph_json, caption):
        graph = pgv.AGraph(labelloc="t", label=caption)
        nodes, edges, n_elt = set(), set(), -1
        for [[node_1, _port], [edge, node_2]] in graph_json["edges"]:
            edges.add((node_1, edge, node_2) if node_1 <
                      node_2 else (node_2, edge, node_1))
            nodes.add(node_1)
            nodes.add(node_2)
            n_elt = max([node_1, edge, node_2])
        for node in nodes:
            graph.add_node(
                node, color=f"0.722 {node / n_elt:1.3f} 0.810")
        for (a, edge, b) in edges:
            graph.add_edge((a, b), label=edge,
                           color=f"0.722 {edge / n_elt:1.3f} 0.810",
                           fontsize=12.0)
        graph.layout()
        return graph

    def structural_prediction(self, latent, selections):
        selections = selections.flatten().tolist()
        restarts = self.restart_manager.find_restarts(
            selections) if self.training else []
        graphs, nfeats, efeats, filenames = [], [], [], []
        for i, selection in enumerate(selections):
            filename = self.representations[selection]
            with open(os.path.join(self.representations_path, filename)) as f:
                repr = json.load(f)
            graph, nfeat, efeat = self.graph_of_json(repr["output"], latent[i])
            graphs.append(graph)
            nfeats.append(nfeat)
            efeats.append(efeat)
            filenames.append(filename)
        graphs, nfeats, efeats = dgl.batch(graphs), th.cat(
            nfeats, dim=0), th.cat(efeats, dim=0),
        for node_dropout, edge_dropout, conv in zip(self.node_dropouts, self.edge_dropouts, self.convs):
            nfeats = conv(graphs, node_dropout(nfeats), edge_dropout(efeats))
        return self.linear_out(self.pool(graphs, nfeats)), restarts, filenames

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
                graph = self.pygraphviz_graph_of_json(contents["output"], name)
                graph.draw(
                    os.path.join(self.visualizations_path, name + ".svg"))
                visualized += 1
        print(f"produced {visualized} visualizations.")
