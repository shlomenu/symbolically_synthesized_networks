import os
import json

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
        self.emb_matrix = self.sincos_embedding_2d(
            10, self.max_conn, self.codebook_dim, device=device)
        self.edge_dropout = nn.Dropout(dropout_rate)
        self.convs = nn.ModuleList([
            dgl.nn.GINEConv(  # type: ignore
                nn.Linear(self.codebook_dim, self.codebook_dim)) for _ in range(depth)
        ])
        self.node_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(depth)])
        self.pool = dgl.nn.AvgPooling()  # type:ignore
        self.linear_out = nn.Linear(self.codebook_dim, output_dim)

    @staticmethod
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

    @staticmethod
    def graph_of_json(graph_json, emb_matrix, device):
        edges, edges_1, edges_2, nodes, efeats, n_elt = set(), [], [], set(), [], -1
        for [[node_1, port], [edge, node_2]] in graph_json["edges"]:
            edge_canonical = (node_1, edge, node_2) if node_1 < node_2 else (
                node_2, node_1)
            if edge_canonical not in edges:
                edges_1.append(node_1)
                edges_2.append(node_2)
                efeats.append([edge, port + 1])
                nodes.add(node_1)
                nodes.add(node_2)
                n_elt = max((node_1, edge, node_2, n_elt))
        edges_1, edges_2 = edges_1 + edges_2, edges_2 + edges_1
        creation_ordered = sorted(nodes)
        unit_spaced_to_creation_ordered = dict(
            zip(range(len(nodes)), creation_ordered))
        nfeats = [[unit_spaced_to_creation_ordered[n], 0]
                  for n in range(len(nodes))]
        creation_ordered_to_unit_spaced = dict(
            zip(creation_ordered, range(len(nodes))))
        data = (
            th.tensor([creation_ordered_to_unit_spaced[node]
                       for node in edges_1]),
            th.tensor([creation_ordered_to_unit_spaced[node]
                       for node in edges_2]))
        if emb_matrix is None or n_elt + 1 > emb_matrix.size(0):
            emb_matrix = GraphQuantizer.sincos_embedding_2d(n_elt + 1, emb_matrix.size(
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
        selections, device = selections.flatten().tolist(), selections.device
        restarts = self.restart_manager.find_restarts(
            selections) if self.training else []
        graphs, nfeats, efeats, filenames = [], [], [], []
        for i, selection in enumerate(selections):
            filename = self.representations[selection]
            with open(os.path.join(self.representations_path, filename)) as f:
                repr = json.load(f)
            self.emb_matrix, graph, nfeat, efeat = self.graph_of_json(
                repr["output"], self.emb_matrix, device)
            graphs.append(graph)
            nfeats.append(nfeat + latent[i])
            efeats.append((efeat + latent[i]).tile((2, 1)))
            filenames.append(filename)
        graphs, nfeats, efeats = (
            dgl.batch(graphs),
            th.cat(nfeats, dim=0),
            self.edge_dropout(th.cat(efeats, dim=0)),
        )
        for node_dropout, conv in zip(self.node_dropouts, self.convs):
            nfeats = conv(graphs, node_dropout(nfeats), efeats)
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
