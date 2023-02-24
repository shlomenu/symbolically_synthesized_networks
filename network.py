from typing import Union
import math

from tqdm import tqdm
import dgl
import torch as th
from torch import nn
from torch.utils.data import DataLoader
from vit_pytorch.simple_vit import posemb_sincos_2d, Transformer
from einops import rearrange
from einops.layers.torch import Rearrange


class PositionalEmbedding2d(nn.Module):

    def forward(self, x):
        return x + rearrange(
            posemb_sincos_2d(x), "(h w) c -> h w c", h=x.size(1), w=x.size(2))


class GlobalAvgPool(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x: th.Tensor):
        return self.net(x.mean(dim=1))


class ConvResidual(nn.Module):

    def __init__(self, channels: int, size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      bias=False), nn.LayerNorm([channels, size, size]),
            nn.Conv2d(channels, channels, kernel_size=1,
                      bias=False), nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False))

    def forward(self, x):
        return x + self.net(x)


class PixelShuffle_ViT_Encoder(nn.Module):

    def __init__(self,
                 input_size,
                 downsampled_size,
                 vit_dim,
                 vit_depth,
                 vit_heads,
                 vit_head_dim,
                 vit_mlp_dim,
                 *,
                 input_channels,
                 conv_depth,
                 output_dim):
        super().__init__()
        assert (input_size >= downsampled_size and vit_depth > 0)
        layers = [
            nn.PixelUnshuffle(input_size // downsampled_size),
            nn.Conv2d(input_channels * (input_size // downsampled_size)**2,
                      vit_dim,
                      kernel_size=1)
        ]
        for _ in range(conv_depth):
            layers.append(ConvResidual(vit_dim, downsampled_size))
        layers.extend([
            Rearrange("b c h w -> b h w c",
                      h=downsampled_size,
                      w=downsampled_size,
                      c=vit_dim),
            PositionalEmbedding2d(),
            Rearrange("b h w c -> b (h w) c",
                      h=downsampled_size,
                      w=downsampled_size,
                      c=vit_dim),
            Transformer(vit_dim, vit_depth, vit_heads, vit_head_dim,
                        vit_mlp_dim),
            GlobalAvgPool(vit_dim, output_dim)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, img):
        return self.net(img)


class GraphStructured(nn.Module):

    def __init__(self,
                 graph_json,
                 max_conn,
                 dim,
                 output_dim,
                 depth,
                 dropout_rate,
                 temperature=10000):
        super().__init__()
        self.dim, self.max_conn, self.output_dim = dim, max_conn, output_dim
        seen_edges, edge_start, edge_end, node_order_ids, self.edge_order_ids, ports = \
            set(), [], [], set(), [], []
        for [[node_1_order_id, port], [edge_order_id, node_2_order_id]] in graph_json["edges"]:
            edge_canonical = (node_1_order_id, edge_order_id, node_2_order_id) \
                if node_1_order_id < node_2_order_id else (
                    node_2_order_id, node_1_order_id)
            if edge_canonical not in seen_edges:
                edge_start.append(node_1_order_id)
                edge_end.append(node_2_order_id)
                self.edge_order_ids.append(edge_order_id)
                ports.append(port + 1)
                node_order_ids.add(node_1_order_id)
                node_order_ids.add(node_2_order_id)
        self.node_order_ids = sorted(node_order_ids)
        self.n_nodes, self.n_edges = len(
            self.node_order_ids), len(self.edge_order_ids)
        self.n_elt = self.n_nodes + self.n_edges
        assert (self.dim %
                4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
        h, w = self.n_elt, self.max_conn + 1
        x, y = th.meshgrid(th.arange(h), th.arange(w), indexing="xy")
        omega = 1 / (temperature ** (th.arange(self.dim // 4) /
                                     (self.dim // 4 - 1)))
        y = rearrange(y.flatten()[:, None] * omega[None, :],
                      "(h w) qc -> h w qc", h=h, w=w, qc=(self.dim // 4))
        x = rearrange(x.flatten()[:, None] * omega[None, :],
                      "(h w) qc -> h w qc", h=h, w=w, qc=(self.dim // 4))
        emb_matrix = th.cat(
            (x.sin(), x.cos(), y.sin(), y.cos()), dim=2).to(dtype=th.float)
        nfeats_posemb_index = th.empty(
            self.n_nodes, 1, self.dim, dtype=th.long)
        efeats_posemb_order_index = th.empty(
            self.n_edges, self.max_conn + 1, self.dim, dtype=th.long)
        efeats_posemb_port_index = th.empty(
            self.n_edges, 1, self.dim, dtype=th.long)
        for i, order_id in enumerate(self.node_order_ids):
            nfeats_posemb_index[i, 0, :] = order_id
        for i, (order_id, port) in enumerate(zip(self.edge_order_ids, ports)):
            for p in range(self.max_conn + 1):
                efeats_posemb_order_index[i, p, :] = order_id
            efeats_posemb_port_index[i, 0, :] = port
        unit_spaced = dict(
            zip(self.node_order_ids, range(len(self.node_order_ids))))
        self.graph = dgl.to_homogeneous(dgl.graph((
            th.tensor([unit_spaced[order_id]
                       for order_id in edge_start + edge_end]),
            th.tensor([unit_spaced[order_id]
                       for order_id in edge_end + edge_start]))))
        self.register_buffer("nfeats_posemb", th.gather(
            emb_matrix, 0, nfeats_posemb_index).squeeze(dim=1))
        self.register_buffer("efeats_posemb", th.gather(
            th.gather(emb_matrix, 0, efeats_posemb_order_index),
            1, efeats_posemb_port_index).squeeze(dim=1))
        self.increment = self.dim // self.n_elt
        remainder = self.dim - self.increment * self.n_elt
        self.increments = list(range(self.increment + remainder,
                                     self.dim + 1, self.increment))
        nfeats_choice_index = th.empty(self.n_nodes, self.dim, dtype=th.long)
        efeats_choice_index = th.empty(self.n_edges, self.dim, dtype=th.long)
        for i, order_id in enumerate(self.node_order_ids):
            nfeats_choice_index[i, :] = order_id
        for i, order_id in enumerate(self.edge_order_ids):
            efeats_choice_index[i, :] = order_id
        self.register_buffer("nfeats_choice_index", nfeats_choice_index)
        self.register_buffer("efeats_choice_index", efeats_choice_index)
        self.convs = nn.ModuleList([
            dgl.nn.GINEConv(  # type: ignore
                nn.Linear(dim, dim)) for _ in range(depth)
        ])
        self.node_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(depth)])
        self.edge_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(depth)]
        )
        self.pool = dgl.nn.AvgPooling()  # type:ignore
        self.linear_out = nn.Linear(dim, output_dim)

    def forward(self, x):
        assert x.size(1) == self.dim
        if x.device != self.graph.device:  # type: ignore
            self.graph = self.graph.to(device=x.device)  # type: ignore
        graphs = dgl.batch([self.graph for _ in range(x.size(0))])
        nfeats, efeats = [], []
        for i in range(x.size(0)):
            choice_matrix = th.cat(
                [x[i, start:stop].tile(
                    (math.ceil(self.dim / self.increment),))[:self.dim].unsqueeze(0)
                    for start, stop in zip([0] + self.increments[:-1], self.increments)],
                dim=0)
            nfeats.append(self.nfeats_posemb +
                          th.gather(choice_matrix, 0, self.nfeats_choice_index))  # type: ignore
            efeats.append((self.efeats_posemb +
                           th.gather(choice_matrix, 0, self.efeats_choice_index)).tile((2, 1)))  # type: ignore
        nfeats, efeats = th.cat(nfeats, dim=0), th.cat(efeats, dim=0)
        for node_dropout, edge_dropout, conv in zip(self.node_dropouts, self.edge_dropouts, self.convs):
            nfeats = conv(graphs, node_dropout(
                nfeats), edge_dropout(efeats))
        return self.linear_out(self.pool(graphs, nfeats))


class TqdmExtraFormat(tqdm):

    def __init__(self, *args, extras={}, **kwargs):
        self.extras = extras
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        d.update(**self.extras)
        return d


class AccuracyMetric:

    def __init__(self, shape, device):
        super().__init__()
        self.correct = th.zeros(shape, device=device)
        self.incorrect = th.zeros(shape, device=device)

    def tally(self, output, target):
        preds = th.round(th.sigmoid(output))
        self.correct += th.count_nonzero(preds == target, dim=0)
        self.incorrect += th.count_nonzero(preds != target, dim=0)

    def measure(self):
        return th.mean(
            self.correct / (self.correct + self.incorrect), dim=0).tolist()


class F1Metric:

    def __init__(self, shape, device):
        self.tp = th.zeros(shape, device=device)
        self.fp = th.zeros(shape, device=device)
        self.fn = th.zeros(shape, device=device)
        self.epsilon = .001 * th.ones(shape, device=device)

    def tally(self, output, target):
        preds = th.round(th.sigmoid(output))
        self.tp += th.count_nonzero(
            th.logical_and(preds == target, preds == 1.), dim=0)
        self.fp += th.count_nonzero(
            th.logical_and(preds != target, preds == 1.), dim=0)
        self.fn += th.count_nonzero(
            th.logical_and(preds != target, preds == 0.), dim=0)

    def measure(self):
        return th.mean(
            (2 * self.tp) / (2 * self.tp + self.fp + self.fn + self.epsilon), dim=0).tolist()


class Network(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.optimizer = th.optim.Adam(self.parameters())
        self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=.5, patience=50,
            verbose=True, threshold=1e-4, cooldown=15)
        self.loss_f = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x):
        return self.net(x)

    def run(self,
            dataset,
            batch_size,
            n_epochs,
            smoothing_factor,
            device,
            perf_metric: Union[AccuracyMetric, F1Metric, None] = None,
            train=True,
            shuffle=True,
            use_scheduler=True):
        assert ((len(dataset) % batch_size) == 0)
        log = {"use_scheduler": use_scheduler}
        if train:
            self.train()
        else:
            self.eval()
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        data_iterator = iter(dataloader)
        steps_per_epoch = len(dataset) // batch_size
        total_steps = n_epochs * steps_per_epoch
        bar_format = \
            "{bar}{r_bar} - " \
            "epoch: {epoch:3.0f}, " \
            "loss: {loss:.4f}, " \
            "perf.: {perf:.4f}"
        extras = {"epoch": 0,
                  "loss": 0.,
                  "perf": 0.}
        with TqdmExtraFormat(total=total_steps, bar_format=bar_format, extras=extras) as pbar:
            for i in range(1, total_steps + 1):
                try:
                    x, target = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    x, target = next(data_iterator)
                x, target = x.to(device), target.to(device)
                out = self(x)
                loss = self.loss_f(out, target)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if use_scheduler:
                        self.scheduler.step(loss)
                pbar.extras["epoch"] = (i // steps_per_epoch)
                if perf_metric is not None:
                    perf_metric.tally(out, target)
                    pbar.extras["perf"] = perf_metric.measure()
                pbar.extras["loss"] = smoothing_factor * \
                    loss.item() + (1 - smoothing_factor) * \
                    pbar.extras["loss"]
                pbar.update()
            log.update(
                perf=(perf_metric.measure()  # type: ignore
                      if perf_metric is not None else None),
                final_loss=pbar.extras["loss"])
        return log

    def run_with_split(self,
                       train,
                       eval,
                       batch_size,
                       iterations,
                       epochs_per_iteration,
                       perf_metric,
                       smoothing_factor,
                       device,
                       use_scheduler=True):
        assert (perf_metric in ("acc", "f1"))
        for i in range(1, iterations + 1):

            label_shape = train[0][1].shape

            def make_perf_metric():
                if perf_metric == "acc":
                    return AccuracyMetric(label_shape, device=device)
                else:  # perf_metric == "f1"
                    return F1Metric(label_shape, device=device)

            training_log = self.run(
                train,
                batch_size,
                epochs_per_iteration,
                smoothing_factor,
                device,
                perf_metric=make_perf_metric(),
                use_scheduler=use_scheduler)

            train_log = self.run(
                train,
                batch_size,
                1,
                smoothing_factor,
                device,
                perf_metric=make_perf_metric(),
                train=False,
                shuffle=False)

            eval_log = self.run(
                eval,
                batch_size,
                1,
                smoothing_factor,
                device,
                perf_metric=make_perf_metric(),
                shuffle=False,
                train=False)

            yield (
                i,
                {
                    "perf_metric": perf_metric,
                    "batch_size": batch_size,
                    "training_set_size": len(train),
                    "evaluation_set_size": len(eval),
                    "epochs_per_iteration": epochs_per_iteration,
                    "training": training_log,
                    "train": train_log,
                    "eval": eval_log
                }
            )


def baseline(
        target_dim,
        input_size,
        downsampled_size,
        vit_dim,
        vit_depth,
        vit_heads,
        vit_head_dim,
        vit_mlp_dim,
        input_channels,
        conv_depth,
        device):
    return Network(PixelShuffle_ViT_Encoder(
        input_size,
        downsampled_size,
        vit_dim,
        vit_depth,
        vit_heads,
        vit_head_dim,
        vit_mlp_dim,
        input_channels=input_channels,
        conv_depth=conv_depth,
        output_dim=target_dim)).to(device)


def program_structured(
        graph_json,
        target_dim,
        *,
        input_size,
        downsampled_size,
        vit_dim,
        vit_depth,
        vit_heads,
        vit_head_dim,
        vit_mlp_dim,
        input_channels,
        conv_depth,
        gnn_depth,
        graph_dim,
        max_conn,
        dropout_rate,
        device):
    return Network(
        nn.Sequential(*[
            PixelShuffle_ViT_Encoder(
                input_size,
                downsampled_size,
                vit_dim,
                vit_depth,
                vit_heads,
                vit_head_dim,
                vit_mlp_dim,
                input_channels=input_channels,
                conv_depth=conv_depth,
                output_dim=graph_dim),
            GraphStructured(
                graph_json,
                max_conn,
                graph_dim,
                target_dim,
                gnn_depth,
                dropout_rate)])).to(device)
