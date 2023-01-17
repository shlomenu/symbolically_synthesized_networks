from collections import defaultdict
import math
import os
from typing import Optional, Tuple, List

import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from einops import rearrange

from restart_manager import RestartManager


class TqdmExtraFormat(tqdm):

    def __init__(self, *args, extras={}, **kwargs):
        self.extras = extras
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        d.update(**self.extras)
        return d


class PSN(nn.Module):

    def __init__(self,
                 out_codebook_size,
                 codebook_dim,
                 dsl_name,
                 pre_quantizer,
                 quantizer,
                 post_quantizer,
                 two_stage_quantization=True,
                 pre_quantizer_kwargs={},
                 quantizer_kwargs={},
                 post_quantizer_kwargs={},
                 beta: float = 0.25):
        super().__init__()
        self.C = codebook_dim
        self.pre_quantizer = pre_quantizer(self.C, **pre_quantizer_kwargs)
        self.post_quantizer = post_quantizer(self.C, **post_quantizer_kwargs)
        self.two_stage_quantization = two_stage_quantization
        assert (self.pre_quantizer.inner_seq_len ==
                self.post_quantizer.inner_seq_len)
        self.E = self.post_quantizer.inner_seq_len
        self.quantizer = quantizer(self.E, self.C, dsl_name,
                                   **quantizer_kwargs)

        self.O = out_codebook_size
        self.out_codebook = self.initial_codebook(self.O, self.C)
        self.P, self.I = (self.E, self.O) if two_stage_quantization else (1, len(
            self.quantizer))
        if two_stage_quantization:
            self.in_codebook = self.initial_codebook(self.I,
                                                     (self.E // self.P) * self.C)
        self.out_restart_manager = RestartManager(8, self.O)
        self.in_restarts: Optional[Tuple[th.Tensor, List[int]]] = None
        self.out_restarts: Optional[Tuple[th.Tensor, List[int]]] = None
        self.beta = beta

    def initial_codebook(self, size, dim):
        codebook = nn.Embedding(size, dim)
        codebook.weight.data.uniform_(-1. / (size if size >
                                      0. else 1.), 1. / (size if size > 0. else 1.))
        codebook.weight.data -= codebook.weight.data.mean(dim=0)
        return codebook

    def forward(self, x, y, quantization_noise_std, mode):
        if self.two_stage_quantization:
            assert (mode in ("none", "nearest"))
        else:
            assert (mode not in ("none", "nearest"))
            mode = "learned"
        if mode == "none":
            if self.in_restarts is not None:
                self.in_restarts = None
            if self.out_restarts is not None:
                self.out_restarts = None
            encoding_inds = None
            out = self.post_quantizer(self.pre_quantizer(x), encoding_inds)
            return (out, self.post_quantizer.loss(out, y), None)
        if mode == "nearest":
            if self.in_restarts is not None:
                self.in_restarts = None
            latents = self.pre_quantizer(x)
            (quantized_latents_det, quantized_latents_noisy), (_, encoding_inds_noisy) = \
                self.nearest_neighbors(
                    latents, self.out_codebook, self.E, quantization_noise_std)
            self._set_out_restarts(latents, encoding_inds_noisy)
            out = self.post_quantizer(latents + (quantized_latents_noisy -
                                                 latents).detach(),
                                      encoding_inds_noisy)
            return (out,
                    self.vqvae_loss(
                        out,
                        y,
                        latents,
                        quantized_latents_det,
                        quantized_latents_noisy=quantized_latents_noisy), None)
        elif mode == "learned":
            pg_in_latents = self.pre_quantizer(x)
            (quantized_pg_in_latents, ), (pg_in_encoding_inds, ) = \
                self.nearest_neighbors(pg_in_latents,
                                       self.in_codebook, self.P, quantization_noise_std, noisy=False)
            pg_out_latents, in_restarts, filenames = self.quantizer(
                quantized_pg_in_latents.detach(), pg_in_encoding_inds)
            if in_restarts:
                self.in_restarts = (
                    rearrange(pg_in_latents.detach(),
                              "b (p r) c -> (b p) (r c)",
                              p=self.P,
                              r=(self.E // self.P),
                              c=self.C),
                    in_restarts)
            else:
                self.in_restarts = None
            (quantized_pg_out_latents_det, quantized_pg_out_latents_noisy), (_, pg_out_encoding_inds_noisy) = \
                self.nearest_neighbors(pg_out_latents, self.out_codebook, self.E,
                                       quantization_noise_std)
            self._set_out_restarts(pg_in_latents, pg_out_encoding_inds_noisy)
            out = self.post_quantizer(pg_in_latents +
                                      (quantized_pg_out_latents_noisy -
                                       pg_in_latents).detach(),
                                      pg_out_encoding_inds_noisy)
            quantized_pg_in_latents_slim = rearrange(quantized_pg_in_latents,
                                                     "b p (r c) -> b (p r) c",
                                                     p=self.P,
                                                     r=(self.E // self.P),
                                                     c=self.C)
            loss = self.vqvae_loss(out,
                                   y,
                                   pg_in_latents,
                                   quantized_pg_out_latents_det,
                                   quantized_latents_noisy=quantized_pg_out_latents_noisy) + \
                (self.non_recon_loss(pg_in_latents, quantized_pg_in_latents_slim) +
                 self.non_recon_loss(pg_out_latents,
                                     quantized_pg_out_latents_det,
                                     quantized_latents_noisy=quantized_pg_out_latents_noisy))

            return out, loss, filenames

    def _set_out_restarts(self, latents, encoding_inds):
        self.out_restarts, out_restarts = None, []
        for i, cmds in enumerate(encoding_inds.tolist()):
            if i == encoding_inds.size(0) - 1:
                out_restarts: List[int] = self.out_restart_manager.find_restarts(
                    cmds)
        if out_restarts:
            self.out_restarts = (
                rearrange(latents.detach(),
                          "b e c -> (b e) c",
                          e=self.E,
                          c=self.C),
                out_restarts)

    def nearest_neighbors(self, latents, codebook, S, noise_std, noisy=True):
        K = codebook.weight.size(0)
        flat_latents = rearrange(latents,
                                 "b (s r) c -> (b s) (r c)",
                                 s=S,
                                 r=(self.E // S),
                                 c=self.C)

        dist = th.sum(flat_latents ** 2, dim=1, keepdim=True) + \
            th.sum(codebook.weight ** 2, dim=1) - \
            2 * th.matmul(flat_latents,
                          codebook.weight.t())  # (b s) k

        if S == 1:
            encoding_inds_det = dist.argmin(
                dim=1).unsqueeze(dim=1)
        else:
            encoding_inds_det = dist.argmin(
                dim=1, keepdim=True)  # b s

        if noisy:
            encoding_inds = (encoding_inds_det, (encoding_inds_det + th.round(
                th.normal(mean=0,
                          std=noise_std,
                          size=encoding_inds_det.shape,
                          device=encoding_inds_det.device,
                          dtype=th.float)).long()).clamp(0, K - 1))
        else:
            encoding_inds = (encoding_inds_det, )

        quantized_latents = []
        for inds in encoding_inds:
            encoding_one_hot = th.zeros(inds.size(0) * inds.size(1),
                                        K,
                                        device=latents.device)  # (b s) k
            encoding_one_hot.scatter_(dim=1, index=inds, value=1)
            codes = th.matmul(encoding_one_hot, codebook.weight)
            quantized_latents.append(
                rearrange(codes,
                          "(b s) (r c) -> b s (r c)",
                          s=S,
                          r=(self.E // S),
                          c=self.C))

        return quantized_latents, [
            rearrange(inds, "(b s) d -> b (s d)", s=S, d=1)
            for inds in encoding_inds
        ]

    def vqvae_loss(self,
                   out,
                   y,
                   latents,
                   quantized_latents_det,
                   quantized_latents_noisy=None,
                   delta=1.):
        return self.post_quantizer.loss(out, y) + self.non_recon_loss(
            latents,
            quantized_latents_det=quantized_latents_det,
            quantized_latents_noisy=quantized_latents_noisy,
            delta=delta)

    def non_recon_loss(self,
                       latents,
                       quantized_latents_det,
                       quantized_latents_noisy=None,
                       delta=1.):
        adjusted_quantized_latents = quantized_latents_det if quantized_latents_noisy is None else quantized_latents_noisy
        return delta * (
            self.beta * F.mse_loss(latents, quantized_latents_det.detach()) +
            F.mse_loss(adjusted_quantized_latents, latents.detach()))

    def run(self,
            dataset,
            batch_size,
            n_epochs,
            mode,
            device,
            quantization_noise_std=.5,
            alpha=.8,
            scheduler=None,
            optimizer=None,
            perf_metric=None,
            shuffle=True,
            train=True,
            use_scheduler=True):
        if train:
            self.train()
        else:
            self.eval()
        self.out_restart_manager.idleness_limit = batch_size * 3
        if hasattr(self.quantizer, "restart_manager"):
            self.quantizer.restart_manager.idleness_limit = batch_size * 3
        if optimizer is not None:
            self.optimizer = optimizer
        elif not hasattr(self, "optimizer") or self.optimizer is None:
            self.optimizer = th.optim.Adam(self.parameters())
        if scheduler is not None:
            self.scheduler = scheduler(self.optimizer)
        elif not hasattr(self, "scheduler") or self.scheduler is None:
            self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=.5, patience=50,
                verbose=True, threshold=1e-4, cooldown=15)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        total_loss, total_mass, repr_usage = 0., 0., defaultdict(
            int)
        assert ((len(dataset) % batch_size) == 0)
        steps_per_epoch = len(dataset) // batch_size
        total_steps = n_epochs * steps_per_epoch
        bar_format = "{bar}{r_bar} - " \
                     "epoch: {epoch:3.0f}, " \
                     "loss: {loss:.4f}, " \
                     "tot. loss: {tot_loss:.4f}, " \
                     "div.: {div:.4f}, " \
                     "tot. div.: {tot_div:.4f}, " \
                     "mass: {mass:.4f}, " \
                     "tot. mass: {tot_mass:.4f}, " \
                     "perf.: {perf:.4f}"
        extras = {"epoch": 0, "loss": 0., "tot_loss": 0.,
                  "div": float("NaN"), "tot_div": float("NaN"),
                  "mass": float("NaN"), "tot_mass": float("NaN"),
                  "perf": float("NaN")}
        data_iterator = iter(dataloader)
        with TqdmExtraFormat(total=total_steps, bar_format=bar_format, extras=extras) as pbar:
            for i in range(1, total_steps + 1):
                try:
                    x, target = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    x, target = next(data_iterator)
                x, target = x.to(device), target.to(device)
                output, loss, filenames = self(
                    x, target, quantization_noise_std, mode)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.apply_restarts()
                    if use_scheduler:
                        self.scheduler.step(loss)
                pbar.extras["epoch"] = (i // steps_per_epoch)
                if perf_metric is not None:
                    perf_metric.tally(output, target)
                    pbar.extras["perf"] = perf_metric.measure()
                loss = loss.item()
                pbar.extras["loss"] = alpha * \
                    loss + (1 - alpha) * pbar.extras["loss"]
                total_loss += loss
                pbar.extras["tot_loss"] = total_loss / i
                if filenames is not None:
                    pbar.extras["div"] = alpha * \
                        (len(set(filenames)) / len(filenames)) + \
                        (1 - alpha) * \
                        (0. if math.isnan(pbar.extras["div"])
                         else pbar.extras["div"])
                    for filename in filenames:
                        repr_usage[filename] += 1
                    pbar.extras["tot_div"] = len(
                        repr_usage) / len(self.quantizer)
                    mass = self.quantizer.mass_of_representations(
                        filenames)
                    pbar.extras["mass"] = mass
                    total_mass += mass
                    pbar.extras["tot_mass"] = total_mass / i
                pbar.update()
        if train:
            self.quantizer.repr_usage = repr_usage
        return {
            "perf": perf_metric.measure() if perf_metric is not None else None,
            "total_loss": pbar.extras["tot_loss"],
            "total_diversity": pbar.extras["tot_div"],
            "total_mass": pbar.extras["tot_mass"],
            "representations_usages": self.quantizer.repr_usage
        }

    def apply_restarts(self):
        if self.in_restarts is not None:
            starts, to_restart = self.in_restarts
            randomized = starts[np.random.choice(starts.size(0),
                                                 min(len(to_restart), starts.size(0)), replace=False)]
            self.in_codebook.weight.data[to_restart] = randomized
            self.quantizer.restart_manager.reset_count(to_restart)
            self.in_restarts = None
        if self.out_restarts is not None:
            starts, to_restart = self.out_restarts
            randomized = starts[np.random.choice(starts.size(0),
                                                 min(len(to_restart), starts.size(0)), replace=False)]
            self.out_codebook.weight.data[to_restart] = randomized
            self.out_restarts = None

    def exploration(self,
                    frontier,
                    exploration_timeout,
                    next_dsl_name="dsl",
                    **kwargs):
        log = self.quantizer.explore(
            frontier, next_dsl_name, exploration_timeout, **kwargs)
        if len(self.quantizer) != self.I:
            I = len(self.quantizer)
            in_codebook = self.initial_codebook(I,
                                                (self.E // self.P) * self.C).to(
                self.in_codebook.weight.device)
            # in_codebook.weight.data[:self.I] = self.in_codebook.weight.data
            self.I, self.in_codebook = I, in_codebook
            for code in range(self.I):
                self.out_restart_manager.add_code(code)
            self.in_restarts, self.out_restarts = None, None
        return log

    def compression(self,
                    frontier,
                    next_dsl_name="dsl",
                    stitch_compression=True,
                    **kwargs):
        if stitch_compression:
            log = self.quantizer.stitch_compress(
                frontier, next_dsl_name, **kwargs)
        else:
            log = self.quantizer.dreamcoder_compress(
                frontier, next_dsl_name, **kwargs)
        log.update(stitch_compression=stitch_compression)
        return log

    def make_frontier(self, repr_usage, frontier_size):
        log = {}
        self.quantizer.repr_usage = repr_usage
        if len(repr_usage) > frontier_size:
            log.update(truncated_usages=True)
            total = sum(repr_usage.values())
            frontier = np.random.choice(
                list(repr_usage.keys()),
                int(frontier_size),
                replace=True,
                p=[count / total for count in repr_usage.values()]).tolist()
            log.update(frontier_div=len(
                repr_usage) / len(self.quantizer))
            log.update(
                frontier_mass=self.quantizer.mass_of_representations(frontier))
        else:
            log.update(truncated_usages=False)
            frontier = []
            for filename, count in repr_usage.items():
                frontier += [filename] * count
            log.update(frontier_div=len(repr_usage) / len(self.quantizer))
            log.update(
                frontier_mass=self.quantizer.mass_of_representations(frontier))
        return frontier, log
