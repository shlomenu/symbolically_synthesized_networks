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
                 program_size=None,
                 pre_quantizer_kwargs={},
                 quantizer_kwargs={},
                 post_quantizer_kwargs={},
                 beta: float = 0.25):
        super().__init__()
        self.C = codebook_dim
        self.pre_quantizer = pre_quantizer(self.C, **pre_quantizer_kwargs)
        self.post_quantizer = post_quantizer(self.C, **post_quantizer_kwargs)
        assert (self.pre_quantizer.inner_seq_len ==
                self.post_quantizer.inner_seq_len)
        self.E = self.post_quantizer.inner_seq_len
        self.quantizer = quantizer(self.E, self.C, dsl_name,
                                   **quantizer_kwargs)

        self.O = out_codebook_size
        self.out_codebook = self.initial_codebook(self.O, self.C)
        self.P, self.I = (self.E, self.O) if program_size is None else (1, len(
            self.quantizer))
        self.in_codebook = self.initial_codebook(self.I,
                                                 (self.E // self.P) * self.C)
        self.out_restart_manager = RestartManager(8, self.O)
        self.in_restarts: Optional[Tuple[th.Tensor, List[int]]] = None
        self.out_restarts: Optional[Tuple[th.Tensor, List[int]]] = None
        self.beta = beta

    def initial_codebook(self, size, dim):
        codebook = nn.Embedding(size, dim)
        codebook.weight.data.uniform_(-1., 1.)
        codebook.weight.data -= codebook.weight.data.mean(dim=0)
        return codebook

    def forward(self, x, y, quantization_noise_std, mode):
        assert (mode in ("none", "nearest", "learned"))
        if mode == "none":
            if self.in_restarts is not None:
                self.in_restarts = None
            if self.out_restarts is not None:
                self.out_restarts = None
            out = self.post_quantizer(self.pre_quantizer(x))
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
                                                 latents).detach())
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
                pg_in_encoding_inds.flatten())
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
                                       pg_in_latents).detach())
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
            optimizer=None,
            shuffle=True,
            train=True):
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
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        smoothed_loss, smoothed_diversity, all_representations = 0, 0, set()
        assert ((len(dataset) % batch_size) == 0)
        steps_per_epoch = len(dataset) // batch_size
        total_steps = n_epochs * steps_per_epoch
        bar_format = "{bar}{r_bar} - " \
                     "epoch: {epoch:3.0f}, " \
                     "loss: {smoothed_loss:.4f}, " \
                     "div.: {smoothed_div:.4f}, " \
                     "tot. div.: {tot_div:.4f}"
        extras = {"epoch": 0, "smoothed_loss": 0.0,
                  "smoothed_div": float("NaN"), "tot_div": float("NaN")}
        data_iterator = iter(dataloader)
        with TqdmExtraFormat(total=total_steps, bar_format=bar_format, extras=extras) as pbar:
            for i in range(1, total_steps + 1):
                try:
                    x, y = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    x, y = next(data_iterator)
                x, y = x.to(device), y.to(device)
                out, loss, filenames = self(
                    x, y, quantization_noise_std, mode)
                py_loss = loss.item()
                smoothed_loss = alpha * py_loss + (1 - alpha) * smoothed_loss
                if filenames is not None:
                    smoothed_diversity = alpha * \
                        (len(set(filenames)) / len(filenames)) + \
                        (1 - alpha) * smoothed_diversity
                    all_representations.update(filenames)
                pbar.extras["epoch"] = (i // steps_per_epoch)
                pbar.extras["smoothed_loss"] = smoothed_loss
                pbar.extras["smoothed_div"] = (
                    float("NaN") if filenames is None else smoothed_diversity)
                pbar.extras["tot_div"] = (
                    float("NaN") if filenames is None else (
                        len(all_representations) / len(self.quantizer)))
                pbar.update()
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.apply_restarts()
                del (x, loss)
                yield (out.detach(), y, py_loss, filenames, all_representations)

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

    def exploration(self, exploration_timeout, program_size, **kwargs):
        new, replaced = self.quantizer.explore(
            exploration_timeout, program_size, **kwargs)
        if len(self.quantizer) != self.I:
            I = len(self.quantizer)
            in_codebook = self.initial_codebook(I,
                                                (self.E // self.P) * self.C).to(
                self.in_codebook.weight.device)
            in_codebook.weight.data[:self.I] = self.in_codebook.weight.data
            self.I, self.in_codebook = I, in_codebook
            for code in range(self.I):
                self.out_restart_manager.add_code(code)
            self.in_restarts, self.out_restarts = None, None
        return new, replaced, self.I

    def compression(self, dataset, batch_size, max_compressed, next_dsl_name, device, **kwargs):
        frontier = []
        for (_, _, _, filenames, _) in self.run(dataset, batch_size, 1,
                                                "learned", device,
                                                quantization_noise_std=0., shuffle=False,
                                                train=False):
            frontier.extend(filenames)
        if len(frontier) > max_compressed:
            frontier = np.random.choice(
                frontier, max_compressed, replace=False).tolist()
        self.quantizer.clear_visualizations()
        self.quantizer.visualize(set(frontier))
        rewritten = self.quantizer.compress(
            frontier, next_dsl_name, **kwargs)
        return rewritten
