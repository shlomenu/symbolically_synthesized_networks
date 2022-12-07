import os

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
                 program_length=None,
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

        if program_length is None:
            self.P = self.E
        else:
            self.P = 2
            while self.P < program_length:
                self.P *= 2
        self.I = self.quantizer.dsl_size if hasattr(
            self.quantizer, "dsl_size") and program_length is not None else self.O
        self.in_codebook = self.initial_codebook(self.I,
                                                 (self.E // self.P) * self.C)
        self.out_restart_manager = RestartManager(8, self.O, contextual=False)
        self.in_restarts = (None, set())
        self.out_restarts = (None, set())
        self.beta = beta

    def initial_codebook(self, size, dim):
        codebook = nn.Embedding(size, dim)
        codebook.weight.data.uniform_(-1., 1.)
        codebook.weight.data -= codebook.weight.data.mean(dim=0)
        return codebook

    def forward(self, x, y, quantization_noise_std, mode, always_cache=False):
        assert (mode in ("none", "nearest", "learned"))
        if mode == "none":
            if self.in_restarts[0] is not None or len(self.in_restarts[1]) > 0:
                self.in_restarts = (None, set())
            if self.out_restarts[0] is not None or len(self.out_restarts[1]) > 0:
                self.out_restarts = (None, set())
            y = self.post_quantizer(self.pre_quantizer(x))
            return (y, self.post_quantizer.loss(x, y), None, None)
        if mode == "nearest":
            if self.in_restarts[0] is not None or len(self.in_restarts[1]) > 0:
                self.in_restarts = (None, set())
            latents = self.pre_quantizer(x)
            (quantized_latents_det, quantized_latents_noisy), _ = \
                self.nearest_neighbors(
                    latents, self.out_codebook, self.E, quantization_noise_std)
            out = self.post_quantizer(latents + (quantized_latents_noisy -
                                                 latents).detach())
            return (out,
                    self.vqvae_loss(
                        out,
                        y,
                        latents,
                        quantized_latents_det,
                        quantized_latents_noisy=quantized_latents_noisy), None, None)
        elif mode == "learned":
            pg_in_latents = self.pre_quantizer(x)
            (quantized_pg_in_latents, ), (pg_in_encoding_inds, ) = \
                self.nearest_neighbors(pg_in_latents,
                                       self.in_codebook, self.P, quantization_noise_std, noisy=False)
            pg_out_latents, in_restarts, hashes, programs = self.quantizer(
                pg_in_encoding_inds, always_cache=always_cache)
            self.in_restarts = (
                rearrange(pg_in_latents.detach(),
                          "b (p r) c -> (b p) (r c)",
                          p=self.P,
                          r=(self.E // self.P),
                          c=self.C),
                in_restarts)
            (quantized_pg_out_latents_det, quantized_pg_out_latents_noisy), (_, pg_out_encoding_inds_noisy) = \
                self.nearest_neighbors(pg_out_latents, self.out_codebook, self.E,
                                       quantization_noise_std)
            out_restarts = set()
            for cmds in pg_out_encoding_inds_noisy.tolist():
                out_restarts.update(
                    self.out_restart_manager.find_restarts(cmds))
            self.out_restarts = (
                rearrange(pg_in_latents.detach(),
                          "b e c -> (b e) c",
                          e=self.E,
                          c=self.C),
                out_restarts)
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
                                   quantized_latents_noisy=quantized_pg_out_latents_noisy,
                                   delta=.33) + \
                .33 * (self.non_recon_loss(pg_in_latents, quantized_pg_in_latents_slim) +
                       self.non_recon_loss(pg_out_latents,
                                           quantized_pg_out_latents_det,
                                           quantized_latents_noisy=quantized_pg_out_latents_noisy))

            return out, loss, hashes, programs

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

        encoding_inds_det = dist.argmin(dim=1, keepdim=True)  # b s
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
            train=True,
            always_cache=False):
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
        smoothed_loss, smoothed_diversity, all_programs = 0, 0, set()
        steps_per_epoch = len(dataset) // batch_size
        assert (steps_per_epoch * batch_size == len(dataset))
        total_steps = n_epochs * steps_per_epoch
        bar_format = "{bar}{r_bar} - " \
                     "epoch: {epoch:3.0f}, " \
                     "loss: {smoothed_loss:.4f}, " \
                     "div.: {smoothed_div:.4f}, " \
                     "tot. div.: {tot_div:.4f}"
        extras = {"epoch": 0, "smoothed_loss": 0.0,
                  "smoothed_div": float("NaN"), "tot_div": float("NaN")}
        with TqdmExtraFormat(total=total_steps, bar_format=bar_format, extras=extras) as pbar:
            for i, (x, y) in zip(range(total_steps), dataloader):
                x, y = x.to(device), y.to(device)
                _, loss, hashes, programs = self(
                    x, y, quantization_noise_std, mode, always_cache=always_cache)
                py_loss = loss.item()
                smoothed_loss = alpha * py_loss + (1 - alpha) * smoothed_loss
                if programs is not None:
                    smoothed_diversity = alpha * \
                        (len(set(programs)) / len(programs)) + \
                        (1 - alpha) * smoothed_diversity
                    all_programs.update(programs)
                if (i % steps_per_epoch) == 0:
                    pbar.extras["epoch"] = (i % steps_per_epoch)
                pbar.extras["smoothed_loss"] = smoothed_loss
                pbar.extras["smoothed_div"] = (
                    float("NaN") if programs is None else smoothed_diversity)
                pbar.extras["tot_div"] = (
                    float("NaN") if programs is None else (
                        len(all_programs) / (max(i * batch_size, 1.))))
                pbar.update()
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.apply_restarts()
                del (x, y, loss)
                yield (py_loss, hashes, programs, all_programs)

    def apply_restarts(self):
        starts, to_restart = self.in_restarts
        for r in to_restart:
            self.in_codebook.weight.data[r] = starts[th.randint(
                starts.size(0), (1, ))]
        self.in_restarts = (None, set())
        starts, to_restart = self.out_restarts
        for r in to_restart:
            self.out_codebook.weight.data[r] = starts[th.randint(
                starts.size(0), (1,))]
        self.out_restarts = (None, set())

    def compression(self, dataset, batch_size, max_compressed, next_dsl_name, device, *args, **kwargs):
        assignment = {}
        for i, (_, hashes, _, _) in enumerate(
                self.run(dataset, batch_size, 1,
                         "learned", device,
                         quantization_noise_std=0., shuffle=False,
                         train=False, always_cache=True)):
            for j, hash in zip(range(i * batch_size, (i + 1) * batch_size), hashes):
                assignment[j] = f"{hash}.json"
        reduction = self.quantizer.deduplicate()
        for j in assignment:
            assignment[j] = reduction[assignment[j]]
        frontier = list(assignment.values())
        if len(frontier) > max_compressed:
            frontier = np.random.choice(
                frontier, max_compressed, replace=False).tolist()
        self.quantizer.clear_visualizations()
        self.quantizer.visualize(frontier)
        rewritten = self.quantizer.compress(
            frontier, next_dsl_name, *args, **kwargs)
        self.quantizer.load_dsl(next_dsl_name)
        self.I = self.quantizer.dsl_size
        in_codebook = self.initial_codebook(self.I,
                                            (self.E // self.P) * self.C)
        in_codebook.weight.data[1:] = self.in_codebook.weight.data
        self.in_codebook = in_codebook
        self.in_restarts, self.out_restarts = (None, set()), (None, set())
        return rewritten
