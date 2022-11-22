from torch.nn import functional as F
import torch.nn as nn
import torch as th
from einops import rearrange


class PSN(nn.Module):

    def __init__(self,
                 program_length,
                 out_codebook_size,
                 codebook_dim,
                 dsl_name,
                 pre_quantizer,
                 quantizer,
                 post_quantizer,
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

        self.P = 2
        while self.P < program_length:
            self.P *= 2
        self.I = self.quantizer.dsl_size
        self.in_codebook = self.initial_codebook(self.I,
                                                 (self.E // self.P) * self.C)

        self.restarts = (None, [])
        self.beta = beta

    def initial_codebook(self, size, dim):
        codebook = nn.Embedding(size, dim)
        codebook.weight.data.uniform_(-1., 1.)
        codebook.weight.data -= codebook.weight.data.mean(dim=0)
        return codebook

    def forward(self, x, y, quantization_noise_std, mode):
        assert (mode in ("nearest", "learned"))
        if mode == "nearest":
            if self.restarts[0] is not None or len(self.restarts[1]) > 0:
                self.restarts = (None, [])
            latents = self.pre_quantizer(x)
            (quantized_latents_det, quantized_latents_noisy), _ = \
                self.nearest_neighbors(latents, self.out_codebook, self.E, quantization_noise_std)
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
            pg_out_latents, to_restart, programs = self.quantizer(
                pg_in_encoding_inds)
            self.restarts = (rearrange(pg_in_latents.detach(),
                                       "b (p r) c -> (b p) (r c)",
                                       p=self.P,
                                       r=(self.E // self.P),
                                       c=self.C), to_restart)
            (quantized_pg_out_latents_det, quantized_pg_out_latents_noisy), _ = \
                self.nearest_neighbors(pg_out_latents, self.out_codebook, self.E,
                    quantization_noise_std)
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

            return out, loss, programs

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

    def apply_restarts(self):
        starts, to_restart = self.restarts
        for r in to_restart:
            self.in_codebook.weight.data[r] = starts[th.randint(
                starts.size(0), (1, ))]

    def compression(self, next_dsl_name, *args, **kwargs):
        rewritten = self.quantizer.compress(next_dsl_name, *args, **kwargs)
        self.quantizer.load_dsl(next_dsl_name)
        self.I = self.quantizer.dsl_size
        in_codebook = self.initial_codebook(self.I,
                                            (self.E // self.P) * self.C)
        in_codebook.weight.data[1:] = self.in_codebook.weight.data
        self.in_codebook = in_codebook
        self.restarts = (None, [])
        return rewritten
