import torch as th
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch.simple_vit import posemb_sincos_2d, Transformer
from einops import rearrange
from einops.layers.torch import Rearrange


class QuantizationEmbedding(nn.Module):

    def __init__(self, E, C, n_representations, temperature=10000, coordinates_only=False):
        super().__init__()
        self.E, self.C, self.temperature = E, C, temperature
        self.coordinates_only = coordinates_only
        self.create_embedding_space(n_representations)

    def create_embedding_space(self, n_representations):
        self.n_representations = n_representations
        h, w, dim = self.E, self.n_representations, self.C
        y, x = th.meshgrid(th.arange(h), th.arange(w), indexing='ij')
        assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
        omega = th.arange(dim // 4) / (dim // 4 - 1)
        omega = 1. / (self.temperature ** omega)

        y = rearrange(y.flatten()[:, None] * omega[None, :],
                      "(h w) qc -> h w qc", h=h, w=w, qc=(dim // 4))
        x = rearrange(x.flatten()[:, None] * omega[None, :],
                      "(h w) qc -> h w qc", h=h, w=w, qc=(dim // 4))
        self.emb = th.cat(
            (x.sin(), x.cos(), y.sin(), y.cos()), dim=2).float()

    def forward(self, inputs):
        latents, selections = inputs
        if selections is not None:
            m = self.E // selections.size(1)
            embs = []
            for s in selections.tolist():
                coordinates = []
                for v in s:
                    coordinates += m * [v]
                embs.append(th.unsqueeze(
                    self.emb[th.arange(self.E), coordinates], dim=0))
            embs = th.cat(embs, dim=0).to(
                device=latents.device, dtype=latents.dtype).detach()
        else:
            assert (
                self.C % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
            omega = th.arange(self.C // 2) / (self.C // 2 - 1)
            omega = 1. / (self.temperature ** omega)
            wave = th.arange(self.E)[:, None] * omega[None, :]
            embs = th.cat((wave.sin(), wave.cos()), dim=1).unsqueeze(
                dim=0).repeat((latents.size(0), 1, 1)).to(device=latents.device, dtype=latents.dtype)

        return embs if self.coordinates_only else latents.reshape(embs.shape) + embs


class PositionalEmbedding2d(nn.Module):

    def forward(self, x):
        return x + rearrange(
            posemb_sincos_2d(x), "(h w) c -> h w c", h=x.size(1), w=x.size(2))


class StrideConv_ViT_Encoder(nn.Module):

    def __init__(self, codebook_dim, input_size, input_channels, upsize_channels,
                 vit_in_size, vit_depth, vit_heads, vit_head_dim, vit_mlp_dim):
        super().__init__()
        assert (input_size >= vit_in_size and vit_depth >= 1)
        stem_dims, HW, C = [], input_size, input_channels
        while HW > vit_in_size:
            out_channels = 2 * C if C != input_channels else upsize_channels
            stem_dims.append((C, out_channels))
            HW, C = HW // 2, out_channels
        assert (HW == vit_in_size and C == codebook_dim)
        layers = []
        for (in_channels, out_channels) in stem_dims:
            layers.extend([
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1,
                          stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ELU()
            ])
        layers.extend([
            nn.Conv2d(C, C, kernel_size=1),
            Rearrange("b c h w -> b h w c", h=vit_in_size, w=vit_in_size, c=C),
            PositionalEmbedding2d(),
            Rearrange("b h w c -> b (h w) c",
                      h=vit_in_size,
                      w=vit_in_size,
                      c=C),
            Transformer(C, vit_depth, vit_heads, vit_head_dim, vit_mlp_dim),
        ])
        self.net = nn.Sequential(*layers)
        self.inner_seq_len = vit_in_size**2

    def forward(self, img):
        return self.net(img)


class StrideConv_ViT_Decoder(nn.Module):

    def __init__(self, codebook_dim, codebook_size, input_size, input_channels, upsize_channels,
                 vit_in_size, vit_depth, vit_heads, vit_head_dim, vit_mlp_dim):
        super().__init__()
        assert (input_size >= vit_in_size and vit_depth >= 1)
        stem_dims, HW, C = [], input_size, input_channels
        while HW > vit_in_size:
            out_channels = 2 * C if C != input_channels else upsize_channels
            stem_dims.append((out_channels, C))
            HW, C = HW // 2, out_channels
        assert (HW == vit_in_size and C == codebook_dim)
        self.inner_seq_len = vit_in_size**2
        layers = [
            QuantizationEmbedding(self.inner_seq_len, C, codebook_size),
            Rearrange("b h w c -> b (h w) c",
                      h=vit_in_size,
                      w=vit_in_size,
                      c=C),
            Transformer(C, vit_depth, vit_heads, vit_head_dim, vit_mlp_dim),
            Rearrange("b (h w) c -> b c h w",
                      h=vit_in_size,
                      w=vit_in_size,
                      c=C)
        ]
        for (in_channels, out_channels) in stem_dims[::-1][:-1]:
            layers.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1,
                          stride=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ])
        in_channels, out_channels = stem_dims[::-1][-1]
        layers.extend([
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Tanh()
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, latents, selections):
        return self.net((latents, selections))

    def loss(self, out, y):
        return F.mse_loss(out, y)


class GlobalAvgPool(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x: th.Tensor):
        return self.net(x.mean(dim=1))


class StrideConv_ViT_Classifier(nn.Module):

    def __init__(self, codebook_dim, codebook_size, target_dim, input_size, input_channels, upsize_channels,
                 vit_in_size, vit_depth, vit_heads, vit_head_dim, vit_mlp_dim):
        super().__init__()
        assert (input_size >= vit_in_size and vit_depth >= 1)
        stem_dims, HW, C = [], input_size, input_channels
        while HW > vit_in_size:
            out_channels = 2 * C if C != input_channels else upsize_channels
            stem_dims.append((out_channels, C))
            HW, C = HW // 2, out_channels
        assert (HW == vit_in_size and C == codebook_dim)
        self.inner_seq_len = vit_in_size**2
        layers = [
            QuantizationEmbedding(self.inner_seq_len, C, codebook_size),
            Rearrange("b h w c -> b (h w) c",
                      h=vit_in_size,
                      w=vit_in_size,
                      c=C),
            Transformer(C, vit_depth, vit_heads, vit_head_dim, vit_mlp_dim),
            GlobalAvgPool(C, target_dim),
        ]
        self.net = nn.Sequential(*layers)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, latents, selections):
        return self.net((latents, selections))

    def loss(self, out: th.Tensor, y: th.Tensor):
        return self.loss_func(out, y.reshape(out.shape))


class ViT_Encoder(nn.Module):

    def __init__(self,
                 codebook_dim,
                 image_size,
                 patch_size,
                 vit_depth,
                 vit_heads,
                 vit_head_dim,
                 vit_mlp_dim,
                 channels=1):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = image_size // patch_size
        patch_dim = channels * patch_size**2
        self.net = nn.Sequential(*[
            Rearrange("b c (h p1) (w p2) -> b h w (p1 p2 c)",
                      h=num_patches,
                      w=num_patches,
                      p1=patch_size,
                      p2=patch_size,
                      c=channels),
            nn.Linear(patch_dim, codebook_dim),
            PositionalEmbedding2d(),
            Rearrange("b ... c -> b (...) c", c=codebook_dim),
            Transformer(codebook_dim, vit_depth, vit_heads, vit_head_dim,
                        vit_mlp_dim)
        ])
        self.inner_seq_len = num_patches**2

    def forward(self, img):
        return self.net(img)


class ViT_Decoder(nn.Module):

    def __init__(self,
                 codebook_dim,
                 codebook_size,
                 image_size,
                 patch_size,
                 vit_depth,
                 vit_heads,
                 vit_head_dim,
                 vit_mlp_dim,
                 channels=2):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = image_size // patch_size
        patch_dim = channels * patch_size**2
        self.inner_seq_len = num_patches**2
        self.net = nn.Sequential(*[
            QuantizationEmbedding(self.inner_seq_len,
                                  codebook_dim, codebook_size),
            Rearrange("b ... c -> b (...) c", c=codebook_dim),
            Transformer(codebook_dim, vit_depth, vit_heads, vit_head_dim,
                        vit_mlp_dim),
            nn.Linear(codebook_dim, patch_dim),
            Rearrange("b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                      h=num_patches,
                      w=num_patches,
                      p1=patch_size,
                      p2=patch_size,
                      c=channels),
            nn.Tanh()
        ])

    def forward(self, latents, selections):
        return self.net((latents, selections))

    def loss(self, out, y):
        return F.mse_loss(out, y)


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

    def __init__(self, codebook_dim, input_size, input_channels,
                 downsampled_size, conv_depth, vit_depth, vit_heads,
                 vit_head_dim, vit_mlp_dim):
        super().__init__()
        assert (input_size >= downsampled_size and vit_depth > 0)
        layers = [
            nn.PixelUnshuffle(input_size // downsampled_size),
            nn.Conv2d(input_channels * (input_size // downsampled_size)**2,
                      codebook_dim,
                      kernel_size=1)
        ]
        for _ in range(conv_depth):
            layers.append(ConvResidual(codebook_dim, downsampled_size))
        layers.extend([
            Rearrange("b c h w -> b h w c",
                      h=downsampled_size,
                      w=downsampled_size,
                      c=codebook_dim),
            PositionalEmbedding2d(),
            Rearrange("b h w c -> b (h w) c",
                      h=downsampled_size,
                      w=downsampled_size,
                      c=codebook_dim),
            Transformer(codebook_dim, vit_depth, vit_heads, vit_head_dim,
                        vit_mlp_dim),
        ])
        self.net = nn.Sequential(*layers)
        self.inner_seq_len = downsampled_size**2

    def forward(self, img):
        return self.net(img)


class PixelShuffle_ViT_Decoder(nn.Module):

    def __init__(self, codebook_dim, codebook_size, input_size, input_channels,
                 downsampled_size, conv_depth, vit_depth, vit_heads,
                 vit_head_dim, vit_mlp_dim):
        super().__init__()
        assert (input_size >= downsampled_size and vit_depth > 0)
        self.inner_seq_len = downsampled_size**2
        layers = [
            QuantizationEmbedding(self.inner_seq_len,
                                  codebook_dim, codebook_size),
            Transformer(codebook_dim, vit_depth, vit_heads, vit_head_dim,
                        vit_mlp_dim),
            Rearrange("b (h w) c -> b c h w",
                      h=downsampled_size,
                      w=downsampled_size,
                      c=codebook_dim)
        ]
        for _ in range(conv_depth):
            layers.append(ConvResidual(codebook_dim, downsampled_size))
        layers.extend([
            nn.Conv2d(codebook_dim,
                      input_channels * (input_size // downsampled_size)**2,
                      kernel_size=1),
            nn.Tanh(),
            nn.PixelShuffle(input_size // downsampled_size)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, latents, selections):
        return self.net((latents, selections))

    def loss(self, out, y):
        return F.mse_loss(out, y)


class PixelShuffle_ViT_Classifier(nn.Module):

    def __init__(self, codebook_dim, codebook_size, target_dim, input_size, input_channels,
                 downsampled_size, conv_depth, vit_depth, vit_heads,
                 vit_head_dim, vit_mlp_dim):
        super().__init__()
        assert (input_size >= downsampled_size and vit_depth > 0)
        self.inner_seq_len = downsampled_size**2
        layers = [
            QuantizationEmbedding(self.inner_seq_len,
                                  codebook_dim, codebook_size),
            Transformer(codebook_dim, vit_depth, vit_heads, vit_head_dim,
                        vit_mlp_dim),
            GlobalAvgPool(codebook_dim, target_dim),
        ]
        self.net = nn.Sequential(*layers)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, latents, selections):
        return self.net((latents, selections))

    def loss(self, out: th.Tensor, y: th.Tensor):
        return self.loss_func(out, y.reshape(out.shape))
