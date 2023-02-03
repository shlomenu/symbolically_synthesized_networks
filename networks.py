import torch as th
import torch.nn as nn
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


class PixelShuffle_ViT_Classifier(nn.Module):

    def __init__(self,
                 input_size,
                 downsampled_size,
                 vit_dim,
                 vit_depth,
                 vit_heads,
                 vit_head_dim,
                 vit_mlp_dim,
                 *,
                 target_dim,
                 no_transformer=False):
        super().__init__()
        assert (input_size >= downsampled_size and vit_depth > 0)
        if no_transformer:
            layers = [GlobalAvgPool(vit_dim, target_dim)]
        else:
            layers = [
                Transformer(vit_dim, vit_depth, vit_heads, vit_head_dim,
                            vit_mlp_dim),
                GlobalAvgPool(vit_dim, target_dim),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, latents):
        return self.net(latents)
