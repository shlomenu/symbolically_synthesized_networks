import torch as th
import numpy as np

from raven import RavenDataset
from autoencoder import (PixelShuffle_ViT_Encoder, PixelShuffle_ViT_Decoder,
                         StrideConv_ViT_Encoder, StrideConv_ViT_Decoder)
from quantizers import GraphQuantizer, NullQuantizer
from psn import PSN

PROGRAM_LENGTH = 8
OUT_CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 512
INITIAL_DSL = "graph_0"
DATASET_DIR = "graph/dataset"

train, val, test = RavenDataset.split(DATASET_DIR, .2, .2, )

strideconv_vit_kwargs = {
    "input_size": 128,
    "input_channels": 1,
    "upsize_channels": 128,
    "vit_in_size": 16,
    "vit_depth": 3,
    "vit_heads": 4,
    "vit_head_dim": 256,
    "vit_mlp_dim": 1024
}

pixelshuffle_vit_kwargs = {
    "input_size": 128,
    "input_channels": 1,
    "downsampled_size": 8,
    "conv_depth": 5,
    "vit_depth": 3,
    "vit_heads": 4,
    "vit_head_dim": 256,
    "vit_mlp_dim": 1024
}


def train_early_stopping(psn: PSN, batch_size, mode, device, stopping_criterion, peak_save_path):
    training_curve, validation_curve, since_last_best, best = [
    ], [], 0, float("inf")
    while since_last_best < stopping_criterion:
        training_nats = 0
        for loss, _, _, _ in psn.run(train, batch_size, 1, mode, device, quantization_noise_std=0., log_frequency=5):
            training_nats += loss
        training_bits_per_dim = training_nats / \
            (len(train) * (128**2) * np.log(2))
        training_curve.append(training_bits_per_dim)
        validation_nats = 0
        for loss, _, _, _ in psn.run(val, batch_size, 1, mode, device, quantization_noise_std=0., log_frequency=5, shuffle=False, train=False):
            validation_nats += loss
        validation_bits_per_dim = validation_nats / \
            (len(val) * (128**2) * np.log(2))
        validation_curve.append(validation_bits_per_dim)
        if validation_bits_per_dim >= best:
            th.save(psn.state_dict(), peak_save_path)
            best, since_last_best = validation_bits_per_dim, 0
        else:
            since_last_best += 1
    psn.load_state_dict(th.load(peak_save_path))
    test_nats = 0
    for loss, _, _, _ in psn.run(test, batch_size, 1, mode, device, quantization_noise_std=0., log_frequency=5):
        test_nats += loss
    test_bits_per_dim = test_nats / \
        (len(test) * (128**2) * np.log(2))
    return training_curve, validation_curve, test_bits_per_dim


def raven_autoencoder_strideconv(batch_size, stopping_criterion=5, device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              StrideConv_ViT_Encoder,
              NullQuantizer,
              StrideConv_ViT_Decoder,
              pre_quantizer_kwargs=strideconv_vit_kwargs,
              post_quantizer_kwargs=strideconv_vit_kwargs)
    psn = psn.to(device)
    training_curve, validation_curve, test_bits_per_dim = \
        train_early_stopping(psn, batch_size, "none",
                             device, stopping_criterion,
                             "peak_raven_autoencoder_strideconv.pth")
    return psn, training_curve, validation_curve, test_bits_per_dim


def raven_vqvae_strideconv(batch_size, stopping_criterion=5, device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              StrideConv_ViT_Encoder,
              NullQuantizer,
              StrideConv_ViT_Decoder,
              pre_quantizer_kwargs=strideconv_vit_kwargs,
              post_quantizer_kwargs=strideconv_vit_kwargs)
    psn = psn.to(device)
    training_curve, validation_curve, test_bits_per_dim = \
        train_early_stopping(psn, batch_size, "nearest",
                             device, stopping_criterion,
                             "peak_raven_vqvae_strideconv.pth")
    return psn, training_curve, validation_curve, test_bits_per_dim


def raven_vqvae_pixelshuffle(batch_size, stopping_criterion=5, device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              NullQuantizer,
              PixelShuffle_ViT_Decoder,
              pre_quantizer_kwargs=pixelshuffle_vit_kwargs,
              post_quantizer_kwargs=pixelshuffle_vit_kwargs)
    psn = psn.to(device)
    training_curve, validation_curve, test_bits_per_dim = \
        train_early_stopping(psn, batch_size, "nearest",
                             device, stopping_criterion,
                             "peak_raven_vqvae_pixelshuffle.pth")
    return psn, training_curve, validation_curve, test_bits_per_dim


def raven_psn_pixelshuffle(batch_size, next_dsl_name, max_compressed=300, stopping_criterion=5, device="cuda:0", psn=None):
    if psn is None:
        psn = PSN(OUT_CODEBOOK_SIZE,
                  CODEBOOK_DIM,
                  INITIAL_DSL,
                  PixelShuffle_ViT_Encoder,
                  GraphQuantizer,
                  PixelShuffle_ViT_Decoder,
                  program_length=PROGRAM_LENGTH,
                  pre_quantizer_kwargs=pixelshuffle_vit_kwargs,
                  quantizer_kwargs={"max_color": 10},
                  post_quantizer_kwargs=pixelshuffle_vit_kwargs)
        psn = psn.to(device)
    training_curve, validation_curve, test_bits_per_dim = \
        train_early_stopping(psn, batch_size, "learned",
                             device, stopping_criterion,
                             "peak_raven_psn_pixelshuffle.pth")
    psn.compression(train, batch_size, max_compressed, next_dsl_name)
    return psn, training_curve, validation_curve, test_bits_per_dim
