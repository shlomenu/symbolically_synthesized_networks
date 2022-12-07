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
BATCH_SIZE = 32

train, val, test = RavenDataset.split(DATASET_DIR, .2, .2, )

portions = RavenDataset.multisplit(DATASET_DIR, BATCH_SIZE, 25)

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


def bpd_of_nats(dataset, nats):
    return nats / (len(dataset) * (128 ** 2) * np.log(2))


def train_portions(psn: PSN, cycles, epochs, lookahead, mode, device):
    for c in range(cycles):
        for i, train_portion in enumerate(portions):
            training_bits_per_dim = bpd_of_nats(train_portion, sum(loss for (loss, _, _, _) in psn.run(
                train_portion, BATCH_SIZE, epochs, mode, device, quantization_noise_std=0.)))
            eval_bits_per_dim = 0
            for j in range(i + 1, i + 1 + lookahead):
                eval_portion = portions[j % len(portions)]
                eval_bits_per_dim += bpd_of_nats(eval_portion, sum(loss for (loss, _, _, _) in psn.run(
                    eval_portion, BATCH_SIZE, 1, mode, device, quantization_noise_std=0., shuffle=False, train=False)))
            eval_bits_per_dim /= lookahead
            yield c, i, training_bits_per_dim, eval_bits_per_dim, psn


def train_early_stopping(psn: PSN, mode, device, max_epochs, stopping_criterion, peak_save_path):
    training_curve, validation_curve, since_last_best, best = [
    ], [], 0, float("inf")
    for _ in range(max_epochs):
        training_nats = 0
        for loss, _, _, _ in psn.run(train, BATCH_SIZE, 1, mode, device, quantization_noise_std=0.):
            training_nats += loss
        training_bits_per_dim = training_nats / \
            (len(train) * (128**2) * np.log(2))
        training_curve.append(training_bits_per_dim)
        validation_nats = 0
        for loss, _, _, _ in psn.run(val, BATCH_SIZE, 1, mode, device, quantization_noise_std=0., shuffle=False, train=False):
            validation_nats += loss
        validation_bits_per_dim = validation_nats / \
            (len(val) * (128**2) * np.log(2))
        validation_curve.append(validation_bits_per_dim)
        if validation_bits_per_dim <= best:
            th.save(psn.state_dict(), peak_save_path)
            best, since_last_best = validation_bits_per_dim, 0
        else:
            since_last_best += 1
        if since_last_best >= stopping_criterion:
            break
    psn.load_state_dict(th.load(peak_save_path))
    test_nats = 0
    for loss, _, _, _ in psn.run(test, BATCH_SIZE, 1, mode, device, quantization_noise_std=0.):
        test_nats += loss
    test_bits_per_dim = test_nats / \
        (len(test) * (128**2) * np.log(2))
    return training_curve, validation_curve, test_bits_per_dim


def raven_autoencoder_pixelshuffle(max_epochs=10, stopping_criterion=3, device="cuda:0"):
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
        train_early_stopping(psn, "none",
                             device, max_epochs, stopping_criterion,
                             "peak_raven_autoencoder_pixelshuffle.pth")
    return psn, training_curve, validation_curve, test_bits_per_dim


def raven_autoencoder_strideconv(max_epochs=10, stopping_criterion=3, device="cuda:0"):
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
        train_early_stopping(psn, "none",
                             device, max_epochs, stopping_criterion,
                             "peak_raven_autoencoder_strideconv.pth")
    return psn, training_curve, validation_curve, test_bits_per_dim


def raven_vqvae_pixelshuffle(max_epochs=10, stopping_criterion=3, device="cuda:0"):
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
        train_early_stopping(psn, "nearest",
                             device, max_epochs, stopping_criterion,
                             "peak_raven_vqvae_pixelshuffle.pth")
    return psn, training_curve, validation_curve, test_bits_per_dim


def raven_bp_vec_pixelshuffle():
    pass


def raven_bp_coord_pixelshuffle():
    pass


def raven_psn_pixelshuffle(max_epochs=10,
                           stopping_criterion=3,
                           device="cuda:0",
                           psn=None):
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
        train_early_stopping(psn, "learned",
                             device, max_epochs, stopping_criterion,
                             "peak_raven_psn_pixelshuffle.pth")
    return psn, training_curve, validation_curve, test_bits_per_dim
