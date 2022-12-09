import pickle

import torch as th
import numpy as np

from raven import RavenDataset
from autoencoder import (PixelShuffle_ViT_Encoder, PixelShuffle_ViT_Decoder,
                         StrideConv_ViT_Encoder, StrideConv_ViT_Decoder)
from quantizers import GraphQuantizer, NullQuantizer
from psn import PSN

PROGRAM_LENGTH = 16
OUT_CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 512
INITIAL_DSL = "simplest"
DATASET_DIR = "graph/dataset"
BATCH_SIZE = 8


def random_portions(batches_per_portion):
    return RavenDataset.multisplit(DATASET_DIR, BATCH_SIZE, batches_per_portion)


def save_portions(portions, path):
    with open(path, "wb") as f:
        pickle.dump([portion.multi_indices for portion in portions], f)


def load_portions(path):
    with open(path, "rb") as f:
        all_multi_indices = pickle.load(f)
    portions = random_portions(len(all_multi_indices[0]) // BATCH_SIZE)
    for portion, multi_indices in zip(portions, all_multi_indices):
        portion.multi_indices = multi_indices
    return portions


strideconv_vit_kwargs = {
    "input_size": 128,
    "input_channels": 1,
    "upsize_channels": 128,
    "vit_in_size": 16,
    "vit_depth": 2,
    "vit_heads": 4,
    "vit_head_dim": 256,
    "vit_mlp_dim": 1024
}

pixelshuffle_vit_kwargs = {
    "input_size": 128,
    "input_channels": 1,
    "downsampled_size": 8,
    "conv_depth": 3,
    "vit_depth": 2,
    "vit_heads": 4,
    "vit_head_dim": 256,
    "vit_mlp_dim": 1024
}


def bpd_of_nats(n, dims, nats):
    return nats / (n * dims * np.log(2))


def train_apportioned(psn: PSN, portions, iterations, epochs, lookahead, mode, device):
    samples_per_portion = len(portions[0])
    for portion in portions[1:]:
        assert len(portion) == samples_per_portion
    for i in range(iterations):
        j = i % len(portions)
        train_portion = portions[j]
        for _ in psn.run(
                train_portion, BATCH_SIZE, epochs, mode, device, quantization_noise_std=0.):
            pass
        training_nats = sum(loss for (loss, _, _, _) in psn.run(
            train_portion, BATCH_SIZE, 1, mode, device, quantization_noise_std=0., shuffle=False, train=False))
        eval_nats = 0
        for j in range(i + 1, i + 1 + lookahead):
            eval_portion = portions[j % len(portions)]
            eval_nats += sum(loss for (loss, _, _, _) in psn.run(
                eval_portion, BATCH_SIZE, 1, mode, device, quantization_noise_std=0., shuffle=False, train=False))
        yield (
            i, j, bpd_of_nats(samples_per_portion, 128**2, training_nats),
            bpd_of_nats(samples_per_portion * lookahead, 128**2, eval_nats), psn)


def train_apportioned_no_compression(psn, portions, iterations, epochs, lookahead, mode, device):
    training_curve, eval_curve = [], []
    for i, j, train_bpd, eval_bpd, _ in train_apportioned(
            psn, portions, iterations, epochs, lookahead, mode, device):
        training_curve.append((i, j, train_bpd))
        eval_curve.append((i, j, eval_bpd))
        print(
            f"cycle: {i}/{iterations}, portion no.: {j}/{len(portions)}, train bpd: {train_bpd:.4E}, eval. bpd (lookahd {lookahead}): {eval_bpd:.4E}")
    return training_curve, eval_curve


def train_apportioned_compression(psn,
                                  portions,
                                  run_name,
                                  max_compressed,
                                  iterations,
                                  epochs,
                                  lookahead,
                                  mode,
                                  device,
                                  compression_intervals=[],
                                  **kwargs):
    for v in compression_intervals:
        assert (v > 0)
    compression_intervals = [int(v) for v in compression_intervals]
    named_intervals = [[f"graph_{run_name}_{i}", interval] for i,
                       interval in enumerate(compression_intervals, start=1)]
    if named_intervals:
        dsl_name, interval = named_intervals[0]
        named_intervals = named_intervals[1:]
    else:
        dsl_name, interval = None, None
    training_curve, eval_curve = [], []
    for i, j, train_bpd, eval_bpd, psn in train_apportioned(
            psn, portions, iterations, epochs, lookahead, mode, device):
        performed_compression = False
        if interval is not None:
            if interval > 0:
                interval -= 1
            if interval == 0:
                performed_compression = psn.compression(portions[i], BATCH_SIZE,
                                                        max_compressed, dsl_name, device, **kwargs)
                if named_intervals:
                    new_dsl_name, interval = named_intervals[0]
                    named_intervals = named_intervals[1:]
                else:
                    new_dsl_name, interval = None, None
                if performed_compression:
                    dsl_name = new_dsl_name
        training_curve.append((i, j, train_bpd, performed_compression))
        eval_curve.append((i, j, eval_bpd, performed_compression))
        print(
            f"cycle: {i}/{iterations}, portion no.: {j}/{len(portions)}, compress: {performed_compression}, train: {train_bpd:.4E}, eval (lookahd {lookahead}): {eval_bpd:.4E}")
    return training_curve, eval_curve


def raven_autoencoder_pixelshuffle(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              NullQuantizer,
              PixelShuffle_ViT_Decoder,
              pre_quantizer_kwargs=pixelshuffle_vit_kwargs,
              post_quantizer_kwargs=pixelshuffle_vit_kwargs)
    return psn.to(device)


def raven_autoencoder_strideconv(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              StrideConv_ViT_Encoder,
              NullQuantizer,
              StrideConv_ViT_Decoder,
              pre_quantizer_kwargs=strideconv_vit_kwargs,
              post_quantizer_kwargs=strideconv_vit_kwargs)
    return psn.to(device)


def raven_vqvae_pixelshuffle(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              NullQuantizer,
              PixelShuffle_ViT_Decoder,
              pre_quantizer_kwargs=pixelshuffle_vit_kwargs,
              post_quantizer_kwargs=pixelshuffle_vit_kwargs)
    return psn.to(device)


def raven_bp_vec_pixelshuffle():
    pass


def raven_bp_coord_pixelshuffle():
    pass


def raven_psn_pixelshuffle(device="cuda:0"):
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
    return psn.to(device)
