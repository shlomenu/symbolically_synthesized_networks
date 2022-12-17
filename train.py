import pickle
from copy import deepcopy

import torch as th
import numpy as np

from raven import RavenDataset
from networks import (PixelShuffle_ViT_Encoder, PixelShuffle_ViT_Decoder,
                      PixelShuffle_ViT_Classifier, StrideConv_ViT_Encoder,
                      StrideConv_ViT_Decoder, StrideConv_ViT_Classifier)
from quantizers import GraphQuantizer, NullQuantizer
from psn import PSN

PROGRAM_SIZE = 12
OUT_CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 512
INITIAL_DSL = "graph_0"
DATASET_DIR = "graph/dataset"
N_TRAINING = 32
BATCH_SIZE = 8


def random_split(classification):
    return RavenDataset.bisplit(
        DATASET_DIR, N_TRAINING / 20000,
        .5 if classification else 1., BATCH_SIZE, classification)


def save_split(x, y, path):
    with open(path, "wb") as f:
        pickle.dump([x.multi_indices, y.multi_indices], f)


def load_split(path, classification):
    with open(path, "rb") as f:
        all_multi_indices = pickle.load(f)
    x, y = random_split(classification)
    x.multi_indices = all_multi_indices[0]
    y.multi_indices = all_multi_indices[1]
    return x, y


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


def train_split(psn: PSN, train, eval, iterations, epochs, mode, device, classification):
    for i in range(iterations):
        for _ in psn.run(train, BATCH_SIZE, epochs, mode, device, quantization_noise_std=0.):
            pass
        training_loss, eval_loss = 0, 0
        if classification:
            training_accuracy, eval_accuracy = 0, 0
            for (out, y, loss, _, _) in psn.run(
                    train, BATCH_SIZE, 1, mode, device, quantization_noise_std=0., shuffle=False, train=False):
                training_loss += loss
                training_accuracy += th.count_nonzero(th.softmax(
                    out, dim=1).argmax(dim=1, keepdim=True) == y).tolist()
            for (out, y, loss, _, _) in psn.run(
                    eval, BATCH_SIZE, 1, mode, device, quantization_noise_std=0., shuffle=False, train=False):
                eval_loss += loss
                eval_accuracy += th.count_nonzero(th.softmax(
                    out, dim=1).argmax(dim=1, keepdim=True) == y).tolist()
            training_accuracy /= len(train)
            eval_accuracy /= len(eval)
        else:
            training_accuracy, eval_accuracy = None, None
            training_loss += sum(loss for (_, _, loss, _, _) in psn.run(
                train, BATCH_SIZE, 1, mode, device, quantization_noise_std=0., shuffle=False, train=False))
            eval_loss += sum(loss for (_, _, loss, _, _) in psn.run(
                eval, BATCH_SIZE, 1, mode, device, quantization_noise_std=0., shuffle=False, train=False))

        if classification:
            yield (i, training_loss / len(train), training_accuracy, eval_loss / len(eval), eval_accuracy, psn)
        else:
            yield (i, bpd_of_nats(len(train), 128**2, training_loss), training_loss,
                   bpd_of_nats(len(eval), 128**2, eval_loss), eval_loss, psn)


def train_split_no_compression(psn, train, eval, iterations, epochs, mode, device, classification):
    training_curve, eval_curve = [], []
    for i, train_loss_metric, train_acc, eval_loss_metric, eval_acc, _ in train_split(psn, train, eval, iterations, epochs, mode, device, classification):
        if classification:
            training_curve.append((train_loss_metric, train_acc))
            eval_curve.append((eval_loss_metric, eval_acc))
            print(
                f"cycle: {i + 1}/{iterations}, train loss: {train_loss_metric:.4E}, train acc.: {train_acc:.4f}, eval loss: {eval_loss_metric:.4E}, eval acc.: {eval_acc:.4f}")
        else:
            training_curve.append(train_loss_metric)
            eval_curve.append(eval_loss_metric)
            print(
                f"cycle: {i + 1}/{iterations}, train bpd: {train_loss_metric:.4E}, eval. bpd: {eval_loss_metric:.4E}")
    return training_curve, eval_curve


def train_split_compression(psn,
                            train,
                            eval,
                            exploration_timeout,
                            run_name,
                            max_compressed,
                            iterations,
                            epochs,
                            mode,
                            device,
                            classification,
                            exploration_intervals=[],
                            compression_intervals=[],
                            exploration_kwargs={},
                            compression_kwargs={}):
    exploration_intervals = [int(v) for v in exploration_intervals]
    for v in exploration_intervals:
        assert (v > 0)
    if exploration_intervals:
        exploration_interval = exploration_intervals[0]
        exploration_intervals = exploration_intervals[1:]
    else:
        exploration_interval = None
    compression_intervals = [int(v) for v in compression_intervals]
    for v in compression_intervals:
        assert (v > 0)
    named_compression_intervals = [[f"graph_{run_name}_{i}", interval] for i,
                                   interval in enumerate(compression_intervals, start=1)]
    if named_compression_intervals:
        dsl_name, compression_interval = named_compression_intervals[0]
        named_compression_intervals = named_compression_intervals[1:]
    else:
        dsl_name, compression_interval = None, None
    training_curve, eval_curve = [], []
    print("performing exploration...")
    new, replaced, total = psn.exploration(
        exploration_timeout, PROGRAM_SIZE, **exploration_kwargs)
    print(f"new: {new}, replaced: {replaced}, total: {total}")
    for i, train_loss_metric, train_acc, eval_loss_metric, eval_acc, _ in train_split(
            psn, train, eval, iterations, epochs, mode, device, classification):
        performed_compression = False
        if compression_interval is not None:
            if compression_interval > 0:
                compression_interval -= 1
            if compression_interval == 0:
                performed_compression = psn.compression(train, BATCH_SIZE,
                                                        max_compressed, dsl_name, device, **compression_kwargs)
                if named_compression_intervals:
                    new_dsl_name, compression_interval = named_compression_intervals[0]
                    named_compression_intervals = named_compression_intervals[1:]
                else:
                    new_dsl_name, compression_interval = None, None
                if performed_compression:
                    dsl_name = new_dsl_name
        if exploration_interval is not None:
            if exploration_interval > 0:
                exploration_interval -= 1
            if exploration_interval == 0:
                print("performing exploration...")
                new, replaced, total = psn.exploration(
                    exploration_timeout, PROGRAM_SIZE, **exploration_kwargs)
                print(f"new: {new}, replaced: {replaced}, total: {total}")
                if exploration_intervals:
                    exploration_interval = exploration_intervals[0]
                    exploration_intervals = exploration_intervals[1:]
                else:
                    exploration_interval = None
        if classification:
            training_curve.append(
                (performed_compression, train_loss_metric, train_acc))
            eval_curve.append(
                (performed_compression, eval_loss_metric, eval_acc))
        print(
            f"cycle: {i + 1}/{iterations}, compress: {performed_compression}, train loss: {train_loss_metric:.4E}, train acc.: {train_acc:.4f}, eval loss: {eval_loss_metric:.4E}, eval acc.: {eval_acc:.4f}")
    return training_curve, eval_curve


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


def raven_classifier_strideconv(device="cuda:0"):
    post_quantizer_kwargs = deepcopy(strideconv_vit_kwargs)
    post_quantizer_kwargs.update(n_classes=2)
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              StrideConv_ViT_Encoder,
              NullQuantizer,
              StrideConv_ViT_Classifier,
              pre_quantizer_kwargs=strideconv_vit_kwargs,
              post_quantizer_kwargs=post_quantizer_kwargs)
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


def raven_vq_classifier_pixelshuffle(device="cuda:0"):
    post_quantizer_kwargs = deepcopy(pixelshuffle_vit_kwargs)
    post_quantizer_kwargs.update(n_classes=2)
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              NullQuantizer,
              PixelShuffle_ViT_Classifier,
              pre_quantizer_kwargs=pixelshuffle_vit_kwargs,
              post_quantizer_kwargs=post_quantizer_kwargs)
    return psn.to(device)


def raven_psn_autoencoder_pixelshuffle(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              GraphQuantizer,
              PixelShuffle_ViT_Decoder,
              program_size=PROGRAM_SIZE,
              pre_quantizer_kwargs=pixelshuffle_vit_kwargs,
              quantizer_kwargs={"max_color": 10},
              post_quantizer_kwargs=pixelshuffle_vit_kwargs)
    return psn.to(device)


def raven_psn_classifier_pixelshuffle(device="cuda:0"):
    post_quantizer_kwargs = deepcopy(pixelshuffle_vit_kwargs)
    post_quantizer_kwargs.update(n_classes=2)
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              GraphQuantizer,
              PixelShuffle_ViT_Classifier,
              program_size=PROGRAM_SIZE,
              pre_quantizer_kwargs=pixelshuffle_vit_kwargs,
              quantizer_kwargs={"max_color": 10},
              post_quantizer_kwargs=post_quantizer_kwargs)
    return psn.to(device)
