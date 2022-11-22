from tqdm import tqdm
import torch as th
from torch.utils.data import DataLoader

from raven import RavenDataset
from autoencoder import (PixelShuffle_ViT_Encoder, PixelShuffle_ViT_Decoder,
                         StrideConv_ViT_Encoder, StrideConv_ViT_Decoder,
                         ViT_Encoder, ViT_Decoder)
from graph_quantizer import GraphQuantizer
from psn import PSN


def fetch_raven_autoencoder(dataset_dir,
                            program_length,
                            out_embedding_size,
                            embedding_dim,
                            dsl_name,
                            max_node_color=8,
                            device="cuda:0"):
    vit_kwargs = {
        "input_size": 128,
        "input_channels": 1,
        # "upsize_c": 128,
        "downsampled_size": 8,
        "conv_depth": 5,
        "vit_depth": 6,
        "vit_heads": 4,
        "vit_head_dim": 256,
        "vit_mlp_dim": 1024
    }
    psn = PSN(program_length,
              out_embedding_size,
              embedding_dim,
              dsl_name,
              PixelShuffle_ViT_Encoder,
              GraphQuantizer,
              PixelShuffle_ViT_Decoder,
              pre_quantizer_kwargs=vit_kwargs,
              quantizer_kwargs={"max_node_color": max_node_color},
              post_quantizer_kwargs=vit_kwargs)
    psn = psn.to(device)
    dataset = RavenDataset(dataset_dir)
    return psn, dataset


def train_raven_autoencoder(psn,
                            dataset,
                            batch_size,
                            n_steps,
                            mode,
                            device,
                            quantization_noise_std=.5,
                            log_frequency=10,
                            optimizer=None):
    optimizer = th.optim.Adam(
        psn.parameters()) if optimizer is None else optimizer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses, diversity, n_elapsed = 0, 0, 0
    all_programs = set()
    for i, (x, _) in tqdm(zip(range(n_steps), dataloader)):
        x = (2. * (x / 255.) - 1.).to(device)
        _, loss, programs = psn(x, x, quantization_noise_std, mode)
        losses += loss.item()
        diversity += len(set(programs)) / len(programs)
        n_elapsed += 1
        all_programs.update(programs)
        if (i % log_frequency) == 0:
            print(f"\nRunning avg. loss: {losses / n_elapsed}")
            print(f"Running avg. program diversity: {diversity / n_elapsed}")
            losses, diversity, n_elapsed = 0, 0, 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del (x, loss)
    print(
        f"total program diversity: {len(all_programs) / (n_steps * batch_size)}"
    )

    return optimizer, all_programs
