import os
import pickle
from copy import deepcopy
from functools import lru_cache
from collections import defaultdict

import torch as th
import numpy as np
from tqdm import trange


from raven import RavenDataset
from raven_gen import Matrix, MatrixType, Ruleset, RuleType, ComponentType, LayoutType
from networks import (PixelShuffle_ViT_Encoder, PixelShuffle_ViT_Decoder,
                      PixelShuffle_ViT_Classifier, StrideConv_ViT_Encoder,
                      StrideConv_ViT_Decoder, StrideConv_ViT_Classifier)
from quantizers import GraphQuantizer, BottleneckQuantizer, NullQuantizer
from psn import PSN


def current_ruleset():
    return Ruleset(size_rules=[RuleType.CONSTANT])


def generate_data(size, dataset_dir, save_pickle=False):
    Matrix.oblique_angle_rotations(allowed=False)
    ruleset = current_ruleset()
    matrix_types = [
        MatrixType.ONE_SHAPE, MatrixType.FOUR_SHAPE, MatrixType.FIVE_SHAPE,
        MatrixType.TWO_SHAPE_VERTICAL_SEP, MatrixType.TWO_SHAPE_HORIZONTAL_SEP,
        MatrixType.SHAPE_IN_SHAPE
    ]
    weights = [.15, .2, .2, .15, .15, .15]
    Matrix.attribute_bounds[MatrixType.FOUR_SHAPE][(
        ComponentType.NONE, LayoutType.GRID_FOUR)]["size_min"] = 3
    Matrix.attribute_bounds[MatrixType.FIVE_SHAPE][(
        ComponentType.NONE, LayoutType.GRID_FIVE)]["size_min"] = 5
    Matrix.attribute_bounds[MatrixType.TWO_SHAPE_VERTICAL_SEP][(
        ComponentType.LEFT, LayoutType.CENTER)]["size_min"] = 3
    Matrix.attribute_bounds[MatrixType.TWO_SHAPE_VERTICAL_SEP][(
        ComponentType.RIGHT, LayoutType.CENTER)]["size_min"] = 3
    Matrix.attribute_bounds[MatrixType.TWO_SHAPE_HORIZONTAL_SEP][(
        ComponentType.UP, LayoutType.CENTER)]["size_min"] = 3
    Matrix.attribute_bounds[MatrixType.TWO_SHAPE_HORIZONTAL_SEP][(
        ComponentType.DOWN, LayoutType.CENTER)]["size_min"] = 3
    Matrix.attribute_bounds[MatrixType.SHAPE_IN_SHAPE][(
        ComponentType.OUT, LayoutType.CENTER)]["size_min"] = 5
    Matrix.attribute_bounds[MatrixType.SHAPE_IN_SHAPE][(
        ComponentType.IN, LayoutType.CENTER)]["size_min"] = 5
    background_colors = list(range(28, 225, 28))
    for i in trange(size):
        rpm = Matrix.make(np.random.choice(matrix_types, p=weights),  # type: ignore
                          ruleset=ruleset,
                          n_alternatives=1)
        background_color = np.random.choice(background_colors)
        rpm.save(os.path.join(dataset_dir, "data"),
                 f"rpm_{i}_background_{background_color}",
                 background_color,
                 image_size=128,
                 line_thickness=1,
                 shape_border_thickness=1)
        with open(
                os.path.join(dataset_dir, "meta", f"rpm_{i}_description.txt"),
                "w") as f:
            f.write(str(rpm))
        with open(os.path.join(dataset_dir, "meta", f"rpm_{i}_rules.txt"),
                  "w") as f:
            f.write(str(rpm.rules))
        if save_pickle:
            with open(os.path.join(dataset_dir, "pkl", f"rpm_{i}.pkl"),
                      "wb") as f:
                pickle.dump(rpm, f)


OUT_CODEBOOK_SIZE = 1024
CODEBOOK_DIM = 512
INITIAL_DSL = "dsl_0"
DATASET_DIR = "graph/dataset"
N_TRAINING = 32
N_EVAL_BATCHES = 150
BATCH_SIZE = 8


def reconstruction(_):
    def f(_instance, _sub_instance, _background_color, img): return img
    return f


CORRECTNESS_TARGET_DIM = 1


def correctness(dataset):
    def f(_instance, sub_instance, _background_color, _img):
        return th.tensor([1] if sub_instance < 0 else [0], dtype=th.float)
    return f


CONSTANT_COLOR_TARGET_DIM = 1


def constant_color(cache):
    def q(dataset):

        def f(instance, sub_instance):
            rpm: Matrix = dataset._rpm(instance)
            patterns = [0]
            for comp_rules in rpm.rules:
                if comp_rules.color.name is RuleType.CONSTANT:
                    patterns[0] = 1
            return patterns

        g = lru_cache(maxsize=len(dataset))(f) if cache else f

        def h(instance, sub_instance, _background_color, _img):
            return th.tensor(g(instance, sub_instance), dtype=th.float)

        return h

    return q


def pattern_description_length():
    ruleset = current_ruleset()
    n_slots = (len(set(ruleset.position_rules).union(ruleset.number_rules)) +
               len(ruleset.shape_rules) + len(ruleset.size_rules) +  # type: ignore
               len(ruleset.color_rules))  # type: ignore
    return 2 + 2 * n_slots


ALL_PATTERNS_TARGET_DIM = pattern_description_length()


def all_patterns(cache):
    def q(dataset):
        ruleset = current_ruleset()
        comparators = (
            tuple(sorted(set(ruleset.position_rules).union(
                ruleset.number_rules), key=lambda r: r.value)),
            tuple(sorted(ruleset.shape_rules, key=lambda r: r.value)),
            tuple(sorted(ruleset.size_rules, key=lambda r: r.value)),
            tuple(sorted(ruleset.color_rules, key=lambda r: r.value)),
        )
        n_slots = sum(len(s) for s in comparators)

        def f(instance, sub_instance):
            rpm: Matrix = dataset._rpm(instance)
            patterns = [0] * (2 + 2 * n_slots)
            # puzzle has two components
            patterns[0] = 1 if sub_instance < 0 else 0
            # puzzle is correctly completed
            patterns[1] = 1 if len(rpm.rules) == 2 else 0
            # puzzle rules
            offset = 2
            for component_rules in rpm.rules:
                for possible_rules, rule in zip(comparators, component_rules.all):
                    for possible_rule in possible_rules:
                        if rule.name is possible_rule:
                            patterns[offset] = 1
                        offset += 1
            return patterns

        g = lru_cache(maxsize=len(dataset))(f) if cache else f

        def h(instance, sub_instance, _background_color, _img):
            return th.tensor(g(instance, sub_instance), dtype=th.float)

        return h

    return q


def random_split(annotate_1, annotate_2, n_training, include_incorrect):
    assert (n_training % BATCH_SIZE) == 0
    return RavenDataset.bisplit(
        DATASET_DIR, annotate_1, annotate_2, n_training /
        (n_training + N_EVAL_BATCHES * BATCH_SIZE),
        (n_training + N_EVAL_BATCHES * BATCH_SIZE) /
        (40000 if include_incorrect else 20000),
        BATCH_SIZE, include_incorrect=include_incorrect)


def save_split(x, y, path):
    with open(path, "wb") as f:
        pickle.dump([x.multi_indices, y.multi_indices], f)


def load_split(path, annotate_1, annotate_2, n_training, include_incorrect):
    with open(path, "rb") as f:
        all_multi_indices = pickle.load(f)
    x, y = random_split(annotate_1, annotate_2, n_training, include_incorrect)
    x.multi_indices = all_multi_indices[0]
    y.multi_indices = all_multi_indices[1]
    return x, y


pre_strideconv_vit_kwargs = {
    "input_size": 128,
    "input_channels": 1,
    "upsize_channels": 128,
    "vit_in_size": 16,
    "vit_depth": 2,
    "vit_heads": 4,
    "vit_head_dim": 256,
    "vit_mlp_dim": 1024
}

post_strideconv_vit_kwargs = deepcopy(pre_strideconv_vit_kwargs)
post_strideconv_vit_kwargs["codebook_size"] = OUT_CODEBOOK_SIZE

pre_pixelshuffle_vit_kwargs = {
    "input_size": 128,
    "input_channels": 1,
    "downsampled_size": 8,
    "conv_depth": 3,
    "vit_depth": 2,
    "vit_heads": 4,
    "vit_head_dim": 256,
    "vit_mlp_dim": 1024
}

post_pixelshuffle_vit_kwargs = deepcopy(pre_pixelshuffle_vit_kwargs)
post_pixelshuffle_vit_kwargs["codebook_size"] = OUT_CODEBOOK_SIZE


class AccuracyMetric:

    def __init__(self, shape, device):
        super().__init__()
        self.correct = th.zeros(shape, device=device)
        self.incorrect = th.zeros(shape, device=device)

    def tally(self, output, target):
        preds = th.round(th.sigmoid(output))
        self.correct += th.count_nonzero(preds == target, dim=0)
        self.incorrect += th.count_nonzero(preds != target, dim=0)

    def measure(self):
        return th.mean(
            self.correct / (self.correct + self.incorrect), dim=0).tolist()


class F1Metric:

    def __init__(self, shape, device):
        self.tp = th.zeros(shape, device=device)
        self.fp = th.zeros(shape, device=device)
        self.fn = th.zeros(shape, device=device)
        self.epsilon = .001 * th.ones(shape, device=device)

    def tally(self, output, target):
        preds = th.round(th.sigmoid(output))
        self.tp += th.count_nonzero(
            th.logical_and(preds == target, preds == 1.), dim=0)
        self.fp += th.count_nonzero(
            th.logical_and(preds != target, preds == 1.), dim=0)
        self.fn += th.count_nonzero(
            th.logical_and(preds != target, preds == 0.), dim=0)

    def measure(self):
        return th.mean(
            (2 * self.tp) / (2 * self.tp + self.fp + self.fn + self.epsilon), dim=0).tolist()


def train_split(psn: PSN, train, eval, iterations, epochs_per_iteration, mode, device, perf_metric, use_scheduler=True):
    assert (perf_metric in ("acc", "f1"))
    for i in range(1, iterations + 1):

        label_shape = train[0][1].shape

        def make_perf_metric():
            if perf_metric == "acc":
                return AccuracyMetric(label_shape, device=device)
            else:  # perf_metric == "f1"
                return F1Metric(label_shape, device=device)

        training_metrics = psn.run(
            train,
            BATCH_SIZE,
            epochs_per_iteration,
            mode,
            device,
            quantization_noise_std=0.,
            perf_metric=make_perf_metric(),
            use_scheduler=use_scheduler)

        train_metrics = psn.run(
            train,
            BATCH_SIZE,
            1,
            mode,
            device,
            quantization_noise_std=0.,
            perf_metric=make_perf_metric(),
            shuffle=False,
            train=False)

        eval_metrics = psn.run(
            eval,
            BATCH_SIZE,
            1,
            mode,
            device,
            quantization_noise_std=0.,
            perf_metric=make_perf_metric(),
            shuffle=False,
            train=False)

        yield (
            i,
            {
                "perf_metric": perf_metric,
                "batch_size": BATCH_SIZE,
                "training_set_size": len(train),
                "evaluation_set_size": len(eval),
                "epochs_per_iteration": epochs_per_iteration,
                "training": training_metrics,
                "train": train_metrics,
                "eval": eval_metrics
            }
        )


def train_split_without_program_synthesis(psn,
                                          train,
                                          eval,
                                          iterations,
                                          epochs_per_iteration,
                                          mode,
                                          device,
                                          perf_metric):
    curve = []
    for i, log in train_split(psn,
                              train,
                              eval,
                              iterations,
                              epochs_per_iteration,
                              mode,
                              device,
                              perf_metric):
        train_loss, train_perf = log["train"]["total_loss"], log["train"]["perf"]
        eval_loss, eval_perf = log["eval"]["total_loss"], log["eval"]["perf"]
        print(
            f"cycle: {i}/{iterations}, "
            f"train loss: {train_loss:.4E}, "
            f"eval loss: {eval_loss:.4E}, "
            f"train perf.: {train_perf:.4f}, "
            f"eval perf.: {eval_perf:.4f}")
        log["iteration"] = i
        curve.append(log)

    return curve


def train_split_with_program_synthesis(psn,
                                       train,
                                       eval,
                                       exploration_timeout,
                                       frontier_size,
                                       iterations,
                                       epochs_per_iteration,
                                       mode,
                                       device,
                                       perf_metric,
                                       frontier_of_training=True,
                                       root_dsl_name="dsl",
                                       use_scheduler=True,
                                       exploration_eval_timeout=.1,
                                       exploration_eval_attempts=1,
                                       compression_iterations=3,
                                       compression_beta_inversions=2,
                                       compression_threads=4,
                                       compression_verbose=True):
    log = {"frontier_of_training": True, "metrics": []}
    exploration_kwargs = {
        "eval_timeout": exploration_eval_timeout,
        "eval_attempts": exploration_eval_attempts
    }
    compression_kwargs = {
        "iterations": compression_iterations,
        "n_beta_inversions": compression_beta_inversions,
        "threads": compression_threads,
        "verbose": compression_verbose
    }
    if not psn.quantizer.representations:
        print(f"initial exploration...")
        exploration_log = psn.exploration(
            [], exploration_timeout, next_dsl_name=root_dsl_name,
            **exploration_kwargs)
        print(
            f"\tnew: {exploration_log['new']}\n"
            f"\treplaced: {exploration_log['replaced']}\n"
            f"\ttotal: {exploration_log['total']}\n"
            f"\tmin. mass: {exploration_log['min_mass']}\n"
            f"\tmax. mass: {exploration_log['max_mass']}\n"
            f"\tavg. mass: {exploration_log['avg_mass']}\n"
        )
        exploration_log["iteration"] = 0
        exploration_log["activity"] = "exploration"
        log["metrics"].append(exploration_log)
        psn.quantizer.clear_visualizations()
        psn.quantizer.visualize()
    for i, psn_log in train_split(
            psn,
            train,
            eval,
            iterations,
            epochs_per_iteration,
            mode,
            device,
            perf_metric,
            use_scheduler=use_scheduler):
        print(f"cycle: {i}/{iterations}")
        print(
            "\taggregate training state.:\n"
            f"\t\tloss: {psn_log['training']['total_loss']:.4f}, "
            f"div.: {psn_log['training']['total_diversity']:.4f}, "
            f"mass: {psn_log['training']['total_mass']:.4f}, "
            f"perf.: {psn_log['training']['perf']:.4f}"
        )
        print(
            "\taggregate training set stat.:\n"
            f"\t\tloss: {psn_log['train']['total_loss']:.4f}, "
            f"div.: {psn_log['train']['total_diversity']:.4f}, "
            f"mass: {psn_log['train']['total_mass']:.4f}, "
            f"perf.: {psn_log['train']['perf']:.4f}"
        )
        print(
            "\taggregate evaluation set stat.:\n"
            f"\t\tloss: {psn_log['eval']['total_loss']:.4f}, "
            f"div.: {psn_log['eval']['total_diversity']:.4f}, "
            f"mass: {psn_log['eval']['total_mass']:.4f}, "
            f"perf.: {psn_log['eval']['perf']:.4f}"
        )
        psn_log["iteration"] = i
        psn_log["activity"] = "nn_training"
        log["metrics"].append(psn_log)
        print("creating frontier...")
        if frontier_of_training:
            repr_usage = psn_log["training"]["representations_usages"]
        else:
            repr_usage = psn_log["train"]["representations_usages"]
        frontier, frontier_log = psn.make_frontier(
            repr_usage, frontier_size)
        print(
            f"\tfrontier created from truncated usages: {frontier_log['truncated_usages']},\n"
            f"\tfrontier diversity: {frontier_log['frontier_div']:.4f},\n"
            f"\tfrontier mass: {frontier_log['frontier_mass']:.4f}")
        frontier_log["iteration"] = i
        frontier_log["activity"] = "frontier_creation"
        log["metrics"].append(frontier_log)
        print("compressing...")
        compression_log = psn.compression(
            frontier, next_dsl_name=root_dsl_name, **compression_kwargs)
        print(f"\tsuccessful compression: {compression_log['success']}")
        if compression_log["success"]:
            print(f"\tnew dsl mass: {compression_log['next_dsl_mass']}")
        compression_log["iteration"] = i
        compression_log["activity"] = "compression"
        log["metrics"].append(compression_log)
        repl = psn.quantizer.replacements
        frontier = [(repl[file] if file in repl else file)
                    for file in frontier]
        print("exploring...")
        exploration_log = psn.exploration(
            frontier, exploration_timeout, next_dsl_name=root_dsl_name,
            **exploration_kwargs)
        print(
            f"\tnew: {exploration_log['new']}\n"
            f"\treplaced: {exploration_log['replaced']}\n"
            f"\ttotal: {exploration_log['total']}\n"
            f"\tmin. mass: {exploration_log['min_mass']}\n"
            f"\tmax. mass: {exploration_log['max_mass']}\n"
            f"\tavg. mass: {exploration_log['avg_mass']}\n"
        )
        exploration_log["iteration"] = i
        exploration_log["activity"] = "exploration"
        log["metrics"].append(exploration_log)
        repl = psn.quantizer.replacements
        frontier = [(repl[file] if file in repl else file)
                    for file in frontier]
        psn.quantizer.clear_visualizations()
        psn.quantizer.visualize(frontier)

    return log


def raven_autoencoder_strideconv(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              StrideConv_ViT_Encoder,
              NullQuantizer,
              StrideConv_ViT_Decoder,
              two_stage_quantization=False,
              pre_quantizer_kwargs=pre_strideconv_vit_kwargs,
              post_quantizer_kwargs=post_strideconv_vit_kwargs)
    return psn.to(device)


def raven_classifier_strideconv(target_dim, device="cuda:0"):
    post_quantizer_kwargs = deepcopy(post_strideconv_vit_kwargs)
    post_quantizer_kwargs.update(target_dim=target_dim)
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              StrideConv_ViT_Encoder,
              NullQuantizer,
              StrideConv_ViT_Classifier,
              two_stage_quantization=False,
              pre_quantizer_kwargs=pre_strideconv_vit_kwargs,
              post_quantizer_kwargs=post_quantizer_kwargs)
    return psn.to(device)


def raven_autoencoder_pixelshuffle(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              NullQuantizer,
              PixelShuffle_ViT_Decoder,
              two_stage_quantization=False,
              pre_quantizer_kwargs=pre_pixelshuffle_vit_kwargs,
              post_quantizer_kwargs=post_pixelshuffle_vit_kwargs)
    return psn.to(device)


def raven_classifier_pixelshuffle(target_dim, device="cuda:0"):
    post_quantizer_kwargs = deepcopy(post_pixelshuffle_vit_kwargs)
    post_quantizer_kwargs.update(target_dim=target_dim)
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              NullQuantizer,
              PixelShuffle_ViT_Classifier,
              two_stage_quantization=False,
              pre_quantizer_kwargs=pre_pixelshuffle_vit_kwargs,
              post_quantizer_kwargs=post_quantizer_kwargs)
    return psn.to(device)


def raven_vqvae_pixelshuffle(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              NullQuantizer,
              PixelShuffle_ViT_Decoder,
              two_stage_quantization=False,
              pre_quantizer_kwargs=pre_pixelshuffle_vit_kwargs,
              post_quantizer_kwargs=post_pixelshuffle_vit_kwargs)
    return psn.to(device)


def raven_vq_classifier_pixelshuffle(target_dim, device="cuda:0"):
    post_quantizer_kwargs = deepcopy(post_pixelshuffle_vit_kwargs)
    post_quantizer_kwargs.update(target_dim=target_dim)
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              NullQuantizer,
              PixelShuffle_ViT_Classifier,
              two_stage_quantization=False,
              pre_quantizer_kwargs=pre_pixelshuffle_vit_kwargs,
              post_quantizer_kwargs=post_quantizer_kwargs)
    return psn.to(device)


def raven_psn_autoencoder_pixelshuffle(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              GraphQuantizer,
              PixelShuffle_ViT_Decoder,
              two_stage_quantization=True,
              pre_quantizer_kwargs=pre_pixelshuffle_vit_kwargs,
              quantizer_kwargs={"max_color": 10},
              post_quantizer_kwargs=post_pixelshuffle_vit_kwargs)
    return psn.to(device)


def raven_bottleneck_classifier_pixelshuffle(coordinates_only, n_representations, target_dim, device="cuda:0"):
    post_quantizer_kwargs = deepcopy(post_pixelshuffle_vit_kwargs)
    post_quantizer_kwargs.update(target_dim=target_dim)
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              BottleneckQuantizer,
              PixelShuffle_ViT_Classifier,
              two_stage_quantization=True,
              pre_quantizer_kwargs=pre_pixelshuffle_vit_kwargs,
              quantizer_kwargs={"coordinates_only": coordinates_only,
                                "n_representations": n_representations},
              post_quantizer_kwargs=post_quantizer_kwargs)
    return psn.to(device)


def raven_psn_classifier_pixelshuffle(target_dim, device="cuda:0"):
    post_quantizer_kwargs = deepcopy(post_pixelshuffle_vit_kwargs)
    post_quantizer_kwargs.update(target_dim=target_dim)
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              PixelShuffle_ViT_Encoder,
              GraphQuantizer,
              PixelShuffle_ViT_Classifier,
              two_stage_quantization=True,
              pre_quantizer_kwargs=pre_pixelshuffle_vit_kwargs,
              quantizer_kwargs={"max_color": 10},
              post_quantizer_kwargs=post_quantizer_kwargs)
    return psn.to(device)
