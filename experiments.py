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


PROGRAM_SIZE = 12
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


def train_split(psn: PSN, train, eval, iterations, epochs, mode, device, perf_metric, use_scheduler=True):
    assert (perf_metric in ("acc", "f1"))
    for i in range(1, iterations + 1):

        label_shape = train[0][1].shape

        def make_perf_metric():
            if perf_metric == "acc":
                return AccuracyMetric(label_shape, device=device)
            else:  # perf_metric == "f1"
                return F1Metric(label_shape, device=device)

        metrics_during_training = psn.run(
            train,
            BATCH_SIZE,
            epochs,
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

        yield (i, metrics_during_training, train_metrics, eval_metrics)


def train_split_no_compression(psn,
                               train,
                               eval,
                               iterations,
                               epochs_per_iteration,
                               mode,
                               device,
                               perf_metric):
    curve = {
        "perf_metric": perf_metric,
        "batch_size": BATCH_SIZE,
        "training_set_size": len(train),
        "evaluation_set_size": len(eval),
        "epochs_per_iteration": epochs_per_iteration,
        "metrics": []
    }
    for i, _, train_metrics, eval_metrics in \
            train_split(psn, train, eval, iterations, epochs_per_iteration, mode, device, perf_metric):
        train_loss, train_perf = train_metrics["total_loss"], train_metrics["perf"]
        eval_loss, eval_perf = eval_metrics["total_loss"], eval_metrics["perf"]
        print(
            f"cycle: {i}/{iterations}, "
            f"train loss: {train_loss:.4E}, "
            f"eval loss: {eval_loss:.4E}, "
            f"train perf.: {train_perf:.4f}, "
            f"eval perf.: {eval_perf:.4f}")
        curve["metrics"].append({
            "iteration": i,
            "training_set_metrics": train_metrics,
            "evaluation_set_metrics": eval_metrics
        })

    return curve


def train_split_compression(psn,
                            train,
                            eval,
                            exploration_timeout,
                            run_name,
                            frontier_size,
                            iterations,
                            epochs_per_iteration,
                            mode,
                            device,
                            perf_metric,
                            exploration_intervals=[],
                            compression_intervals=[],
                            exploration_eval_timeout=.1,
                            exploration_eval_attempts=1,
                            compression_iterations=1,
                            compression_beam_size=3,
                            compression_invention_sizes=1,
                            compression_beta_inversions=1,
                            compression_verbosity=1):
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
    curve = {
        "perf_metric": perf_metric,
        "batch_size": BATCH_SIZE,
        "training_set_size": len(train),
        "evaluation_set_size": len(eval),
        "epochs_per_iteration": epochs_per_iteration,
        "exploration": {
            "exploration_timeout": exploration_timeout,
            "program_size": PROGRAM_SIZE,
            "eval_timeout": exploration_eval_timeout,
            "attempts": exploration_eval_attempts,
        },
        "compression": {
            "iterations": compression_iterations,
            "beam_size": compression_beam_size,
            "n_invention_sizes": compression_invention_sizes,
            "n_beta_inversions": compression_beta_inversions,
        },
        "frontier_size": frontier_size,
        "metrics": []
    }
    exploration_kwargs = {
        "eval_timeout": exploration_eval_timeout,
        "attempts": exploration_eval_attempts
    }
    compression_kwargs = {
        "iterations": compression_iterations,
        "beam_size": compression_beam_size,
        "n_invention_sizes": compression_invention_sizes,
        "n_beta_inversions": compression_beta_inversions
    }
    dsl_mass = psn.quantizer.dsl_mass
    print(f"dsl mass: {dsl_mass:.4f}\nexploring...")
    new, replaced, total, mass_statistics = psn.exploration(
        exploration_timeout, PROGRAM_SIZE, **exploration_kwargs)
    min_mass, max_mass, avg_mass = mass_statistics
    print(
        f"\tnew: {new}\n"
        f"\treplaced: {replaced}\n"
        f"\ttotal: {total}\n"
        f"\tmin. mass: {min_mass:.4f}\n"
        f"\tmax. mass: {max_mass:.4f}\n"
        f"\tavg mass: {avg_mass:.4f}"
    )
    curve["metrics"].append({
        "activity": "exploration",
        "iteration": .5,
        "new": new,
        "replaced": replaced,
        "total": total,
        "min_mass": min_mass,
        "max_mass": max_mass,
        "avg_mass": avg_mass
    })
    for (i, training_metrics, train_metrics, eval_metrics) in \
            train_split(psn, train, eval, iterations, epochs_per_iteration, mode, device, perf_metric):
        print(f"cycle: {i}/{iterations}:")
        print("\taggregate training statistics:")
        training_loss, training_div, training_mass, training_perf = (
            training_metrics["total_loss"],
            training_metrics["total_diversity"],
            training_metrics["total_mass"],
            training_metrics["perf"]
        )
        print(
            f"\t\tloss: {training_loss:.4f}, div.: {training_div:.4f}, mass: {training_mass:.4f}, perf.: {training_perf:.4f}")
        print("\taggregate training set statistics:")
        train_loss, train_div, train_mass, train_perf = (
            train_metrics["total_loss"],
            train_metrics["total_diversity"],
            train_metrics["total_mass"],
            train_metrics["perf"]
        )
        print(
            f"\t\tloss: {train_loss:.4f}, div.: {train_div:.4f}, mass: {train_mass:.4f}, perf.: {train_perf:.4f}")
        print("\taggregate evaluation set statistics:")
        eval_loss, eval_div, eval_mass, eval_perf = (
            eval_metrics["total_loss"],
            eval_metrics["total_diversity"],
            eval_metrics["total_mass"],
            eval_metrics["perf"]
        )
        print(
            f"\t\tloss: {eval_loss:.4f}, div.: {eval_div:.4f}, mass: {eval_mass:.4f}, perf.: {eval_perf:.4f}")
        curve["network_metrics"].append({
            "iteration": i,
            "training_metrics": training_metrics,
            "training_set_metrics": train_metrics,
            "evaluation_set_metrics": eval_metrics
        })
        performed_compression = False
        if compression_interval is not None:
            if compression_interval > 0:
                compression_interval -= 1
            if compression_interval == 0:
                dsl_mass, frontier_div, frontier_mass = psn.compression(
                    training_metrics["representations_usages"],
                    frontier_size, dsl_name, **compression_kwargs)
                print(
                    f"performed compression..\n"
                    f"\tdsl mass: {dsl_mass:.4f}\n"
                    f"\ttruncated frontier: {frontier_div is None and frontier_mass is None}\n"
                    f"\tfrontier diversity: {frontier_div}\n"
                    f"\tfrontier mass: {frontier_mass}\n")
                curve["metrics"].append({
                    "activity": "compression",
                    "iteration": i + .25,
                    "dsl_mass": dsl_mass,
                    "truncated_frontier": frontier_div is None and frontier_mass is None,
                    "frontier_div":
                        frontier_div if frontier_div is not None else training_metrics[
                            "total_diversity"],
                    "frontier_mass":
                        frontier_mass if frontier_mass is not None else training_metrics[
                            "total_mass"]
                })
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
                print("exploring...")
                new, replaced, total, mass_statistics = psn.exploration(
                    exploration_timeout, PROGRAM_SIZE, **exploration_kwargs)
                min_mass, max_mass, avg_mass = mass_statistics
                print(
                    f"\tnew: {new}\n"
                    f"\treplaced: {replaced}\n"
                    f"\ttotal: {total}\n"
                    f"\tmin. mass: {min_mass:.4f}\n"
                    f"\tmax. mass: {max_mass:.4f}\n"
                    f"\tavg mass: {avg_mass:.4f}"
                )
                curve["program_synthesis_metrics"].append({
                    "activity": "exploration",
                    "iteration": i + .5,
                    "new": new,
                    "replaced": replaced,
                    "total": total,
                    "min_mass": min_mass,
                    "max_mass": max_mass,
                    "avg_mass": avg_mass
                })

                if exploration_intervals:
                    exploration_interval = exploration_intervals[0]
                    exploration_intervals = exploration_intervals[1:]
                else:
                    exploration_interval = None

    return curve


def raven_autoencoder_strideconv(device="cuda:0"):
    psn = PSN(OUT_CODEBOOK_SIZE,
              CODEBOOK_DIM,
              INITIAL_DSL,
              StrideConv_ViT_Encoder,
              NullQuantizer,
              StrideConv_ViT_Decoder,
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
              program_size=PROGRAM_SIZE,
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
              program_size=PROGRAM_SIZE,
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
              program_size=PROGRAM_SIZE,
              pre_quantizer_kwargs=pre_pixelshuffle_vit_kwargs,
              quantizer_kwargs={"max_color": 10},
              post_quantizer_kwargs=post_quantizer_kwargs)
    return psn.to(device)
