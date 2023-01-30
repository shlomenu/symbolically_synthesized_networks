from typing import Optional
import os
import pickle
import math
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache

from tqdm import tqdm, trange
import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader

from raven import RavenDataset
from raven_gen import Matrix, MatrixType, Ruleset, RuleType, ComponentType, LayoutType
from networks import PixelShuffle_ViT_Encoder, PixelShuffle_ViT_Classifier
from quantizer import Quantizer
from graph_quantizer import GraphQuantizer


class TqdmExtraFormat(tqdm):

    def __init__(self, *args, extras={}, **kwargs):
        self.extras = extras
        super().__init__(*args, **kwargs)

    @property
    def format_dict(self):
        d = super().format_dict
        d.update(**self.extras)
        return d


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


class ExperimentalModel(nn.Module):

    def __init__(self, pre_quantizer, post_quantizer, quantizer=None):
        super().__init__()
        self.pre_quantizer = pre_quantizer
        self.quantizer: Optional[Quantizer] = quantizer
        self.post_quantizer = post_quantizer
        self.optimizer = th.optim.Adam(self.parameters())
        self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=.5, patience=50,
            verbose=True, threshold=1e-4, cooldown=15)
        self.loss_f = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x):
        if self.quantizer is not None:
            return self.post_quantizer(self.quantizer(self.pre_quantizer(x)))
        else:
            return self.post_quantizer(self.pre_quantizer)

    def _run(self,
             dataset,
             batch_size,
             n_epochs,
             device,
             alpha=.8,
             perf_metric=None,
             shuffle=True,
             train=True,
             use_scheduler=True):
        if train:
            self.train()
        else:
            self.eval()
        if self.quantizer is not None:
            self.quantizer.in_restart_manager.idleness_limit = batch_size * 3
            self.quantizer.out_restart_manager.idleness_limit = batch_size * 3
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        total_loss, total_mass, repr_usage = 0., 0., defaultdict(int)
        assert ((len(dataset) % batch_size) == 0)
        steps_per_epoch = len(dataset) // batch_size
        total_steps = n_epochs * steps_per_epoch
        bar_format = "{bar}{r_bar} - " \
            "epoch: {epoch:3.0f}, " \
            "loss: {loss:.4f}, " \
            "tot. loss: {tot_loss:.4f}, " \
            "div.: {div:.4f}, " \
            "tot. div.: {tot_div:.4f}, " \
            "mass: {mass:.4f}, " \
            "tot. mass: {tot_mass:.4f}, " \
            "perf.: {perf:.4f}"
        extras = {"epoch": 0, "loss": 0., "tot_loss": 0.,
                  "div": float("NaN"), "tot_div": float("NaN"),
                  "mass": float("NaN"), "tot_mass": float("NaN"),
                  "perf": float("NaN")}
        data_iterator = iter(dataloader)
        with TqdmExtraFormat(total=total_steps, bar_format=bar_format, extras=extras) as pbar:
            for i in range(1, total_steps + 1):
                try:
                    x, target = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    x, target = next(data_iterator)
                x, target = x.to(device), target.to(device)
                out = self(x, target)
                loss = self.loss_f(out, target)
                if self.quantizer is not None:
                    loss += self.quantizer.loss
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.quantizer is not None:
                        self.quantizer.apply_restarts()
                    if use_scheduler:
                        self.scheduler.step(loss)
                pbar.extras["epoch"] = (i // steps_per_epoch)
                if perf_metric is not None:
                    perf_metric.tally(out, target)
                    pbar.extras["perf"] = perf_metric.measure()
                loss = loss.item()
                pbar.extras["loss"] = alpha * \
                    loss + (1 - alpha) * pbar.extras["loss"]
                total_loss += loss
                pbar.extras["tot_loss"] = total_loss / i
                if self.quantizer is not None and self.quantizer.filenames is not None:
                    pbar.extras["div"] = alpha * \
                        (len(set(self.quantizer.filenames)) / len(self.quantizer.filenames)) + \
                        (1 - alpha) * \
                        (0. if math.isnan(pbar.extras["div"])
                            else pbar.extras["div"])
                    for filename in self.quantizer.filenames:
                        repr_usage[filename] += 1
                    pbar.extras["tot_div"] = len(
                        repr_usage) / len(self.quantizer)
                    mass = self.quantizer.mass_of_representations(
                        self.quantizer.filenames)
                    pbar.extras["mass"] = mass
                    total_mass += mass
                    pbar.extras["tot_mass"] = total_mass / i
                pbar.update()
        if train and self.quantizer is not None:
            self.quantizer.repr_usage = repr_usage
        return {
            "perf": perf_metric.measure() if perf_metric is not None else None,
            "total_loss": pbar.extras["tot_loss"],
            "total_diversity": pbar.extras["tot_div"],
            "total_mass": pbar.extras["tot_mass"],
            "representations_usages": repr_usage
        }

    def _run_split(self,
                   train,
                   eval,
                   batch_size,
                   iterations,
                   epochs_per_iteration,
                   device,
                   perf_metric,
                   use_scheduler=True):
        assert (perf_metric in ("acc", "f1"))
        for i in range(1, iterations + 1):

            label_shape = train[0][1].shape

            def make_perf_metric():
                if perf_metric == "acc":
                    return AccuracyMetric(label_shape, device=device)
                else:  # perf_metric == "f1"
                    return F1Metric(label_shape, device=device)

            training_metrics = self._run(
                train,
                batch_size,
                epochs_per_iteration,
                device,
                perf_metric=make_perf_metric(),
                use_scheduler=use_scheduler)

            train_metrics = self._run(
                train,
                batch_size,
                1,
                device,
                perf_metric=make_perf_metric(),
                shuffle=False,
                train=False)

            eval_metrics = self._run(
                eval,
                batch_size,
                1,
                device,
                perf_metric=make_perf_metric(),
                shuffle=False,
                train=False)

            yield (
                i,
                {
                    "perf_metric": perf_metric,
                    "batch_size": batch_size,
                    "training_set_size": len(train),
                    "evaluation_set_size": len(eval),
                    "epochs_per_iteration": epochs_per_iteration,
                    "training": training_metrics,
                    "train": train_metrics,
                    "eval": eval_metrics
                }
            )

    def train_split(self,
                    train,
                    eval,
                    batch_size,
                    iterations,
                    epochs_per_iteration,
                    device,
                    perf_metric,
                    *,
                    exploration_timeout,
                    frontier_size,
                    frontier_of_training=True,
                    root_dsl_name="dsl",
                    use_scheduler=True,
                    exploration_eval_timeout=.1,
                    exploration_eval_attempts=1,
                    compression_iterations=3,
                    compression_beta_inversions=2,
                    compression_threads=4,
                    compression_verbose=True):
        if self.quantizer is None:
            curve = []
            for i, log in self._run_split(
                    train,
                    eval,
                    batch_size,
                    iterations,
                    epochs_per_iteration,
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
        else:

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
            if not self.quantizer.representations:
                print(f"initial exploration...")
                exploration_log = self.quantizer.explore(
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
                self.quantizer.clear_visualizations()
                self.quantizer.visualize()
            for i, psn_log in self._run_split(
                    train,
                    eval,
                    batch_size,
                    iterations,
                    epochs_per_iteration,
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
                frontier, frontier_log = self.quantizer.make_frontier(
                    repr_usage, frontier_size)
                print(
                    f"\tfrontier created from truncated usages: {frontier_log['truncated_usages']},\n"
                    f"\tfrontier diversity: {frontier_log['frontier_div']:.4f},\n"
                    f"\tfrontier mass: {frontier_log['frontier_mass']:.4f}")
                frontier_log["iteration"] = i
                frontier_log["activity"] = "frontier_creation"
                log["metrics"].append(frontier_log)
                print("compressing...")
                compression_log = self.quantizer.compress(
                    frontier, next_dsl_name=root_dsl_name, **compression_kwargs)
                print(
                    f"\nnumber of primitives added during compression: {compression_log['n_added']}")
                if compression_log["n_added"] > 0:
                    print(
                        f"\tnew dsl mass: {compression_log['next_dsl_mass']}")
                compression_log["iteration"] = i
                compression_log["activity"] = "compression"
                log["metrics"].append(compression_log)
                repl = self.quantizer.replacements
                frontier = [(repl[file] if file in repl else file)
                            for file in frontier]
                print("exploring...")
                exploration_log = self.quantizer.explore(
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
                repl = self.quantizer.replacements
                frontier = [(repl[file] if file in repl else file)
                            for file in frontier]
                self.quantizer.clear_visualizations()
                self.quantizer.visualize(frontier)

            return log


def raven_baseline(
        target_dim,
        input_size=128,
        downsampled_size=8,
        vit_dim=512,
        vit_depth=2,
        vit_heads=4,
        vit_head_dim=256,
        vit_mlp_dim=1024,
        input_channels=1,
        conv_depth=3,
        device="cuda:0"):
    pre_quantizer = PixelShuffle_ViT_Encoder(
        input_size,
        downsampled_size,
        vit_dim,
        vit_depth,
        vit_heads,
        vit_head_dim,
        vit_mlp_dim,
        input_channels=input_channels,
        conv_depth=conv_depth)
    post_quantizer = PixelShuffle_ViT_Classifier(
        input_size,
        downsampled_size,
        vit_dim,
        vit_depth,
        vit_heads,
        vit_head_dim,
        vit_mlp_dim,
        target_dim=target_dim)
    return ExperimentalModel(pre_quantizer, post_quantizer).to(device)


def raven_psn(
        target_dim,
        input_size=128,
        downsampled_size=8,
        vit_dim=512,
        vit_depth=2,
        vit_heads=4,
        vit_head_dim=256,
        vit_mlp_dim=1024,
        input_channels=1,
        conv_depth=3,
        gnn_depth=3,
        graph_dim=512,
        max_conn=10,
        device="cuda:0"):
    pre_quantizer = PixelShuffle_ViT_Encoder(
        input_size,
        downsampled_size,
        vit_dim,
        vit_depth,
        vit_heads,
        vit_head_dim,
        vit_mlp_dim,
        input_channels=input_channels,
        conv_depth=conv_depth)
    post_quantizer = PixelShuffle_ViT_Classifier(
        input_size,
        downsampled_size,
        vit_dim,
        vit_depth,
        vit_heads,
        vit_head_dim,
        vit_mlp_dim,
        target_dim=target_dim)
    quantizer = GraphQuantizer(
        "dsl_0",
        codebook_dim=((downsampled_size**2) * vit_dim),
        beta=.25,
        depth=gnn_depth,
        graph_dim=graph_dim,
        max_conn=max_conn)
    return ExperimentalModel(pre_quantizer, post_quantizer, quantizer).to(device)


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


def random_split(dataset_dir,
                 annotate_1,
                 annotate_2,
                 n_training,
                 *,
                 batch_size=8,
                 n_eval_batches=150,
                 include_incorrect):
    assert (n_training % batch_size) == 0
    return RavenDataset.bisplit(
        dataset_dir, annotate_1, annotate_2, n_training /
        (n_training + n_eval_batches * batch_size),
        (n_training + n_eval_batches * batch_size) /
        (40000 if include_incorrect else 20000),
        batch_size, include_incorrect=include_incorrect)


def save_split(x, y, path):
    with open(path, "wb") as f:
        pickle.dump([x.multi_indices, y.multi_indices], f)


def load_split(path,
               dataset_dir,
               annotate_1,
               annotate_2,
               n_training,
               *,
               batch_size=8,
               n_eval_batches=150,
               include_incorrect):
    with open(path, "rb") as f:
        all_multi_indices = pickle.load(f)
    x, y = random_split(dataset_dir, annotate_1, annotate_2, n_training,
                        batch_size=batch_size, n_eval_batches=n_eval_batches,
                        include_incorrect=include_incorrect)
    x.multi_indices = all_multi_indices[0]
    y.multi_indices = all_multi_indices[1]
    return x, y
