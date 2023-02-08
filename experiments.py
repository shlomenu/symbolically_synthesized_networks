from typing import Optional
import os
import pickle
from functools import lru_cache

from tqdm import tqdm, trange
import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader

from raven import RavenDataset
from raven_gen import Matrix, MatrixType, Ruleset, RuleType, ComponentType, LayoutType
from networks import PixelShuffle_ViT_Encoder
from quantizer import Quantizer, VQBaseline
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
            return self.post_quantizer(self.pre_quantizer(x))

    def _run(self,
             dataset,
             batch_size,
             n_epochs,
             smoothing_factor,
             device,
             perf_metric=None,
             train=True,
             shuffle=True,
             use_scheduler=True,
             n_archetypes=None,
             n_preserved=None,
             usage_smoothing_factor=None):
        assert ((len(dataset) % batch_size) == 0)
        log = {
            "smoothing_factor": smoothing_factor,
            "use_scheduler": use_scheduler}
        if train:
            self.train()
        else:
            self.eval()
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        data_iterator = iter(dataloader)
        steps_per_epoch = len(dataset) // batch_size
        total_steps = n_epochs * steps_per_epoch
        if self.quantizer is not None:
            bar_format = \
                "{bar}{r_bar} - " \
                "epoch: {epoch:3.0f}, " \
                "loss: {loss:.4f}, " \
                "div.: {div:.4f}, " \
                "mass: {mass:.4f}, " \
                "perf.: {perf:.4f}"
            extras = {"epoch": 0,
                      "loss": 0.,
                      "div": 0.,
                      "mass": 0.,
                      "perf": 0.}
            self.quantizer.clear_usages()
            self.quantizer.restart_manager.idleness_limit = batch_size * 3
            if train:
                assert type(n_archetypes) is int and n_archetypes <= len(
                    dataset) and n_archetypes > 0
                assert type(n_preserved) is int
                log.update(usage_smoothing_factor=usage_smoothing_factor,
                           n_archetypes=n_archetypes, n_preserved=n_preserved)
        else:
            bar_format = \
                "{bar}{r_bar} - " \
                "epoch: {epoch:3.0f}, " \
                "loss: {loss:.4f}, " \
                "perf.: {perf:.4f}"
            extras = {"epoch": 0,
                      "loss": 0.,
                      "perf": float("NaN")}
        with TqdmExtraFormat(total=total_steps, bar_format=bar_format, extras=extras) as pbar:
            for i in range(1, total_steps + 1):
                try:
                    x, target = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    x, target = next(data_iterator)
                    if self.quantizer is not None:
                        self.quantizer.finish_epoch(
                            usage_smoothing_factor, initial_unsmoothed=(not train))
                x, target = x.to(device), target.to(device)
                out = self(x)
                loss = self.loss_f(out, target)
                if self.quantizer is not None:
                    loss += self.quantizer.loss
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if use_scheduler:
                        self.scheduler.step(loss)
                    if self.quantizer is not None:
                        self.quantizer.apply_restarts()
                pbar.extras["epoch"] = (i // steps_per_epoch)
                if perf_metric is not None:
                    perf_metric.tally(out, target)
                    pbar.extras["perf"] = perf_metric.measure()
                pbar.extras["loss"] = smoothing_factor * \
                    loss.item() + (1 - smoothing_factor) * \
                    pbar.extras["loss"]
                if self.quantizer is not None:
                    pbar.extras["div"] = smoothing_factor * \
                        (len(set(self.quantizer.filenames)) / len(self.quantizer.filenames)) + \
                        (1 - smoothing_factor) * pbar.extras["div"]
                    pbar.extras["mass"] = self.quantizer.mass_of_representations(
                        self.quantizer.filenames)
                pbar.update()
            log.update(
                perf=(perf_metric.measure()
                      if perf_metric is not None else None),
                final_loss=pbar.extras["loss"])
            if self.quantizer is not None:
                log.update(
                    final_diversity=pbar.extras["div"],
                    final_mass=pbar.extras["mass"])
        if self.quantizer is not None:
            try:
                _ = next(data_iterator)
            except StopIteration:
                self.quantizer.finish_epoch(
                    usage_smoothing_factor, initial_unsmoothed=(not train))
            log.update(usages=sorted(
                ((r, usage)
                    for r, usage in self.quantizer.smoothed_usages.items()),
                key=lambda x: x[1], reverse=True))
            if train:
                n_discarded = self.quantizer.finish_training(
                    n_archetypes, n_preserved)
                log.update(n_discarded=n_discarded)
        return log

    def _run_with_split(self,
                        train,
                        eval,
                        batch_size,
                        iterations,
                        epochs_per_iteration,
                        perf_metric,
                        smoothing_factor,
                        device,
                        n_archetypes=None,
                        n_preserved=None,
                        usage_smoothing_factor=None,
                        use_scheduler=True):
        assert (perf_metric in ("acc", "f1"))
        for i in range(1, iterations + 1):

            label_shape = train[0][1].shape

            def make_perf_metric():
                if perf_metric == "acc":
                    return AccuracyMetric(label_shape, device=device)
                else:  # perf_metric == "f1"
                    return F1Metric(label_shape, device=device)

            training_log = self._run(
                train,
                batch_size,
                epochs_per_iteration,
                smoothing_factor,
                device,
                perf_metric=make_perf_metric(),
                use_scheduler=use_scheduler,
                n_archetypes=n_archetypes,
                n_preserved=n_preserved,
                usage_smoothing_factor=usage_smoothing_factor)

            train_log = self._run(
                train,
                batch_size,
                1,
                smoothing_factor,
                device,
                perf_metric=make_perf_metric(),
                train=False,
                shuffle=False)

            eval_log = self._run(
                eval,
                batch_size,
                1,
                smoothing_factor,
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
                    "training": training_log,
                    "train": train_log,
                    "eval": eval_log
                }
            )

    def train_with_split(self,
                         train,
                         eval,
                         batch_size,
                         iterations,
                         epochs_per_iteration,
                         device,
                         perf_metric,
                         retain_rate=2.,
                         preservation_rate=.75,
                         archetype_rate=.25,
                         exploration_timeout=15.,
                         smoothing_factor=.25,
                         usage_smoothing_factor=.8,
                         max_diff=.5,
                         program_size_limit=150,
                         root_dsl_name="dsl",
                         use_scheduler=True,
                         exploration_eval_timeout=.1,
                         exploration_eval_attempts=1,
                         compression_iterations=3,
                         compression_beta_inversions=2,
                         compression_threads=4,
                         compression_verbose=True,
                         no_program_synthesis=False):
        n_archetypes = int(len(train) * archetype_rate)
        if self.quantizer is None or no_program_synthesis:
            logs = []
            for i, log in self._run_with_split(
                    train,
                    eval,
                    batch_size,
                    iterations,
                    epochs_per_iteration,
                    perf_metric,
                    smoothing_factor,
                    device,
                    n_archetypes=n_archetypes,
                    usage_smoothing_factor=usage_smoothing_factor,
                    use_scheduler=use_scheduler):
                print(
                    f"cycle: {i}/{iterations}, "
                    f"train loss: {log['train']['final_loss']:.4E}, "
                    f"eval loss: {log['eval']['final_loss']:.4E}, "
                    f"train perf.: {log['train']['perf']:.4f}, "
                    f"eval perf.: {log['eval']['perf']:.4f}")
                log["iteration"] = i
                logs.append(log)

            return logs
        else:
            logs = []
            n_retained = int(len(train) * retain_rate)
            n_preserved = int(preservation_rate * n_retained)
            assert n_preserved < n_retained
            exploration_kwargs = {
                "exploration_timeout": exploration_timeout,
                "program_size_limit": program_size_limit,
                "eval_timeout": exploration_eval_timeout,
                "eval_attempts": exploration_eval_attempts,
                "max_diff": max_diff
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
                    n_retained,
                    next_dsl_name=root_dsl_name,
                    **exploration_kwargs)
                print(
                    f"\treplaced: {exploration_log['replaced']}\n"
                    f"\tmax. novel representations: {exploration_log['max_novel_representations']}\n"
                    f"\tnew: {exploration_log['new']}\n"
                    f"\ttotal: {exploration_log['total']}\n"
                    f"\tmin. mass: {exploration_log['min_mass']}\n"
                    f"\tmax. mass: {exploration_log['max_mass']}\n"
                    f"\tavg. mass: {exploration_log['avg_mass']}\n"
                )
                exploration_log["iteration"] = 0
                exploration_log["activity"] = "exploration"
                logs.append(exploration_log)
                self.quantizer.clear_visualizations()
                self.quantizer.visualize()
            for i, psn_log in self._run_with_split(
                    train,
                    eval,
                    batch_size,
                    iterations,
                    epochs_per_iteration,
                    perf_metric,
                    smoothing_factor,
                    device,
                    n_archetypes=n_archetypes,
                    n_preserved=n_preserved,
                    usage_smoothing_factor=usage_smoothing_factor,
                    use_scheduler=use_scheduler):
                print(f"cycle: {i}/{iterations}")
                print(
                    "\taggregate training state.:\n"
                    f"\t\tloss: {psn_log['training']['final_loss']:.4f}, "
                    f"div.: {psn_log['training']['final_diversity']:.4f}, "
                    f"mass: {psn_log['training']['final_mass']:.4f}, "
                    f"perf.: {psn_log['training']['perf']:.4f}, "
                    f"discarded: {psn_log['training']['n_discarded']}"
                )
                print(
                    "\taggregate training set stat.:\n"
                    f"\t\tloss: {psn_log['train']['final_loss']:.4f}, "
                    f"div.: {psn_log['train']['final_diversity']:.4f}, "
                    f"mass: {psn_log['train']['final_mass']:.4f}, "
                    f"perf.: {psn_log['train']['perf']:.4f}"
                )
                print(
                    "\taggregate evaluation set stat.:\n"
                    f"\t\tloss: {psn_log['eval']['final_loss']:.4f}, "
                    f"div.: {psn_log['eval']['final_diversity']:.4f}, "
                    f"mass: {psn_log['eval']['final_mass']:.4f}, "
                    f"perf.: {psn_log['eval']['perf']:.4f}"
                )
                psn_log["iteration"] = i
                psn_log["activity"] = "nn_training"
                logs.append(psn_log)
                self.quantizer.clear_visualizations()
                self.quantizer.visualize(self.quantizer.archetypes)
                print("compressing...")
                compression_log = self.quantizer.compress(
                    next_dsl_name=root_dsl_name, **compression_kwargs)
                print(
                    f"\nnumber of primitives added during compression: {compression_log['n_added']}")
                if compression_log["n_added"] > 0:
                    print(
                        f"\tnew dsl mass: {compression_log['next_dsl_mass']}")
                compression_log["iteration"] = i
                compression_log["activity"] = "compression"
                logs.append(compression_log)
                print("exploring...")
                exploration_log = self.quantizer.explore(
                    n_retained,
                    next_dsl_name=root_dsl_name,
                    **exploration_kwargs)
                print(
                    f"\treplaced: {exploration_log['replaced']}\n"
                    f"\tmax. new: {exploration_log['max_novel_representations']}\n"
                    f"\tnew: {exploration_log['new']}\n"
                    f"\ttotal: {exploration_log['total']}\n"
                    f"\tmin. mass: {exploration_log['min_mass']}\n"
                    f"\tmax. mass: {exploration_log['max_mass']}\n"
                    f"\tavg. mass: {exploration_log['avg_mass']}\n"
                )
                exploration_log["iteration"] = i
                exploration_log["activity"] = "exploration"
                logs.append(exploration_log)

            return logs


def raven_baseline(
        target_dim,
        input_size=128,
        downsampled_size=8,
        vit_dim=256,
        vit_depth=2,
        vit_heads=4,
        vit_head_dim=128,
        vit_mlp_dim=512,
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
        conv_depth=conv_depth,
        output_dim=target_dim)
    post_quantizer = nn.Identity()
    return ExperimentalModel(pre_quantizer, post_quantizer).to(device)


def raven_vq_baseline(
        target_dim,
        codebook_size,
        input_size=128,
        downsampled_size=8,
        vit_dim=256,
        vit_depth=2,
        vit_heads=4,
        vit_head_dim=128,
        vit_mlp_dim=512,
        input_channels=1,
        conv_depth=3,
        codebook_dim=512,
        beta=3.,
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
        conv_depth=conv_depth,
        output_dim=codebook_dim)
    post_quantizer = nn.Linear(codebook_dim, target_dim)
    quantizer = VQBaseline(codebook_size, codebook_dim, beta)
    return ExperimentalModel(pre_quantizer, post_quantizer, quantizer).to(device)


def raven_psn(
        target_dim,
        input_size=128,
        downsampled_size=8,
        vit_dim=256,
        vit_depth=2,
        vit_heads=4,
        vit_head_dim=128,
        vit_mlp_dim=512,
        input_channels=1,
        conv_depth=3,
        gnn_depth=3,
        graph_dim=512,
        max_conn=10,
        beta=3.,
        dropout_rate=.3,
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
        conv_depth=conv_depth,
        output_dim=graph_dim)
    post_quantizer = nn.Identity()
    quantizer = GraphQuantizer(
        "dsl_0",
        codebook_dim=graph_dim,
        beta=beta,
        dropout_rate=dropout_rate,
        depth=gnn_depth,
        max_conn=max_conn,
        output_dim=target_dim,
        device=device)
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
