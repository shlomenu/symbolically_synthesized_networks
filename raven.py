import os
import pickle
import math
from copy import deepcopy

import numpy as np
from tqdm import trange
import torch as th
from functools import lru_cache
from torch.utils.data import Dataset
from torchvision.io import read_image

from raven_gen import Matrix, MatrixType, Ruleset, RuleType, ComponentType, LayoutType


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


class RavenDataset(Dataset):

    def __init__(self, dataset_dir, annotate, include_incorrect=False):
        self.dataset_dir = dataset_dir
        self.include_incorrect = include_incorrect
        self.data_dir = os.path.join(self.dataset_dir, "data")
        self.meta_dir = os.path.join(self.dataset_dir, "meta")
        self.pkl_dir = os.path.join(self.dataset_dir, "pkl")
        multi_indices = set()
        for filename in os.listdir(self.data_dir):
            no_ext = filename.split(".")[0]
            parts = no_ext.split("_")[1:]
            if len(parts) == 4:
                instance_index, background_color, label = int(parts[0]), int(
                    parts[2]), parts[3]
                assert label == "answer"
                multi_indices.add((instance_index, -1, background_color))
            elif len(parts) == 5 and self.include_incorrect:
                instance_index, background_color, label, alternative_index = int(
                    parts[0]), int(parts[2]), parts[3], int(parts[4])
                assert label == "alternative"
                multi_indices.add(
                    (instance_index, alternative_index, background_color))
        self.multi_indices = sorted(multi_indices)
        self.annotate = annotate(self)
        self.annotations = {
            multi_index: None for multi_index in self.multi_indices}

    def __len__(self):
        return len(self.multi_indices)

    def __getitem__(self, i):
        instance, sub_instance, background_color = self.multi_indices[i]
        if sub_instance < 0:
            filename = f"rpm_{instance}_background_{background_color}_answer.png"
        else:
            filename = f"rpm_{instance}_background_{background_color}_alternative_{sub_instance}.png"
        img = (
            2. * (read_image(os.path.join(self.data_dir, filename)) / 255.) - 1.).float()
        return img, self.annotate(instance, sub_instance, background_color, img)

    def _description(self, instance):
        filename = f"rpm_{instance}_description.txt"
        with open(os.path.join(self.meta_dir, filename)) as f:
            return f.read()

    def description(self, i):
        instance, _, _ = self.multi_indices[i]
        return self._description(instance)

    def _rules(self, instance):
        filename = f"rpm_{instance}_rules.txt"
        with open(os.path.join(self.meta_dir, filename)) as f:
            return f.read()

    def rules(self, i):
        instance, _, _ = self.multi_indices[i]
        return self._rules(instance)

    def _rpm(self, instance):
        filename = f"rpm_{instance}.pkl"
        with open(os.path.join(self.pkl_dir, filename), "rb") as f:
            return pickle.load(f)

    def rpm(self, i):
        instance, _, _ = self.multi_indices[i]
        return self._rpm(instance)

    @classmethod
    def bisplit(cls, dataset_dir, annotate_1, annotate_2, prop, load_prop, batch_size, include_incorrect=False):
        """
        Splits data in dataset_dir randomly into two portions.
        """
        assert (0 <= prop and prop <= 1 and 0 < load_prop and load_prop <= 1)
        dataset = cls(dataset_dir, annotate_1,
                      include_incorrect=include_incorrect)
        n_instances = len(
            {instance for (instance, _, _) in dataset.multi_indices}
        )
        assert ((len(dataset) % n_instances) == 0)
        puzzles_per_instance = len(dataset) // n_instances
        n_instances = math.ceil(load_prop * n_instances)
        assert (batch_size >= 1 and ((batch_size % puzzles_per_instance) == 0))
        instances_per_batch = batch_size // puzzles_per_instance
        n_instances -= (n_instances % instances_per_batch)
        n_instances_x: int = math.ceil(prop * n_instances)
        n_instances_y: int = math.floor((1 - prop) * n_instances)
        n_instances_x -= (n_instances_x % instances_per_batch)
        n_instances_y -= (n_instances_y % instances_per_batch)
        assert (n_instances_x >= 1 and n_instances_y >= 1)
        multi_indices = deepcopy(dataset.multi_indices)
        del dataset.multi_indices[:]
        dataset_x = dataset
        dataset_y = cls(dataset_dir, annotate_2,
                        include_incorrect=include_incorrect)
        del dataset_y.multi_indices[:]
        x_instances = sorted(np.random.choice(
            n_instances, n_instances_x, replace=False))
        for i in range(n_instances):
            if x_instances and i == x_instances[0]:
                dataset_x.multi_indices.extend(
                    multi_indices[i * puzzles_per_instance:(i + 1) * puzzles_per_instance])
                del x_instances[0]
            else:
                dataset_y.multi_indices.extend(
                    multi_indices[i * puzzles_per_instance:(i + 1) * puzzles_per_instance])
        return dataset_x, dataset_y


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
