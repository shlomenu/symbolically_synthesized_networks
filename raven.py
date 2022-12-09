import os

from raven_gen import Matrix, MatrixType, Ruleset, RuleType, ComponentType, LayoutType

from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from tqdm import trange
import pickle
from copy import deepcopy


class RavenDataset(Dataset):

    def __init__(self, dataset_dir, load_correct=True, load_incorrect=False):
        self.dataset_dir = dataset_dir
        self.data_dir = os.path.join(self.dataset_dir, "data")
        self.meta_dir = os.path.join(self.dataset_dir, "meta")
        self.pkl_dir = os.path.join(self.dataset_dir, "pkl")
        multi_indices = set()
        for filename in os.listdir(self.data_dir):
            no_ext = filename.split(".")[0]
            parts = no_ext.split("_")[1:]
            if len(parts) == 4 and load_correct:
                instance_index, background_color, label = int(parts[0]), int(
                    parts[2]), parts[3]
                assert label == "answer"
                multi_indices.add((instance_index, -1, background_color))
            elif len(parts) == 5 and load_incorrect:
                instance_index, background_color, label, alternative_index = int(
                    parts[0]), int(parts[2]), parts[3], int(parts[4])
                assert label == "alternative"
                multi_indices.add(
                    (instance_index, alternative_index, background_color))
        self.multi_indices = sorted(multi_indices)

    def __len__(self):
        return len(self.multi_indices)

    def __getitem__(self, i):
        instance, sub_instance, background_color = self.multi_indices[i]
        if sub_instance < 0:
            filename = f"rpm_{instance}_background_{background_color}_answer.png"
        else:
            filename = f"rpm_{instance}_background_{background_color}_alternative_{sub_instance}.png"
        img = read_image(os.path.join(self.data_dir, filename))
        img = (2. * (img / 255.) - 1.).float()
        return img, img

    def description(self, i):
        instance, _, _ = self.multi_indices[i]
        filename = f"rpm_{instance}_description.txt"
        with open(os.path.join(self.meta_dir, filename)) as f:
            return f.read()

    def rules(self, i):
        instance, _, _ = self.multi_indices[i]
        filename = f"rpm_{instance}_rules.txt"
        with open(os.path.join(self.meta_dir, filename)) as f:
            return f.read()

    def rpm(self, i):
        instance, _, _ = self.multi_indices[i]
        filename = f"rpm_{instance}.pkl"
        with open(os.path.join(self.pkl_dir, filename), "rb") as f:
            return pickle.load(f)

    @classmethod
    def multisplit(cls, dataset_dir, batch_size, batches_per_portion, load_correct=True, load_incorrect=False):
        """
        Splits data in dataset_dir randomly into portions of size `batch_size * batches_per_portion`. 
        """
        dataset = cls(dataset_dir, load_correct=load_correct,
                      load_incorrect=load_incorrect)
        n_instances = len(
            {instance for (instance, _, _) in dataset.multi_indices})
        assert ((len(dataset) % n_instances) == 0)
        puzzles_per_instance = len(dataset) // n_instances
        assert ((batch_size % puzzles_per_instance) == 0)
        instances_per_batch = batch_size // puzzles_per_instance
        instances_per_portion = batches_per_portion * instances_per_batch
        n_portions = n_instances // instances_per_portion
        n_used: int = n_portions * instances_per_portion
        multi_indices = deepcopy(dataset.multi_indices)
        del dataset.multi_indices[:]
        portions = [deepcopy(dataset) for _ in range(n_portions)]
        for i, inst in enumerate(np.random.choice(n_instances, n_used, replace=False)):
            portions[i // instances_per_portion].multi_indices.extend(
                multi_indices[inst * puzzles_per_instance:(inst + 1) * puzzles_per_instance])
        for portion in portions:
            portion.multi_indices.sort()
        return portions


def generate_data(size, dataset_dir, save_pickle=False):
    Matrix.oblique_angle_rotations(allowed=False)
    ruleset = Ruleset(size_rules=[RuleType.CONSTANT])
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
        rpm = Matrix.make(np.random.choice(matrix_types, p=weights),
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
