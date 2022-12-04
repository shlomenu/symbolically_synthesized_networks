import os

from raven_gen import Matrix, MatrixType, Ruleset, RuleType, ComponentType, LayoutType

from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from tqdm import trange
import pickle
import math
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
        with open(os.path.join(self.pkl_dir, filename)) as f:
            return pickle.load(f)

    @classmethod
    def split(cls, dataset_dir, val_prop, test_prop, load_correct=True, load_incorrect=False):
        """
        Creates (train, test, val) split of all data under dataset_dir.  The test instances are 
        enumerated from the end of the sorted list of multi-indices, which dependent only on the names
        of the files not the order in which the operating system displays files.  Among the remaining, 
        the validation set is chosen randomly.  The proportion of test data is interpreted as a 
        proportion of the total; the proportion of validation data is interpreted as the proportion of 
        what remains after test data is set aside.  To perform K-fold cross validation with leftover test 
        data, for example, you would use `val_prop=.33`, not `val_prop=(.33 * (1 - test_prop)).  
        """
        assert (val_prop < 1. and test_prop < 1.)
        dataset = cls(dataset_dir, load_correct=load_correct,
                      load_incorrect=load_incorrect)
        indices = deepcopy(dataset.multi_indices)
        del dataset.multi_indices[:]
        train, val, test = deepcopy(dataset), deepcopy(
            dataset), deepcopy(dataset)
        n_instances = len(
            {instance for (instance, _, _) in indices})
        puzzles_per_instance = len(indices) // n_instances
        assert (n_instances * puzzles_per_instance == len(indices))
        n_test_instances = math.ceil(test_prop * n_instances)
        n_train_val_instances = math.floor((1 - test_prop) * n_instances)
        test_split = -(n_test_instances * puzzles_per_instance)
        test.multi_indices = deepcopy(indices[test_split:])
        n_val_instances = math.ceil(val_prop * n_train_val_instances)
        val_instances_idx = sorted(np.random.choice(
            n_train_val_instances, n_val_instances, replace=False))
        val.multi_indices = [indices[idx] for idx in val_instances_idx]
        for i, datum in enumerate(indices[:test_split]):
            if val_instances_idx and i == val_instances_idx[0]:
                del val_instances_idx[0]
                continue
            else:
                train.multi_indices.append(datum)
        return train, val, test


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
