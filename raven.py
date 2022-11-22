import os

from raven_gen import Matrix, MatrixType, Ruleset, RuleType, ComponentType, LayoutType

from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from tqdm import trange
import pickle


class RavenDataset(Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
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
            elif len(parts) == 5:
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
            label = 1
        else:
            filename = f"rpm_{instance}_background_{background_color}_alternative_{sub_instance}.png"
            label = 0
        img = read_image(os.path.join(self.data_dir, filename))
        return img.float(), label

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
