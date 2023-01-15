import os
import pickle
import math
from copy import deepcopy


from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np


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
