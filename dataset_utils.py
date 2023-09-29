import os
from pathlib import Path
import json

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as T
from PIL import Image

IGBP_map = {
    1: "Evergreen Needleleaf FOrests",
    2: "Evergreen Broadleaf Forests",
    3: "Deciduous Needleleaf Forests",
    4: "Deciduous Broadleaf Forests",
    5: "Mixed Forests",
    6: "Closed (Dense) Shrublands",
    7: "Open (Sparse) Shrublands",
    8: "Woody Savannas",
    9: "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-Up Lands",
    14: "Croplands/Natural Vegetation Mosaics",
    15: "Permanent Snow and Ice",
    16: "Barren",
    17: "Water Bodies",
}

DFC_map = {
    1: "Forest",
    2: "Shrubland",
    3: "Savanna",
    4: "Grassland",
    5: "Wetlands",
    6: "Croplands",
    7: "Urban/Built-up",
    8: "Snow/Ice",
    9: "Barren",
    10: "Water",
}

# this is what we use in this work
DFC_map_clean = {
    0: "Forest",
    1: "Shrubland",
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban/Built-up",
    6: "Barren",
    7: "Water",
    255: "Invalid",
}

s1_mean = [0.7326, 0.3734]
s1_std = [0.1634, 0.1526]
s2_mean = [
    80.2513,
    67.1305,
    61.9878,
    61.7679,
    73.5373,
    105.9787,
    121.4665,
    118.3868,
    132.6419,
    42.9694,
    1.3114,
    110.6207,
    74.3797,
]
s2_std = [
    4.5654,
    7.4498,
    9.4785,
    14.4985,
    14.3098,
    20.0204,
    24.3366,
    25.5085,
    27.1181,
    7.5455,
    0.1892,
    24.8511,
    20.4592,
]

# Remapping IGBP classes to simplified DFC classes
IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])

METER_MEANS = {
    "sentinel-1.npy":     [1652.7944, 1258.3584],
    "sentinel-2-10m.npy": [1025.2214, 1138.3673, 1194.6796, 2637.0828],
    "naip.png":           [1025.2214, 1138.3673, 1194.6796, 2637.0828],
}
METER_STDS = {
    "sentinel-1.npy":     [290.2874, 280.4149],
    "sentinel-2-10m.npy": [724.7973, 545.1207, 514.9276, 1051.9325],
    "naip.png":           [724.7973, 545.1207, 514.9276, 1051.9325],
}


class NDatasets(torch.utils.data.Dataset):
    def __init__(self, datasets, use_pairs):
        """
        A dataset class that combines multiple datasets into one.

        Parameters:
        - datasets: Tuple[Dataset]
            The datasets to be combined.
        - use_pairs: bool
            Whether to align the sample indexing across datasets.

        """

        super().__init__()
        assert len(datasets) >= 1, "expected at least one dataset"
        self.n = len(datasets[0])
        self.datasets = datasets
        assert all(self.n == len(d) for d in self.datasets)

        if use_pairs:
            self.perms = [
                np.arange(self.n)
                for _ in self.datasets
            ]
        else:
            self.perms = [
                np.random.permutation(self.n)
                for _ in self.datasets
            ]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        """
        Get a sample from the combined dataset.

        Parameters:
        - index: int
            The index of the sample to retrieve.

        Returns:
        - Tuple: A tuple of samples from each dataset, aligned based on the index.
        """
        return tuple(
            dataset[perm[index]]
            for dataset, perm in zip(self.datasets, self.perms)
        )


class TwoDatasets(NDatasets):
    def __init__(self, d1, d2, use_pairs):
        super().__init__((d1,d2), use_pairs=use_pairs)


class LimitedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        ratio: float = 1.,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
            A dataset class that limits the number of items returned from the original dataset.

            Parameters:
            - dataset: torch.utils.data.Dataset
                The original dataset from which items will be sampled.
            - ratio: float, optional (default=1.0)
                The ratio of items to be sampled from the original dataset. Value should be between 0 and 1.
            - shuffle: bool, optional (default=True)
                Whether to shuffle the sampled items.
            - seed: int, optional (default=42)
                The seed value for random number generation.

        """
        super().__init__()

        filename = Path("/share/projects/ottopia/perms/"
                        f"dataset_{seed}_{shuffle}_{len(dataset)}_{ratio}.npy")
        filename.parent.mkdir(exist_ok=True)

        self.ratio = ratio
        self.dataset = dataset
        self.n_items = int(ratio * len(self.dataset))

        if filename.exists():
            print(f">> Loading permutation from file {filename}")
            self.perm = np.load(filename)
        else:
            if shuffle:
                self.perm = np.random.permutation(self.n_items)
            else:
                self.perm = np.arange(self.n_items)
            print(f">> Saving permutation to file {filename}")
            np.save(filename, self.perm)
        #self._validate_counts()

    def _validate_counts(self):
        counts = {}
        for _, y in self:
            while isinstance(y, tuple): y = y[-1]
            assert isinstance(y, int)
            if y not in counts: counts[y] = 0
            counts[y] += 1
        all_classes = set(y for _, y in self.dataset)
        for c in all_classes:
            assert c in counts, f"Class {c} not sampled"
            assert counts[c] >= 2, f"Less than two samples for class {c} ({counts[c]} samples)"

    def __len__(self):
        return self.n_items

    def __getitem__(self, index):
        new_index = self.perm[index]
        return self.dataset[new_index]
