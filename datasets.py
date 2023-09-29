import os
from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from albumentations.pytorch import ToTensorV2
import albumentations as A
from rasterio.windows import Window
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils import AlbumentationsToTorchTransform, identity,normalize_channel_sen12_dfc,RandomAugmentations,DFCRandomAugmentations
from dfc_sen12ms_dataset import DFCSEN12MSDataset, Seasons, S1Bands, S2Bands, LCBands

from dataset_utils import (
    DFC_map_clean,
    METER_MEANS,
    NDatasets,
    TwoDatasets,
    METER_STDS,)



METER_DATA_LOCATION = Path(os.getenv("SCRATCH", "/share/projects/ottopia/")) / "meter-ml"
SEN12MS_DATA_LOCATION = Path(os.getenv("SCRATCH", "/share/projects/ottopia/")) / "SEN12MS"
DFC2020_DATA_LOCATION = Path(os.getenv("SCRATCH", "/share/projects/ottopia/")) / "DFC2020"


class MeterMLDataset(torch.utils.data.Dataset):
    classes_labels = {
        "CAFOs": 0,
        "CAFOs-Landfills": 0,
        "CAFOs-WWTreatment": 0,
        "Landfills": 1,
        "Landfills-RefineriesAndTerminals": 1,
        "Landfills-WWTreatment": 1,
        "Mines": 2,
        "Mines-Landfills": 2,
        "Mines-ProcessingPlants": 2,
        "Mines-RefineriesAndTerminals": 2,
        "Negative": 3,
        "ProcessingPlants": 4,
        "ProcessingPlants-Landfills": 4,
        "RefineriesAndTerminals": 5,
        "RefineriesAndTerminals-Landfills": 5,
        "RefineriesAndTerminals-ProcessingPlants": 5,
        "RefineriesAndTerminals-WWTreatment": 5,
        "WWTreatment": 6,
        "WWTreatment-Landfills": 6,
        "WWTreatment-Mines": 6,
        "WWTreatment-ProcessingPlants": 6,
        "WWTreatment-RefineriesAndTerminals": 6,
        "WWTreatment-RefineriesAndTerminals-Landfills": 6,
    }

    @staticmethod
    def sentinel1(json_file, transform=identity, **kwargs):
        return MeterMLDataset(METER_DATA_LOCATION / json_file, "sentinel-1.npy", transform, **kwargs)

    @staticmethod
    def sentinel2(json_file, transform=identity, **kwargs):
        return MeterMLDataset(METER_DATA_LOCATION / json_file, "sentinel-2-10m.npy", transform, **kwargs)

    @staticmethod
    def train_sentinel1(transform=identity, **kwargs):
        return MeterMLDataset.sentinel1("train_dataset.json", transform, **kwargs)

    @staticmethod
    def validate_sentinel1(transform=identity, **kwargs):
        return MeterMLDataset.sentinel1("val_dataset.json", transform, **kwargs)

    @staticmethod
    def train_sentinel2(transform=identity, **kwargs):
        return MeterMLDataset.sentinel2("train_dataset.json", transform, **kwargs)

    @staticmethod
    def validate_sentinel2(transform=identity, **kwargs):
        return MeterMLDataset.sentinel2("val_dataset.json", transform, **kwargs)



    def __init__(
        self,
        json_file_path: str,
        filename: str,
        transform=identity,
        exclude_negatives: bool=False,
        **kwargs
    ):
        super().__init__()
        self.filename = filename
        json_file_path = Path(json_file_path)

        with open(json_file_path) as f:
            self.data = json.load(f)["features"]

        self.data_folder = json_file_path.parent


        self.data = [
            data
            for data in self.data
            if (
                (self.data_folder / data["properties"]["Image_Folder"]).exists() and
                (exclude_negatives and data["properties"]["Type"] != "Negative" or
                    not exclude_negatives)
            )
        ]

        self.transform = transform

        self.normalizer = T.Normalize(
            mean=METER_MEANS[self.filename],
            std=METER_STDS[self.filename],
        )        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]["properties"]
        path = self.data_folder / sample["Image_Folder"] / self.filename

        if self.filename.endswith(".npy"):
            x = np.load(path).astype(np.float32)
        else:
            x = np.asarray(Image.open(path))
        label = MeterMLDataset.classes_labels[sample["Type"]]
        x = torch.tensor(x).float().permute(2, 0, 1)
        x = self.normalizer(x)

        return self.transform(x), label


class DFCDataset(Dataset):
    """Pytorch wrapper for DFCSEN12MSDataset"""

    def __init__(
        self,
        base_dir,
        mode="dfc",
        transforms=None,
        clip_sample_values=True,
        used_data_fraction=1.0,
        image_px_size=256,
        cover_all_parts=False,
        balanced_classes=False,
        use_sup_con=False,
        sampling_seed=42,
        normalize=False,
    ):
        """cover_all_parts: if image_px_size is not 256, this makes sure that during validation the entire image is used
        during training, we read image parst at random parts of the original image, during vaildation, use a non-overlapping sliding window to cover the entire image"""
        super(DFCDataset, self).__init__()

        self.clip_sample_values = clip_sample_values
        self.used_data_fraction = used_data_fraction
        self.image_px_size = image_px_size
        self.cover_all_parts = cover_all_parts
        self.balanced_classes = balanced_classes
        self.normalize = normalize
        self.use_sup_con = use_sup_con


        if mode == "dfc":
            self.seasons = [
                Seasons.AUTUMN_DFC,
                Seasons.SPRING_DFC,
                Seasons.SUMMER_DFC,
                Seasons.WINTER_DFC,
            ]
        elif mode == "test":
            self.seasons = [Seasons.TESTSET]
        elif mode == "validation":
            self.seasons = [Seasons.VALSET]
        elif mode == "sen12ms":
            self.seasons = [
                Seasons.SPRING,
                Seasons.SUMMER,
                Seasons.FALL,
                Seasons.WINTER,
            ]
        else:
            raise ValueError(
                "Unsupported mode, must be in ['dfc', 'sen12ms', 'test', 'validation']"
            )

        self.data = DFCSEN12MSDataset(base_dir)

        if self.balanced_classes:
            self.observations = pd.read_csv(
                os.path.join(f"splits/{mode}_observations_balanced_classes.csv"),
                header=0,
                # names=["Season", "Scene", "ID", "dfc_label", "copy_nr"],
            )
        else:
            self.observations = pd.read_csv(
                os.path.join(f"splits/{mode}_observations.csv"),
                header=None,
                names=["Season", "Scene", "ID"],
            )

        if self.cover_all_parts:
            num_img_parts = int(256**2 / self.image_px_size**2)
            obs = []
            for season, scene, idx in self.observations.values:
                for i in range(num_img_parts):
                    obs.append([season, scene, idx, i])

            self.observations = pd.DataFrame(
                obs, columns=["Season", "Scene", "ID", "ScenePart"]
            )

        self.observations = self.observations.sample(
            frac=self.used_data_fraction, random_state=sampling_seed
        ).sort_index()
        self.transforms = transforms
        self.mode = mode


        if self.use_sup_con:
            ## random augmentation give us the same image of 2 random augmentation
            self.train_transforms = DFCRandomAugmentations()
            
        base_aug = A.Compose([ToTensorV2()])

        self.base_transform = AlbumentationsToTorchTransform(base_aug)

    def __getitem__(self, idx, s2_bands=S2Bands.ALL, transform=True, normalize=True):
        
        obs = self.observations.iloc[idx]
        season = Seasons[obs.Season[len("Seasons.") :]]

        if self.image_px_size != 256:
            # crop the data to self.image_px_size times self.image_px_size (e.g. 128x128)
            x_offset, y_offset = np.random.randint(0, 256 - self.image_px_size, 2)
            window = Window(x_offset, y_offset, self.image_px_size, self.image_px_size)

        else:
            window = None

        if self.mode != "sen12ms":
            # high-resolution LC (dfc) labels are not available for the entire dataset
            s1, s2, lc, dfc, bounds = [
                x.astype(np.float32) if type(x) == np.ndarray else x
                for x in self.data.get_s1_s2_lc_dfc_quad(
                    season,
                    obs.Scene,
                    int(obs.ID),
                    s1_bands=S1Bands.ALL,
                    s2_bands=s2_bands,
                    lc_bands=LCBands.LC,
                    dfc_bands=LCBands.DFC,
                    include_dfc=True,
                    window=window,
                    mode=self.mode
                )
            ]
            dfc[dfc == 3] = 0
            dfc[dfc == 8] = 0
            dfc[dfc >= 3] -= 1
            dfc[dfc >= 8] -= 1
            dfc -= 1
            dfc[dfc == -1] = 255

            dfc_unique, dfc_counts = np.unique(dfc, return_counts=True)
            dfc_label = dfc_unique[
                dfc_counts.argmax()
            ]  # this is already mapped to dfc in data.get_s1_s2_lc_dfc_quad
            dfc_label_str = DFC_map_clean[int(dfc_label)]

            dfc_multilabel = torch.tensor(
                [
                    class_idx
                    for class_idx, num in zip(dfc_unique, dfc_counts)
                    if num / self.image_px_size**2 >= 0.1 and class_idx != 255
                ]
            ).long()
            dfc_multilabel_one_hot = torch.nn.functional.one_hot(
                dfc_multilabel.flatten(), num_classes=8
            ).float()
            dfc_multilabel_one_hot = dfc_multilabel_one_hot.sum(dim=0)  # create one one-hot label for all classes
            # all classes which make up more than 10% of a scene, as per https://arxiv.org/pdf/2104.00704.pdf

        else:
            s1, s2, lc, bounds = [
                x.astype(np.float32) if type(x) == np.ndarray else x
                for x in self.data.get_s1_s2_lc_dfc_quad(
                    season,
                    obs.Scene,
                    int(obs.ID),
                    s1_bands=S1Bands.ALL,
                    s2_bands=s2_bands,
                    lc_bands=LCBands.LC,
                    dfc_bands=LCBands.DFC,
                    include_dfc=False,
                    window=window,
                    mode=self.mode
                )
            ]

            dfc = None

        lc[lc == 3] = 0
        lc[lc == 8] = 0
        lc[lc >= 3] -= 1
        lc[lc >= 8] -= 1
        lc -= 1
        # print("Number of invalid pixels:", lc[lc == -1].size)
        lc[lc == -1] = 255

        # use the most frequent MODIS class as pseudo label
        lc_unique, lc_counts = np.unique(lc, return_counts=True)
        lc_label = lc_unique[
            lc_counts.argmax()
        ]  # this is already mapped to dfc in data.get_s1_s2_lc_dfc_quad
        lc_label_str = DFC_map_clean[int(lc_label)]

        lc_multilabel = torch.tensor(
            [
                class_idx
                for class_idx, num in zip(lc_unique, lc_counts)
                if num / self.image_px_size**2 >= 0.1 and class_idx != 255
            ]
        ).long()
        lc_multilabel_one_hot = torch.nn.functional.one_hot(
            lc_multilabel.flatten(), num_classes=8
        ).float()
        lc_multilabel_one_hot = lc_multilabel_one_hot.sum(dim=0)
        # all classes which make up more than 10% of a scene, as per https://arxiv.org/pdf/2104.00704.pdf

        # as per the baseline paper https://arxiv.org/pdf/2002.08254.pdf
        if self.clip_sample_values:
            s1 = np.clip(s1, a_min=-25, a_max=0)
            s1 = (
                s1 + 25
            )  # go from [-25,0] to [0,25] interval to make normalization easier
            s2 = np.clip(s2, a_min=0, a_max=1e4)


        if self.use_sup_con != True:
            s1 = self.base_transform(np.moveaxis(s1, 0, -1))
            s2 = self.base_transform(np.moveaxis(s2, 0, -1))

        elif self.use_sup_con == True: 
            s1 = self.train_transforms(s1)
            s2 = self.train_transforms(s2)     
            ## here s1 and s2 represent an array of 2 images


        # normalize images channel wise
        if (normalize or self.normalize):
            if type(s1) == list:
                ## this means s1 has 2 images 
                s1[0] = normalize_channel_sen12_dfc(s1[0])
                s1[1] = normalize_channel_sen12_dfc(s1[1])

                s2[0] = normalize_channel_sen12_dfc(s2[0])
                s2[1] = normalize_channel_sen12_dfc(s2[1])
            else:
                s1 = normalize_channel_sen12_dfc(s1)
                s2 = normalize_channel_sen12_dfc(s2)
        
        output = {
            "s1": s1,
            "s2": s2,
            "lc": lc,
            "bounds": bounds,
            "idx": idx,
            "lc_label": lc_label,
            "lc_label_str": lc_label_str,
            "lc_multilabel": lc_multilabel.numpy().tolist(),
            "lc_multilabel_one_hot": lc_multilabel_one_hot,
            "season": str(season.value),
            "scene": obs.Scene,
            "id": obs.ID,
        }

        output_tensor = {
            "s1": s1,
            "s2": s2,
            "lc": lc,
            "idx": idx,
            "lc_label": lc_label,
            "lc_multilabel_one_hot": lc_multilabel_one_hot,
        }  # new pytorch version does not allow non-tensor values in dataloader

        if dfc is not None:
            output.update(
                {
                    "dfc": dfc,
                    "dfc_label": dfc_label,
                    "dfc_label_str": dfc_label_str,
                    "dfc_multilabel_one_hot": dfc_multilabel_one_hot,
                }
            )  # , "dfc_multilabel" : dfc_multilabel.numpy().tolist()})

            output_tensor.update(
                {
                    "dfc": dfc,
                    "dfc_label": dfc_label,
                    "dfc_multilabel_one_hot": dfc_multilabel_one_hot,
                }
            )  # , "dfc_multilabel" : dfc_multilabel.numpy().tolist()})
            # print(",".join([k + " : " + str(np.array(v).shape) for k,v in output_tensor.items()]))
            return output_tensor
        else:
            # print(",".join([k + " : " + str(np.array(v).shape) for k,v in output_tensor.items()]))
            return output_tensor

    def __len__(self):
        return self.observations.shape[0]



