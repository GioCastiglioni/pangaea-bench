###
# Modified version of the PASTIS-HD dataset
# original code https://github.com/gastruc/OmniSat/blob/main/src/data/Pastis.py
###

import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from einops import rearrange

from pangaea.datasets.base import RawGeoFMDataset


def prepare_dates(date_dict, reference_date):
    """Date formating."""
    if type(date_dict) is str:
        date_dict = json.loads(date_dict)
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return torch.tensor(d.values)

# def prepare_dates(date_dict, reference_date):
#     d = pd.DataFrame().from_dict(date_dict, orient="index")
#     d = d[0].apply(
#         lambda x: (
#             datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
#             - reference_date
#         ).days
#     )
#     return d.values


# def compute_norm_vals(folder, sat):
#     norm_vals = {}
#     for fold in range(1, 6):
#         dt = Dummy(folder=folder, norm=False, folds=[fold], sats=[sat])
#         means = []
#         stds = []
#         for i, b in enumerate(dt):
#             print("{}/{}".format(i, len(dt)), end="\r")
#             data = b[0][0][sat]  # T x C x H x W
#             data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
#             means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
#             stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

#         mean = np.stack(means).mean(axis=0).astype(float)
#         std = np.stack(stds).mean(axis=0).astype(float)

#         norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

#     with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
#         file.write(json.dumps(norm_vals, indent=4))


# def split_image(image_tensor, nb_split, id):
#     """
#     Split the input image tensor into four quadrants based on the integer i.
#     To use if Pastis data does not fit in your GPU memory.
#     Returns the corresponding quadrant based on the value of i
#     """
#     if nb_split == 1:
#         return image_tensor
#     i1 = id // nb_split
#     i2 = id % nb_split
#     height, width = image_tensor.shape[-2:]
#     half_height = height // nb_split
#     half_width = width // nb_split
#     if image_tensor.dim() == 4:
#         return image_tensor[
#             :,
#             :,
#             i1 * half_height : (i1 + 1) * half_height,
#             i2 * half_width : (i2 + 1) * half_width,
#         ].float()
#     if image_tensor.dim() == 3:
#         return image_tensor[
#             :,
#             i1 * half_height : (i1 + 1) * half_height,
#             i2 * half_width : (i2 + 1) * half_width,
#         ].float()
#     if image_tensor.dim() == 2:
#         return image_tensor[
#             i1 * half_height : (i1 + 1) * half_height,
#             i2 * half_width : (i2 + 1) * half_width,
#         ].float()


class Dummy(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        reference_date: str = ""
    ):
        """Initialize the PASTIS dataset.

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image.
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
        """
        super(Dummy, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
        )

        assert split in ["train", "val", "test"], "Split must be train, val or test"
        if split == "train":
            folds = [1, 2, 3]
        elif split == "val":
            folds = [4]
        else:
            folds = [5]
        self.modalities = ["S2"]

        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.num_classes = 11

        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(root_path, "metadata.geojson"))
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        self.memory_dates = {}

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        self.date_tables = {s: None for s in self.modalities}

        for s in self.modalities:
            dates = self.meta_patch["dates-{}".format(s)]
            self.date_range = np.array(range(-200, 600))

            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                if type(date_seq) == str:
                    date_seq = json.loads(date_seq)
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

    def __len__(self):
        return self.len
    
    def get_dates(self, id_patch, sat):
        indices = np.where(self.date_tables[sat][id_patch] == 1)[0]
        indices = indices[indices < len(self.date_range)]  # Ensure indices are within bounds
        return torch.tensor(self.date_range[indices], dtype=torch.int32)


    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get the item at index i.

        Args:
            i (int): index of the item.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {"optical": torch.Tensor,
                 "sar": torch.Tensor},
            "target": torch.Tensor,
             "metadata": dict}.
        """
        line = self.meta_patch.iloc[i]
        id_patch = self.id_patches[i]
        name = line["ID_PATCH"]
        target = torch.from_numpy(
            np.load(
                os.path.join(self.root_path, "ANNOTATIONS/TARGET_" + str(name) + ".npy")
            )
        )
        # only for s2
        modality_name = self.modalities[0]
        data = {
                modality_name: np.load(
                    os.path.join(
                        self.root_path,
                        "DATA_{}".format(modality_name),
                        "{}_{}.npy".format(modality_name, name),
                    )
                )
        }

        is_s2 = lambda x: x=="S2"

        data = {"optical" if is_s2(s) else "": torch.from_numpy(a) for s, a in data.items()}

        metadata = {
                "dates": self.get_dates(id_patch, s) for s in self.modalities
            }

        # change the temporal axis
        indexes = {s: torch.linspace(
                0, a.shape[1] - 1, self.multi_temporal, dtype=torch.long
            ) for s, a in data.items()}

        data = {s: rearrange(a.to(torch.float32)[indexes[s]], "t c h w -> c t h w")  for s, a in data.items()}
        
        return_dict = {
            "image": data,
            "target": target.to(torch.int64),
            "metadata": metadata["dates"][indexes["optical"]].to(torch.int32),
        }

        return return_dict

    @staticmethod
    def download():
        pass