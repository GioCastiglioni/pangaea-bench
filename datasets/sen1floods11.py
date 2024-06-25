# Obtained from: https://github.com/synativ/RSFMs/blob/main/src/rsfms/datamodules/sen1floods11.py

import dataclasses
import glob
import os
from pathlib import Path
from typing import Any, List, Optional

import geopandas
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections.abc import Iterator
from functools import partial
import numpy as np
import pandas as pd
import rioxarray
import torch
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torch.utils.data import Dataset


def filter_valid_files(
    files, valid_files: Optional[Iterator[str]] = None, ignore_extensions: bool = False, allow_substring: bool = True
):
    if valid_files is None:
        return sorted(files)
    valid_files = list(valid_files)
    if ignore_extensions:
        valid_files = [os.path.splitext(sub)[0] for sub in valid_files]
    filter_function = partial(
        _split_filter_function,
        valid_files=valid_files,
        ignore_extensions=ignore_extensions,
        allow_substring=allow_substring,
    )
    # TODO fix this
    filtered = filter(filter_function, files)

    return sorted(filtered)


def _split_filter_function(file_name, valid_files: list[str], ignore_extensions=False, allow_substring=True):
    base_name = os.path.basename(file_name)
    if ignore_extensions:
        base_name = os.path.splitext(base_name)[0]
    if not allow_substring:
        return base_name in valid_files

    for valid_file in valid_files:
        if valid_file in base_name:
            return True
    return False


class Sen1Floods11(Dataset):
    """NonGeo dataset implementation for fire scars."""

    def __init__(self, data_root: str, split="train", bands: Optional[List[int]] = None) -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            msg = "Split must be one of train, test, val."
            raise Exception(msg)
        if split == "val":
            split = "valid"

        self.bands = bands
        data_root = Path(data_root)
        data_directory = data_root / "data/flood_events/HandLabeled/"
        input_directory = data_directory / "S2Hand"
        label_directory = data_directory / "LabelHand"

        split_file = data_root / f"splits/flood_handlabeled/flood_{split}_data.txt"
        metadata_file = data_root / "Sen1Floods11_Metadata.geojson"
        self.metadata = geopandas.read_file(metadata_file)

        self.image_files = sorted(glob.glob(os.path.join(input_directory, "*.tif")))
        self.segmentation_mask_files = sorted(glob.glob(os.path.join(label_directory, "*.tif")))

        with open(split_file) as f:
            split = f.readlines()
        valid_files = {rf"{substring.strip()}" for substring in split}
        self.image_files = filter_valid_files(
            self.image_files,
            valid_files=valid_files,
            ignore_extensions=True,
            allow_substring=True,
        )
        self.segmentation_mask_files = filter_valid_files(
            self.segmentation_mask_files,
            valid_files=valid_files,
            ignore_extensions=True,
            allow_substring=True,
        )

        self.rgb_indices = [2, 1, 0]

    def __len__(self) -> int:
        return len(self.image_files)

    def _get_date(self, index) -> np.ndarray:
        # move this logic to the model?
        file_name = self.image_files[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            date = pd.to_datetime("13-10-1998", dayfirst=True)
        else:
            date = pd.to_datetime(self.metadata[self.metadata["location"] == location]["s2_date"].item())
        date_np = np.zeros((1, 3))
        date_np[0, 0] = date.year
        date_np[0, 1] = date.dayofyear - 1  # base 0
        date_np[0, 2] = date.hour
        return date_np

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = self._load_file(self.image_files[index]).astype(np.float32) * 0.0001
        if self.bands:
            image = image[self.bands, ...]

        output = {
            'image': {
                'optical':image,
            },
            'target': self._load_file(self.segmentation_mask_files[index]).astype(np.int64),
            'metadata': {
                "timestamp": self._get_date(index).astype(np.float32),
            }
        }
        return output

    def _load_file(self, path: Path):
        data = rioxarray.open_rasterio(path)
        return data.to_numpy()

    def plot(self, sample: dict[str, Tensor], suptitle: Optional[str] = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"]
        if torch.is_tensor(image):
            image = image.numpy()
        image = image.take(self.rgb_indices, axis=0)
        image = np.transpose(image, (1, 2, 0))
        image = (image - image.min(axis=(0, 1))) * (1 / image.max(axis=(0, 1)))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        label_mask = np.transpose(label_mask, (1, 2, 0))

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]

        return self._plot_sample(
            image,
            label_mask,
            prediction=prediction_mask if showing_predictions else None,
            suptitle=suptitle,
        )

    @staticmethod
    def _plot_sample(image, label, num_classes=3, prediction=None, suptitle=None, class_names=None):
        num_images = 5 if prediction else 4
        fig, ax = plt.subplots(1, num_images, figsize=(8, 6))

        # for legend
        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=num_classes - 1)
        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(label, cmap="jet", norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(label, cmap="jet", alpha=0.3, norm=norm)

        if prediction:
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)

        cmap = plt.get_cmap("jet")
        legend_data = []
        for i, _ in enumerate(range(num_classes)):
            class_name = class_names[i] if class_names else str(i)
            data = [i, cmap(norm(i)), class_name]
            legend_data.append(data)
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
    

if __name__ == "__main__":
    import pdb
    dataset = Sen1Floods11(data_root="/geomatics/gpuserver-0/yjia/datasets/sen1floods11_v1.1", split="train")
    sample = dataset[0]
    pdb.set_trace()
    fig = dataset.plot(sample)
    plt.show()
    fig.savefig('output.png')