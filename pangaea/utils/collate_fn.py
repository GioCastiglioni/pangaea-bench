from typing import Callable

import torch
import torch.nn.functional as F


def get_collate_fn(modalities: list[str]) -> Callable:
    def collate_fn(
        batch: list[dict[str, dict[str, torch.Tensor] | torch.Tensor]],
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        """Collate function for torch DataLoader
        
        Args:
            batch: list of dictionaries with keys 'image', 'target', and optionally 'dates'.
                   'image' is a dict with keys corresponding to modalities and values being torch.Tensor.
                   For single images: shape (C, H, W), for time series: (C, T, H, W).
                   'target' is a torch.Tensor.
                   'dates' is a torch.Tensor if present.
        Returns:
            A dictionary with keys 'image', 'target', and 'dates' (if available).
        """
        # Compute maximum temporal dimension across modalities for time series images
        T_max = 0
        for modality in modalities:
            for x in batch:
                # Check if the image is a time series (has 4 dimensions)
                if len(x["image"][modality].shape) == 4:
                    T_max = max(T_max, x["image"][modality].shape[1])
                    
        # Pad all images to the same temporal dimension if needed
        for modality in modalities:
            for i, x in enumerate(batch):
                if len(x["image"][modality].shape) == 4:
                    T = x["image"][modality].shape[1]
                    if T < T_max:
                        padding = (0, 0, 0, 0, 0, T_max - T)
                        batch[i]["image"][modality] = F.pad(
                            x["image"][modality], padding, "constant", 0
                        )

        # Build the batch dictionary for images and target
        batch_out = {
            "image": {
                modality: torch.stack([x["image"][modality] for x in batch])
                for modality in modalities
            },
            "target": torch.stack([x["target"] for x in batch]),
        }
        
        # Conditionally add "metadata" if present in the first sample
        if "metadata" in batch[0]:
            if isinstance(x, torch.Tensor):
                batch_out["metadata"] = torch.stack([x["metadata"] for x in batch])
            else:
                batch_out["metadata"] = torch.stack([torch.linspace(0, 999, batch_out["image"][modalities[0]].shape[2]).long() for _ in range(batch_out["image"][modalities[0]].shape[0])])
            
        return batch_out

    return collate_fn

