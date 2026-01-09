import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import rasterio
import shapely
from lightning import LightningDataModule

from src.data.urur_datamodule import URURDataModule

id_to_trainid = {
    0: 255,  # unlabeled
    1: 255,  # ego vehicle
    2: 255,  # rectification border
    3: 255,  # out of roi
    4: 255,  # static
    5: 255,  # dynamic
    6: 255,  # ground
    7: 0,  # road
    8: 1,  # sidewalk
    9: 255,  # parking
    10: 255,  # rail track
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,  # pole
    18: 255,  # polegroup
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: -1,  # license plate
}
inverted_mapping = {v: k for k, v in id_to_trainid.items() if v != 255}


class CityDataModule(URURDataModule):
    """`LightningDataModule` for Cityscapes dataset"""

    def __init__(
        self,
        data_pth: str = "datasets/",
        batch_size: int = 16,
        sample_multiplier: int = 1,
        imageside: int = 256,
        imagesize: int = 256,
        test_overlap: float = 0,
        val_overlap: float = 0,
        mean: float = 0,
        std: float = 255,
        mean_type: Union[str, List[str]] = "global",
        num_workers: int = 0,
        iinter: int = 1,  # LINEAR
        pin_memory=True,
        tsize_base=None,
        tsize_enum_sizes=[1],
        tsize_enum_probs=None,
        tsize_range_frac=0,
        tsize_range_sizes=[0.5, 2],
        trot_angle=90,
        trot_prob=0.5,
        min_overlap=0.2,
        generate_targets=True,
        factors_m: Union[None, List[int]] = None,
        factors_px: Union[None, List[int]] = None,
        cache_dir="./.cache",
        **kwargs,
    ) -> None:
        """Initialize a `LightningDataModule` for Cityscapes dataset."""
        # We override the id_to_trainid with our own specific mapping
        super().__init__(
            data_pth=data_pth,
            batch_size=batch_size,
            sample_multiplier=sample_multiplier,
            imageside=imageside,
            imagesize=imagesize,
            test_overlap=test_overlap,
            val_overlap=val_overlap,
            mean=mean,
            std=std,
            mean_type=mean_type,
            num_workers=num_workers,
            iinter=iinter,
            pin_memory=pin_memory,
            tsize_base=tsize_base,
            tsize_enum_sizes=tsize_enum_sizes,
            tsize_enum_probs=tsize_enum_probs,
            tsize_range_frac=tsize_range_frac,
            tsize_range_sizes=tsize_range_sizes,
            trot_angle=trot_angle,
            trot_prob=trot_prob,
            min_overlap=min_overlap,
            generate_targets=generate_targets,
            factors_m=factors_m,
            factors_px=factors_px,
            cache_dir=cache_dir,
            id_to_trainid=id_to_trainid,
            trainid_to_id=inverted_mapping,
            **kwargs,
        )

        # We need to store the raster profile for the train_dataloader
        self.raster_profile = None

    def get_data_split(self) -> pd.DataFrame:
        """Get image paths, labels and splits for Cityscapes dataset.

        Returns:
            pd.DataFrame: DataFrame with columns 'image', 'split', 'label'
        """
        # list all image files
        images = list(
            (Path(self.hparams.data_pth) / "leftImg8bit").rglob("*.png")
        )
        assert (
            len(images) != 0
        ), f"No image detected in the dataset at {Path(self.hparams.data_pth) / 'leftImg8bit'}"

        # get split from directory structure (train/val/test)
        split = [x.parts[-3] for x in images]

        # get label
        def get_associated_label(image_path):
            label_path = (
                image_path.parent.parent.parent.parent
                / "gtFine"
                / image_path.parts[-3]
                / image_path.parts[-2]
                / image_path.parts[-1].replace(
                    "leftImg8bit", "gtFine_labelIds"
                )
            )
            return label_path

        label = [get_associated_label(x) for x in images]

        # create a dataframe
        data = pd.DataFrame({"image": images, "split": split, "label": label})

        # set test to pred and duplicate val as test
        data.loc[data["split"] == "test", "split"] = "pred"

        # Create test set from frankfurt validation images
        val_data = data[data["split"] == "val"].copy()
        val_test = val_data[
            val_data["image"].astype(str).str.contains("frankfurt")
        ]
        val_test.loc[:, "split"] = "test"
        data = pd.concat([data, val_test], ignore_index=True)

        # Store the first label's profile for later use
        with rasterio.open(data["label"][0]) as src:
            self.raster_profile = src.profile

        return data
