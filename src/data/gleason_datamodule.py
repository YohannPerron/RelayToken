import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import rasterio
import shapely
from lightning import LightningDataModule

from src.data.urur_datamodule import URURDataModule, inverted_mapping

id_to_trainid = {1: 0, 2: 0, 6: 0, 3: 1, 4: 2, 5: 3}  # building


class GleasonDataModule(URURDataModule):
    """`LightningDataModule` for Gleason dataset"""

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
        """Initialize a `LightningDataModule` for Gleason dataset.

        Args:
            pixel_size (float, optional): Size of a pixel in micrometers. Defaults to 0.3.

            Other parameters are inherited from URURDataModule.
        """
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
        self.pixel_size = 1  # Store the pixel size

    def get_data_split(self) -> pd.DataFrame:
        """Get image paths, labels and splits for Gleason dataset.

        Returns:
            pd.DataFrame: DataFrame with columns 'image', 'split', 'label'
        """
        # list all image files
        images = list(
            (Path(self.hparams.data_pth) / "Train Imgs").rglob("*.jpg")
        )
        assert (
            len(images) != 0
        ), f"No image detected in the dataset at {Path(self.hparams.data_pth) / 'Train Imgs'}"

        # get split (slide001_core003.jpg)
        split = []
        for image_path in images:
            match = re.search(r"slide(\d+)_core(\d+)", image_path.stem)
            if match:
                slide = int(match.group(1))
                core = int(match.group(2))
                # Example logic: assign "train" if the slide number is even, "val" otherwise.
                if slide == 7:
                    split.append("test")
                elif core <= 15:
                    split.append("val")
                else:
                    split.append("train")
            else:
                warnings.warn(
                    f"Filename {image_path.name} does not match expected pattern"
                )
                exit()

        # Print the counts of images per set
        print("Number of images per set:")
        print("Train:", split.count("train"))
        print("Validation:", split.count("val"))
        print("Test:", split.count("test"))

        # get label
        def get_associated_label(image_path):
            stem = image_path.stem  # e.g. "slide001_core003"
            new_stem = f"{stem}_classimg_nonconvex"  # e.g. "slide001_core003_classimg_nonconvex"
            label_path = (
                image_path.parent.parent / "Maps_majority" / f"{new_stem}.png"
            )
            return label_path

        label = [get_associated_label(x) for x in images]

        # create and return a dataframe
        return pd.DataFrame({"image": images, "split": split, "label": label})
