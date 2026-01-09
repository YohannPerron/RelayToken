import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
import rasterio
import shapely
import torch
from joblib import Memory
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components import (
    TDataset_gtiff,
    TiffSampler,
    VectorSampler,
    apply_split,
    compute_data_stat,
    prepare_target_profiles,
    set_data_path,
)
from src.data.components.writer import FileWriter
from src.utils import mkdir, pylogger
from src.utils.data import get_feats_with_proper_labels, gpkg_save
from src.utils.geoaffine import (
    Interpolation,
    itm_collate,
    sample_grid_squares_from_aoi_v2,
    sample_random_squares_from_aoi_v2,
)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Create mapping from RGB values to class IDs
# fmt: off
id_to_trainid = {
    (0, 0, 0):           0,   # others
    (230, 230, 230):     1,   # building
    (100, 100, 100):     2,   # greenhouse
    (200, 230, 160):     3,   # woodland
    (95, 163, 7):        4,   # farmland
    (255, 255, 100):     5,   # bareland
    (150, 200, 250):     6,   # water
    (240, 100, 80):      7,   # road
    (255, 255, 255):   255,   # ignore
}
# fmt: on
inverted_mapping = {v: k for k, v in id_to_trainid.items()}


class URURDataModule(LightningDataModule):
    """`LightningDataModule`"""

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
        iinter: Interpolation = 1,  # LINEAR
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
        id_to_trainid=id_to_trainid,
        trainid_to_id=inverted_mapping,
        **kwargs,
    ) -> None:
        """Initialize a `LightningDataModule`.

        Args:
            data_pth (str, optional): Path to aois. Defaults to "data/".
            batch_size (int, optional): Batch size. Defaults to 16.
            sample_multiplier (int, optional): How many time an area should be sample (in esperance) per epoch. Defaults to 1.
            imageside (int, optional): Size of a sample in meters. Defaults to 256.
            imagesize (int, optional): Size of a sample in pixel. Defaults to 256.
            test_overlap (float, optional): Overlap for test samples. Defaults to 0.
            mean (float, optional): Mean for global normalization (if None computed). Defaults to None.
            std (float, optional): Standard deviation for normalization(if None computed). Defaults to None.
            mean_type (Union[str, List[str]], optional): Type of mean to use for normalization. Can be a list with a different value for each input channel. Either global, local, avg_pool or max_pool. Defaults to "local".
            num_workers (int, optional): Number of parrarel worker to load the datasets. Defaults to 0.
            iinter (Interpolation, optional): Interpolation type. Defaults to 1 ~ LINEAR.
            pin_memory (bool, optional): Use pin memory. Defaults to True.
            tsize_base (_type_, optional): For training : Default image side before augment.None for equal to image size dimension in meters. Defaults to None.
            tsize_enum_sizes (list, optional): For training : Randomly multiply the size by a factor in the sizeswith probs. Defaults to [1].
            tsize_enum_probs (_type_, optional): For training : Randomly multiply the size by a factor in the sizes with probs. Defaults to None.
            tsize_range_frac (int, optional): For training : Randomly sample frac of the train sample with unform size in the range. Defaults to 0.
            tsize_range_sizes (list, optional): For training : Randomly sample frac of the train sample with unform size in the range. Defaults to [0.5, 2].
            trot_angle (int, optional): For training : Randomly rotate. Defaults to 90.
            trot_prob (float, optional): For training : Randomly rotate. Defaults to 0.5.
            min_overlap (float, optional): For training : Minimum area of a sample that must be inside the raster. Defaults to 0.2.
            generate_targets (bool, optional): Generate target or not (used to optimize MAE). Defaults to True.
            factors_m (Union[None, List[int]], optional): Factors for downsample in meters. Each value is an additional scale with imageside multiply by the factor. Defaults to None.
            factors_px (Union[None, List[int]], optional): Factors for downsample in pixels. Each value is an additional scale with imagesize multiply by the factor. Defaults to None.
            cache_dir (str, optional): Directory where to store the cache. Defaults to "./.cache".
        """

        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.epoch = 0

        self.rgen = np.random.default_rng(np.random.randint(0, 2**32 - 1))
        log.info(f"Using {self.rgen=} for training dataset")

        try:
            self.output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
        except:
            self.output_dir = "./logs"
        log.info(f"Using {self.output_dir=}")

        self.factors_m = factors_m
        self.factors_px = factors_px

        self.generate_targets = generate_targets

        self.id_to_trainid = id_to_trainid
        self.trainid_to_id = trainid_to_id
        self.colormap = None  # Use for saving with indexes colors

        self.memory = Memory(cache_dir, verbose=0)

        self.pixel_size = 1  # m

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        pass

    def get_data_split(self) -> pd.DataFrame:
        # list all image file
        ext = "tif"
        image_train = list(
            (Path(self.hparams.data_pth) / "train" / "image").rglob(f"*.{ext}")
        )
        image_val = list(
            (Path(self.hparams.data_pth) / "val" / "image").rglob(f"*.{ext}")
        )
        image_test = list(
            (Path(self.hparams.data_pth) / "test" / "image").rglob(f"*.{ext}")
        )
        assert (
            len(image_train) != 0
        ), f"No image detected in the dataset at {Path(self.hparams.data_pth) / 'train' / 'image'}"

        split_train = ["train"] * len(image_train)
        split_val = ["val"] * len(image_val)
        split_test = ["test"] * len(image_test)

        # get label
        def get_associated_label(image_path):
            # test/image/00000.tif -> test/label/00000.png
            label_path = image_path.parent.parent / "label" / image_path.parts[-1]
            label_path = label_path.with_suffix(".png")
            return label_path

        # append all images
        images = image_train + image_val + image_test
        split = split_train + split_val + split_test
        label = [get_associated_label(x) for x in images]

        # load colormap from a test file
        with rasterio.open(get_associated_label(image_test[0])) as src:
            self.colormap = src.colormap(1)  # i ->(R,G,B,A)
        # get the mapping from RGB to class ID
        id_to_trainid = {}
        for i, (r, g, b, a) in self.colormap.items():
            if self.id_to_trainid[(r, g, b)] != 255:
                id_to_trainid[i] = self.id_to_trainid[(r, g, b)]
        self.trainid_to_id = {v: k for k, v in id_to_trainid.items()}
        # add cutoff
        id_to_trainid["cutoffid"] = 8
        id_to_trainid["cutoffto"] = 255
        self.id_to_trainid = id_to_trainid
        log.info(f"Using {self.id_to_trainid=}")

        # create and return a dataframe
        return pd.DataFrame({"image": images, "split": split, "label": label})

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        Args:
            stage(str): The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        fold_evalprep = Path(self.output_dir + "/eval_prepared")

        if not hasattr(self, "data"):
            self.data = self.get_data_split()

        # Assign number of samples to each gf_aois proportional to area
        batch_size = self.hparams.batch_size
        imageside = self.hparams.imageside
        sample_multiplier = self.hparams.sample_multiplier
        # get image dimension (same for all images)
        with rasterio.open(self.data["image"][0]) as src:
            area = (src.width * self.pixel_size) * (src.height * self.pixel_size)

        self.data["train_samples"] = int(area / imageside**2 * sample_multiplier)

        # raster targets profile
        log.info("Preparing target profiles")
        self.data["raster_profile"] = None
        self.data["bounds"] = None
        self.data["raster_profile_label"] = None
        for id, row_aoi in self.data.iterrows():
            with rasterio.open(row_aoi["image"]) as src:
                self.data.at[id, "raster_profile"] = src.profile
                self.data.at[id, "bounds"] = shapely.geometry.box(*src.bounds)
            with rasterio.open(row_aoi["label"]) as src:
                self.data.at[id, "raster_profile_label"] = src.profile

        # Compute global mean and std
        mean, std = self.hparams.mean, self.hparams.std
        log.info(f"global {mean=}, {std=}")

        log.info("Creating samplers")
        iinter = Interpolation(self.hparams.iinter)
        # Create sampler for each aois
        samplers = {}
        for id, row_aoi in self.data.iterrows():
            # Create Sampler
            inputSamplers = []
            inputSamplers.append(
                TiffSampler(
                    row_aoi["image"],
                    id,
                    iinter,
                    factors_m=self.factors_m,
                    factors_px=self.factors_px,
                )
            )
            targetSamplers = [
                TiffSampler(
                    row_aoi["label"],
                    id,
                    Interpolation(0),  # Nearest
                    factors_m=self.factors_m,
                    factors_px=self.factors_px,
                    out_type="uint8",
                )
            ]

            samplers[id] = [
                inputSamplers,
                targetSamplers,
            ]
        self.data["sampler"] = samplers

        # Create file writers
        log.info("Creating file writers")
        output_dir = f"{self.output_dir}/predictions"
        mkdir(Path(output_dir))

        self.data["file_writer"] = None
        for id, row_aoi in self.data.iterrows():
            # Créer le file writer pour chaque AOI
            name = row_aoi["image"].stem
            file_writer = FileWriter(
                output_dir,
                row_aoi["raster_profile_label"],
                name,
                threaded=True,
                mapping=self.trainid_to_id,
                colormap=self.colormap,
            )
            self.data.loc[id, "file_writer"] = file_writer

        # Instantiate reference grid-evaluating dataloaders for evaluation
        log.info("Creating grid sampling dataloaders")
        self.datasets_gridsampling = {}

        @self.memory.cache
        def get_square(split, poly_aoi, profile, imageside, val_overlap, test_overlap):
            raster_afft = profile["transform"]
            if split == "train":
                eval_stride = imageside
                shift = (0, 0)
            if split == "val":
                eval_stride = imageside - int(imageside * val_overlap)
                shift = (0, 0)
            elif split == "test" or split == "pred":
                eval_stride = imageside - int(imageside * test_overlap)
                shift = (-eval_stride // 2, -eval_stride // 2)
            gf_squares = sample_grid_squares_from_aoi_v2(
                poly_aoi,
                imageside,
                None,
                stride=eval_stride,
                shift=shift,
                raster_afft=raster_afft,
            )
            return gf_squares

        WH = (self.hparams.imagesize, self.hparams.imagesize)  # in Pixel
        global_rank = self.trainer.global_rank if self.trainer is not None else 0
        for id, row_aoi in self.data.iterrows():

            gf_squares = get_square(
                row_aoi["split"],
                row_aoi["bounds"],
                row_aoi["raster_profile"],
                self.hparams.imageside,
                self.hparams.val_overlap,
                self.hparams.test_overlap,
            )
            if id == 0:
                gpkg_save(
                    gf_squares,
                    mkdir(fold_evalprep / "squares"),
                    f"{id}_squares_r{global_rank}",
                )

            inputSamplers, targetSamplers = row_aoi["sampler"]
            tdata_eval = TDataset_gtiff(
                gf_squares,
                inputSamplers,
                targetSamplers,
                mean,
                std,
                self.hparams.mean_type,
                label_map=self.id_to_trainid,
                WH=WH,
                generate_targets=self.generate_targets,
                return_debug_info=False,
                file_writer=row_aoi["file_writer"],
            )
            self.datasets_gridsampling[id] = tdata_eval

        log.info("DataModule setup complete")

        # # Check targets, assign number of squares inside
        # self.data["eval_samples"] = {
        #     k: len(v.gf_squares) for k, v in self.datasets_gridsampling.items()
        # }

        # nb_train_samples = 0
        # # Print combined data stats
        # for split, iids in self.data.groupby("split").groups.items():
        #     gf = self.data.loc[iids]
        #     with pd.option_context(
        #         "display.max_rows",
        #         None,
        #         "display.max_columns",
        #         None,
        #         "display.width",
        #         512,
        #     ):
        #         log.info(
        #             "Split: {}, Samples train={} eval={}:\n{}".format(
        #                 split,
        #                 gf.train_samples.sum(),
        #                 gf.eval_samples.sum(),
        #                 gf,
        #             )
        #         )
        #     nb_train_samples += gf.train_samples.sum()

        self.mean = mean
        self.std = std
        self.iinter = iinter
        self.WH = WH
        self.nb_train_samples = self.data.query("split == 'train'")[
            "train_samples"
        ].sum()

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        tsize_base = self.hparams.tsize_base
        if tsize_base is None:
            tsize_base = self.hparams.imageside
        tsize_enum_sizes = self.hparams.tsize_enum_sizes
        tsize_enum_probs = self.hparams.tsize_enum_probs
        tsize_range_frac = self.hparams.tsize_range_frac
        tsize_range_sizes = self.hparams.tsize_range_sizes
        trot_angle = self.hparams.trot_angle
        trot_prob = self.hparams.trot_prob
        min_overlap = self.hparams.min_overlap

        # Create squares dataset and appropriate dataset/dataloader
        rgen = self.rgen
        gf_train_squares = []
        train_datasets = []
        for id, row_aoi in self.data[self.data["split"] == "train"].iterrows():
            poly_aoi = row_aoi["bounds"]
            raster_afft = row_aoi["raster_profile"]["transform"]
            n_samples = row_aoi["train_samples"]
            squares = sample_random_squares_from_aoi_v2(
                poly_aoi,
                n_samples,
                rgen,
                tsize_base,
                tsize_enum_sizes,
                tsize_enum_probs,
                tsize_range_frac,
                tsize_range_sizes,
                trot_angle,
                trot_prob,
                min_overlap,
                raster_afft,
            )
            gf_aoi_squares = gpd.GeoDataFrame(geometry=squares, crs=None)
            gf_aoi_squares["id"] = id

            # create aoi dataloader
            inputSamplers, targetSamplers = self.data["sampler"][id]
            tdata_train = TDataset_gtiff(
                gf_aoi_squares,
                inputSamplers,
                targetSamplers,
                self.mean,
                self.std,
                self.hparams.mean_type,
                label_map=self.id_to_trainid,
                WH=self.WH,
                generate_targets=self.generate_targets,
                return_debug_info=False,
            )
            train_datasets.append(tdata_train)

            gf_train_squares.append(gf_aoi_squares)

        # gf_train_squares = pd.concat(
        #     gf_train_squares, axis=0, ignore_index=True
        # )
        # # Log for debugging
        # dep_fold = (
        #     self.output_dir
        #     + f"/runtime/e{self.epoch}r{self.trainer.global_rank}"
        # )
        # gpkg_save(gf_train_squares, dep_fold, "train_squares")

        data_train = torch.utils.data.ConcatDataset(train_datasets)
        dload_train = DataLoader(
            data_train,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=itm_collate,
            drop_last=True,  # Just a safeguard
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
        )

        self.epoch += 1
        return dload_train

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        validation_names = self.data.query("split == 'val'").index.tolist()
        train_names = []  # self.gf_aois.query("split == 'train'").index.tolist()
        val_dataset = torch.utils.data.ConcatDataset(
            [self.datasets_gridsampling[name] for name in validation_names]
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=itm_collate,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
        )
        if len(train_names) == 0:
            return val_dataloader
        else:
            evaltrain_datasets = torch.utils.data.ConcatDataset(
                [self.datasets_gridsampling[name] for name in train_names]
            )
            evaltrain_dataloader = DataLoader(
                evaltrain_datasets,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                collate_fn=itm_collate,
                drop_last=False,
                pin_memory=self.hparams.pin_memory,
                # persistent_workers=True,
            )
            return [evaltrain_dataloader, val_dataloader]

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        test_names = self.data.query("split == 'test'").index.tolist()
        test_datasets = torch.utils.data.ConcatDataset(
            [self.datasets_gridsampling[name] for name in test_names]
        )
        test_dataloader = DataLoader(
            test_datasets,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=itm_collate,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
        )
        return test_dataloader

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        pred_names = self.data.query("split == 'pred'").index.tolist()
        pred_datasets = torch.utils.data.ConcatDataset(
            [self.datasets_gridsampling[name] for name in pred_names]
        )
        pred_dataloader = DataLoader(
            pred_datasets,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=itm_collate,
            drop_last=False,
            pin_memory=self.hparams.pin_memory,
            # persistent_workers=True,
        )
        return pred_dataloader

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage (Optional[str]): The stage being torn down. Either "fit", "validate", "test", or "predict".
                Defaults to None.
        """
        pass

        # Fermer tous les file writers
        for writer in self.data["file_writer"]:
            writer.close()

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns:
                A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict (Dict[str, Any]): The datamodule state returned by `self.state_dict()`.

        Returns:
            None
        """
        pass
