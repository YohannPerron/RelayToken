#!/usr/bin/env python3
"""
Usage:
    eval_vlad.py run <ckpt> [options] [--] [<cfg_overrides>...]
    eval_vlad.py augment <old> <new>
    eval_vlad.py (-h | --help)

Options:
    --cfg <cfg>             Train config  (default: <ckpt>/../train_config.yaml)
    --over <csv_overrides>  Dotlist omniconf overrides like K1=V1,K2=V2, etc
    --id <id>               ID that allows optional reruning
                                No ID - > Create new exp (LAST_ID+1)
                                ID -> Run (ID) exp if exists, or create one
    -l, --label <label>              Label the experiment
    -h, --help              Show help

    Routes:
    --eval <mode>            (test_predict, test, predict) How to eval [default: test]

"""

import itertools
import re
import logging
import os
import sys
from typing import Optional, Tuple
from pathlib import Path

import torch
from lightning import LightningDataModule, LightningModule, Trainer

import cv2
import numpy as np
import hydra
import rootutils
from omegaconf import DictConfig
from omegaconf import OmegaConf as OC
from docopt import docopt
import pandas as pd

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.utils import printlog_config_tree, mkdir
from src.utils.utils_vlad import (
    reasonable_logging_setup,
    LogCaptorToRecords,
    get_experiment_id_string,
    release_logcaptor_to_folder,
    git_repo_query_v2,
    add_logging_filehandlers,
    Isaver_fast,
)

log = logging.getLogger(__name__)

# (a, b, d, e, x_off, y_off)
Affine_transform_shapely = Tuple[int, int, int, int, int, int]


def afft_shapely_to_ocv(safft: Affine_transform_shapely) -> np.ndarray:
    (a, b, d, e, x_off, y_off) = safft
    return np.array([[a, b, x_off], [d, e, y_off]])


def _get_data_split_cityscapes(data_pth) -> pd.DataFrame:
    """Get image paths, labels and splits for Cityscapes dataset.

    Returns:
        pd.DataFrame: DataFrame with columns 'image', 'split', 'label'
    """
    # list all image files
    images = list((Path(data_pth) / "leftImg8bit").rglob("*.png"))
    assert (
        len(images) != 0
    ), f"No image detected in the dataset at {Path(data_pth) / 'leftImg8bit'}"

    # get split from directory structure (train/val/test)
    split = [x.parts[-3] for x in images]

    # get label
    def get_associated_label(image_path):
        label_path = (
            image_path.parent.parent.parent.parent
            / "gtFine"
            / image_path.parts[-3]
            / image_path.parts[-2]
            / image_path.parts[-1].replace("leftImg8bit", "gtFine_labelIds")
        )
        return label_path

    label = [get_associated_label(x) for x in images]

    # create a dataframe
    data = pd.DataFrame({"image": images, "split": split, "label": label})

    # set test to pred and duplicate val as test
    data.loc[data["split"] == "test", "split"] = "pred"

    # Create test set from frankfurt validation images
    val_data = data[data["split"] == "val"].copy()
    val_test = val_data[val_data["image"].astype(str).str.contains("frankfurt")]
    val_test.loc[:, "split"] = "test"
    data = pd.concat([data, val_test], ignore_index=True)
    # data["image"] = data.image.astype(str)
    # data["label"] = data.label.astype(str)

    # # Store the first label's profile for later use
    # with rasterio.open(data["label"][0]) as src:
    #     raster_profile = src.profile
    return data


def derive_exp_folder_path(
    p_project_root: Path,
    args_id: Optional[str],
    args_l: Optional[str],
    sha8: Optional[str],  # sha8 + possibly DIRTY prefix
):
    # Avoiding hydra for clean experiments
    # Find experiments
    p_exp_parfold = mkdir(p_project_root / "outputs_vlad")
    runs = {}
    # pattern = re.compile(r"^E(\d+):(.*?)(?::.*)?:(.*)$")
    pattern = re.compile(r"^E(\d+):([^:]+)(?::([^:]+))?:(.*)$")
    for fold in p_exp_parfold.iterdir():
        if m := pattern.match(fold.name):
            runs[int(m.group(1))] = {
                "label": m.group(2),
                "sha8": m.group(3),
                "path": fold,
            }
    exp_id = (max(runs.keys()) + 1) if args_id is None else int(args_id)
    if exp_id in runs:
        # Selecting existing run
        p_exp_fold = runs[exp_id]["path"]
        log.info(f"Using id {exp_id} to select existing run:\n{p_exp_fold}")
        exp_label = runs[exp_id]["label"]
        if args_l and args_l != exp_label:
            log.warning(f"Label mismatch ({args_l} != {exp_label}) for run {exp_id}")
        exp_sha = runs[exp_id]["sha8"]
        if sha8 and sha8 != exp_sha:
            log.warning(f"SHA mismatch ({sha8} != {exp_sha}) selecting run {exp_id}")
    else:
        # Creating new run
        exp_label = args_l if args_l else "new"
        exp_name = "E{:04d}:{}".format(exp_id, exp_label)
        if sha8 is not None:
            exp_name += ":" + sha8
        exp_name += ":" + get_experiment_id_string()
        p_exp_fold = mkdir(p_exp_parfold / exp_name)
        log.info(f"Using id {exp_id} to create new run {p_exp_fold}")
    return p_exp_fold


def main(args):
    """Run experiment manually without hydra, maximising control"""
    # Simple logging and output folder setup
    with LogCaptorToRecords(pause_others=True) as lctr:
        p_project_root = Path(os.getcwd())
        repo, short_sha = git_repo_query_v2(p_project_root)
        p_exp_fold = derive_exp_folder_path(
            p_project_root, args["--id"], args["--label"], short_sha
        )
    release_logcaptor_to_folder(lctr, p_exp_fold)

    # Load training hydra config for the experiment
    p_ckpt = Path(args["<ckpt>"])
    if args["--cfg"] is None:
        p_cfg = p_ckpt.parent / "train_config.yaml"
    else:
        p_cfg = Path(args["--cfg"])
    assert p_ckpt.exists()
    assert p_cfg.exists()

    # Fake hydra runtime for URURDataModule.__init__ get call
    fake = OC.create({"hydra": {"runtime": {"output_dir": str(p_exp_fold)}}})
    hydra.core.hydra_config.HydraConfig.instance().cfg = fake

    cfg: DictConfig = OC.load(p_cfg)
    # Replace Hydra fields to prevent crashes during comprehension resolution
    cfg["paths"]["work_dir"] = str(p_project_root)
    cfg["paths"]["output_dir"] = str(p_exp_fold)
    # Ensure that devices and nodes equals 1
    cfg["trainer"]["devices"] = 1
    cfg["trainer"]["num_nodes"] = 1
    cfg["trainer"]["strategy"] = "auto"
    # Allow overrides from argv
    if len(orr := args["<cfg_overrides>"]):
        cfg_orr = OC.from_dotlist(orr)
        printlog_config_tree(cfg_orr, "overrides")
        cfg = OC.merge(cfg, cfg_orr)

    log.info("Original argv:\n{}".format(" ".join(sys.orig_argv)))
    printlog_config_tree(cfg)
    # Manually print the new config tree (reset)
    printlog_config_tree(cfg, "train_adjusted")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Loading model from checkpoint")
    state_dict = torch.load(p_ckpt)["state_dict"]
    model.load_state_dict(state_dict=state_dict, strict=False)

    if args["--eval"] in ["test", "predict", "test_predict"]:
        if "test" in args["--eval"]:
            log.info("Testing")  # This is fast
            trainer.test(
                model=model,
                datamodule=datamodule,
            )
        if "predict" in args["--eval"]:
            log.info("Predicting")  # This is slow
            trainer.predict(
                model=model,
                datamodule=datamodule,
                ckpt_path=None,
            )
    else:
        raise ValueError("Wrong --eval value")


def aug_to_str(aug: dict) -> str:
    s = "s{:.2f}".format(aug["scale"])
    f = "hf" if aug["hflip"] else ""
    return "_".join(filter(None, [s, f]))


def get_ocv_transform_for_augmentations(scale, hflip, src_W, src_H):
    # OCV Mask is  [a, b, x_off]
    #              [d, e, y_off]

    new_W, new_H = int(round(src_W * scale)), int(round(src_H * scale))
    cx, cy = (src_W - 1) / 2.0, (src_H - 1) / 2.0
    # Center translate
    Tc = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    # Horizontal flip
    if hflip:
        F = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    else:
        F = np.eye(3, dtype=np.float32)
    # Scale
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    # Translate back
    Tb = np.array(
        [[1, 0, (new_W - 1) / 2.0], [0, 1, (new_H - 1) / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )
    M = Tb @ S @ F @ Tc
    return M[:2], new_W, new_H


def apply_ocv_transform(img, M, new_W, new_H, flags=cv2.INTER_LINEAR):
    mean_value = int(np.round(img.mean()))
    new_WH = (new_W, new_H)
    new_img = cv2.warpAffine(
        img,
        M,
        new_WH,
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=mean_value,
    )
    return new_img


def hack_augment_cityscapes(args):
    p_old_fold = Path(args["<old>"])
    assert (p_old_fold / "gtFine").exists()
    assert (p_old_fold / "leftImg8bit").exists()
    df_old = _get_data_split_cityscapes(p_old_fold)

    p_new_fold = mkdir(args["<new>"])
    add_logging_filehandlers(p_new_fold)

    # augtransform_cfg = {
    #     "scale": [1.0, 1.25, 1.5, 1.75, 2.0, 2.25],
    #     "hflip": [False, True],
    # }
    augtransform_cfg = {
        "scale": [1.0, 1.25],
        "hflip": [False, True],
    }
    augtransform_list = [
        dict(zip(augtransform_cfg, xs))
        for xs in itertools.product(*augtransform_cfg.values())
    ]

    def create_augmented_images(row):
        dflist_augmented_images = []
        # Source image
        src_img = cv2.imread(str(row.image))  # H, W, C (BGR)
        src_H, src_W = src_img.shape[:2]
        # New (relative) folder
        p_nrel_fold = mkdir(p_new_fold / row.image.parent.relative_to(p_old_fold))
        for augtransform in augtransform_list:
            augtransform_str = aug_to_str(augtransform)
            new_path = p_nrel_fold / (
                row.image.stem + "_" + augtransform_str + row.image.suffix
            )
            M, new_W, new_H = get_ocv_transform_for_augmentations(
                augtransform["scale"], augtransform["hflip"], src_W, src_H
            )
            new_img = apply_ocv_transform(src_img, M, new_W, new_H)
            cv2.imwrite(str(new_path), new_img)  # H, W, C (BGR)
            dflist_augmented_images.append(
                dict(
                    df_old_idx=row.Index,
                    augtransform_str=augtransform_str,
                    new_path=str(new_path),
                    src_H=src_H,
                    src_W=src_W,
                    M=M,
                    augtransform=augtransform,
                )
            )
        return dflist_augmented_images

    # Go over old dataset, create new files for each one
    # Since images fit in RAM, pure OCV is the simplest solution
    isaver = Isaver_fast(
        p_new_fold / "_isaver",
        [[x] for x in df_old.itertuples()],
        create_augmented_images,
        progress="saving_augmented_images",
    )
    dflist_augmented_images_grp = isaver.run()
    df_augmented_images = pd.DataFrame(itertools.chain(*dflist_augmented_images_grp))

    # Test projecting back some images
    # Project back every image
    temp = mkdir(p_new_fold / "_test_backproject")
    unaug_imgs = {}
    for row in df_augmented_images.query("df_old_idx==0").itertuples():
        orig_row = df_old.iloc[row.df_old_idx]
        orig_img = cv2.imread(str(orig_row.image))
        aug_img = cv2.imread(str(row.new_path))
        unaug_W, unaug_H = (row.src_W, row.src_H)

        M_invert = cv2.invertAffineTransform(row.M)

        unaug_img = apply_ocv_transform(aug_img, M_invert, unaug_W, unaug_H)
        unaug_imgs[row.Index] = unaug_img

        cv2.imwrite(str(temp / Path(row.new_path).name), unaug_img)  # H, W, C (BGR)
        cv2.imwrite(
            str(temp / (Path(row.new_path).name + "_diff.jpg")),
            np.abs(unaug_img - orig_img),
        )  # H, W, C (BGR)


if __name__ == "__main__":
    log = reasonable_logging_setup(logging.INFO)
    args = docopt(__doc__)
    if args["augment"]:
        hack_augment_cityscapes(args)
    else:
        main(args)
