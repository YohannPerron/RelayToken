import multiprocessing as mp
import os
from pathlib import Path
from turtle import down
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

from src.utils import mkdir, pylogger
from src.utils.data import read_tif_downscaled

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def prepare_downsample_data(
    gf_aois: gpd.GeoDataFrame,
    downsample_factor: int,
    folder: Path,
    layer_name: str = "rasterpath",
    statistics: List[str] = ["avg"],  # ["avg", "std", "max", "min"]
    num_workers: int = 1,
) -> Dict[str, Dict]:
    """
    Prepare downsample data targets to help evaluation and training
    - Extract to disk, return paths and profiles
    """
    # pool = mp.Pool(num_workers)

    downsample_path = {}
    downsample_folder = mkdir(folder / f"downsample_{layer_name}")
    if not os.path.exists(downsample_folder):
        os.makedirs(downsample_folder)

    for aoi_name, row_aoi in gf_aois.iterrows():
        downsample_path[aoi_name] = os.path.join(
            downsample_folder,
            f"{aoi_name}_downsample_x{downsample_factor}.tif",
        )
        if os.path.exists(downsample_path[aoi_name]):
            continue
        downsample_single_aoi_gdal(
            row_aoi,
            layer_name,
            downsample_factor,
            statistics,
            downsample_path[aoi_name],
            num_workers=num_workers,
        )
    #     pool.apply_async(
    #         downsample_single_aoi_rasterio,
    #         args=(
    #             row_aoi,
    #             layer_name,
    #             downsample_factor,
    #             statistics,
    #             downsample_path[aoi_name],
    #         ),
    #     )
    # pool.close()
    # pool.join()
    return downsample_path


def downsample_single_aoi_gdal(
    row_aoi,
    layer_name,
    downsample_factor,
    statistics,
    downpath,
    windows_size=1024,
    num_workers=1,
):
    downsample_factor = int(downsample_factor)
    original_file = str(row_aoi[layer_name])
    # open the original raster
    with rasterio.open(str(row_aoi[layer_name])) as src:
        original_resolution = src.res
    down_resolution = (
        original_resolution[0] * downsample_factor,
        original_resolution[1] * downsample_factor,
    )
    # call gdal
    os.system(
        f"gdalwarp -r average -tr {down_resolution[0]} {down_resolution[1]} "
        f"-wm 2048 -multi -wo NUM_THREADS={num_workers} "
        f"-co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co TILED=YES "
        f"-co NUM_THREADS={num_workers} -co COMPRESS=LZW "
        f"{original_file} {downpath}"
    )
    # gdalwarp -r average -ts 1295 464 \
    # -wm 2048 -multi -wo NUM_THREADS=ALL_CPUS \
    # -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co TILED=YES \
    # -co NUM_THREADS=ALL_CPUS -co COMPRESS=LZW \
    # world.tif downsized_world.tif


def downsample_single_aoi_rasterio(
    row_aoi,
    layer_name,
    downsample_factor,
    statistics,
    downpath,
    windows_size=1024,
):
    """Downsample a single aoi raster (for multiprocessing)"""
    downsample_factor = int(downsample_factor)
    windows_mult_size = int(windows_size // downsample_factor)
    windows_size = int(windows_mult_size * downsample_factor)
    with rasterio.open(str(row_aoi[layer_name])) as src:
        # downsample raster
        downsample_profile = src.profile.copy()
        height, width = int(src.height // downsample_factor), int(
            src.width // downsample_factor
        )
        layer_count = src.count * len(statistics)
        data = np.zeros((layer_count, height, width), dtype=src.dtypes[0])
        for h in range(int(height // windows_mult_size + 1)):
            for w in range(int(width // windows_mult_size + 1)):
                window = rasterio.windows.Window(
                    w * windows_size,
                    h * windows_size,
                    windows_size,
                    windows_size,
                )
                original = src.read(window=window, masked=True)
                for x in range(windows_mult_size):
                    for y in range(windows_mult_size):
                        coords_h = h * windows_mult_size + y
                        coords_w = w * windows_mult_size + x
                        if coords_h >= height or coords_w >= width:
                            continue
                        ### Doesn't go futher
                        y_min = y * downsample_factor
                        y_max = (y + 1) * downsample_factor
                        x_min = x * downsample_factor
                        x_max = (x + 1) * downsample_factor
                        # print(y_min, y_max, x_min, x_max, original.shape)
                        sub_window = original[:, y_min:y_max, x_min:x_max]
                        i = 0
                        if "avg" in statistics:
                            data[
                                i * src.count : (i + 1) * src.count,
                                coords_h,
                                coords_w,
                            ] = np.ma.mean(sub_window, axis=(1, 2))
                            i += 1
                        if "std" in statistics:
                            data[
                                i * src.count : (i + 1) * src.count,
                                coords_h,
                                coords_w,
                            ] = np.ma.std(sub_window, axis=(1, 2))
                            i += 1
                        if "max" in statistics:
                            data[
                                i * src.count : (i + 1) * src.count,
                                coords_h,
                                coords_w,
                            ] = np.ma.max(sub_window, axis=(1, 2))
                            i += 1
                        if "min" in statistics:
                            data[
                                i * src.count : (i + 1) * src.count,
                                coords_h,
                                coords_w,
                            ] = np.ma.min(sub_window, axis=(1, 2))
                            i += 1

        downsample_profile["transform"] = src.transform * src.transform.scale(
            (src.width / data.shape[-1]), (src.height / data.shape[-2])
        )

        downsample_profile["height"] = data.shape[-2]
        downsample_profile["width"] = data.shape[-1]
        downsample_profile["count"] = layer_count
        downsample_profile["dtype"] = data.dtype

        # save downsampled raster
        with rasterio.open(downpath, "w", **downsample_profile) as dst:
            dst.write(data)
        print(f"Downsampled {downpath}")


def prepare_target_profiles(
    gf_aois: gpd.GeoDataFrame,
    layer_name: str,
) -> Dict[str, Dict]:
    """
    Prepare raster targets to help evaluation
    - Extract to disk, return paths and profiles
    """
    targets_profile = {}
    for aoi_name, row_aoi in gf_aois.iterrows():
        poly_aoi = row_aoi["geometry"]
        with rasterio.open(str(row_aoi[layer_name])) as src:
            raster_profile = src.profile
        raster_profile["count"] = 1
        raster_profile["dtype"] = "uint8"
        targets_profile[aoi_name] = {"profile": raster_profile, "path": None}
    return targets_profile


def apply_split(
    gf_aois: gpd.GeoDataFrame,
    subset_train: List[str],
    subset_val: List[str],
    subset_test: List[str],
    subset_pred: List[str],
):
    """But the correct split on the aois.

    Args:
        gf_aois (gpd.GeoDataFrame): Area of interest
        subset_train (List[str]): Names of the aois to be used for training
        subset_val (List[str]): Names of the aois to be used for validation
        subset_test (List[str]): Names of the aois to be used for testing
        subset_pred (List[str]): Names of the aois to be used for prediction

    Returns:
        gpd.GeoDataFrame: Area of interest with split
    """
    # make a deep copy of the set
    subset_train = subset_train.copy()
    subset_val = subset_val.copy()
    subset_test = subset_test.copy()
    subset_pred = subset_pred.copy()

    # if there are dublicate names, duplicate the column adding _te _va _tr
    all_aois = set(subset_train + subset_val + subset_test + subset_pred)
    for name in all_aois:
        i = (
            (name in subset_train)
            + (name in subset_val)
            + (name in subset_test)
            + (name in subset_pred)
        )
        if i > 1:
            if name in subset_train:
                subset_train.pop(subset_train.index(name))
                subset_train.append(name + "_tr")
                rwo = gf_aois.loc[name].copy()
                rwo.name = name + "_tr"
                gf_aois = pd.concat([gf_aois, rwo.to_frame().T])
            if name in subset_val:
                subset_val.pop(subset_val.index(name))
                subset_val.append(name + "_va")
                rwo = gf_aois.loc[name].copy()
                rwo.name = name + "_va"
                gf_aois = pd.concat([gf_aois, rwo.to_frame().T])
            if name in subset_test:
                subset_test.pop(subset_test.index(name))
                subset_test.append(name + "_te")
                rwo = gf_aois.loc[name].copy()
                rwo.name = name + "_te"
                gf_aois = pd.concat([gf_aois, rwo.to_frame().T])
            if name in subset_pred:
                subset_pred.pop(subset_pred.index(name))
                gf_aois = subset_pred.append(name + "_pr")
                rwo = gf_aois.loc[name].copy()
                rwo.name = name + "_pr"
                gf_aois = pd.concat([gf_aois, rwo.to_frame().T])

    gf_aois["split"] = ""
    gf_aois.loc[subset_train, "split"] = "train"
    gf_aois.loc[subset_val, "split"] = "val"
    gf_aois.loc[subset_test, "split"] = "test"
    gf_aois.loc[subset_pred, "split"] = "pred"
    # Keep only aois belonging to a split
    gf_aois = gf_aois.query("split in ['train', 'val', 'test', 'pred']").copy()
    return gf_aois


def set_data_path(
    gf_aois: gpd.GeoDataFrame,
    data_path: Path,
    layers_names: List[str],
):
    for layer in layers_names:
        gf_aois[layer] = ""
        for idx, row in gf_aois.iterrows():
            name = idx
            if idx.endswith("_tr") or idx.endswith("_va"):
                name = idx[:-3]
            if idx.endswith("_te") or idx.endswith("_pr"):
                name = idx[:-3]
            gf_aois.loc[idx, layer] = data_path / layer / f"{name}.tif"

            assert gf_aois.loc[
                idx, layer
            ].exists(), f"File {gf_aois.loc[idx, layer]} does not exist"
    return gf_aois


def compute_data_stat(gf_aois, layers_names, scale=4):
    """Compute AOIs mean and variance."""
    mean = []
    std = []
    for layer in layers_names:
        stats = {}
        for aoi_name, row in gf_aois.iterrows():
            try:
                from osgeo import gdal

                ds = gdal.Open(str(row[layer]))
                mean_list, std_list = [], []
                for i in range(1, ds.RasterCount + 1):
                    band = ds.GetRasterBand(i)
                    stats_dict = band.GetStatistics(True, True)
                    mean_value = stats_dict[2]
                    std_value = stats_dict[3]
                    mean_list.append(mean_value)
                    std_list.append(std_value)
                stats[aoi_name] = dict(
                    mean=np.array(mean_list), var=np.array(std_list) ** 2
                )
            except ImportError:
                X = read_tif_downscaled(row[layer], scale=scale, crop=False)
                stats[aoi_name] = dict(
                    mean=X.mean(axis=(1, 2)), var=X.var(axis=(1, 2))
                )
        # Weighted mean and std with respect to the number of samples (~area)
        stats = pd.DataFrame(stats).T.assign(n=gf_aois["train_samples"])
        log.info(f"Estimating from per-aoi stats:\n{stats}")
        _mean = (stats["mean"] * stats["n"]).sum() / stats["n"].sum()
        _std = np.sqrt((stats["var"] * stats["n"]).sum() / stats["n"].sum())
        if isinstance(_mean, np.ndarray):
            mean.extend(_mean)
            std.extend(_std)
        else:
            mean.append(_mean)
            std.append(_std)
    return mean, std


def separate_geometry(
    multipolygon: shapely.MultiPolygon, distance: float = 1000
):
    """Simplify multypoligon geometry in a list of few simpler multypolygone.

    Args:
        multipolygon (shapely.MultiPolygon): Shape to divide
        distance (float, optional): distance under which polygo,e should still be kept together. Defaults to 1000.

    Returns:
        List[shapely.MultiPolygon]: List of simplified multipolygon
    """
    polygones = list(multipolygon.geoms)
    areas = []
    while len(polygones) > 0:
        poly = polygones.pop()
        last_removed = 0
        while last_removed > -1:
            last_removed = -1
            for i, p in enumerate(polygones):
                if shapely.dwithin(poly, p, distance):
                    poly = poly.union(p)
                    last_removed = i
                    break
            if last_removed > -1:
                polygones.pop(last_removed)
        areas.append(poly)
    return areas
