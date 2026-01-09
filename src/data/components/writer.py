import multiprocessing as mp
from ast import Dict

import numpy as np
import rasterio

from src.utils.data import apply_mapping, set_sparse_rasterio_profile
from src.utils.geoaffine import Interpolation, reproject_to_tcell_rasterize


def _worker_write(writer_args):
    (
        raster_profile,
        output_path,
        data_window_pairs,
        save_pred_as_logits,
        nb_classes,
        mapping,
        colormap,
    ) = writer_args
    with rasterio.open(output_path, "w+", **raster_profile) as writer:
        for func, func_arg, window in data_window_pairs:
            data = func(*func_arg)
            FileWriter._write_to_disk(
                data, window, writer, save_pred_as_logits, nb_classes, mapping
            )
        if colormap is not None:
            writer.write_colormap(1, colormap)


class FileWriter:
    def __init__(
        self,
        output_dir,
        raster_profile,
        aoi_name,
        threaded=True,
        mapping=None,
        colormap=None,
    ):
        self.output_dir = output_dir
        self.raster_profile = raster_profile
        self.aoi_name = aoi_name
        self.threaded = threaded
        self.mapping = mapping
        self.colormap = colormap

        self.rasterio_writer = None
        self.save_pred_as_logits = None
        self.nb_classes = 1
        self.random_id = np.random.randint(0, 1000000)
        self.list_of_write = []
        self.worker_process = None
        self.output_path = None

    def open(self, prefix, save_pred_as_logits, nb_classes=1):
        self.save_pred_as_logits = save_pred_as_logits
        if prefix is None:
            self.output_path = f"{self.output_dir}/{self.aoi_name}."
        else:
            self.output_path = f"{self.output_dir}/{prefix}_{self.aoi_name}."

        if self.raster_profile["driver"] == "GTiff":
            self.output_path += "tif"
            if save_pred_as_logits:
                self.raster_profile = set_sparse_rasterio_profile(
                    self.raster_profile, "float32"
                )
                self.raster_profile.update({"count": nb_classes})
                self.nb_classes = nb_classes
                # compression
                # self.raster_profile.update("compress", "lzw")
                # self.raster_profile.update("predictor", 2)
            else:
                self.raster_profile = set_sparse_rasterio_profile(
                    self.raster_profile, "uint8"
                )
                self.raster_profile.update({"count": 1})
                # compression
                if nb_classes <= 2:
                    self.raster_profile.update(
                        {"compress": "CCITTFAX4", "nbits": 1}
                    )
                else:
                    self.raster_profile.update(
                        {"compress": "lzw", "predictor": 2}
                    )
        elif (
            self.raster_profile["driver"] == "PNG"
            or self.raster_profile["driver"] == "JPEG"
        ):
            self.raster_profile.update(
                {"driver": "PNG"}
            )  # force PNG driver as JPEG is not supported by rasterio
            self.output_path += "png"
            if save_pred_as_logits:
                self.raster_profile.update({"dtype": "float32"})
                self.raster_profile.update({"count": nb_classes})
                self.nb_classes = nb_classes
            else:
                self.raster_profile.update({"dtype": "uint8"})
                self.raster_profile.update({"count": 1})

            if "tiled" in self.raster_profile:
                self.raster_profile.pop("tiled")
            if "blockysize" in self.raster_profile:
                self.raster_profile.pop("blockysize")
            if "compress" in self.raster_profile:
                self.raster_profile.pop("compress")
            if "interleave" in self.raster_profile:
                self.raster_profile.pop("interleave")
            if "photometric" in self.raster_profile:
                self.raster_profile.pop("photometric")
        else:
            raise Exception(
                f"Unsupported raster driver: {self.raster_profile['driver']}"
            )

        self.rasterio_writer = rasterio.open(
            self.output_path, "w+", **self.raster_profile
        )
        if self.colormap is not None:
            self.rasterio_writer.write_colormap(1, self.colormap)

    @staticmethod
    def _write_to_disk(
        data, window, writer, save_pred_as_logits, nb_classes=1, mapping=None
    ):
        if writer.closed:
            raise Exception("FileWriter is not open.")

        if save_pred_as_logits:
            data = data.astype("float32")
            data_to_write = data
            if isinstance(data, np.ma.MaskedArray):
                mask = data.mask
            else:
                mask = np.zeros_like(data, dtype=bool)
            indexes = range(nb_classes)
        else:
            data_to_write = np.ma.argmax(data, axis=0).astype("uint8")
            if isinstance(data, np.ma.MaskedArray):
                mask = np.any(data.mask, axis=0)
            else:
                mask = np.zeros_like(data_to_write, dtype=bool)
            indexes = 1

        if mapping is not None:
            data_to_write = apply_mapping(data_to_write, mapping)

        data = np.ma.MaskedArray(data=data_to_write, mask=mask)
        dst_data = writer.read(window=window, indexes=indexes, masked=True)
        dst_data[~data.mask] = data[~data.mask]
        writer.write(dst_data, window=window, indexes=indexes)

    def write(self, func, func_arg, window):
        if not self.threaded:
            self.__class__._write_to_disk(
                func(*func_arg),
                window,
                self.rasterio_writer,
                self.save_pred_as_logits,
                nb_classes=self.nb_classes,
                mapping=self.mapping,
            )
        else:
            self.list_of_write.append((func, func_arg, window))

    def close(self):
        if self.is_open():
            try:
                self.rasterio_writer.close()
            except Exception as e:
                print(f"Error closing rasterio writer: {e}")

        if self.threaded and self.list_of_write:
            if self.worker_process is not None:
                self.worker_process.join()

            worker_args = (
                self.raster_profile.copy(),
                self.output_path,
                self.list_of_write,
                self.save_pred_as_logits,
                self.nb_classes,
                self.mapping,
                self.colormap,
            )
            self.worker_process = mp.Process(
                target=_worker_write, args=(worker_args,)
            )
            self.worker_process.start()
            self.worker_process.join()  # Vlad: You really have to close here!
            self.list_of_write = []

    def is_open(self):
        return self.rasterio_writer and not self.rasterio_writer.closed

    def is_equal(self, other):
        return (
            self.output_dir == other.output_dir
            and self.aoi_name == other.aoi_name
            and self.random_id == other.random_id
        )


class ArchaeoWindowWriter:
    def __init__(
        self,
        file_writer,
        window,
        square,
        safft_world_to_icell,
        safft_icell_to_tcell,
        poly_aoi,
    ):
        self.file_writer = file_writer
        self.window = window
        self.square = square
        self.safft_world_to_icell = safft_world_to_icell
        self.safft_icell_to_tcell = safft_icell_to_tcell
        self.poly_aoi = poly_aoi

    @staticmethod
    def process_data(
        data,
        window,
        safft_icell_to_tcell,
        safft_world_to_icell,
        square,
        poly_aoi,
        overlap,
    ):
        if overlap > 0:
            # extract the center part of the square (square is an shapely polygon)
            square_side = square.area**0.5
            square = square.buffer(-overlap * square_side / 2.0)

        if poly_aoi is None:
            goodpoly_world = square
        else:
            goodpoly_world = poly_aoi & square
        if goodpoly_world.is_empty:
            return

        scores_icell = np.ma.stack(
            [
                reproject_to_tcell_rasterize(
                    score,
                    window,
                    safft_icell_to_tcell,
                    goodpoly_world,
                    safft_world_to_icell,
                    Interpolation.LINEAR,
                )
                for score in data
            ]
        )

        return scores_icell

    def write(self, data, overlap=0):

        func_arg = (
            data,
            self.window,
            self.safft_icell_to_tcell,
            self.safft_world_to_icell,
            self.square,
            self.poly_aoi,
            overlap,
        )
        self.file_writer.write(
            self.__class__.process_data, func_arg, window=self.window
        )

    def open(self, prefix, save_pred_as_logits, nb_classes=1):
        self.file_writer.open(
            prefix, save_pred_as_logits, nb_classes=nb_classes
        )

    def close(self):
        self.file_writer.close()

    def is_open(self):
        return self.file_writer.is_open()


class EmptyWindowWriter:
    def __init__(self):
        pass

    def write(self, data, overlap=0):
        pass

    def open(self, prefix, save_pred_as_logits, nb_classes=1):
        pass

    def close(self):
        pass

    def is_open(self):
        return True


class FlairWindowWriter:
    def __init__(self, aoi_name, raster_profile, output_dir="pred"):
        self.file_writer = FileWriter(output_dir, raster_profile, aoi_name)
        self.aoi_name = aoi_name

    def write(self, data, overlap=0):
        self.file_writer.write(data, window=None)

    def open(self, prefix, save_pred_as_logits, nb_classes=1):
        self.file_writer.open(prefix, save_pred_as_logits, nb_classes)

    def close(self):
        self.file_writer.close()

    def is_open(self):
        return self.file_writer.is_open()
