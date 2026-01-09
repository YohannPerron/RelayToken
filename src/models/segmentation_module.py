from math import isnan
from typing import Any, Dict, Iterable, List, Tuple

import einops
import hydra
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import JaccardIndex
from torchmetrics.classification.accuracy import Accuracy

from src.utils import mkdir, pylogger
from src.utils.flops import get_new_ops

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    FLOP_ANALYSIS = True
except ImportError:
    FLOP_ANALYSIS = False
    log.warning("fvcore not installed, flop count analysis will not be available.")


def extract_center(tensor, keep_fraction):
    """Extract the center region of a tensor based on the given overlap.

    Args:
        tensor (torch.Tensor): The input tensor.
        keep_fraction (float): The proportion of the tensor to keep.

    Returns:
        torch.Tensor: The center region of the input tensor.
    """
    if tensor is None:
        return None
    overlap = 1 - keep_fraction
    h, w = tensor.shape[-2:]
    new_h, new_w = h - int(h * overlap), w - int(w * overlap)
    h_start, w_start = int(h * overlap) // 2, int(w * overlap) // 2
    return tensor[..., h_start : h_start + new_h, w_start : w_start + new_w]


class SegmentationModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        warmup_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        metric_monitored: str = None,
        loss_ce_weight: torch.Tensor = None,
        num_classes: int = 10,
        val_overlap: float = 0.0,
        test_overlap: float = 0.0,
        save_freq: int = 1,
        save_eval_only: bool = False,
        save_pred_as_logits: bool = False,
        index_to_ignore: int = 255,
        context_factor_m: int = 1,
        context_factor_px: int = 1,
        loss_context_factor: float = 0.0,
        loss_co_pred_factor: float = 0.0,
        loss_co_embed_factor: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize a `SegmentationModule`.

        Args:
            net (torch.nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): The optimizer.
            compile (bool): Whether to compile the model.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.
            warmup_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate warmup scheduler. Defaults to None.
            metric_monitored (str, optional): The metric to monitor. Defaults to None.
            loss_ce_weight (torch.Tensor, optional): The class weights for the cross-entropy loss. Defaults to None.
            num_classes (int, optional): The number of classes. Defaults to 10.
            val_overlap (float, optional): The overlap to use during validation. Defaults to 0.0.
            test_overlap (float, optional): The overlap to use during testing. Defaults to 0.0.
            save_freq (int, optional): The frequency to save the predictions. Defaults to 1.
            save_eval_only (bool, optional): Whether to only save the predictions for the evaluation set. Defaults to False.
            save_pred_as_logits (bool, optional): Whether to save the predictions as logits or just the argmax. Defaults to False.
            index_to_ignore (int, optional): The index to ignore in the loss computation. Defaults to 255.
            context_factor (int, optional): The factor by which to downsample the context. Defaults to 1.
            loss_context_factor (float, optional): The factor to multiply the context loss by. Defaults to 0.0.
            loss_co_pred_factor (float, optional): The factor to multiply the co-prediction loss by. Defaults to 0.0.
            loss_co_embed_factor (float, optional): The factor to multiply the co-embedding loss by. Defaults to 0.0.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.targets_to_net = (
            hasattr(self.net, "need_targets") and self.net.need_targets
        )

        self.save_freq = save_freq
        self.save_val_only = save_eval_only
        self.save_pred_as_logits = save_pred_as_logits

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(
            loss_ce_weight,
            reduction="mean",
            ignore_index=self.hparams.index_to_ignore,
        )

        # auxiliary loss
        self.auxloss = torch.nn.CrossEntropyLoss(
            loss_ce_weight,
            reduction="mean",
        )
        if isinstance(context_factor_m, Iterable) and len(context_factor_m) == 1:
            context_factor_m = context_factor_m[0]
        if isinstance(context_factor_px, Iterable) and len(context_factor_px) == 1:
            context_factor_px = context_factor_px[0]
        self.context_factor_m = context_factor_m
        self.context_factor_down = int(context_factor_m / context_factor_px)
        self.loss_context_factor = loss_context_factor
        self.loss_co_pred_factor = loss_co_pred_factor
        self.loss_co_embed_factor = loss_co_embed_factor

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=self.hparams.index_to_ignore,
        )
        self.eval_train_acc = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=self.hparams.index_to_ignore,
        )
        self.val_acc = Accuracy(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=self.hparams.index_to_ignore,
        )
        # self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_aux_loss = MeanMetric()
        self.eval_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        # self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # for per aoi evaluation
        self.last_writer = None
        self.val_JaccardIndex = torch.nn.ModuleDict(
            {
                "Total": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    average="none",
                    ignore_index=self.hparams.index_to_ignore,
                )
            }
        )
        self.train_JaccardIndex = torch.nn.ModuleDict(
            {
                "Total": JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    average="none",
                    ignore_index=self.hparams.index_to_ignore,
                )
            }
        )

        try:
            self.output_dir = (
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )
        except ValueError:
            self.output_dir = "./logs"

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def forward(
        self, x: torch.Tensor, metas: List = None, targets: torch.Tensor = None
    ) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if self.targets_to_net:
            # the model expects targets (probably for computing auxiliary losses)
            return self.net(x, metas=metas, targets=targets)
        return self.net(x, metas=metas)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        for metric in self.val_JaccardIndex.values():
            metric.reset()
        for metric in self.train_JaccardIndex.values():
            metric.reset()

        self.net.train()

    def model_step(self, batch, overlap=0.0):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        data, targets, metas = batch
        if targets.dtype == torch.uint8:
            targets = targets.to(torch.int64)

        output = self.forward(data, metas=metas, targets=targets)

        logics = output["out"]
        logics_contex = output.get("out_context", None)
        embed = output.get("embed", None)
        embed_context = output.get("embed_context", None)
        if isinstance(embed, list) and isinstance(embed_context, list):
            embed = embed[-1]
            embed_context = embed_context[-1]

        # extract relevent part of the image
        if overlap > 0.0:
            logics = extract_center(logics, 1 - overlap)
            embed = extract_center(embed, 1 - overlap)
            targets = extract_center(targets, 1 - overlap)

        if logics_contex is not None:
            keep = 1
            if self.context_factor_m > 1.0:
                keep *= 1 / self.context_factor_m
            if overlap > 0.0:
                keep *= 1 - overlap

            logics_contex = extract_center(logics_contex, keep)
            embed_context = extract_center(embed_context, keep)

        if "loss" in output:
            # if the model returns a loss, it is already computed
            loss = output["loss"]
        else:
            loss = self.criterion(logics, targets)

            def get_context_loss(logics, target, mask=None, sm_target=False):
                # downsample by the context factor using avg
                if sm_target:
                    target = torch.nn.functional.softmax(target, dim=1)
                target = torch.nn.functional.avg_pool2d(
                    target, self.context_factor_down, self.context_factor_down
                )
                # compute loss
                if mask is None:
                    loss_context = self.auxloss(logics, target)
                else:
                    mask = einops.repeat(mask, "B H W -> B C H W", C=logics.shape[1])
                    logics = torch.where(mask, logics, torch.zeros_like(logics))

                    target = torch.where(mask, target, torch.zeros_like(target))

                    loss_context = self.auxloss(logics, target)
                return loss_context

            if self.loss_context_factor > 0.0 and logics_contex is not None:
                # get low dimmension mask
                mask = einops.rearrange(
                    targets != self.hparams.index_to_ignore,
                    "B (H k) (W l) -> B H W k l",
                    k=self.context_factor_down,
                    l=self.context_factor_down,
                )
                # B H_context W_context
                mask = mask.all(dim=-1, keepdim=False).all(dim=-1, keepdim=False)

                # convert target to one hot
                targets_for_one_hot = torch.where(
                    targets == self.hparams.index_to_ignore,
                    torch.zeros_like(targets),
                    targets,
                )
                targets_one_hot = (
                    torch.nn.functional.one_hot(
                        targets_for_one_hot, self.hparams.num_classes
                    )
                    .permute(0, 3, 1, 2)
                    .float()
                )  # BCHW

                loss_context = get_context_loss(logics_contex, targets_one_hot, mask)
                loss += self.loss_context_factor * loss_context

            if self.loss_co_pred_factor > 0.0 and logics_contex is not None:
                # compute loss
                loss_co_pred = get_context_loss(logics_contex, logics, sm_target=True)
                loss += self.loss_co_pred_factor * loss_co_pred

            if self.loss_co_embed_factor > 0.0 and embed_context is not None:
                # compute loss
                loss_co_embed = get_context_loss(embed_context, embed, sm_target=True)
                loss += self.loss_co_embed_factor * loss_co_embed

        preds = torch.nn.functional.softmax(logics, dim=1)

        # if target are probability convert the to max class
        if len(targets.shape) == len(preds.shape):
            targets = targets.argmax(axis=1)

        if overlap >= 0.0:
            preds_full = torch.nn.functional.softmax(output["out"], dim=1)
            return loss, preds, targets, metas, preds_full
        return loss, preds, targets, metas, preds

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, _, _ = self.model_step(batch)
        preds_int = preds.argmax(axis=1)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds_int, targets)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def _common_step(self, batch, batch_idx, step_type, overlap=-1.0):
        """Common logic for validation, test, and predict steps.

        Args:
            batch (Tuple): A batch of data containing images, targets, and metadata.
            batch_idx (int): The index of the current batch.
            step_type (str): The type of step being performed ('validation', 'test', or 'predict').
            overlap (float, optional): The overlap to use when writing the predictions to a raster. Defaults to no overlap.
        Returns:
            Tuple or None: Returns (loss, preds_int, targets, aoi_name) if processing should continue,
                           or None if the step should be skipped.
        """
        # common per aoi
        epoch = self.trainer.current_epoch
        aoi_name = batch[2][0]["aoi_name"]

        # all the righting to raster is done on rank 0
        if step_type == "validation":
            # validation is saved only if it is the last epoch or if the save frequency is reached
            correct_split = True
            correct_epoch = (self.trainer.max_epochs - epoch) % self.save_freq == 0
        elif step_type == "test" or step_type == "predict":
            # test and predict are always saved
            correct_split = True
            correct_epoch = True
        rank_zero = self.trainer.global_rank == 0
        multi_gpu = self.trainer.world_size > 1

        loss, preds, targets, metas, preds_full = self.model_step(
            batch, overlap=overlap
        )
        preds_int = preds.argmax(axis=1)

        if correct_split and correct_epoch:
            if multi_gpu:
                # Gather from all workers
                preds_all = self.all_gather(preds_full.detach())
                metas_all = [_ for _ in range(self.trainer.world_size)]
                torch.distributed.all_gather_object(metas_all, metas)
                preds_all = preds_all.reshape(-1, *preds_all.shape[2:])
                metas_all = [item for sublist in metas_all for item in sublist]
            else:
                preds_all = preds_full
                metas_all = metas

            if rank_zero:
                preds_to_save = preds_all.detach().cpu().numpy()
                for i in range(len(preds_to_save)):
                    writer = metas_all[i]["window_writer"]
                    # keep last writer if it is the same aoi
                    if (
                        self.last_writer is not None
                        and hasattr(self.last_writer, "file_writer")
                        and writer.file_writer.is_equal(self.last_writer.file_writer)
                    ):
                        writer.file_writer = self.last_writer.file_writer
                    # if this is a new writer, close the previous one and open a new one
                    else:
                        if self.last_writer is not None:
                            self.last_writer.close()
                        self.last_writer = writer
                        writer.open(
                            (
                                f"{step_type[:5]}_{epoch}"
                                if step_type != "test" and step_type != "predict"
                                else None
                            ),
                            self.save_pred_as_logits,
                            nb_classes=self.hparams.num_classes,
                        )

                    writer.write(preds_to_save[i], overlap=overlap)
        return loss, preds_int, targets, aoi_name

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a single validation step.

        Args:
            batch (Tuple): A batch of data containing images, targets, and metadata.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader. Defaults to 0.

        Returns:
            None
        """
        result = self._common_step(
            batch, batch_idx, "validation", overlap=self.hparams.val_overlap
        )
        if result is not None:
            loss, preds_int, targets, aoi_name = result
            self._update_val_metrics(loss, preds_int, targets, aoi_name)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a single test step.

        Args:
            batch (Tuple): A batch of data containing images, targets, and metadata.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int, optional): The index of the dataloader. Defaults to 0.

        Returns:
            None
        """
        if self.analyse_flop:
            input = batch[0]
            if isinstance(input, list) or isinstance(input, tuple):
                single_image_input = [i[:1] for i in input]
            else:
                single_image_input = batch[0][:1]
            flop = FlopCountAnalysis(self.net, (single_image_input,))
            flop.set_op_handle(**get_new_ops())
            log.info("FLOP count analysis")
            log.info(f"{flop_count_table(flop)}")
            log.info(
                f"FLOP count: {flop.by_operator()}, unsupported: {flop.unsupported_ops()}"
            )
            self.analyse_flop = False

        result = self._common_step(
            batch, batch_idx, "test", overlap=self.hparams.test_overlap
        )
        if result is not None:
            loss, preds_int, targets, aoi_name = result
            self._update_test_metrics(loss, preds_int, targets, aoi_name)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a single predict step.

        Args:
            batch (Tuple): A batch of data containing images, targets, and metadata.
            batch_idx (int): The index of the current batch.

        Returns:
            None
        """
        result = self._common_step(
            batch, batch_idx, "predict", overlap=self.hparams.test_overlap
        )

    def _update_train_metrics(self, loss, preds_int, targets, aoi_name):
        """Update training metrics.

        Args:
            loss (torch.Tensor): The loss value for the current batch.
            preds_int (torch.Tensor): The integer predictions.
            targets (torch.Tensor): The target labels.
            aoi_name (str): The name of the area of interest.

        Returns:
            None
        """
        self.eval_train_loss(loss)
        self.eval_train_acc(preds_int, targets)
        # IoU
        if aoi_name not in self.train_JaccardIndex:
            self.train_JaccardIndex[aoi_name] = JaccardIndex(
                task="multiclass",
                num_classes=self.hparams.num_classes,
                average="none",
                ignore_index=self.hparams.index_to_ignore,
            ).to(self.device)
        self.train_JaccardIndex[aoi_name].update(preds_int, targets)
        self.train_JaccardIndex["Total"].update(preds_int, targets)

    def _update_val_metrics(self, loss, preds_int, targets, aoi_name):
        """Update validation metrics.

        Args:
            loss (torch.Tensor): The loss value for the current batch.
            preds_int (torch.Tensor): The integer predictions.
            targets (torch.Tensor): The target labels.
            aoi_name (str): The name of the area of interest.

        Returns:
            None
        """
        self.val_loss(loss)
        self.val_acc(preds_int, targets)
        # IoU
        if isinstance(aoi_name, str):
            if aoi_name not in self.val_JaccardIndex:
                self.val_JaccardIndex[aoi_name] = JaccardIndex(
                    task="multiclass",
                    num_classes=self.hparams.num_classes,
                    average="none",
                    ignore_index=self.hparams.index_to_ignore,
                ).to(self.device)
            self.val_JaccardIndex[aoi_name].update(preds_int, targets)
        self.val_JaccardIndex["Total"].update(preds_int, targets)

    def _update_test_metrics(self, loss, preds_int, targets, aoi_name):
        """Update test metrics.

        Args:
            loss (torch.Tensor): The loss value for the current batch.
            preds_int (torch.Tensor): The integer predictions.
            targets (torch.Tensor): The target labels.
            aoi_name (str): The name of the area of interest.

        Returns:
            None
        """

        # create metrics if not already created
        if not hasattr(self, "test_loss"):
            self.test_loss = MeanMetric().to(self.device)
            self.test_acc = Accuracy(
                task="multiclass",
                num_classes=self.hparams.num_classes,
                ignore_index=self.hparams.index_to_ignore,
            ).to(self.device)
            self.test_JaccardIndex = torch.nn.ModuleDict(
                {
                    "Total": JaccardIndex(
                        task="multiclass",
                        num_classes=self.hparams.num_classes,
                        average="none",
                        ignore_index=self.hparams.index_to_ignore,
                    )
                }
            ).to(self.device)

        if isnan(loss):
            return

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds_int, targets)
        # IoU
        if isinstance(aoi_name, str):
            # if aoi_name is a number we don't want to create a separate metric
            if aoi_name not in self.test_JaccardIndex:
                self.test_JaccardIndex[aoi_name] = JaccardIndex(
                    task="multiclass",
                    num_classes=self.hparams.num_classes,
                    average="none",
                    ignore_index=self.hparams.index_to_ignore,
                ).to(self.device)
            self.test_JaccardIndex[aoi_name].update(preds_int, targets)
        self.test_JaccardIndex["Total"].update(preds_int, targets)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends.

        This method computes and logs various metrics, including accuracy, loss, and IoU scores.

        Returns:
            None
        """
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss", self.val_loss.compute(), sync_dist=True, prog_bar=True)
        self.log("val/acc", self.val_acc.compute(), sync_dist=True, prog_bar=True)
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        # self.log(
        #     "eval_train/loss",
        #     self.eval_train_loss.compute(),
        #     sync_dist=True,
        #     prog_bar=False,
        # )
        # self.log(
        #     "eval_train/acc",
        #     self.eval_train_acc.compute(),
        #     sync_dist=True,
        #     prog_bar=True,
        # )

        # log IoU to logger
        # for aoi_name, metric in self.train_JaccardIndex.items():
        #     # self.log(f"Train IoU {aoi_name}", metric.compute(), sync_dist=True, prog_bar=False)
        # for aoi_name, metric in self.eval_JaccardIndex.items():
        #     # self.log(f"Eval IoU {aoi_name}", metric.compute(), sync_dist=True, prog_bar=False)

        # logging to info
        epoch = self.trainer.current_epoch
        log.info(f"{epoch=}")
        log.info(
            f"eval_train/loss={self.train_loss.compute()}, eval_train/acc={self.eval_train_acc.compute()} val/loss={self.val_loss.compute()}, val/acc={self.val_acc.compute()}"
        )

        # IoU
        # log.info('labels_names')
        metrics = pd.DataFrame()
        if hasattr(self.trainer.datamodule, "label_names"):
            label_names = self.trainer.datamodule.label_names
        else:
            label_names = [str(i) for i in range(self.hparams.num_classes)]
        # set data columns names to label names
        for aoi_name, metric in self.val_JaccardIndex.items():
            metric_l = metric.compute().detach().cpu().numpy()
            row = {label_names[i]: metric_l[i] for i in range(self.hparams.num_classes)}
            row.update({"mean": metric_l.mean()})
            df = pd.DataFrame(row, index=[f"Eval IoU {aoi_name}"])
            metrics = pd.concat([metrics, df])
            # Log to logger
            if aoi_name == "Total":
                self.log(
                    "val/IoU/mean",
                    metric_l.mean(),
                    sync_dist=True,
                    prog_bar=True,
                )
                for i in range(self.hparams.num_classes):
                    self.log(
                        f"val/IoU/{label_names[i]}",
                        metric_l[i],
                        sync_dist=True,
                        prog_bar=False,
                    )
        for aoi_name, metric in self.train_JaccardIndex.items():
            metric_l = metric.compute().detach().cpu().numpy()
            row = {label_names[i]: metric_l[i] for i in range(self.hparams.num_classes)}
            row.update({"mean": metric_l.mean()})
            df = pd.DataFrame(row, index=[f"Train IoU {aoi_name}"])
            metrics = pd.concat([metrics, df])
            # Log to logger
            if aoi_name == "Total":
                self.log(
                    "eval_train/IoU/mean",
                    metric_l.mean(),
                    sync_dist=True,
                    prog_bar=False,
                )
                for i in range(self.hparams.num_classes):
                    self.log(
                        f"eval_train/IoU/{label_names[i]}",
                        metric_l[i],
                        sync_dist=True,
                        prog_bar=False,
                    )
        print("")
        log.info("\n" + metrics.to_markdown())

        # reset metrics
        self.val_loss.reset()
        self.val_acc.reset()
        self.eval_train_loss.reset()
        self.eval_train_acc.reset()
        for metric in self.val_JaccardIndex.values():
            metric.reset()
        for metric in self.train_JaccardIndex.values():
            metric.reset()

        # Close writer
        if self.last_writer is not None:
            self.last_writer.close()
            self.last_writer = None

    def on_test_epoch_start(self):
        self.analyse_flop = FLOP_ANALYSIS

        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends.

        This method computes and logs test metrics, including loss, accuracy, and IoU scores.

        Returns:
            None
        """
        # log metrics
        self.log("test/loss", self.test_loss.compute(), sync_dist=True)
        self.log("test/acc", self.test_acc.compute(), sync_dist=True)
        # log metrics to info
        log.info(
            f"test/loss={self.test_loss.compute()}, test/acc={self.test_acc.compute()}"
        )
        # log IoU to logger
        # log.info('labels_names')
        metrics = pd.DataFrame()
        if hasattr(self.trainer.datamodule, "label_names"):
            label_names = self.trainer.datamodule.label_names
        else:
            label_names = [str(i) for i in range(self.hparams.num_classes)]
        # set data columns names to label names
        for aoi_name, metric in self.test_JaccardIndex.items():
            metric_l = metric.compute().detach().cpu().numpy()
            row = {label_names[i]: metric_l[i] for i in range(self.hparams.num_classes)}
            row.update({"mean": metric_l.mean()})
            df = pd.DataFrame(row, index=[f"Eval IoU {aoi_name}"])
            metrics = pd.concat([metrics, df])
            # Log to logger
            if aoi_name == "Total":
                self.log(
                    "test/IoU/mean",
                    metric_l.mean(),
                    sync_dist=True,
                    prog_bar=False,
                )
                for i in range(self.hparams.num_classes):
                    self.log(
                        f"test/IoU/{label_names[i]}",
                        metric_l[i],
                        sync_dist=True,
                        prog_bar=False,
                    )
        log.info("\n" + metrics.to_markdown())
        # Close writer
        if self.last_writer is not None:
            self.last_writer.close()
            self.last_writer = None

    def on_predict_epoch_end(self):
        # Close writer
        if self.last_writer is not None:
            self.last_writer.close()
            self.last_writer = None
        return super().on_predict_epoch_end()

    def on_before_optimizer_step(self, _) -> None:
        has_nan = []
        is_inf = []
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    has_nan.append(name)
                if torch.isinf(param.grad).any():
                    is_inf.append(name)
                torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        if len(has_nan) > 0:
            print(f"Found NaN in {has_nan}")
        if len(is_inf) > 0:
            print(f"Found Inf in {is_inf}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        if self.hparams.head_lr is not None:
            # if head_lr is set, we use a different learning rate for the head
            params = [
                {
                    "params": self.net.model.parameters(),
                    "lr": self.hparams.optimizer.keywords["lr"],
                },
                {
                    "params": self.net.seg_head.parameters(),
                    "lr": self.hparams.head_lr,
                },
            ]
        else:
            # otherwise we use the same learning rate for all parameters
            params = self.trainer.model.parameters()
        optimizer = self.hparams.optimizer(params=params)
        list_schedulers = []
        if self.hparams.scheduler is not None:
            assert (
                self.hparams.metric_monitored is not None
            ), "A metric must be monitored to use a scheduler."
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            list_schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": self.hparams.metric_monitored,
                    "interval": "epoch",
                    "frequency": 1,
                }
            )
        if self.hparams.warmup_scheduler is not None:
            if hasattr(self.trainer.datamodule, "nb_train_samples"):
                total_step = (
                    self.trainer.datamodule.nb_train_samples
                    // self.trainer.datamodule.hparams.batch_size
                )
                total_step = total_step * self.trainer.max_epochs
            else:
                total_step = self.trainer.estimated_stepping_batches
            warmup = self.hparams.warmup_scheduler(
                optimizer=optimizer,
                total_step=total_step,
            )
            list_schedulers.append(
                {
                    "scheduler": warmup,
                    "interval": "step",
                    "frequency": 1,
                }
            )
        print(f"{list_schedulers=}")
        return [optimizer], list_schedulers


if __name__ == "__main__":
    _ = SegmentationModule(None, None, None, None)
