"""
- PAPER
https://arxiv.org/pdf/2008.07357.pdf
- CODE
https://github.com/kechua/DART20/blob/master/damri/model/unet.py
"""
from __future__ import annotations

from typing import Dict
from datetime import datetime
from omegaconf import OmegaConf
import wandb
import torch
from torch import (
    log, 
    tensor
)
from torch.special import entr
from torch.nn.functional import one_hot
from monai.networks.nets import UNet
from monai.networks.layers.factories import Norm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping



def get_unet_module_trainer(
    data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    trainer_cfg: OmegaConf
):
    # infered variables
    patience = model_cfg.patience * 2
    now = datetime.now()
    dropout_str = str(model_cfg.dropout).replace('.', '-')
    filename = f'{data_cfg.dataset}_{data_cfg.domain.lower()}_dropout-{dropout_str}_{now.strftime("%Y-%m-%d-%H-%M")}'

    # init logger
    if trainer_cfg.logging:
        wandb.finish()
        logger = WandbLogger(
            project="MIDL25", 
            log_model=True, 
            name=filename
        )
    else:
        logger = None

    # return trainer
    return L.Trainer(
        limit_train_batches=trainer_cfg.limit_train_batches,
        max_epochs=trainer_cfg.max_epochs,
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor=trainer_cfg.early_stopping.monitor, 
                mode=trainer_cfg.early_stopping.mode, 
                patience=patience
            ),
            ModelCheckpoint(
                dirpath=trainer_cfg.model_checkpoint.dirpath,
                filename=filename,
                save_top_k=trainer_cfg.model_checkpoint.save_top_k, 
                monitor=trainer_cfg.model_checkpoint.monitor,
            )
        ],
        precision='16-mixed',
        gradient_clip_val=0.5,
        devices=[0],
        limit_test_batches=50
    )



def get_unet_module(
    cfg: OmegaConf,
    metadata: Dict,
    load_from_checkpoint: bool = False
):
    unet = UNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        channels=[cfg.n_filters_init * 2 ** i for i in range(cfg.depth)],
        strides=[2] * (cfg.depth - 1),
        num_res_units=4,
        norm=Norm.INSTANCE,
        dropout=cfg.dropout,
    )
    if load_from_checkpoint:
        return LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            model=unet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            map_location='cuda:0'
        )
    else:
        return LightningSegmentationModel(
            model=unet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            metadata=metadata
        )


class LightningSegmentationModel(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        patience: int = 5,
        binary_target: bool = False,
        metadata: Dict[str, OmegaConf] = None
    ):
        super().__init__()
        # this would save the model as hyperparameter, not desired!
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.patience = patience
        self.metadata = metadata
        self.loss = DiceCELoss(
            softmax=False if binary_target else True,
            sigmoid=True if binary_target else False,
            to_onehot_y=False if binary_target else True,
        )
        self.dsc = DiceMetric(include_background=False, reduction="none")
        self.hausdorff = HausdorffDistanceMetric(include_background=False, reduction="none", percentile=95)
        if self.metadata['unet']['out_channels'] == 1:
            num_classes = 2
        else:
            num_classes = self.metadata['unet']['out_channels']
        self.sdsc = SurfaceDiceMetric(include_background=False, class_thresholds=(num_classes-1)*[3], reduction="none")

    def forward(self, inputs):        
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0.5) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean()

        self.log_dict({
            'train_loss': loss,
            'train_dsc': dsc,
        })
        return {
            'loss': loss
        }
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean().nan_to_num(0)

        self.log_dict({
            'val_loss': loss,
            'val_dsc': dsc,
        })
        return {
            'loss': loss,
        }
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        loss = self.loss(outputs, target)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1
        outputs = torch.nn.functional.one_hot(outputs, num_classes=num_classes).moveaxis(-1, 1)
        dsc = self.dsc(outputs, target).nanmean().nan_to_num(0)


        self.log_dict({
            'test_loss': loss,
            'test_dsc': dsc,
        })
        return {
            'loss': loss,
            'dsc': dsc
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            probs = torch.softmax(outputs, 1).detach()
            outputs = outputs.argmax(dim=1, keepdim=True).detach()
        else:
            probs = outputs.sigmoid().detach()
            outputs = (outputs > 0).long().detach()
        predicted_segmentation = one_hot(outputs.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
        target_segmentation = one_hot(target.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
        dice = self.dsc(predicted_segmentation, target_segmentation).nanmean(-1).nan_to_num(0).cpu().detach()
        hausdorff = self.hausdorff(predicted_segmentation, target_segmentation).nanmean(-1).nan_to_num(0).cpu().detach()
        surface_dice = self.sdsc(predicted_segmentation, target_segmentation).nanmean(-1).nan_to_num(0).cpu().detach()
        entropy = 1 - (entr(probs).sum(1).mean((-1, -2)) / log(tensor(num_classes)))

        metrics = {
            'dice': dice,
            'hausdorff': hausdorff,
            'surface_dice': surface_dice,
            'entropy': entropy
        }
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }