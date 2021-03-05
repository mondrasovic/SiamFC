#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import multiprocessing
import os
import sys
from typing import cast, Optional, Sequence, Tuple

import click
import numpy as np
import torch
import tqdm
from got10k.datasets import GOT10k, OTB, VOT, ImageNetVID
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common import DatasetType
from sot.cfg import TrackerConfig
from sot.dataset import SiamesePairwiseDataset
from sot.losses import WeightedBCELoss
from sot.tracker import TrackerSiamFC
from sot.utils import create_ground_truth_mask_and_weight


class SiamFCTrainer:
    def __init__(
            self, cfg: TrackerConfig, train_dataset_type: DatasetType,
            train_dataset_dir_path: str,
            val_dataset_type: Optional[DatasetType],
            val_dataset_dir_path: Optional[str],
            checkpoint_dir_path: Optional[str] = None,
            log_dir_path: Optional[str] = None) -> None:
        self.cfg: TrackerConfig = cfg
        
        self.train_dataset_type: DatasetType = train_dataset_type
        self.train_dataset_dir_path: str = train_dataset_dir_path
        
        self.val_dataset_type: Optional[DatasetType] = val_dataset_type
        self.val_dataset_dir_path: Optional[str] = val_dataset_dir_path
        
        self.checkpoint_dir_path: Optional[str] = checkpoint_dir_path
        self.log_dir_path: Optional[str] = log_dir_path
        
        self.device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tracker: TrackerSiamFC = TrackerSiamFC(cfg, self.device)
        
        response_map_size = (self.cfg.response_size, self.cfg.response_size)
        mask_mat, weight_mat = create_ground_truth_mask_and_weight(
            response_map_size, self.cfg.positive_class_radius,
            self.cfg.total_stride, self.cfg.batch_size)
        
        self.mask_mat = torch.from_numpy(mask_mat).float().to(self.device)
        weight_mat = torch.from_numpy(weight_mat).float().to(self.device)
        
        self.criterion = WeightedBCELoss(weight_mat).to(self.device)
        
        self.optimizer = optim.SGD(
            self.tracker.model.parameters(), lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        
        self.lr_scheduler = self._create_exponential_lr_scheduler(
            self.optimizer, self.cfg.initial_lr, self.cfg.ultimate_lr,
            self.cfg.n_epochs)
        
        self.epoch: int = 0
    
    def run(self, checkpoint_file_path: Optional[str] = None) -> None:
        if self.log_dir_path is None:
            writer = None
        else:
            writer = SummaryWriter(self.log_dir_path)
        
        n_workers = max(
            1, min(self.cfg.n_workers,
                   multiprocessing.cpu_count() - self.cfg.free_cpus))
        pin_memory = torch.cuda.is_available()
        train_loader, val_loader = self._init_data_loaders(
            n_workers, pin_memory)
        
        if checkpoint_file_path is None:
            self.epoch = 1
        else:
            self._load_checkpoint(checkpoint_file_path)
        
        try:
            while self.epoch <= self.cfg.n_epochs:
                train_loss = self._run_epoch(train_loader)
                
                if writer is not None:
                    writer.add_scalar('Loss/train', train_loss, self.epoch)
                
                self.lr_scheduler.step()
                
                if self.checkpoint_dir_path is not None:
                    self._save_checkpoint(
                        train_loss, self._build_checkpoint_file_path_and_init())
                
                if self._should_validate(val_loader):
                    eval_loss = self._run_epoch(val_loader, backward=False)

                    if writer is not None:
                        writer.add_scalar(
                            'Loss/val', eval_loss, self.epoch)
    
                self.epoch += 1
                
                print("-" * 80)
        except KeyboardInterrupt:
            print("interrupting...")
        finally:
            if writer is not None:
                writer.close()
    
    def _should_validate(self, val_loader: DataLoader) -> bool:
        return val_loader and \
               (self.cfg.n_epochs_val > 0) and \
               (self.epoch % self.cfg.n_epochs_val == 0)
    
    def _run_epoch(
            self, data_loader: DataLoader, *, backward: bool = True) -> float:
        self.tracker.model.train(backward)
        
        losses_sum = 0.0
        n_batches = len(data_loader)
        
        mode_text = "train" if backward else "valid"
        epoch_text = f"[{mode_text}] epoch: {self.epoch:3d}/{self.cfg.n_epochs}"
        
        tqdm_pbar = tqdm.tqdm(total=n_batches, file=sys.stdout)
        with torch.set_grad_enabled(backward), tqdm_pbar as pbar:
            for batch, (exemplar, instance) in enumerate(data_loader, start=1):
                exemplar = exemplar.to(self.device)
                instance = instance.to(self.device)
    
                pred_response_maps = self.tracker.model(exemplar, instance)
                loss = self.criterion(pred_response_maps, self.mask_mat)
    
                if backward:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
    
                curr_loss = loss.item()
                losses_sum += curr_loss
                curr_batch_loss = losses_sum / batch
    
                loss_text = f"loss: {curr_loss:.5f} ({curr_batch_loss:.5f})"
                pbar.set_description(f"{epoch_text} | {loss_text}")
                pbar.update()
        
        batch_loss = losses_sum / n_batches
        
        return batch_loss
    
    def _init_data_loaders(
            self, n_workers: int,
            pin_memory: bool) -> Tuple[DataLoader, Optional[DataLoader]]:
        def create_dataloader(dataset):
            return DataLoader(
                dataset, batch_size=self.cfg.batch_size, shuffle=True,
                num_workers=n_workers, pin_memory=pin_memory, drop_last=True)
        
        train_kwargs = {}
        
        if self.train_dataset_type == DatasetType.GOT10k:
            train_kwargs['subset'] = 'train'
        if self.val_dataset_type:
            val_kwargs = {}
            
            if self.val_dataset_type == DatasetType.GOT10k:
                val_kwargs['subset'] = 'val'
            
            val_dataset = self._init_pairwise_dataset(
                self.val_dataset_type, self.val_dataset_dir_path, **val_kwargs)
            val_loader = create_dataloader(val_dataset)
        else:
            val_loader = None
        
        train_dataset = self._init_pairwise_dataset(
            self.train_dataset_type, self.train_dataset_dir_path,
            **train_kwargs)
        
        train_loader = create_dataloader(train_dataset)
        
        return train_loader, val_loader
    
    @staticmethod
    def _init_pairwise_dataset(
            dataset_type: DatasetType, dir_path: str,
            **kwargs) -> SiamesePairwiseDataset:
        if dataset_type == DatasetType.GOT10k:
            data_seq = GOT10k(root_dir=dir_path, **kwargs)
        elif dataset_type == DatasetType.OTB13:
            data_seq = OTB(root_dir=dir_path, version=2013, **kwargs)
        elif dataset_type == DatasetType.OTB15:
            data_seq = OTB(root_dir=dir_path, version=2015, **kwargs)
        elif dataset_type == DatasetType.VOT15:
            data_seq = VOT(dir_path, version=2015, **kwargs)
        elif dataset_type == DatasetType.ILSVRC15:
            data_seq = ImageNetVID(root_dir=dir_path, subset='train', **kwargs)
        else:
            raise ValueError(f"unsupported dataset type: {dataset_type}")
        
        pairwise_dataset = SiamesePairwiseDataset(
            cast(Sequence, data_seq), TrackerConfig())
        
        return pairwise_dataset
    
    @staticmethod
    def _create_exponential_lr_scheduler(
            optimizer, initial_lr: float, ultimate_lr: float,
            n_epochs: int) -> optim.lr_scheduler.ExponentialLR:
        assert n_epochs > 0
        
        # Learning rate is geometrically annealed at each epoch. Starting from
        # A and terminating at B, then the known gamma factor x for n epochs
        # is computed as
        #         A * x ^ n = B,
        #                 x = (B / A)^(1 / n).
        gamma = np.power(ultimate_lr / initial_lr, 1.0 / n_epochs)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        
        return lr_scheduler
    
    def _build_checkpoint_file_path_and_init(self) -> str:
        os.makedirs(self.checkpoint_dir_path, exist_ok=True)
        file_name = f"checkpoint_{self.epoch:03d}.pth"
        return os.path.join(self.checkpoint_dir_path, file_name)
    
    def _save_checkpoint(self, loss: float, checkpoint_file_path: str) -> None:
        checkpoint = {
            'model': self.tracker.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'loss': loss,
        }
        torch.save(checkpoint, checkpoint_file_path)
    
    def _load_checkpoint(self, checkpoint_file_path: str) -> None:
        checkpoint = torch.load(checkpoint_file_path)
        self.tracker.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1


@click.command()
@click.argument("train_dataset_name")
@click.argument("train_dataset_dir_path")
@click.option(
    "-v", "--val-dataset-name",
    help="validation dataset name")
@click.option(
    "-p", "--val-dataset-dir-path",
    help="validation dataset directory path")
@click.option(
    "-l", "--log-dir-path",
    help="directory path to save the tensorboard logs")
@click.option(
    "-d", "--checkpoints-dir-path", help="directory path to save checkpoints")
@click.option(
    "-c", "--checkpoint-file-path",
    help="checkpoint file path to start the training from")
def main(
        train_dataset_name: str, train_dataset_dir_path: str,
        val_dataset_name: Optional[str], val_dataset_dir_path: Optional[str],
        log_dir_path: Optional[str], checkpoints_dir_path: Optional[str],
        checkpoint_file_path: Optional[str]) -> int:
    """
    Starts a SiamFC training with the specific DATASET_NAME
    (GOT10k | OTB13 | OTB15 | VOT15) located in the DATASET_DIR_PATH.
    """
    
    train_dataset_type = DatasetType.decode_dataset_type(train_dataset_name)
    val_dataset_type = None
    if val_dataset_name:
        val_dataset_type = DatasetType.decode_dataset_type(val_dataset_name)
    
    cfg = TrackerConfig()
    trainer = SiamFCTrainer(
        cfg, train_dataset_type, train_dataset_dir_path, val_dataset_type,
        val_dataset_dir_path, checkpoints_dir_path, log_dir_path)
    trainer.run(checkpoint_file_path)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
