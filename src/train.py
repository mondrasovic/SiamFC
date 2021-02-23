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
from got10k.datasets import GOT10k, OTB, VOT
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from common import DatasetType
from sot.cfg import TrackerConfig
from sot.dataset import SiamesePairwiseDataset
from sot.losses import WeightedBCELoss
from sot.tracker import TrackerSiamFC
from sot.utils import create_ground_truth_mask_and_weight


class SiamFCTrainer:
    def __init__(
            self, cfg: TrackerConfig, dataset_dir_path: str,
            dataset_type: DatasetType,
            checkpoint_dir_path: Optional[str] = None,
            log_dir_path: Optional[str] = None) -> None:
        self.cfg: TrackerConfig = cfg
        self.dataset_dir_path: str = dataset_dir_path
        self.dataset_type: DatasetType = dataset_type
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
        
        # TODO Use weight decay once again!
        
        # self.optimizer = optim.SGD(
        #     self.tracker.model.parameters(), lr=self.cfg.initial_lr,
        #     weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        self.optimizer = optim.SGD(
            self.tracker.model.parameters(), lr=self.cfg.initial_lr,
            momentum=self.cfg.momentum)
        
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
                
                if self.checkpoint_dir_path is not None:
                    self._save_checkpoint(
                        train_loss, self._build_checkpoint_file_path_and_init())
                
                if self.cfg.n_epochs_eval > 0:
                    if (self.epoch % self.cfg.n_epochs_eval) == 0:
                        eval_loss = self._run_epoch(val_loader, backward=False)
    
                        if writer is not None:
                            writer.add_scalar(
                                'Loss/eval', eval_loss, self.epoch)
    
                self.lr_scheduler.step()
                self.epoch += 1
                print("-" * 80)
        except KeyboardInterrupt:
            print("interrupting...")
        finally:
            if writer is not None:
                writer.close()
    
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
            pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
        pairwise_dataset = self._init_pairwise_dataset()
        
        n_total_samples = len(pairwise_dataset)
        n_valid_samples = int(round(
            n_total_samples * self.cfg.validation_split))
        n_train_samples = n_total_samples - n_valid_samples
        
        train_dataset, val_dataset = random_split(
            pairwise_dataset, (n_train_samples, n_valid_samples),
            generator=torch.Generator())
        
        def create_dataloader(dataset):
            return DataLoader(
                dataset, batch_size=self.cfg.batch_size, shuffle=True,
                num_workers=n_workers, pin_memory=pin_memory, drop_last=True)
        
        train_loader = create_dataloader(train_dataset)
        val_loader = create_dataloader(val_dataset)
        
        return train_loader, val_loader
    
    def _init_pairwise_dataset(self) -> SiamesePairwiseDataset:
        if self.dataset_type == DatasetType.GOT10K:
            data_seq = GOT10k(root_dir=self.dataset_dir_path, subset='train')
        elif self.dataset_type == DatasetType.OTB13:
            data_seq = OTB(root_dir=self.dataset_dir_path, version=2013)
        elif self.dataset_type == DatasetType.OTB15:
            data_seq = OTB(root_dir=self.dataset_dir_path, version=2015)
        elif self.dataset_type == DatasetType.VOT15:
            data_seq = VOT(root_dir=self.dataset_dir_path, version=2015)
        else:
            raise ValueError(f"unsupported dataset type: {self.dataset_type}")
        
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
@click.argument("dataset_name")
@click.argument("dataset_dir_path")
@click.option(
    "-l", "--log-dir-path",
    help="directory path to save the tensorboard logs")
@click.option(
    "-d", "--checkpoints-dir-path", help="directory path to save checkpoints")
@click.option(
    "-c", "--checkpoint-file-path",
    help="checkpoint file path to start the training from")
def main(
        dataset_name: str, dataset_dir_path: str, log_dir_path: Optional[str],
        checkpoints_dir_path: Optional[str],
        checkpoint_file_path: Optional[str]) -> int:
    """
    Starts a SiamFC training with the specific DATASET_NAME
    (GOT10k | OTB13 | OTB15 | VOT15) located in the DATASET_DIR_PATH.
    """
    # np.random.seed(731995)
    
    dataset_type = DatasetType.decode_dataset_type(dataset_name)
    cfg = TrackerConfig()
    trainer = SiamFCTrainer(
        cfg, dataset_dir_path, dataset_type, checkpoints_dir_path, log_dir_path)
    trainer.run(checkpoint_file_path)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
