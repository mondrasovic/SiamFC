import os
import sys
import enum
import click
import multiprocessing
from typing import cast, Sequence, Optional

import numpy as np
import torch
import tqdm
from got10k.datasets import GOT10k, OTB
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sot.cfg import TrackerConfig
from sot.dataset import SiamesePairwiseDataset
from sot.losses import WeightedBCELoss
from sot.tracker import TrackerSiamFC
from sot.utils import create_ground_truth_mask_and_weight


@enum.unique
class DatasetType(enum.Enum):
    GOT10K = 'GOT10k'
    OTB13 = 'OTB13'
    OTB15 = 'OTB15'


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
        weight_mat = torch.from_numpy(weight_mat).float()
        
        self.criterion = WeightedBCELoss(weight_mat).to(self.device)
        
        self.optimizer = optim.SGD(
            self.tracker.model.parameters(), lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        
        self.lr_scheduler = self.create_exponential_lr_scheduler(
            self.optimizer, self.cfg.initial_lr, self.cfg.ultimate_lr,
            self.cfg.n_epochs)
        
        self.epoch: int = 0
    
    def run(self, checkpoint_file_path: Optional[str] = None) -> None:
        if self.log_dir_path is None:
            writer = None
        else:
            writer = SummaryWriter(self.log_dir_path)
        
        pairwise_dataset = self.init_pairwise_dataset()
        n_workers = max(
            1, min(self.cfg.n_workers, multiprocessing.cpu_count() - 1))
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            pairwise_dataset, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=pin_memory, drop_last=True)
        
        if checkpoint_file_path is None:
            self.epoch = 1
        else:
            self.load_checkpoint(checkpoint_file_path)
        
        self.tracker.model.train()
        
        while self.epoch <= self.cfg.n_epochs:
            loss = self._run_epoch(train_loader)
            
            if writer is not None:
                writer.add_scalar('Loss/train', loss, self.epoch)
            
            if self.checkpoint_dir_path is not None:
                self.save_checkpoint(
                    loss, self.build_checkpoint_file_path_and_init())
            
            self.epoch += 1
        
        if writer is not None:
            writer.close()
    
    def _run_epoch(self, train_loader: DataLoader) -> float:
        losses_sum = 0.0
        n_batches = len(train_loader)
        
        epoch_text = f"epoch: {self.epoch}/{self.cfg.n_epochs}"
        
        with tqdm.tqdm(total=n_batches, file=sys.stdout) as pbar:
            for batch, (exemplar, instance) in enumerate(train_loader, start=1):
                exemplar = exemplar.to(self.device)
                instance = instance.to(self.device)
                
                self.optimizer.zero_grad()
                pred_response_maps = self.tracker.model(exemplar, instance)
                
                loss = self.criterion(pred_response_maps, self.mask_mat)
                loss.backward()
                self.optimizer.step()
                
                curr_loss = loss.item()
                losses_sum += curr_loss
                curr_batch_loss = losses_sum / batch
                
                loss_text = f"loss: {curr_loss:.5f} [{curr_batch_loss:.5f}]"
                pbar.set_description(f"{epoch_text} | {loss_text}")
                pbar.update()
        
        self.lr_scheduler.step()
        batch_loss = losses_sum / n_batches
        
        return batch_loss
    
    def init_pairwise_dataset(self) -> SiamesePairwiseDataset:
        if self.dataset_type == DatasetType.GOT10K:
            data_seq = GOT10k(root_dir=self.dataset_dir_path, subset='train')
        elif self.dataset_type == DatasetType.OTB13:
            data_seq = OTB(root_dir=self.dataset_dir_path, version=2013)
        elif self.dataset_type == DatasetType.OTB15:
            data_seq = OTB(root_dir=self.dataset_dir_path, version=2015)
        else:
            raise ValueError(f"unsupported dataset type: {self.dataset_type}")
        
        pairwise_dataset = SiamesePairwiseDataset(
            cast(Sequence, data_seq), TrackerConfig())
        
        return pairwise_dataset
    
    @staticmethod
    def create_exponential_lr_scheduler(
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
    
    def build_checkpoint_file_path_and_init(self) -> str:
        os.makedirs(self.checkpoint_dir_path, exist_ok=True)
        file_name = f"checkpoint_{self.epoch:03d}.pth"
        return os.path.join(self.checkpoint_dir_path, file_name)
    
    def save_checkpoint(self, loss: float, checkpoint_file_path: str) -> None:
        checkpoint = {
            'model': self.tracker.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'loss': loss,
        }
        torch.save(checkpoint, checkpoint_file_path)

    def load_checkpoint(self, checkpoint_file_path: str) -> None:
        checkpoint = torch.load(checkpoint_file_path)
        self.tracker.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1


def decode_dataset_type(dataset_name: str) -> DatasetType:
    for dataset_item in DatasetType:
        if dataset_item.value == dataset_name:
            return dataset_item
    raise ValueError("unrecognized dataset type")


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
    (GOT10k | OTB13 | OTB15) located in the DATASET_DIR_PATH.
    """
    np.random.seed(731995)
    
    dataset_type = decode_dataset_type(dataset_name)
    cfg = TrackerConfig()
    trainer = SiamFCTrainer(
        cfg, dataset_dir_path, dataset_type, checkpoints_dir_path, log_dir_path)
    trainer.run(checkpoint_file_path)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
