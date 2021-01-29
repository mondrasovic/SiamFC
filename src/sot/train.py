import click
import pickle
import pathlib
import multiprocessing

import torch

import numpy as np

from torch import optim
from torch.utils.data import DataLoader

from sot.cfg import TrackerConfig
from sot.tracker import TrackerSiamFC
from sot.losses import WeightedBCELoss
from sot.dataset import (
    build_dataset_and_init, ImageNetVideoDataset, SiamesePairwiseDataset)
from sot.utils import create_ground_truth_mask_and_weight


class SiamFCTrainer:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg: TrackerConfig = cfg
        
        self.device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tracker: TrackerSiamFC = TrackerSiamFC(cfg, self.device)

        response_map_size = (self.cfg.response_size, self.cfg.response_size)
        mask_mat, weight_mat = create_ground_truth_mask_and_weight(
            response_map_size, self.cfg.positive_class_radius,
            self.cfg.total_stride, self.cfg.batch_size)
        print(mask_mat)
        self.mask_mat = torch.from_numpy(mask_mat).float().to(self.device)
        weight_mat = torch.from_numpy(weight_mat).float()
        
        self.optimizer = optim.SGD(
            self.tracker.model.parameters(), lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        self.criterion = WeightedBCELoss(weight_mat).to(self.device)
        
        self.lr_scheduler = self.create_exponential_lr_scheduler(
            self.optimizer, self.cfg.initial_lr, self.cfg.ultimate_lr,
            self.cfg.n_epochs)
    
    def run(self) -> None:
        pairwise_dataset = self.init_pairwise_dataset()
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            pairwise_dataset, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        for epoch in range(1, self.cfg.n_epochs + 1):
            print(f"epoch: {epoch}/{self.cfg.n_epochs}")
            
            for exemplar, instance in train_loader:
                exemplar = exemplar.to(self.device)
                instance = instance.to(self.device)
                
                self.optimizer.zero_grad()
                pred_response_maps = self.tracker.model(exemplar, instance)
                
                loss = self.criterion(pred_response_maps, self.mask_mat)
                loss.backward()
                self.optimizer.step()
                
                print(f"loss: {loss.item():.6f}")
            
            self.lr_scheduler.step()
    
    def init_pairwise_dataset(self) -> SiamesePairwiseDataset:
        cache_file = pathlib.Path('../../dataset_train_dump.bin')
        if cache_file.exists():
            with open(str(cache_file), 'rb') as in_file:
                data_seq = pickle.load(in_file)
        else:
            dataset_path = '../../../../datasets/ILSVRC2015_VID_small'
            data_seq = build_dataset_and_init(
                ImageNetVideoDataset, dataset_path, 'train')
            with open(str(cache_file), 'wb') as out_file:
                pickle.dump(data_seq, out_file,
                            protocol=pickle.HIGHEST_PROTOCOL)
    
        pairwise_dataset = SiamesePairwiseDataset(data_seq, TrackerConfig())
        
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


@click.command()
def main() -> int:
    cfg = TrackerConfig()
    trainer = SiamFCTrainer(cfg)
    trainer.run()
    
    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())
