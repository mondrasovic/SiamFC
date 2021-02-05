import sys
import multiprocessing
import pathlib
import pickle

import click
import numpy as np
import cv2 as cv
import torch
import tqdm

from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from sot.cfg import TrackerConfig
from sot.dataset import (
    build_dataset_and_init, OTBDataset,
    SiamesePairwiseDataset
)
from sot.losses import WeightedBCELoss
from sot.tracker import TrackerSiamFC
from sot.utils import create_ground_truth_mask_and_weight


LOG_DIR = "../../logs"
MODEL_DIR = "../../model.pth"
# DATASET_DIR = "../../../../datasets/simple_shape_dataset"
DATASET_DIR = "../../../../datasets/OTB_2013"
DATASET_CACHE_FILE = "../../dataset_train_dump.bin"


def cv_show_tensor_as_img(img: torch.Tensor, win_name: str):
    img = img.cpu().detach().squeeze(0).numpy()
    img = np.transpose(img, axes=(1, 2, 0))
    cv.imshow(win_name, img)


def cv_wait_key_and_destroy_all(delay: int = 0, quit_key: str = 'q') -> bool:
    key = cv.waitKey(delay) & 0xff
    cv.destroyAllWindows()
    return key == ord(quit_key)


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
        self.mask_mat = torch.from_numpy(mask_mat).float().to(self.device)
        weight_mat = torch.from_numpy(weight_mat).float()
        
        self.optimizer = optim.SGD(
            self.tracker.model.parameters(), lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        # self.optimizer = optim.SGD(
        #     self.tracker.model.parameters(), lr=self.cfg.initial_lr)
        self.criterion = WeightedBCELoss(weight_mat).to(self.device)
        
        self.lr_scheduler = self.create_exponential_lr_scheduler(
            self.optimizer, self.cfg.initial_lr, self.cfg.ultimate_lr,
            self.cfg.n_epochs)
    
    def run(self) -> None:
        writer = SummaryWriter(LOG_DIR)
        
        pairwise_dataset = self.init_pairwise_dataset()
        num_workers = max(1, min(3, multiprocessing.cpu_count() - 1))
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            pairwise_dataset, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        
        self.tracker.model.train()
        
        for epoch in range(1, self.cfg.n_epochs + 1):
            loss = self._run_epoch(epoch, train_loader)
            writer.add_scalar("Loss/train", loss, epoch)
            torch.save(self.tracker.model.state_dict(), MODEL_DIR)
        
        writer.close()
    
    def _run_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        losses_sum = 0.0
        n_batches = len(train_loader)
        
        epoch_descr = f"epoch: {epoch}/{self.cfg.n_epochs}"
        
        with tqdm.tqdm(total=n_batches, file=sys.stdout) as pbar:
            for batch, (exemplar, instance) in enumerate(train_loader, start=1):
                # cv_show_tensor_as_img(exemplar[0], "exemplar image")
                # cv_show_tensor_as_img(instance[0], "instance image")
                # if cv_wait_key_and_destroy_all():
                #     return  # =====>
            
                exemplar = exemplar.to(self.device)
                instance = instance.to(self.device)
            
                self.optimizer.zero_grad()
                pred_response_maps = self.tracker.model(exemplar, instance)
            
                loss = self.criterion(pred_response_maps, self.mask_mat)
                loss.backward()
                # for param in self.tracker.model.parameters():
                #     print("param", np.linalg.norm(param.grad.cpu().detach().numpy()))
                self.optimizer.step()
                
                curr_loss = loss.item()
                losses_sum += curr_loss
                curr_batch_loss = losses_sum / batch
                
                loss_descr = f"loss: {curr_loss:.5f} [{curr_batch_loss:.4f}]"
                pbar.set_description(f"{epoch_descr} | {loss_descr}")
                pbar.update()
        
        self.lr_scheduler.step()
        batch_loss = losses_sum / n_batches
        
        return batch_loss
    
    def init_pairwise_dataset(self) -> SiamesePairwiseDataset:
        cache_file = pathlib.Path(DATASET_CACHE_FILE)
        if cache_file.exists():
            with open(str(cache_file), 'rb') as in_file:
                data_seq = pickle.load(in_file)
        else:
            # dataset_path = '../../../../datasets/ILSVRC2015_VID'
            # data_seq = build_dataset_and_init(
            #     ImageNetVideoDataset, dataset_path, 'train')
            # dataset_path = '../../../../datasets/OTB_2013_small'
            dataset_path = DATASET_DIR
            data_seq = build_dataset_and_init(OTBDataset, dataset_path)
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
    np.random.seed(731995)
    cfg = TrackerConfig()
    trainer = SiamFCTrainer(cfg)
    trainer.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
