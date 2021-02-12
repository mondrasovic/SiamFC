import click
import torch
from got10k.experiments import ExperimentGOT10k

from sot.cfg import DATASET_DIR, MODEL_DIR, TrackerConfig
from sot.tracker import TrackerSiamFC


@click.command()
def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrackerConfig()
    tracker = TrackerSiamFC(cfg, device, MODEL_DIR)
    experiment = ExperimentGOT10k(DATASET_DIR, subset='val')
    experiment.run(tracker, visualize=False)
    
    experiment.report([tracker.name])
    
    return 0


if __name__ == '__main__':
    import sys
    
    
    sys.exit(main())
