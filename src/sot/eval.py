import click

import torch

from got10k.experiments import ExperimentGOT10k

from sot.cfg import TrackerConfig, MODEL_DIR
from sot.tracker import TrackerSiamFC


@click.command()
def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrackerConfig()
    tracker = TrackerSiamFC(cfg, device, MODEL_DIR)
    experiment = ExperimentGOT10k("../../../../datasets/GOT10k", subset='test')
    experiment.run(tracker, visualize=True)
    
    experiment.report([tracker.name])
    
    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())
