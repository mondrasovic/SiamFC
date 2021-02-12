from typing import Optional

import click
import torch
from got10k.experiments import ExperimentGOT10k

from sot.cfg import DATASET_DIR, TrackerConfig
from sot.tracker import TrackerSiamFC


@click.command()
@click.argument("results_dir_path")
@click.argument("reports_dir_path")
@click.option("--model-file-path", help="model file path")
def main(
        results_dir_path: str, reports_dir_path: str,
        model_file_path: Optional[str]) -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrackerConfig()
    tracker = TrackerSiamFC(cfg, device, model_file_path)
    experiment = ExperimentGOT10k(
        DATASET_DIR, subset='val', result_dir=results_dir_path,
        report_dir=reports_dir_path)
    experiment.run(tracker, visualize=False)
    
    experiment.report([tracker.name])
    
    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())
