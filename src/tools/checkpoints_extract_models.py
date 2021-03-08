#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import sys
import tqdm
import click
import pathlib
from typing import Optional

from utils import extract_model_from_checkpoint


@click.command()
@click.argument("checkpoints_dir_path", type=click.Path(exists=True))
@click.argument("models_output_dir_path", type=click.Path())
@click.option(
    "--tracker-name",
    help="tracker name (for producing results and reports) to which a suffix "
         "will be added")
def main(
        checkpoints_dir_path: str, models_output_dir_path: str,
        tracker_name: Optional[str]) -> int:
    """
    Extracts the saved models stored in checkpoints given by the
    CHECKPOINTS_DIR_PATH as separate files generated in the
    MODEL_OUTPUT_DIR_PATH so that they can be loaded directly for evaluation or
    fine-tuning.
    """
    checkpoints_dir = pathlib.Path(checkpoints_dir_path)
    models_output_dir = pathlib.Path(models_output_dir_path)
    models_output_dir.mkdir(parents=True, exist_ok=True)
    
    for checkpoint_file in tqdm.tqdm(checkpoints_dir.iterdir()):
        epoch_suffix_sep_pos = checkpoint_file.stem.rfind("_")
        epoch_suffix = checkpoint_file.stem[epoch_suffix_sep_pos + 1:]
        
        curr_tracker_name = tracker_name if tracker_name else "SiamFC"
        model_file_name = f"{curr_tracker_name}_{epoch_suffix}.pth"
        model_output_file_path = str(models_output_dir / model_file_name)
        
        extract_model_from_checkpoint(
            str(checkpoint_file), model_output_file_path)
    
    return 0

    
if __name__ == '__main__':
    sys.exit(main())
