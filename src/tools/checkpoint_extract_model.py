import sys
import click

import torch


@click.command()
@click.argument("checkpoint_file_path")
@click.argument("model_output_file_path")
def main(checkpoint_file_path, model_output_file_path):
    """
    Extracts the saved model in a checkpoint given by the CHECKPOINT_FILE_PATH
    as a separate MODEL_OUTPUT_FILE_PATH so that it can be loaded directly.
    """
    checkpoint = torch.load(checkpoint_file_path)
    model_state = checkpoint['model']
    torch.save(model_state, model_output_file_path)


if __name__ == '__main__':
    sys.exit(main())
