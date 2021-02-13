import click

import torch


@click.command()
@click.argument("checkpoint_file_path")
@click.argument("model_output_file_path")
def main(checkpoint_file_path, model_output_file_path):
    checkpoint = torch.load(checkpoint_file_path)
    model_state = checkpoint['model']
    torch.save(model_state, model_output_file_path)


if __name__ == '__main__':
    import sys
    
    sys.exit(main())
