#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import torch


def extract_model_from_checkpoint(
        checkpoint_file_path: str, model_output_file_path: str) -> None:
    """
    Extracts the saved model in a checkpoint as a separate file so that it can
    be loaded directly. The checkpoint consists of a dictionary where the model
    state dictionary is saved under the 'model' key.
    
    :param checkpoint_file_path: checkpoint file path (produced during training)
    :param model_output_file_path: output file path to save the extracted model
    """
    checkpoint = torch.load(checkpoint_file_path)
    model_state = checkpoint['model']
    torch.save(model_state, model_output_file_path)
