import os
import torch
import yaml

def normalize_data(inputs: torch.Tensor = None, outputs: torch.Tensor = None, range_info_path: str = "/data/min_max.yaml"):
    """
    From the range information file, normalize the data.

    Parameters
    ----------
    inputs : torch.Tensor or None
        Input data with format (batch_size, 1, nx, ny) representing the horizontal
        velocity field
    outputs : torch.Tensor or None
        Output data with format (batch_size, 2, nx, ny) representing the horizontal 
        and vertical velocity fields
    range_info_path : str
        Path to the min-max range information file
    
    Returns
    -------
    inputs_norm : torch.Tensor
        Normalized input data
    outputs_norm : torch.Tensor
        Normalized output data
    """
    # esnure file exists
    if not os.path.exists(range_info_path):
        raise FileNotFoundError("Range information file not found")

    # load the range information file
    with open(range_info_path, "r") as f:
        range_info = yaml.safe_load(f)

    # clone the data to avoid modifying the original data
    inputs_norm = inputs.clone() if inputs is not None else None
    outputs_norm = outputs.clone() if outputs is not None else None
    
    # normalize the data
    if inputs_norm is not None:
        inputs_norm[:, 0, :, :] = (
            (inputs[:, 0, :, :] - range_info["inputs"]["u"]["min"]) / 
            (range_info["inputs"]["u"]["max"] - range_info["inputs"]["u"]["min"])
        )
        if inputs.shape[1] > 1:
            inputs_norm[:, 1, :, :] = (
                (inputs[:, 1, :, :] - range_info["inputs"]["v"]["min"]) / 
                (range_info["inputs"]["v"]["max"] - range_info["inputs"]["v"]["min"])
            )
    if outputs_norm is not None:
        outputs_norm[:, 0, :, :] = (
            (outputs[:, 0, :, :] - range_info["labels"]["u"]["min"]) / 
            (range_info["labels"]["u"]["max"] - range_info["labels"]["u"]["min"])
        )
        outputs_norm[:, 1, :, :] = (
            (outputs[:, 1, :, :] - range_info["labels"]["v"]["min"]) / 
            (range_info["labels"]["v"]["max"] - range_info["labels"]["v"]["min"]) 
        )
    
    return inputs_norm, outputs_norm


def denormalize_data(inputs: torch.Tensor = None, outputs: torch.Tensor = None,
                     range_info_path: str = "/data/min_max.yaml"):
    """
    From the range information file, denormalize the data.

    Parameters
    ----------
    inputs : torch.Tensor
        Normalized input data with format (batch_size, 1, nx, ny) representing the horizontal
        velocity field
    outputs : torch.Tensor
        Normalized output data with format (batch_size, 2, nx, ny) representing the horizontal 
        and vertical velocity fields
    range_info_path : str
        Path to the min-max range information file
    
    Returns
    -------
    inputs_denorm : torch.Tensor
        Denormalized input data
    outputs_denorm : torch.Tensor
        Denormalized output data
    """
    # esnure file exists
    if not os.path.exists(range_info_path):
        raise FileNotFoundError("Range information file not found")

    # load the range information file
    with open(range_info_path, "r") as f:
        range_info = yaml.safe_load(f)

    # clone the data to avoid modifying the original data
    inputs_denorm = inputs.clone() if inputs is not None else None
    outputs_denorm = outputs.clone() if outputs is not None else None
    
    # denormalize the data
    if inputs_denorm is not None:
        inputs_denorm[:, 0, :, :] = (
            inputs[:, 0, :, :] * (range_info["inputs"]["u"]["max"] - range_info["inputs"]["u"]["min"]) + 
            range_info["inputs"]["u"]["min"]
        )
        if inputs.shape[1] > 1:
            inputs_denorm[:, 1, :, :] = (
                inputs[:, 1, :, :] * (range_info["inputs"]["v"]["max"] - range_info["inputs"]["v"]["min"]) + 
                range_info["inputs"]["v"]["min"]
            )
    if outputs_denorm is not None:
        outputs_denorm[:, 0, :, :] = (
            outputs[:, 0, :, :] * (range_info["labels"]["u"]["max"] - range_info["labels"]["u"]["min"]) + 
            range_info["labels"]["u"]["min"]
        )
        outputs_denorm[:, 1, :, :] = (
            outputs[:, 1, :, :] * (range_info["labels"]["v"]["max"] - range_info["labels"]["v"]["min"]) + 
            range_info["labels"]["v"]["min"]
        )
    
    return inputs_denorm, outputs_denorm


def is_data_normalized(dataset: torch.Tensor):
    """
    Check if the data is normalized.
    """
    return dataset.min() == 0 and dataset.max() == 1
