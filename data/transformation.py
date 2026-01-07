import os
import torch


def reflectX(inputs, outputs):
    # reflect along the x axis
    reflected_inputs = inputs.clone()
    reflected_outputs = outputs.clone()
    for i in range(inputs.shape[0]):
        for channel_idx in range(inputs.shape[1]):
            reflected_inputs[i, channel_idx] = torch.flip(inputs[i, channel_idx], dims=(0,))
        for channel_idx in range(outputs.shape[1]):
            reflected_outputs[i, channel_idx] = torch.flip(outputs[i, channel_idx], dims=(0,))
    
    return reflected_inputs, reflected_outputs


def add_data_reflection_x(inputs, outputs):
    reflected_inputs, reflected_outputs = reflectX(inputs, outputs)
    
    # join the original and the reflected data
    inputs = torch.cat((inputs, reflected_inputs), dim=0)
    outputs = torch.cat((outputs, reflected_outputs), dim=0)

    return inputs, outputs

def rotate90(inputs, outputs):
    # rotate by 90 degrees
    rotated_inputs = inputs.clone()
    rotated_outputs = outputs.clone()
    for i in range(inputs.shape[0]):
        for channel_idx in range(inputs.shape[1]):
            rotated_inputs[i, channel_idx] = torch.rot90(inputs[i, channel_idx], k=1, dims=(0, 1))
        for channel_idx in range(outputs.shape[1]):
            rotated_outputs[i, channel_idx] = torch.rot90(outputs[i, channel_idx], k=1, dims=(0, 1))

    return rotated_inputs, rotated_outputs

def exchange_channels(dataset):
    """
    Interchange the u-v channels of the given data tensor
    in case of only one channel, do nothing
    """
    if dataset.shape[1] == 1:
        return dataset
    
    # exchange the channels
    exchanged_dataset = dataset.clone()
    exchanged_dataset[:, 0] = dataset[:, 1]
    exchanged_dataset[:, 1] = dataset[:, 0]
    
    return exchanged_dataset

def add_zeros_channel(dataset):
    # add an extra channel of zeros to the dataset
    return torch.cat((dataset, torch.zeros(dataset.shape[0], 1, dataset.shape[2], dataset.shape[3])), dim=1)

def add_data_open_left_wall(inputs, outputs):
    # transform the data to represent exchange of u and v channels
    # with open left-wall
    reflected_inputs, reflected_outputs = reflectX(inputs, outputs)
    reflected_rotated_inputs, reflected_rotated_outputs = rotate90(reflected_inputs, reflected_outputs)
    reflected_rotated_reflected_inputs, reflected_rotated_reflected_outputs = reflectX(reflected_rotated_inputs, reflected_rotated_outputs)
    open_left_inputs = exchange_channels(reflected_rotated_reflected_inputs)
    open_left_outputs = exchange_channels(reflected_rotated_reflected_outputs)
    
    # join the original and the transformed data
    inputs = torch.cat((inputs, open_left_inputs), dim=0)
    outputs = torch.cat((outputs, open_left_outputs), dim=0)

    return inputs, outputs
    
def add_data_open_right_wall(inputs, outputs):
    # transform the data to represent exchange of u and v channels
    # with open right wall
    rotated_inputs, rotated_outputs = rotate90(inputs, outputs)
    rotated_reflected_inputs, rotated_reflected_outputs = reflectX(rotated_inputs, rotated_outputs)
    open_right_inputs = exchange_channels(rotated_reflected_inputs)
    open_right_outputs = exchange_channels(rotated_reflected_outputs)

    # join the original and the transformed data
    inputs = torch.cat((inputs, open_right_inputs), dim=0)
    outputs = torch.cat((outputs, open_right_outputs), dim=0)

    return inputs, outputs


if __name__ == "__main__":

    nx = 20
    ny = 20

    n_input_channels = 2
    n_output_channels = 2

    add_open_bottom_wall = True
    add_open_left_wall = True
    add_open_right_wall = True

    dataset_path = os.path.join(".", "resources", "datasets")

    dataset_name = ""

    # create output directory
    output_dir = os.path.join(".", "main_dataset", "base_open_walls")
    os.makedirs(output_dir, exist_ok=True)
    
    # load input/output points and reshape them into the correct shape
    inputs = torch.load(os.path.join(dataset_path, "dataset_" + dataset_name + "in.pt"))
    outputs = torch.load(os.path.join(dataset_path, "dataset_" + dataset_name + "out.pt"))

    # for the inputs we only want the horizontal velocity, thus we take the first n rows
    inputs = inputs[:, :n_input_channels, :]
    inputs = inputs.reshape(inputs.shape[0], n_input_channels, (nx + 1), (ny + 1))

    # for the outputs we want both horizontal and vertical velocity, thus we take the
    # first n rows
    outputs = outputs[:, :n_output_channels, :]
    outputs = outputs.reshape(outputs.shape[0], n_output_channels, (nx + 1), (ny + 1))

    if add_open_bottom_wall:
        inputs, outputs = add_data_reflection_x(inputs, outputs)
    
    if add_open_left_wall:
        inputs, outputs = add_data_open_left_wall(inputs, outputs)
    
    if add_open_right_wall:
        inputs, outputs = add_data_open_right_wall(inputs, outputs)
    
    # save the transformed data
    torch.save(inputs, os.path.join(output_dir, "inputs.pt"))
    torch.save(outputs, os.path.join(output_dir, "outputs.pt"))
