import os
import torch

if __name__ == "__main__":

    nx = 20
    ny = 20

    dataset_path = os.path.join(".", "resources", "datasets", "ood")

    dataset_name = "bc_ood_left"

    # create output directory
    output_dir = os.path.join(".", "ood_dataset", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # load input/output points and reshape them into the correct shape
    inputs = torch.load(os.path.join(dataset_path, "dataset_" + dataset_name + "_in.pt"))
    outputs = torch.load(os.path.join(dataset_path, "dataset_" + dataset_name + "_out.pt"))

    # for the inputs we only want the horizontal velocity, thus we take the first row
    inputs = inputs[:, 0, :]
    inputs = inputs.reshape(inputs.shape[0], 1, (nx + 1), (ny + 1))

    # for the outputs we want both horizontal and vertical velocity, thus we take the
    # first two rows
    outputs = outputs[:, :2, :]
    outputs = outputs.reshape(outputs.shape[0], 2, (nx + 1), (ny + 1))

    # save the transformed data
    torch.save(inputs, os.path.join(output_dir, "inputs.pt"))
    torch.save(outputs, os.path.join(output_dir, "outputs.pt"))
