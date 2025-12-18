import torch

if __name__ == "__main__":

    nx = 20
    ny = 20
    
    # load input/output points and reshape them into the correct shape
    inputs = torch.load("./resources/dataset_in.pt")
    outputs = torch.load("./resources/dataset_out.pt")

    # for the inputs we only want the horizontal velocity, thus we take the first row
    inputs = inputs[:, 0, :]
    inputs = inputs.reshape(inputs.shape[0], 1, (nx + 1), (ny + 1))

    # for the outputs we want both horizontal and vertical velocity, thus we take the
    # first two rows
    outputs = outputs[:, :2, :]
    outputs = outputs.reshape(outputs.shape[0], 2, (nx + 1), (ny + 1))

    # save the transformed data
    torch.save(inputs, "./inputs.pt")
    torch.save(outputs, "./outputs.pt")
