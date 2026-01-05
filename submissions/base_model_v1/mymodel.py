import os
import torch
import torch.nn as nn
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
        inputs_norm = (
            (inputs - range_info["inputs"]["u"]["min"]) / 
            (range_info["inputs"]["u"]["max"] - range_info["inputs"]["u"]["min"])
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
        inputs_denorm = (
            inputs * (range_info["inputs"]["u"]["max"] - range_info["inputs"]["u"]["min"]) + 
            range_info["inputs"]["u"]["min"]
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


class Model(nn.Module):
    """
    Class which implements the model architecture of the
    flow simulator accelerator. It is based on nested 
    2D convolutional layers.
    """
    def __init__(self):
        super(Model, self).__init__()

        ##################
        ## Model Layers ##
        ##################

        # First convolutional layer:
        # Conv2d(in=1, out=16, kernel=7x7, padding="same", stride=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, 
                               stride=1, padding='same')
        # Generic activation layer:
        self.relu = nn.ReLU()
        # Hidden layers:
        n_hidden = 5
        self.hidden = nn.ModuleList([
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=7,
                stride=1, padding='same'
            )
        for _ in range(n_hidden)
        ])
        # Output layer:
        self.output_layer = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=7,
            stride=1, padding='same'
        )

        #######################
        ## Training Metadata ##
        #######################

        self._training_loss = []
        self._validation_loss = []

    def fit(self, train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader = None,
            learning_rate: float = 0.001, epochs: int = 100,
            normalize: bool = False, range_info_path: str = "./data/min_max.yaml"):
        """Fit the model on the given training data"""
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # ensure data is normalized if requested
        if normalize:
            inputs, outputs = train_dataloader.dataset.tensors
            inputs, outputs = normalize_data(inputs, outputs, range_info_path)
            train_dataloader.dataset.tensors = (inputs, outputs)

        # Train model: for each epoch, evaluate the model, 
        # compute the loss and update the weights (i.e. step)
        self.train()
        self._training_loss = []
        self._validation_loss = []
        for epoch in range(epochs):
            for batch_x, batch_y in train_dataloader:
                # Ensure data is float32
                batch_x = batch_x.float()
                batch_y = batch_y.float()

                optimizer.zero_grad()
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            self._training_loss.append(loss.item())

            if val_dataloader is not None:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_dataloader:
                        batch_x = batch_x.float()
                        batch_y = batch_y.float()
                        outputs = self.forward(batch_x)
                        val_loss += criterion(outputs, batch_y).item()
                val_loss /= len(val_dataloader)
                self._validation_loss.append(val_loss)
                self.train() # reset module to training mode

            # Log training loss every 50 epochs
            if epoch % 50 == 0:
                print("-"*10)
                print(f"Epoch {epoch}")
                print(f"Training loss: {loss.item()}")
                if val_dataloader is not None:
                    print(f"Validation loss: {val_loss}")
                print("-"*10)



    def forward(self, x: torch.Tensor):
        """Evaluate the model on a given input"""
        # Apply first + hidden layers following of an
        # activation relu-layer
        x = self.conv1(x)
        x = self.relu(x)
        for layer in self.hidden:
            x = layer(x)
            x = self.relu(x)

        # Apply output layer and return
        return self.output_layer(x)

    def inference(self, inputs: torch.Tensor,
                  range_info_path: str = "./data/min_max.yaml"):
        """
        Perform inference on the given inputs by applying the model
        and (de)normalization to the data.
        """
        inputs, _ = normalize_data(inputs, None, range_info_path)
        outputs = self.forward(inputs)
        _, outputs = denormalize_data(None, outputs, range_info_path)
        return outputs
        
    def load_model(self, path: str = "./models/", name: str = "model"):
        """Load model weights from binary file"""
        # check that model file exist in the given path
        if not os.path.exists(os.path.join(path, name + ".pt")):
            raise FileNotFoundError("Model file not found")
        
        self.load_state_dict(torch.load(os.path.join(path, name + ".pt")))

    def export_model(self, path: str = "./models/", name: str = "model"):
        """Export model weights to binary file"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, name + ".pt"))


def init_my_model():
    model = Model()
    return model
