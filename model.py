import logging
import os
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Class which implements the model architecture of the
    flow simulator accelerator. It is based on nested 
    2D convolutional layers.
    """
    def __init__(self):
        super(Model, self).__init__()

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

    def fit(self, train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader = None,
            learning_rate: float = 0.001, epochs: int = 100):
        """Fit the model on the given training data"""
        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # Train model: for each epoch, evaluate the model, 
        # compute the loss and update the weights (i.e. step)
        self.train()
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

            # Log training loss
            logging.info(f"Epoch {epoch}, Loss: {loss.item()}")
            print(f"Epoch {epoch}, Loss: {loss.item()}")


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
