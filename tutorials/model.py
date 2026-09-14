import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    """Simple neural network with a configurable number of layers and neurons.
    """
    
    def __init__(self, input_size, num_layers, num_neurons):
        """Initialize the neural network.

        Args:
            input_size (int): Number of input features.
            num_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
        """
        super().__init__()
        layers = [nn.Flatten()]

        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(nn.ReLU())
            input_size = num_neurons  # Set input size for the next layer

        layers.append(nn.Linear(num_neurons, 10))  # Output layer for 10 classes
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SimpleCNN(nn.Module):
    """Simple convolutional neural network with a configurable number of layers and neurons.
    """

    dropout_magnitude = 0.2

    def __init__(
        self,
        input_channels: int,
        num_layers: int,
        num_classes: int,
        hidden_dim: int,
        channel_factor: int = 2,
        image_height: int = 28,
        image_width: int = 28,
        dropout: bool = True,
    ):
        """Initialize the neural network.
        
        Args:
            input_channels (int): Number of input channels
            num_layers (int): Number of convolutional layers
            num_classes (int): Number of classes to predict
            hidden_dim (int): Number of neurons in the projection of the final layer
            channel_factor (int): Factor to multiply input channels by
            image_height (int): Height of the input image
            image_width (int): Width of the input image
            dropout (bool): Whether to use dropout
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.channel_factor = channel_factor
        self.image_height = image_height
        self.image_width = image_width
        self.dropout = dropout
        
        # Building model architecture
        layers = []
        for _ in range(num_layers):
            _out_channels = input_channels * 2 if input_channels > 1 else self.channel_factor
            layers.append(
                nn.Conv2d(input_channels, _out_channels, kernel_size=3)
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(5, stride=1))
            if dropout:
                layers.append(nn.Dropout(self.dropout_magnitude))
            input_channels = _out_channels  # Set input channels for the next layer

        layers.append(nn.Flatten(1))  # Flatten the output of the conv layers, except the batch dim

        _model = nn.Sequential(*layers)  # Dummy model to dynamically infer size of flattened dim
        with torch.no_grad():
            dummy_data = torch.randn(1, self.input_channels, image_height, image_width)
            dummy_output = _model(dummy_data)
            flattened_dim = dummy_output.shape[1]
        
        layers.append(nn.Linear(flattened_dim, hidden_dim))  # Projection layer
        layers.append(nn.Linear(hidden_dim, num_classes))  # Output layer for num classes
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)