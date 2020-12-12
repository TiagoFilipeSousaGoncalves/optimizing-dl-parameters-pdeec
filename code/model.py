import torch
import torch.nn as nn
from collections import OrderedDict
from code.utilities import utils


# Define the Model Class
class Model(nn.Module):
    def __init__(self, input_shape, number_of_labels, solution):
        super(Model, self).__init__()

        # Process Input Shape
        self.channels = input_shape[0]
        self.rows = input_shape[1]
        self.columns = input_shape[2]

        # Convolutional Block Size
        self.nr_conv_layers = solution[0].size(0)
        self.nr_fc_layers = solution[1].size(0)

        # Go through the convolutional block of the solution (index: 0)
        # Check if we have convolutional layers
        if self.nr_conv_layers > 0:
            # Create Convolutional Block
            self.convolutional_layers = OrderedDict()
            input_channels = self.channels
            for conv_idx, layer in enumerate(solution[0]):
                self.convolutional_layers[f'conv_{conv_idx}'] = nn.Conv2d(in_channels=input_channels,
                                                                          out_channels=int(layer[0].item()),
                                                                          kernel_size=int(layer[1].item()))
                self.convolutional_layers[f'conv_act_{utils.inv_conv_activ_functions[int(layer[2].item())]}{conv_idx}'] = utils.conv_activ_functions[int(layer[2].item())]
                self.convolutional_layers[f'conv_dropout{conv_idx}'] = nn.Dropout2d(layer[3].item())
                self.convolutional_layers[f'conv_pool_{utils.inv_conv_pooling_types[int(layer[4].item())]}{conv_idx}'] = utils.conv_pooling_types[int(layer[4].item())]
                input_channels = int((layer[0].item()))

            # Convert into a conv-layer
            self.convolutional_layers = nn.Sequential(self.convolutional_layers)

        # Now, we have to compute the linear dimensions for the first FC Layer
        aux_tensor = torch.rand(1, self.channels, self.rows, self.columns)
        if self.nr_conv_layers > 0:
            aux_tensor = self.convolutional_layers(aux_tensor)

        # Input features
        input_features = aux_tensor.size(0) * aux_tensor.size(1) * aux_tensor.size(2) * aux_tensor.size(3)

        # Check if we have convolutional layers
        if self.nr_fc_layers > 0:
            # Now, we build the FC-Block
            self.fc_layers = OrderedDict()

            # Go through the fully-connected block of the solution (index: 1)
            for fc_idx, layer in enumerate(solution[1]):
                self.fc_layers[f'fc_{fc_idx}'] = nn.Linear(in_features=input_features,
                                                           out_features=int(layer[0].item()))
                self.fc_layers[f'fc_act_{utils.inv_fc_activ_functions[int(layer[1].item())]}{fc_idx}'] = utils.fc_activ_functions[int(layer[1].item())]
                self.fc_layers[f'fc_dropout{fc_idx}'] = nn.Dropout(layer[2])
                input_features = int(layer[0].item())

            # Now convert this into an fc layer
            self.fc_layers = nn.Sequential(self.fc_layers)

        # Last FC-layer
        self.fc_labels = nn.Linear(in_features=input_features, out_features=number_of_labels)

        # Add learning rate
        self.learning_rate = solution[2].item()

    def forward(self, x):
        # Go through convolutional block
        if self.nr_conv_layers > 0:
            x = self.convolutional_layers(x)
        # print(x.size())

        # Flatten
        x = x.view(x.size(0), -1)
        # print(x.size())

        # Go through fc block
        if self.nr_fc_layers > 0:
            x = self.fc_layers(x)
        # print(x.size())

        # Apply last FC layer
        x = self.fc_labels(x)
        # print(x.size())

        return x

