import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 64, 128, 128, 128, 128], dropout=0.2):
        super(MLP, self).__init__()
        self.layers = self.initialize_layers(input_size, hidden_sizes, dropout)


    def initialize_layers(self, input_size, hidden_sizes, dropout):
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Dropout(p=dropout))
        
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        return nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)