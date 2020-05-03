import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[128, 64, 32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (array int): Units for the linear layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        linear_sizes = zip([state_size] + fc_units, fc_units + [action_size])
        self.linears = nn.ModuleList([nn.Linear(input_size, output_size) for input_size, output_size in linear_sizes])

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.linears:
            x = F.relu(layer(x))
        
        return x