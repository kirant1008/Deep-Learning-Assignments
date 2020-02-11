# importing all the required libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNeuralNetwork, self).__init__()
        # Linear Function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-Linearity
        self.relu = nn.ReLU()
        # Linear Function
        self.fc2 = nn.Linear(hidden_dim, 252)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(252, output_dim)

    def forward(self, x):
        # Linear Fucntion
        out = self.fc1(x)
        # Non Linear Function
        out = self.relu(out)
        # Linear Function
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out