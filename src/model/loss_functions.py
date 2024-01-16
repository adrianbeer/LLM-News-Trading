import torch
import torch.nn as nn

class WeightedSquaredLoss(nn.Module):
    def __init__(self, gamma):
        super(WeightedSquaredLoss, self).__init__()
        self.gamma = gamma

    def forward(self, output, target):
        flat_output = torch.flatten(output)
        flat_target = torch.flatten(target)
        N = len(output)
        loss = torch.dot(torch.pow(torch.add(torch.abs(flat_output), 1), self.gamma), torch.square(flat_output - flat_target)) / N
        return loss
    
