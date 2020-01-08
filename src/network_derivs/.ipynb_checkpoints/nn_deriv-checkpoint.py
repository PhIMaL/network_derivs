import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def forward(self, input):
        z = [F.linear(input[0], self.weight, self.bias)]
        dz = [F.linear(dx, self.weight) for dx in input[1:]]
        return z + dz

class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        sigma = torch.tanh(input[0])
        dsigma = 1 / torch.cosh(input[0])**2
        d2sigma = -2 * sigma * dsigma
        d3sigma = d2sigma**2 / dsigma - 2*dsigma**2
        
        df = input[1] * dsigma[:, None, :]
        d2f = input[1]**2 * d2sigma[:, None, :] + input[2] * dsigma[:, None, :]
        d3f = 3 * input[1] * input[2] * d2sigma[:, None, :] + input[1]**3 * d3sigma[:, None, :] + input[3] * dsigma[:, None, :]        
        
        return [sigma, df, d2f, d3f]
