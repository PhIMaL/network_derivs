import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear layer
class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def forward(self, input):
        X, dX = input
        z = F.linear(X, self.weight, self.bias)
        dz = F.linear(dX, self.weight)
        return (z, dz)

# Activation function 
def activation_func_derivs(input, dsigma):
    X, dX = input
    
    df = dX[:, 0, :, :] * dsigma[:, 0:1, :]
    d2f = dX[:, 0, :, :]**2 * dsigma[:, 1:2, :] + dX[:, 1, :, :] * dsigma[:, 0:1, :]
    d3f = 3 * dX[:, 0, :, :] * dX[:, 1, :, :] * dsigma[:, 1:2, :] + dX[:, 0, :, :]**3 * dsigma[:, 2:3, :] + dX[:, 2, :, :] * dsigma[:, 0:1, :]        
    dF = torch.stack((df, d2f, d3f), dim=1)
    
    return dF

# Specific activation function implementations
class Tanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        sigma = torch.tanh(input[0])
        dsigma = 1 / torch.cosh(input[0])**2
        d2sigma = -2 * sigma * dsigma
        d3sigma = d2sigma**2 / dsigma - 2*dsigma**2
        dSigma = torch.stack((dsigma, d2sigma, d3sigma), dim=1)
        
        dF= activation_func_derivs(input, dSigma)
        
        return (sigma, dF)
