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
        self.activation_func_derivs = [lambda x, ds: torch.tanh(x[0]),
                                       lambda x, ds: 1 / torch.cosh(x[0])**2,
                                       lambda x, ds: -2 * ds[0] * ds[1],
                                       lambda x, ds: ds[2]**2 / ds[1] - 2* ds[1]**2]
        
    def forward(self, input):
        dsigma = []
        for order in range(input[1].shape[1]+1):
            dsigma.append(self.activation_func_derivs[order](input, dsigma))
        z = dsigma[0]
        dsigma = torch.stack(dsigma[1:], dim=1)
        dz = activation_func_derivs(input, dsigma)
        
        return (z, dz)


        
        