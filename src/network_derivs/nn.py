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
def activation_func_derivs(dsigma, input):
    layer_func_derivs = [lambda f, g: g[:, 0, :, :] * f[:, 1:2, :],
                         lambda f, g: g[:, 0, :, :]**2 * f[:, 2:3, :] + g[:, 1, :, :] * f[:, 1:2, :],
                         lambda f, g: 3 * g[:, 0, :, :] * g[:, 1, :, :] * f[:, 2:3, :] + g[:, 0, :, :]**3 * f[:, 3:4, :] + g[:, 2, :, :] * f[:, 1:2, :]]
    
    df = []
    for order in range(input[1].shape[1]):
        df.append(layer_func_derivs[order](dsigma, input[1]))
    df = torch.stack(df, dim=1)
    f = dsigma[:, 0, :]
    
    return (f, df)

# Specific activation function implementations
class Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_func_derivs = [lambda ds, x: torch.tanh(x),
                                       lambda ds, x: 1 / torch.cosh(x)**2,
                                       lambda ds, x: -2 * ds[0] * ds[1],
                                       lambda ds, x: ds[2]**2 / ds[1] - 2* ds[1]**2]
        
    def forward(self, input):
        dsigma = []
        for order in range(input[1].shape[1]+1):
            dsigma.append(self.activation_func_derivs[order](dsigma, input[0]))
        dsigma = torch.stack(dsigma, dim=1)
        z, dz = activation_func_derivs(dsigma, input)
        
        return (z, dz)


        
        