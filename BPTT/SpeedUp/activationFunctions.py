import torch
from torch.nn import Module, ReLU
from torch.nn import _reduction as _Reduction
from torch import Tensor
ReLu = ReLU()

class Tanh_shifted_and_scaled(Module):
    def forward(self, x: Tensor, x_0: float = -0.5) -> Tensor:
        x_0 = torch.tensor([x_0])
        return (torch.tanh(ReLu(x)+x_0)-torch.tanh(x_0))/(1-torch.tanh(x_0))

# ################################################################################
# ################################################################################
# ################### firing rate functions with PyTorch #########################
# ################################################################################
# ################################################################################
# def tanh_shifted_and_scaled2one(x, x0=-0.5):
#     """ [TanH(ReLU(x)+x_0) - TanH(x_0)] / [1 - TanH(x_0)] """
#     x_0 = torch.tensor(x0)
#     return (torch.tanh(ReLu(x)+x_0)-torch.tanh(x_0))/(1-torch.tanh(x_0))

# def bounded_ReLu_Power(x, x_bound=2.0, p=1.0):
#     """ Min(ReLU(x)^p, x_bound) """
#     return torch.min(torch.pow(ReLu(x),p), torch.tensor(x_bound))

# def bounded_ReLu(x, x_bound=2.0):
#     """ Min(ReLU(x), x_bound) """
#     return bounded_ReLu_Power(x, x_bound, 1)

# def bounded_ReLu_squared(x, x_bound=2.0):
#     """ Min(ReLU(x)^2, x_bound) """
#     return bounded_ReLu_Power(x, x_bound, 2)

# def bounded_ReLu_cube(x, x_bound=2.0):
#     """ Min(ReLU(x)^3, x_bound) """
#     return bounded_ReLu_Power(x, x_bound, 3)