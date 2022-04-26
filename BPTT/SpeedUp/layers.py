import torch
from torch.nn.parameter import Parameter
from torch.sparse import mm as SMM
from torch.nn import Module
from torch import Tensor


class SparseLinear(Module):
    def __init__(self, ids, vals, size: tuple):
        super(SparseLinear, self).__init__()
        self.ids  = ids
        self.vals = vals
        self.size = size
        temp_tensor = torch.sparse_coo_tensor(self.ids, self.vals, self.size).coalesce()
        temp_ids = temp_tensor.indices()
        temp_val = temp_tensor.values()
        temp_tensor = torch.sparse_coo_tensor(temp_ids, temp_val, self.size)
        self.weight = Parameter(temp_tensor)


    def forward(self, input: Tensor) -> Tensor:
        return SMM(self.weight, input)



class FullLinear(Module):
    def __init__(self, in_size:int, out_size:int):
        super(FullLinear, self).__init__()
        self.in_size  = in_size
        self.out_size = out_size
        self.weight = Parameter(torch.ones(self.out_size, self.in_size))


    def forward(self, input: Tensor) -> Tensor:
        return torch.mm(self.weight, input)