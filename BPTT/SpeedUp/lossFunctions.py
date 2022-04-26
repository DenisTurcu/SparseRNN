import torch
from torch.nn import Module, ReLU
from torch.nn import _reduction as _Reduction
from torch import Tensor
ReLu = ReLU()


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class LossTargets(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:

        super(LossTargets, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, ids_plus: Tensor, ids_minus: Tensor, start_id: int = 0) -> Tensor:
        target = target[:,start_id:]
        loss_plus = ReLu( target[torch.zeros(ids_plus.shape[0],dtype=int)] - input[ids_plus, start_id:]).sum()
        loss_minus= ReLu(-target[torch.ones(ids_minus.shape[0],dtype=int)] + input[ids_minus,start_id:]).sum()
        return loss_plus + loss_minus


# ################################################################################
# ################################################################################
# ###################### Loss functions for BPTT #################################
# ################################################################################
# ################################################################################
# # loss function to follow given targets
# def loss_ReLU_on_targets(**inp):
#     """ Computes the loss function given target dynamics. "+" patterns need to \
# be above their target (target_dynamics[0]) and "-" patterns need to be below \
# their target (target_dynamics[1]).  """
#     #####
#     # MAY WANT TO INCLUDE SOME FORM OF TIME DISCOUNTING ....???
#     #####
#     loss_plus = ReLu( inp['target_dynamics'][np.zeros(inp['ids_plus' ].shape[0])].T - inp['activity'][:,inp['ids_plus'] ]).sum()
#     loss_minus= ReLu(-inp['target_dynamics'][np.ones( inp['ids_minus'].shape[0])].T + inp['activity'][:,inp['ids_minus']]).sum()
#     return loss_plus + loss_minus

# # loss functions that trains directly on labels
# def loss_helper_train_on_labels(x):
#     return torch.tensor(1) - (torch.tanh(ReLu(x*10)) - ReLu(-x*10))

# def loss_train_on_labels(**inp):
#     """ Computes the loss based on the labels directly. Allows network to use \
# arbitrary dynamics to solve the task. Does not make network follow some \
# target dynamics """
#     #####
#     # MAY WANT TO INCLUDE SOME FORM OF TIME DISCOUNTING ....???
#     #####
#     start_time_id = -inp['thresholds'].shape[0]
#     loss_plus = loss_helper_train_on_labels( (inp['activity'][start_time_id:, inp['ids_plus'] ].T - inp['thresholds'])).sum()
#     loss_minus= loss_helper_train_on_labels(-(inp['activity'][start_time_id:, inp['ids_minus']].T - inp['thresholds'])).sum()
#     return loss_plus + loss_minus