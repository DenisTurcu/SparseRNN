import torch
import numpy as np
import dill
import os
import sys
from srnn_helpers import *

gpu_id = int(sys.argv[1])
torch.cuda.set_device(gpu_id)
print('GPU: ', torch.cuda.current_device())

targets = np.load('matlab_generated_targets.npy')

srnn = SRNN(computing_mode="GPU",  # "CPU" or "GPU" modes
            N=200,  # number of neurons in the SRNN
            connections_denseness=0.01,  # percentage of existing connections, among the possible N^2
            alpha=1.0,  # P / (denseness * N^2) -> overwritten if P is provided
            targets=targets,  # readout activity targets
            P=None,  # number of patterns to store
            readout_requires_grad=False,  # choose whether readout weights are trained or not
            learning_rate=0.15,  # learning rate for the autograd optimizer
            firing_rate_function=tanh_shifted_and_scaled2one,  # non-linearity of the network dynamics
            loss_function=loss_ReLU_on_targets,  # loss function that BPTT uses
            E_to_I_neurons_ratio=None,  # ratio between number of E neurons and I neurons in the recurrent network
            readout_regularization=0.5,  # regularization for the readout weights
            initialization_type_for_J='covariance',  # initialization type for recurrent connections: "random" or "covariance"
            N_epochs=1500,  # number of training epochs
            do_plot=False,  # choose whether to do plots, and select period in terms of epochs
            dt=0.1,  # time constant of the dynamics
            ON_time=10,  # how many time steps "dt" the input stays ON
            x_init=None,  # initial starting point for the internal dynamics: "zeros" or normaly distributed
            readout_sparse_thresh=0.1,  # threshold for the readout weights to be considered non-zero
            classification_thresh=None,  # neural activity based desired classification threshold
            readout_projection_type='binary',  # type of projection for the learned readout weights: "binary" -> 0/1, "ternary" -> -1/0/1
            total_time_multiple=1,  # factor that multiplies total training time so that testing is extended over longer times
            store_history=['accuracy_history'], # store history of given variable names: 
                        # "losses"                              : Ne
                        # "readout_sparsity"                    : Ne
                        # "accuracy_history"                    : Ne x Nt
                        # "accuracy_history_proj_W_out"         : Ne x Nt
                        # "J_history"                           : Ne x N  x N
                        # "W_out_history"                       : Ne x 1  x N
                        # "W_out_actual_history"                : Ne x 1  x N
                        # "rates_all_history"                   : Ne x Nt x P x N
                        # "readout_activity_history"            : Ne x Nt x P
                        # "readout_activity_history_proj_W_out" : Ne x Nt x P
                        # "thresholds_history"                  : Ne x Nt
                        # "thresholds_history_proj_W_out"       : Ne x Nt
            verbose=False)
srnn.train()

dill.dump(srnn, open('percolation_study_large/Xtrained_N200_alpha10_denseness10_Nepochs1500_GPU%d.pkl'%gpu_id, 'wb'))
