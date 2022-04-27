import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from lossFunctions import LossTargets
from activationFunctions import Tanh_shifted_and_scaled
from layers import SparseLinear, FullLinear
import numpy as np
from time import time as tm


class SRNN(nn.Module):
    def __init__(self, init_info:dict = dict(N=100, 
                                             f = 0.05, 
                                             activation=Tanh_shifted_and_scaled(),
                                             J_init_type='covariance',
                                             dt=0.1,
                                             x_init=None,
                                             verbose=True,
                                           	 GPU=0), 
                       input_info:dict = dict(alpha=0.5,
                                              P=None,
                                              ON_time=10,), 
                       output_info:dict = dict(targets=torch.zeros(20),
                                               learn_readout=False,
                                               start_id=10,
                                               f_out=0.05,), 
                       training_info:dict = dict(learning_rate=0.1,
                                                 loss_fn=LossTargets(),
                                                 N_epochs=200,), 
                       history_info:dict = dict(losses=[],
                                                acuracy=[],
                                                readout=[],
                                                etc='each key will have the history of what it means',)) -> None:
        super(SRNN, self).__init__()
        # store the input information
        self.init_info      = init_info
        self.input_info     = input_info
        self.output_info    = output_info
        self.training_info  = training_info
        self.history_info   = history_info
        self.default_tensor = 'torch.DoubleTensor'
        if init_info['GPU'] is not None:
            self.default_tensor = 'torch.cuda.DoubleTensor'
            torch.cuda.set_device(init_info['GPU'] )
        self.verbose = init_info['verbose']

        # store the number of training epochs completed and the maximum accuracy
        self.trained_epochs   = 0
        self.max_accuracy     = 0
        self.max_accuracy_dyn = 0

        self.init_network()
        self.init_params()


    def forward(self, N_time_steps) -> Tensor:
        torch.set_default_tensor_type(self.default_tensor)

        N = self.init_info['N']
        P = self.P
        dt = self.init_info['dt']
        patterns = self.patterns
        ON_time = self.input_info['ON_time']
        activation = self.init_info['activation']
        f_out = self.output_info['f_out']

        # run the network through all the patterns
        readout = torch.Tensor(P, N_time_steps)
        for pi in range(P):  # "pi" is pattern index
            rates_all = torch.Tensor(N, N_time_steps)
            x = torch.zeros(N,1)   # initialize activity
            rates = activation(x)  # compute initial rates

            # run the netork for current pattern for the total duration
            for ti in range(N_time_steps):  # "ti" is time index
                input_current = patterns[:,pi].reshape(-1,1) if ti < ON_time else 0
                x = x + dt*(-x + self.J(rates) + input_current)  # network dynamics equation
                rates = activation(x)  # compute the new rates
                rates_all[:, ti] = rates.reshape(-1)
            readout[pi] = self.w_out(rates_all)
        return readout/N


    def train_model(self, epochs:int = 0, 
                          stopping_accuracy:float = -1) -> None:
        torch.set_default_tensor_type(self.default_tensor)

        targets = torch.Tensor(self.output_info['targets'])
        thresholds = targets.mean(0)
        lr = self.training_info['learning_rate']
        loss_function = self.training_info['loss_fn']
        start_id = self.output_info['start_id']
        ids_plus = self.ids_positive_label
        ids_minus = self.ids_negative_label
        N_epochs = self.training_info['N_epochs'] if epochs == 0 else epochs
        P = self.P
        labels = self.labels
        stopping_accuracy = stopping_accuracy if stopping_accuracy > 0 else self.training_info['stopping_accuracy']

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for ei in range(N_epochs):
            start_time = tm()
            if self.verbose:
                print(f'Epoch {self.trained_epochs}.', end=' ')
            model_output = self(targets.shape[1])
            loss = loss_function(model_output, targets, ids_plus, ids_minus, start_id)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # update trianing details:
            self.trained_epochs += 1
            accuracy = ((((model_output > thresholds) * 2 - 1) * labels.reshape(-1,1) == 1).sum(0) / P).max()
            if self.max_accuracy < accuracy:
                self.max_accuracy = accuracy
            if self.max_accuracy >= stopping_accuracy:
                print(f'Training reached {stopping_accuracy*100:.2f} % accuracy in {self.trained_epochs} epochs. ')
                break
            # printing
            if self.verbose:
                print(f'Elapsed time: {tm()-start_time:.2f} s. Maximum accuracy so far: {self.max_accuracy*100:.2f}%.')

        print(f'Completed {self.trained_epochs} training epochs.')


    def init_params(self) -> None:
        torch.set_default_tensor_type(self.default_tensor)

        N = self.init_info['N']
        f = self.init_info['f']
        g = self.g
        patterns = self.patterns
        labels = self.labels
        f_out = self.output_info['f_out']

        # define the recurrent matrix J
        if self.init_info['J_init_type'] == 'random':
            temp_J = torch.normal(0, g, [N,N])  # random normal initialization for the initial connectivity
        elif self.init_info['J_init_type'] == 'covariance':
            temp_J = g*((patterns*labels)@patterns.T)  # initialization based on the covariance matrix of the patterns to be stored
        else:
            raise ValueError('The variable "initialization_type_for_J" must be either "random" or "covariance".')

        # generate the sparse ids
        paired_ids = np.unravel_index(np.random.permutation(N**2)[:int(N**2 * f)], (N,N))
        values_ids = temp_J[paired_ids]

        self.J = SparseLinear(ids=paired_ids, vals=values_ids, size=(N,N))
        
        # define output weights
        paired_ids_out = np.unravel_index(np.random.permutation(N)[:int(N * f_out)], (1,N))
        if f_out < 1:
            self.w_out = SparseLinear(ids=paired_ids_out, vals=torch.ones(paired_ids_out[0].shape[0]), size=(1,N))
        else:
            self.w_out = FullLinear(in_size=N, out_size=1)
        if self.output_info['learn_readout'] == False:
            for param in self.w_out.parameters():
                param.requires_grad = False

        # reset the training flags
        self.trained_epochs = 0
        self.max_accuracy   = 0


    def init_network(self) -> None:
        torch.set_default_tensor_type(self.default_tensor)

        N = self.init_info['N']
        f = self.init_info['f']

        # adjust the number of patterns to be stored P according to the chosen 
        # alpha value, unless P is specifically chosen; in the latter case, 
        # modify alpha for consistency
        P = self.input_info['P']
        self.P = P
        alpha = self.input_info['alpha']
        if self.input_info['P'] is None:
            self.P = max(int(alpha * f * (N**2)), 1)
            P = self.P
        else:
            self.alpha = P / (f * (N**2))

        # initial synaptic strengths
        self.g = 10 * 0.1**(1/4) / np.sqrt(N * P * np.sqrt(f))

        # define the total number of timesteps for one run of the simulation
        self.N_time_steps = self.output_info['targets'].shape[1]

        # define the P random patterns and their random label
        self.patterns = torch.tensor(2 * np.random.rand(N,P) - 1)
        self.labels = 2 * torch.randint(0,2,(1,P)) - 1
        self.ids_positive_label = torch.where(self.labels== 1)[1]  # index location of "+" patterns
        self.ids_negative_label = torch.where(self.labels==-1)[1]  # index location of "-" patterns

