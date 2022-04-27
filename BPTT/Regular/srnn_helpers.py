import torch
import time
from torch.nn import ReLU
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, ListedColormap
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
ReLu = ReLU()

################################################################################
################### firing rate functions with PyTorch #########################
################################################################################
def tanh_shifted_and_scaled2one(x, x0=-0.5):
    """ [TanH(ReLU(x)+x_0) - TanH(x_0)] / [1 - TanH(x_0)] """
    x_0 = torch.tensor(x0)
    return (torch.tanh(ReLu(x)+x_0)-torch.tanh(x_0))/(1-torch.tanh(x_0))

def bounded_ReLu_Power(x, x_bound=2.0, p=1.0):
    """ Min(ReLU(x)^p, x_bound) """
    return torch.min(torch.pow(ReLu(x),p), torch.tensor(x_bound))

def bounded_ReLu(x, x_bound=2.0):
    """ Min(ReLU(x), x_bound) """
    return bounded_ReLu_Power(x, x_bound, 1)

def bounded_ReLu_squared(x, x_bound=2.0):
    """ Min(ReLU(x)^2, x_bound) """
    return bounded_ReLu_Power(x, x_bound, 2)

def bounded_ReLu_cube(x, x_bound=2.0):
    """ Min(ReLU(x)^3, x_bound) """
    return bounded_ReLu_Power(x, x_bound, 3)

################################################################################
###################### Loss functions for BPTT #################################
################################################################################
# loss function to follow given targets
def loss_ReLU_on_targets(**inp):
    """ Computes the loss function given target dynamics. "+" patterns need to \
be above their target (target_dynamics[0]) and "-" patterns need to be below \
their target (target_dynamics[1]).  """
    loss_plus = ReLu( inp['target_dynamics'][np.zeros(inp['ids_plus' ].shape[0])].T - inp['activity'][:,inp['ids_plus'] ]).sum()
    loss_minus= ReLu(-inp['target_dynamics'][np.ones( inp['ids_minus'].shape[0])].T + inp['activity'][:,inp['ids_minus']]).sum()
    return loss_plus + loss_minus

# loss functions that trains directly on labels
def loss_helper_train_on_labels(x):
    return torch.tensor(1) - (torch.tanh(ReLu(x*10)) - ReLu(-x*10))

def loss_train_on_labels(**inp):
    """ Computes the loss based on the labels directly. Allows network to use \
arbitrary dynamics to solve the task. Does not make network follow some \
target dynamics """
    start_time_id = -inp['thresholds'].shape[0]
    loss_plus = loss_helper_train_on_labels( (inp['activity'][start_time_id:, inp['ids_plus'] ].T - inp['thresholds'])).sum()
    loss_minus= loss_helper_train_on_labels(-(inp['activity'][start_time_id:, inp['ids_minus']].T - inp['thresholds'])).sum()
    return loss_plus + loss_minus



################################################################################
############################## SRNN Class/Object ###############################
################################################################################
class SRNN():
    def __init__(self, 
                  computing_mode,               # "CPU" or "GPU" modes
                  N,                            # number of neurons in the SRNN
                  connections_denseness,        # percentage of existing connections, among the possible N^2
                  alpha,                        # P / (denseness * N^2) -> overwritten if P is provided
                  targets,                      # readout activity targets
                  P=None,                       # number of patterns to store
                  readout_requires_grad=False,  # choose whether readout weights are trained or not
                  learning_rate=0.15,           # learning rate for the autograd optimizer
                  firing_rate_function=(lambda x: tanh_shifted_and_scaled2one(x, -0.5)),  # non-linearity of the network dynamics
                  loss_function=loss_ReLU_on_targets,  # loss function that BPTT uses
                  E_to_I_neurons_ratio=None,    # ratio between number of E neurons and I neurons in the recurrent network
                  readout_regularization=0.5,   # regularization for the readout weights
                  initialization_type_for_J='random',  # initialization type for recurrent connections: "random" or "covariance"
                  N_epochs=200,                 # number of training epochs
                  do_plot=0,                    # choose whether to do plots, and select period in terms of epochs
                  dt=0.1,                       # time constant of the dynamics
                  ON_time=10,                   # how many time steps "dt" the input stays ON
                  x_init=None,                  # initial starting point for the internal dynamics: "zeros" or normaly distributed
                  readout_sparse_thresh=0.1,    # threshold for the readout weights to be considered non-zero
                  classification_thresh=None,   # neural activity based desired classification threshold
                  readout_projection_type='binary',  # type of projection for the learned readout weights: "binary" -> 0/1, "ternary" -> -1/0/1
                  total_time_multiple=2,        # factor that multiplies total training time so that testing is extended over longer times
                  sparse_readout_mask=None,
                  store_history=[], # store history of given variable names: 
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
                  verbose=True  # print progress statements
                  ):
        # store the initialization parameters
        self.mode   = computing_mode
        self.gpu_id = None
        #  set the default torch tensor type for computing on the correct cores
        if self.mode == 'CPU':
            self.defaultTensor = 'torch.DoubleTensor'
        elif self.mode == 'GPU':
            self.defaultTensor = 'torch.cuda.DoubleTensor'
            self.gpu_id        = torch.cuda.current_device()
        else:
            raise ValueError('The variable "computing_mode" must be either "CPU" or "GPU".')
        torch.set_default_tensor_type(self.defaultTensor)

        self.N                      = N
        self.f                      = connections_denseness
        self.alpha                  = alpha
        self.targets                = torch.tensor(targets)

        self.P                      = P
        self.readout_requires_grad  = readout_requires_grad
        self.lr                     = learning_rate
        self.activation             = firing_rate_function
        self.loss_function          = loss_function
        self.E_to_I_ratio           = E_to_I_neurons_ratio
        self.readout_regularization = readout_regularization
        self.J_initialization       = initialization_type_for_J
        self.N_epochs               = N_epochs
        self.do_plot                = do_plot
        self.dt                     = dt
        self.ON_time                = ON_time
        self.readout_sparse_thresh  = readout_sparse_thresh
        self.classification_thresh  = self.targets.mean(0) if classification_thresh is None else torch.tensor(classification_thresh)
        self.readout_projection_type= readout_projection_type
        self.time_multiple          = total_time_multiple
        self.store_history          = store_history
        if 'all' in store_history:
            self.store_history = ["losses", "readout_sparsity", "accuracy_history", 
                                  "accuracy_history_proj_W_out", "J_history" , "W_out_history", 
                                  "W_out_actual_history", "rates_all_history", "readout_activity_history", 
                                  "readout_activity_history_proj_W_out", "thresholds_history", 
                                  "thresholds_history_proj_W_out"]
        self.verbose                = verbose                                  
        
        # adjust the number of patterns to be stored P according to the chosen 
        # alpha value, unless P is specifically chosen; in the latter case, 
        # modify alpha to achieve consistency
        if P is None:
            self.P = max(int(alpha * self.f * (self.N**2)), 1) 
        else:
            self.alpha = P / (self.f * (self.N**2))

        # initial synaptic strengths - may be somewhat useless for BPTT because
        # it will train network to achieve desired neural activity value
        self.g = 10 * 0.1**(1/4) / np.sqrt(self.N * self.P * np.sqrt(self.f))

        # define the total number of timesteps for one run of the simulation
        self.N_time_steps = targets.shape[1]

        # define the P random patterns and their random label
        self.patterns = torch.tensor(2 * np.random.rand(self.N,self.P) - 1)
        self.labels = 2 * np.random.randint(0,2,(1,self.P)) - 1
        self.ids_positive_label = np.where(self.labels== 1)[1]  # index location of "+" patterns
        self.ids_negative_label = np.where(self.labels==-1)[1]  # index location of "-" patterns

        # define the sparsity mask matrix
        temp_mask = np.zeros([self.N,self.N])
        number_non_zero_ids = int(self.N**2 * self.f)
        linear_non_zero_ids = np.random.permutation(self.N**2)[:number_non_zero_ids]  # linear index of the non-zero values
        temp_mask[np.unravel_index(linear_non_zero_ids, temp_mask.shape)] = 1  # map from linear to matrix indices and make the non-zero values equal to 1
        self.mask = torch.tensor(temp_mask)

        # define the conectivity matrix J
        if self.J_initialization == 'random':
            self.initial_J = torch.normal(0,self.g,[self.N,self.N])  # random normal initialization for the initial connectivity
        elif self.J_initialization == 'covariance':
            self.initial_J = self.g*((self.patterns*torch.tensor(self.labels))@self.patterns.T)  # initialization based on the covariance matrix of the patterns to be stored
        else:
            raise ValueError('The variable "initialization_type_for_J" must be either "random" or "covariance".')
        # mask the connectivity matrix J appropriately based on whether it is random or E/I network
        if self.E_to_I_ratio is None:
            self.neurons_type = None
            self.J = (self.mask*self.initial_J).requires_grad_(True)
        else:
            self.neurons_type = torch.ones([1,self.N])
            self.neurons_type[:,np.random.permutation(self.N)[:int(self.N / (1 + self.E_to_I_ratio))]] = -1
            self.J = (self.mask*self.initial_J*torch.sign(self.initial_J*self.neurons_type)).requires_grad_(True)

        # define the projection function for the readout weights
        self.project_readout    = None
        self.project_readout_np = None
        if self.readout_projection_type == 'binary':
            self.project_readout    = (lambda x: torch.abs(torch.sign(x)))
            self.project_readout_np = (lambda x: np.abs(np.sign(x)))
        elif self.readout_projection_type == 'ternary':
            self.project_readout    = (lambda x: torch.sign(x))
            self.project_readout_np = (lambda x: np.sign(x))
        else:
            raise ValueError('The variable "readout_projection_type" must be either "binary" or "ternary".')

        # define the readout sparsity mask
        if sparse_readout_mask is None:
            self.f_out = None
            self.mask_out = torch.ones([1,self.N])
        else:
            self.f_out = self.f if sparse_readout_mask == 'yes' else sparse_readout_mask
            temp_mask = np.zeros([1,self.N])
            number_non_zero_ids = int(self.N * self.f_out)
            linear_non_zero_ids = np.random.permutation(self.N)[:number_non_zero_ids]  # linear index of the non-zero values
            temp_mask[np.unravel_index(linear_non_zero_ids, temp_mask.shape)] = 1  # map from linear to matrix indices and make the non-zero values equal to 1
            self.mask_out = torch.tensor(temp_mask)

        # define the readout weights
        self.W_out = torch.ones([1,self.N], requires_grad=readout_requires_grad) if self.neurons_type is None else self.neurons_type.clone().detach().requires_grad_(readout_requires_grad)

        # define input ON/OFF step function
        self.W_in = torch.zeros(self.N_time_steps-1)
        self.W_in[:self.ON_time] = 1

        # define the activity initialization for each network start
        self.x_init = torch.zeros(self.N) if ((x_init is None) | (x_init=='zeros')) else torch.randn(self.N)

        # initialize various other variables to store after training
        self.losses                             = None  # store losses
        self.readout_sparsity                   = None  # store readout sparsity
        self.accuracy_history                   = None  # store the history of the accuracy for each epoch
        self.accuracy_history_proj_W_out        = None  # store the history of the accuracy, computed with projected (binary/ternary) output weights, for each epoch
        self.J_history                          = None  # store history of the connectivity matrix J
        self.W_out_history                      = None  # store values of the learned output weights
        self.W_out_actual_history               = None  # store values of the actually used learned output weights, accounting for pruned weights
        self.rates_all_history                  = None  # store the history of network rates in epochs and time for each pattern
        self.readout_activity_history           = None  # store the readout activity for each epoch
        self.readout_activity_history_proj_W_out= None  # store the readout activity, computed with projected (binary/ternary) output weights, for each epoch
        self.thresholds_history                 = None  # store the classification thresholds for each epoch
        self.thresholds_history_proj_W_out      = None  # store the classification thresholds, computed with projected (binary/ternary) output weights, for each epoch

        self.training_completed = False

        # connectivity matrix J and readout weights W_out after training
        self.trained_J = None
        self.trained_W_out = None

        self.N_time_steps_test = int(self.N_time_steps * self.time_multiple)  # total time steps for testing
        self.W_in_test         = torch.zeros(self.N_time_steps_test-1)  # input weights for testing
        self.W_in_test[:self.ON_time] = 1

        # define the discrimination thresholds dynamicaly from the test run on stored patterns
        self.discrimination_thresh      = None
        self.discrimination_thresh_proj = None

        # define the accuracy at the end of training for the stored patterns, as a function of time
        self.final_accuracy      = None
        self.final_accuracy_proj = None

        # store the test results as a function of sigma_patterns and sigma_noise
        self.test_results                  = {}
        self.test_results[(0.0, 0.0)]      = []
        self.test_results_proj             = {}
        self.test_results_proj[(0.0, 0.0)] = []

        # define the binary readout weights
        self.binary_W_out = None


    def train(self):
        """ Train the current instance of the SRNN. """
        torch.set_default_tensor_type(self.defaultTensor)

        # initialize optimizer for autograd
        if self.readout_requires_grad:
            optimizer = torch.optim.Adam([self.J, self.W_out], lr=self.lr)
        else:
            optimizer = torch.optim.Adam([self.J], lr=self.lr)

        # initialize the sizes for the variables to be stored after training, if they are selected for history storage
        self.losses                             = np.zeros(self.N_epochs) * np.nan                                          if 'losses'                             in self.store_history else None  # store losses
        self.readout_sparsity                   = np.zeros(self.N_epochs) * np.nan                                          if 'readout_sparsity'                   in self.store_history else None  # store readout sparsity
        self.accuracy_history                   = np.zeros([self.N_epochs, self.N_time_steps]) * np.nan                     if 'accuracy_history'                   in self.store_history else None  # store the history of the accuracy for each epoch
        self.accuracy_history_proj_W_out        = np.zeros([self.N_epochs, self.N_time_steps]) * np.nan                     if 'accuracy_history_proj_W_out'        in self.store_history else None  # store the history of the accuracy, computed with projected (binary/ternary) output weights, for each epoch
        self.J_history                          = np.zeros([self.N_epochs, self.N, self.N]) * np.nan                        if 'J_history'                          in self.store_history else None  # store history of the connectivity matrix J
        self.W_out_history                      = np.zeros([self.N_epochs, 1, self.N]) * np.nan                             if 'W_out_history'                      in self.store_history else None  # store values of the learned output weights
        self.W_out_actual_history               = np.zeros([self.N_epochs, 1, self.N]) * np.nan                             if 'W_out_actual_history'               in self.store_history else None  # store values of the actually used learned output weights, accounting for pruned weights
        self.rates_all_history                  = np.zeros([self.N_epochs, self.N_time_steps, self.P, self.N]) * np.nan     if 'rates_all_history'                  in self.store_history else None  # store the history of network rates in epochs and time for each pattern
        self.readout_activity_history           = np.zeros([self.N_epochs, self.N_time_steps, self.P]) * np.nan             if 'readout_activity_history'           in self.store_history else None  # store the readout activity for each epoch
        self.readout_activity_history_proj_W_out= np.zeros([self.N_epochs, self.N_time_steps, self.P]) * np.nan             if 'readout_activity_history_proj_W_out'in self.store_history else None  # store the readout activity, computed with projected (binary/ternary) output weights, for each epoch
        self.thresholds_history                 = np.zeros([self.N_epochs, self.N_time_steps]) * np.nan                     if 'thresholds_history'                 in self.store_history else None  # store the classification thresholds for each epoch
        self.thresholds_history_proj_W_out      = np.zeros([self.N_epochs, self.N_time_steps]) * np.nan                     if 'thresholds_history_proj_W_out'      in self.store_history else None  # store the classification thresholds, computed with projected (binary/ternary) output weights, for each epoch

        W_out_actual = self.compute_actual_W_out()
        training_start_time = time.time()  # time the running time for training
        current_epoch_time = time.time()  # time the current epoch
        for ei in range(self.N_epochs):  # "ei" is epoch index
            if self.verbose: print('EPOCH %d. Running dynamics ' % ei, end='')
            start_dynamics_time = time.time()  # time the dynamics run
            # rates at all times and neurons for all patterns:
            rates_all = torch.zeros(self.N_time_steps, self.P, self.N, dtype=torch.float64)
            # factor that says whether the network is arbitrary or E/I type:
            EI_dynamics_factor = 1 if self.neurons_type is None else torch.sign(self.J*self.neurons_type)
            # actual connectivity matrix being used for the dynamics, accounting for network type and sparsity
            J_actual = self.J*EI_dynamics_factor*self.mask

            # run the network through all the patterns
            for pi in range(self.P):  # "pi" is pattern index
                # printing progress
                if self.verbose: 
                    if (pi+1) % 100 == 0:
                        print('|',end='')
                    elif (pi+1) % 10 == 0:
                        print('.',end='')

                x = self.x_init  # initialize activity
                rates = self.activation(x)  # compute initial rates
                rates_all[0,pi] = rates  # store initial rates

                # run the netork for current pattern for the total duration
                for ti in range(1, self.N_time_steps):  # "ti" is time index
                    x = x + self.dt*(-x + J_actual @ rates + self.W_in[ti-1] * self.patterns[:,pi])  # network dynamics equation
                    rates = self.activation(x)  # compute the new rates
                    rates_all[ti,pi] = rates  # store the new rates
            end_dynamics_time = time.time()  # time the dynamics run
            
            # compute the readout activity
            readout_activity            = torch.matmul(rates_all, W_out_actual.T).squeeze()/self.N  # radout activity using the actual readout weights
            readout_activity_proj_W_out = torch.matmul(rates_all, self.project_readout(W_out_actual).T).squeeze()/self.N  # readout activity using the projected readout weights

            # store the epoch history of some variables
            if 'J_history'                          in self.store_history: self.J_history[ei]           = J_actual.data.cpu().numpy()
            if 'W_out_history'                      in self.store_history: self.W_out_history[ei]       = self.W_out.data.cpu().numpy()
            if 'W_out_actual_history'               in self.store_history: self.W_out_actual_history[ei]= W_out_actual.data.cpu().numpy()
            if 'rates_all_history'                  in self.store_history: self.rates_all_history[ei]   = rates_all.data.cpu().numpy()
            if 'readout_sparsity'                   in self.store_history: self.readout_sparsity[ei]    = 100*np.sum(W_out_actual.data.cpu().numpy()==0)/self.N
            if 'readout_activity_history'           in self.store_history: self.readout_activity_history[ei]            = readout_activity.data.cpu().numpy()
            if 'readout_activity_history_proj_W_out'in self.store_history: self.readout_activity_history_proj_W_out[ei] = readout_activity_proj_W_out.data.cpu().numpy()
            
            # compute the accuracy based on dynamic threshold for both regular W_out and for projected W_out
            # compute accuracy using W_out for readout weights
            if ('accuracy_history'   in self.store_history) or ('thresholds_history' in self.store_history) or self.do_plot or self.verbose: 
                accuracy, thresh = self.compute_accuracy(readout_activity.data.cpu().numpy().T)
            if 'accuracy_history'   in self.store_history: self.accuracy_history[ei]   = accuracy  # store the accuracy
            if 'thresholds_history' in self.store_history: self.thresholds_history[ei] = thresh  # store the dynamics thresholds
            # compute accuracy using the projected W_out for readout weights
            if ('accuracy_history_proj_W_out'   in self.store_history) or ('thresholds_history_proj_W_out' in self.store_history) or self.do_plot or self.verbose: 
                accuracy_proj, thresh_proj = self.compute_accuracy(readout_activity_proj_W_out.data.cpu().numpy().T)
            if 'accuracy_history_proj_W_out'   in self.store_history: self.accuracy_history_proj_W_out[ei] = accuracy
            if 'thresholds_history_proj_W_out' in self.store_history: self.thresholds_history_proj_W_out[ei] = thresh


            start_BP_time = time.time()  # time the BPTT step
            # compute the loss value
            loss = self.loss_function(activity=readout_activity, 
                                      target_dynamics=self.targets, 
                                      thresholds=self.classification_thresh,
                                      ids_plus=self.ids_positive_label, 
                                      ids_minus=self.ids_negative_label)
            # include readout weights L1 regularization
            if self.readout_regularization is not None: loss += self.readout_regularization * torch.sum(torch.abs(self.W_out))
                
            # do BPTT
            if self.verbose: print('Performing BP; ', end='')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            W_out_actual = self.compute_actual_W_out()

            if 'losses' in self.store_history: self.losses[ei] = loss.data.cpu().numpy()  # store the loss value
            end_BP_time = time.time()  # time the BPTT step            

            if self.verbose: print('Current accuracy: %.1f%%.' % (np.max(accuracy)))
            print('Epoch %d time: %.2fs, Dynamics time: %.2fs, BP time: %.2fs.' 
                    %(ei, time.time() - current_epoch_time, end_dynamics_time - start_dynamics_time, end_BP_time - start_BP_time))

            current_epoch_time = time.time()  # time the current epoch
            if self.verbose: print()
        self.trained_J = J_actual.data
        self.trained_W_out = W_out_actual.data
        print("Train time: ", time.time() - training_start_time)  # time the training time
        self.training_completed = True
        print("Prepare network for robustness tests ... ", end='')
        self.prepare_for_tests()
        print("Done.")
        return


    def compute_actual_W_out(self):
        W_out_actual = self.mask_out * self.W_out  # actual readout weights, if they are not trained
        if (self.readout_requires_grad) and (self.f_out is None):  # need to adjust readout weights for being able to learn sparse readout
            EI_readout_factor = self.W_out if self.neurons_type is None else self.neurons_type
            EI_readout_factor = self.project_readout(EI_readout_factor)  # this factor describes the type of readout weight, E(+) or I(-)
            W_out_mask = self.W_out * EI_readout_factor - self.readout_sparse_thresh  # this mask helps ignore the values of W_out that we want to be 0
            W_out_actual = self.W_out * ReLu(W_out_mask) / W_out_mask  # actual trained sparse readout weights 
        return W_out_actual


    def prepare_for_tests(self, time_multiple=None):
        """ Prepare this instance of the SRNN for testing its robustness. """
        if not self.training_completed:
            raise ValueError('This SRNN instance has not been trained. Train it before testing.')
        torch.set_default_tensor_type(self.defaultTensor)

        if time_multiple is None:
            N_time_steps = self.N_time_steps_test
            W_in         = self.W_in_test
        else:
            N_time_steps = int(self.N_time_steps * time_multiple)  # total time steps for testing
            W_in         = torch.zeros(N_time_steps-1)  # input weights for testing
            W_in[:self.ON_time] = 1

        # rates at all times and neurons for all patterns:
        rates_all = torch.zeros(N_time_steps, self.P, self.N, dtype=torch.float64)

        # run the network through all the patterns
        for pi in range(self.P):  # "pi" is pattern index
            x = self.x_init  # initialize activity
            rates = self.activation(x)  # compute initial rates
            rates_all[0,pi] = rates  # store initial rates

            # run the netork for current pattern for the total duration
            for ti in range(1, N_time_steps):  # "ti" is time index
                x = x + self.dt*(-x + self.trained_J @ rates + W_in[ti-1] * self.patterns[:,pi])  # network dynamics equation
                rates = self.activation(x)  # compute the new rates
                rates_all[ti,pi] = rates  # store the new rates
        # compute the readout activity
        readout_activity            = torch.matmul(rates_all, self.trained_W_out.T).squeeze()/self.N  # radout activity using the actual readout weights
        readout_activity_proj_W_out = torch.matmul(rates_all, self.project_readout(self.trained_W_out).T).squeeze()/self.N  # readout activity using the projected readout weights

        # compute the accuracy and discrimination threshold for regular W_out and projected W_out
        final_accuracy,      discrimination_thresh      = self.compute_accuracy(readout_activity.data.cpu().numpy().T)
        final_accuracy_proj, discrimination_thresh_proj = self.compute_accuracy(readout_activity_proj_W_out.data.cpu().numpy().T)

        if time_multiple is None:
            # store the results at the end of training
            self.final_accuracy             = final_accuracy
            self.final_accuracy_proj        = final_accuracy_proj
            self.discrimination_thresh      = discrimination_thresh
            self.discrimination_thresh_proj = discrimination_thresh_proj
            # store these 0-sigmas results
            self.test_results[(0.0, 0.0)].append(self.final_accuracy)
            self.test_results_proj[(0.0, 0.0)].append(self.final_accuracy_proj)

        return final_accuracy, final_accuracy_proj, discrimination_thresh, discrimination_thresh_proj


    def test(self, sigma_patterns=0.0, sigma_neurons=0.0, pattern_repeats=1):
        """ Test the performance of the current instance of the SRNN. """
        if not self.training_completed:
            raise ValueError('This SRNN instance has not been trained. Train it before testing.')
        torch.set_default_tensor_type(self.defaultTensor)

        # store the number of correct discriminations in time
        N_correct_discriminations      = np.zeros(self.N_time_steps_test)
        N_correct_discriminations_proj = np.zeros(self.N_time_steps_test)

        # run the network through all the noisy patterns, with neuron noise, and 
        # incldue different pattern noise with reach repeat
        for ri in range(pattern_repeats):  # "ri" is repeat index
            # generate displacements from original patterns
            displacements = torch.normal(0, sigma_patterns+1e-20, [self.N, self.P])
            # rates at all times and neurons for all patterns:
            rates_all = torch.zeros(self.N_time_steps_test, self.P, self.N, dtype=torch.float64)
            for pi in range(self.P):  # "pi" is pattern index
                x = self.x_init  # initialize activity
                rates = self.activation(x)  # compute initial rates
                rates_all[0,pi] = rates  # store initial rates

                # run the netork for current pattern for the total duration
                for ti in range(1, self.N_time_steps_test):  # "ti" is time index
                    # network dynamics equation with the appropriate type of dynamics noise
                    x = x + torch.normal(0, sigma_neurons+1e-20, [self.N]) + self.dt*(-x + self.trained_J @ rates + self.W_in_test[ti-1] * (self.patterns[:,pi]+displacements[:,pi]))                       

                    rates = self.activation(x)  # compute the new rates
                    rates_all[ti,pi] = rates  # store the new rates
            readout_activity            = torch.matmul(rates_all, self.trained_W_out.T).squeeze()/self.N  # radout activity using the actual readout weights
            readout_activity_proj_W_out = torch.matmul(rates_all, self.project_readout(self.trained_W_out).T).squeeze()/self.N  # readout activity using the projected readout weights

            readout_activity_numpy            = readout_activity.data.cpu().numpy().T
            readout_activity_proj_W_out_numpy = readout_activity_proj_W_out.data.cpu().numpy().T

            output_label            = 2 * (readout_activity_numpy            > self.discrimination_thresh) - 1
            output_label_proj_W_out = 2 * (readout_activity_proj_W_out_numpy > self.discrimination_thresh_proj) - 1

            N_correct_discriminations      += np.sum((output_label            * self.labels.T)>0, 0)
            N_correct_discriminations_proj += np.sum((output_label_proj_W_out * self.labels.T)>0, 0)

        N_correct_discriminations_proj / (self.P * pattern_repeats)
        
        # innitialize the test results, if the pair of the sigmas has not yet been encountered 
        try: 
            self.test_results[(sigma_patterns, sigma_neurons)]
        except KeyError:
            self.test_results[(sigma_patterns, sigma_neurons)] = []
        try: 
            self.test_results_proj[(sigma_patterns, sigma_neurons)]
        except KeyError:
            self.test_results_proj[(sigma_patterns, sigma_neurons)] = []

        # populate the test results
        self.test_results[(sigma_patterns, sigma_neurons)].append(100*N_correct_discriminations / (self.P * pattern_repeats))
        self.test_results_proj[(sigma_patterns, sigma_neurons)].append(100*N_correct_discriminations_proj / (self.P * pattern_repeats))
        return 


    def compute_accuracy(self, readout_activity_numpy):
        """ Computes the accuracy and dynamics thresholds for discrimination. """
        thresh = np.percentile(readout_activity_numpy, 100 * (np.sum(self.labels==-1)/self.P), axis=0)  # compute dynamic discrimination threshold 
        output_label = 2 * (readout_activity_numpy > thresh) - 1  # compute the predicted label for each pattern
        accuracy = 100*np.sum((output_label * self.labels.T)>0,0)/self.P  # compute the accuracy
        return accuracy, thresh

