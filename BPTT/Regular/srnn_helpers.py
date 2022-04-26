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
################################################################################
################### firing rate functions with PyTorch #########################
################################################################################
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
################################################################################
###################### Loss functions for BPTT #################################
################################################################################
################################################################################
# loss function to follow given targets
def loss_ReLU_on_targets(**inp):
    """ Computes the loss function given target dynamics. "+" patterns need to \
be above their target (target_dynamics[0]) and "-" patterns need to be below \
their target (target_dynamics[1]).  """
    #####
    # MAY WANT TO INCLUDE SOME FORM OF TIME DISCOUNTING ....???
    #####
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
    #####
    # MAY WANT TO INCLUDE SOME FORM OF TIME DISCOUNTING ....???
    #####
    start_time_id = -inp['thresholds'].shape[0]
    loss_plus = loss_helper_train_on_labels( (inp['activity'][start_time_id:, inp['ids_plus'] ].T - inp['thresholds'])).sum()
    loss_minus= loss_helper_train_on_labels(-(inp['activity'][start_time_id:, inp['ids_minus']].T - inp['thresholds'])).sum()
    return loss_plus + loss_minus



################################################################################
################################################################################
############################## SRNN Class/Object ###############################
################################################################################
################################################################################
class SRNN():
    def __init__(self, 
                  computing_mode,  # "CPU" or "GPU" modes
                  N,  # number of neurons in the SRNN
                  connections_denseness,  # percentage of existing connections, among the possible N^2
                  alpha,  # P / (denseness * N^2) -> overwritten if P is provided
                  targets,  # readout activity targets
                  P=None,  # number of patterns to store
                  readout_requires_grad=False,  # choose whether readout weights are trained or not
                  learning_rate=0.15,  # learning rate for the autograd optimizer
                  firing_rate_function=(lambda x: tanh_shifted_and_scaled2one(x, -0.5)),  # non-linearity of the network dynamics
                  loss_function=loss_ReLU_on_targets,  # loss function that BPTT uses
                  E_to_I_neurons_ratio=None,  # ratio between number of E neurons and I neurons in the recurrent network
                  readout_regularization=0.5,  # regularization for the readout weights
                  initialization_type_for_J='random',  # initialization type for recurrent connections: "random" or "covariance"
                  N_epochs=200,  # number of training epochs
                  do_plot=0,  # choose whether to do plots, and select period in terms of epochs
                  dt=0.1,  # time constant of the dynamics
                  ON_time=10,  # how many time steps "dt" the input stays ON
                  x_init=None,  # initial starting point for the internal dynamics: "zeros" or normaly distributed
                  readout_sparse_thresh=0.1,  # threshold for the readout weights to be considered non-zero
                  classification_thresh=None,  # neural activity based desired classification threshold
                  readout_projection_type='binary',  # type of projection for the learned readout weights: "binary" -> 0/1, "ternary" -> -1/0/1
                  total_time_multiple=2,  # factor that multiplies total training time so that testing is extended over longer times
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

        # define input ON/OFF step function (roughly the input weights)
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

        # print information about the created object
        if self.verbose: self.print_info()
        pass


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



            # plot relevant figures
            # "and" opperation stops if first argument is false - important for self.do_plot being 0
            if self.do_plot and ((ei+1) % self.do_plot == 0):
                fig, axs = plt.subplots(4,4, figsize=(28,28))
                if 'losses' in self.store_history: 
                    self.plot_loss(axs[0,0])
                else:
                    axs[0,0].axis('off')
                self.plot_readout_activity(readout_activity.data.cpu().numpy(), 
                                           thresh, axs[1,0])
                self.plot_readout_activity(readout_activity_proj_W_out.data.cpu().numpy(), 
                                           thresh_proj, axs[1,1],
                                           extra_title='Projected Readout Weights')
                if 'accuracy_history' in self.store_history: 
                    self.plot_best_accuracy_in_time(self.accuracy_history, axs[2,0])
                else:
                    axs[2,0].axis('off')
                if 'accuracy_history_proj_W_out' in self.store_history:
                    self.plot_best_accuracy_in_time(self.accuracy_history_proj_W_out, axs[2,1],
                                                    extra_title='Projected Readout Weights')
                else:
                    axs[2,1].axis('off')
                self.plot_J_values(J_actual.data.cpu().numpy(), axs[0,1])
                self.plot_W_out_weights(self.W_out.data.cpu().numpy(), W_out_actual.data.cpu().numpy(), axs[0,2])
                if 'readout_sparsity' in self.store_history:
                    self.plot_readout_sparsity(axs[0,3])
                else:
                    axs[0,3].axis('off')
                self.plot_PCA_of_given_variable([0,1], rates_all.data.cpu().numpy()[-1], 'PCA Of Firing Rates At Final Time', axs[1,2])
                self.plot_PCA_of_given_variable([1,2], rates_all.data.cpu().numpy()[-1], 'PCA Of Firing Rates At Final Time', axs[1,3])

                self.plot_PCA_of_given_variable([0,1], readout_activity.data.cpu().numpy().T, 'PCA Of Readout Activity Timeseries', axs[2,2])
                self.plot_PCA_of_given_variable([1,2], readout_activity.data.cpu().numpy().T, 'PCA Of Readout Activity Timeseries', axs[2,3])
                self.plot_readout_activity_violin(readout_activity.data.cpu().numpy(), 
                                                  thresh, ax=axs[3,0])
                self.plot_readout_activity_violin(readout_activity_proj_W_out.data.cpu().numpy(), 
                                                  thresh_proj, ax=axs[3,1], extra_title='Projected Readout Weights')
                axs[3,2].axis('off')
                axs[3,3].axis('off')
                plt.draw()
                plt.pause(0.01)

            if self.verbose: print('Current accuracy: (%.1f%%, %.1f%%).' % (np.max(accuracy), np.max(accuracy_proj)))
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


    def run_post_training(self, time_multiple):
        return self.prepare_for_tests(time_multiple)


    def test(self, sigma_patterns=0.0, sigma_neurons=0.0, pattern_repeats=1,
             neurons_noise_type='simple'):
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
                    if neurons_noise_type=='simple':
                        x = x + torch.normal(0, sigma_neurons+1e-20, [self.N]) + self.dt*(-x + self.trained_J @ rates + self.W_in_test[ti-1] * (self.patterns[:,pi]+displacements[:,pi]))
                    elif neurons_noise_type=='self_neuron_noise':
                        raise NotImplementedError('"self_neuron_noise" value for the variable "neurons_noise_type" not yet implemented.')
                    elif neurons_noise_type=='weights_noise':
                        raise NotImplementedError('"weights_noise" value for the variable "neurons_noise_type" not yet implemented.')
                    elif neurons_noise_type=='all_noise':
                        raise NotImplementedError('"all_noise" value for the variable "neurons_noise_type" not yet implemented.')
                    else:
                        raise ValueError('The variable "neurons_noise_type" must be "simple", "self_neuron_noise", "weights_noise" or "all_noise".')                        

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


    def print_info(self):
        """ Printing function to show info about the current SRNN instance. """
        print('SRNN initialized:')
        print('     Computing mode: %s%s' % (self.mode[:3],'' if self.mode=='CPU' else ', ID %d' % self.gpu_id))
        print('     Number of neurons: %d. Denseness of connections: %.3f.' % (self.N, self.f))
        print('     Ratio between number of E neurons and number of I neurons: %.1f.' % (np.nan if self.E_to_I_ratio is None else self.E_to_I_ratio))
        print('     Number of patterns: %d. Alpha ratio: %.4f' % (self.P, self.alpha))
        print('     Training epochs: %d.' % self.N_epochs)
        print('     Time step: %.2f. Stimulus ON time: %.1f. Total time: %.1f.'
            % (self.dt, self.ON_time*self.dt, self.N_time_steps*self.dt))
        print('     Activation function documentation:\n        "%s".' % self.activation.__doc__)
        print('     Loss function documentation:\n        "%s".' % self.loss_function.__doc__)
        return



    # plotting functions for each type of plot
    def plot_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        if 'losses' not in self.store_history:
            ax.axis('off')
            print('Losses history was not stored.')
            return
        ax.plot(self.losses)
        ax.set_title('Loss', fontsize=15)
        ax.set_xlabel('Epoch #', fontsize=15)
        ax.set_ylabel('Loss Value', fontsize=15)
        ax.set_ylim(0,np.nanmax(self.losses)*1.05)            
        return


    def plot_readout_activity(self, readout_activity, 
                              actual_classification_thresholds, 
                              ax=None, extra_title=''):
        if ax is None:
            fig, ax = plt.subplots()        
        ax.plot([self.ON_time, self.ON_time], 
                [np.min(readout_activity), np.max(readout_activity)], 'yellow', linewidth=2)
        ax.plot(self.targets.data.cpu().numpy().T, '-k', linewidth=2)
        ax.plot(readout_activity[:,self.ids_positive_label], 'r', alpha=0.6, linewidth=0.3)
        ax.plot(readout_activity[:,self.ids_negative_label], 'b', alpha=0.6, linewidth=0.3)
        ax.scatter(np.arange(len(actual_classification_thresholds)), 
                   actual_classification_thresholds, marker='4', s=500, c='orange')
        ax.scatter(np.arange(readout_activity.shape[0]-len(self.classification_thresh), readout_activity.shape[0]),
                   self.classification_thresh.data.cpu().numpy(), marker='4', s=500, c='green')
        ax.set_title('Readout Activity by pattern type (B+, R-)\nBlack lines are targets%s' % ('\n'+extra_title), fontsize=15)
        ax.set_xlabel('Time Steps (int)', fontsize=15)
        ax.set_ylabel('Readout Activity for each pattern', fontsize=15)
        ax.set_xlim([0,readout_activity.shape[0]-1])
        ax.set_ylim([np.min(readout_activity), np.max(readout_activity)])
        ax.set_xticks(np.arange(0,readout_activity.shape[0]+1,2))
        ax.grid(linewidth=0.1)
        return


    def plot_readout_activity_violin(self, readout_activity, 
                                     actual_classification_thresholds,
                                     labels=None, extra_title='', ax=None):

        readout_activity = temp_readout_activity if readout_activity is None else readout_activity
        actual_classification_thresholds = temp_actual_classification_thresholds if actual_classification_thresholds is None else actual_classification_thresholds
        labels = self.labels[0] if labels is None else labels

        readout_activity_df = pd.DataFrame(readout_activity).stack().to_frame(name='readout_activity')
        readout_activity_df = readout_activity_df.reset_index()
        readout_activity_df = readout_activity_df.rename(columns={'level_0':'Time Steps', 'level_1':'Pattern Label'})
        readout_activity_df['Pattern Label'] = readout_activity_df['Pattern Label'].map(lambda x: labels[x])

        ax = sns.violinplot(x='Time Steps', y='readout_activity', hue='Pattern Label',
                   data=readout_activity_df, palette=['r','b'], split=True, inner="quartile", ax=ax)
        ax.scatter(np.arange(len(actual_classification_thresholds)), 
                   actual_classification_thresholds, marker='4', s=500, c='orange')
        ax.set_xlabel('Time steps (int)', fontsize=15)
        ax.set_ylabel('Readout activity', fontsize=15)
        ax.set_title('Readout Activity by pattern type (B+, R-)%s' % ('\n'+extra_title), fontsize=15)
        # ax.set_xlim([0,readout_activity.shape[0]-1])
        ax.set_ylim([np.min(readout_activity), np.max(readout_activity)])
        ax.set_xticks(np.arange(0,readout_activity.shape[0]+1,2))
        ax.set_xticklabels(np.arange(0,readout_activity.shape[0]+1,2))
        ax.grid(linewidth=0.2)
        return


    def plot_best_accuracy_in_time(self, accuracy_history_selected, ax=None, extra_title=''):
        c_map = plt.get_cmap('gist_ncar')  # colormap for observing accuracy epoch
        norm = Normalize(vmin=0, vmax=1)  # normalizer for the colormapplt.figure(fig_num)
        if ax is None:
            fig, ax = plt.subplots()
        ax.grid()
        ax.set_xlim([0,accuracy_history_selected.shape[1]-1])
        ax.set_ylim(45,100.5)
        ax.set_title('Best Accuracy in time per Epoch (colorbar)%s' % ('\n'+extra_title), fontsize=15)
        for ei, aa in enumerate(accuracy_history_selected):
            ax.plot(aa, color=c_map(norm(1-ei/self.N_epochs)))
        ax.plot([self.ON_time, self.ON_time], [0, 100], 'yellow', linewidth=2)
        if accuracy_history_selected.shape[1] > self.N_time_steps:
            ax.plot([self.N_time_steps-1, self.N_time_steps-1], [0, 100], 'orange', linewidth=2)
        ax.set_ylabel('Accuracy (%)', fontsize=15)
        ax.set_xlabel('Time Steps (int)', fontsize=15)
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('gist_ncar_r')), ax=ax)
        cbar.set_ticks(np.arange(0,1.001,1./8))
        cbar.set_ticklabels(np.int16(np.arange(0,1.001,1./8)*self.N_epochs))
        ax.set_xticks(np.arange(0,self.N_time_steps+1,2))
        pass

    
    def plot_W_out_weights(self, W_out, W_out_actual, ax=None):
        W_out        = W_out.reshape(-1)
        W_out_actual = W_out_actual.reshape(-1)
        if ax is None:
            fig, ax = plt.subplots()
        readout_neuron_type = self.project_readout_np(W_out) if self.neurons_type is None else self.project_readout(self.neurons_type).cpu().detach().numpy().reshape(-1)
        # ax.plot(np.maximum(W_out * readout_neuron_type,0)/W_out, 'bo', markersize=5)
        ax.plot(readout_neuron_type, 'bo', markersize=5)
        ax.plot(W_out, 'r*', markersize=8)
        ax.plot(W_out_actual, 'g+', markersize=10)
        ax.plot(np.array([np.arange(self.N),np.arange(self.N)]),
                np.array([np.maximum(W_out_actual * readout_neuron_type,0)/W_out, 
                          W_out]))
        ax.plot([0,self.N],[0,0])
        ax.set_xlabel('Neuron ID', fontsize=15)
        ax.set_ylabel('Readout Synapse Weight', fontsize=15)
        pass


    def plot_readout_sparsity(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.readout_sparsity)
        ax.set_title('Readout Sparsity', fontsize=15)
        ax.set_xlabel('Epoch #', fontsize=15)
        ax.set_ylabel('Sparsity Value', fontsize=15)
        ax.set_ylim(np.nanmin(self.readout_sparsity)-0.01,np.nanmax(self.readout_sparsity)+0.01)
        pass
    

    def plot_J_values(self, J=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        J = self.J_history[-1] if J is None else J
        lim = np.max(np.abs([J.min(), J.max()]))
        ax.imshow(J, cmap='bwr', vmin=-lim, vmax=lim)
        ax.set_xlabel('Pre-synaptic Neuron ID', fontsize=15)
        ax.set_ylabel('Post-synaptic Neuron ID', fontsize=15)
        pass


    def plot_PCA_of_given_variable(self, pcs_to_show=[0,1], variable=None, plot_title=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            
        if variable is None:
            variable = self.rates_all_history[-1,-1]
            plot_title = 'PCA Of Firing Rates At Final Time'

        pca = PCA(n_components=(np.max(pcs_to_show)+1))
        variable = pca.fit_transform(variable)
        ax.scatter(variable[self.ids_positive_label,pcs_to_show[0]], 
                   variable[self.ids_positive_label,pcs_to_show[1]], c='b')
        ax.scatter(variable[self.ids_negative_label,pcs_to_show[0]], 
                   variable[self.ids_negative_label,pcs_to_show[1]], c='r')
        ax.set_title(plot_title, fontsize=15)
        ax.set_xlabel('PC %d' % pcs_to_show[0], fontsize=15)
        ax.set_ylabel('PC %d' % pcs_to_show[1], fontsize=15)
        pass


################################################################################
################################################################################
#### implement a 0/1 "sparse" perceptron as in Baldassi, Braunstein paper: ##
#### "Efficient supervised learning in networks with binary synapses" 4/6  #####
################################################################################
################################################################################
def binary_perceptron(patterns, labels, hidden=None, activity_threshold=.15, error_threshold=1.0, max_iter=10000):
    N = patterns.shape[0]  # number of neurons
    P = patterns.shape[1]  # number of patterns
    h = np.ones(N) if hidden is None else hidden  # initial hidden states for each weight
    w = (np.sign(h)+1)/2  # initial weights
    theta = activity_threshold * N
    theta_m = error_threshold
    
    count = 0
    while count < max_iter:
        patterns_order = np.random.permutation(P)
        pat_correct = 0
        for i in range(P):
            pi = patterns_order[i]
            pat = patterns[:,pi]
            sig = labels[0, pi]
            
            current = pat.dot(w)
            delta = sig * (current - theta)
            
            if delta <= 0:
                h += 2 * sig * pat 
            elif delta <= theta_m:
                pat_correct += 1
                if sig == -1:
                    if np.random.rand() < 0.4:
                        h -= 2 * pat
            else:
                pat_correct += 1
                
            w = (np.sign(h)+1)/2
        if pat_correct == P:
            print('All patterns correctly classified; ', end='')
            break
        count += 1
    print('Max_Iter reached; ', end='')
    return w, h

