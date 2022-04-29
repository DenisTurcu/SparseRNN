# Sparse RNNs can support high-capacity classification

The BPTT Python + Pytorch simulations, Hebbian Learning Matlab simulations and Directed Graph Empirical Study results can be found in their respective folders.

## BPTT
The "Regular" code works via sparsity and EI architecture Mask, while the "SpeedUp" code works via PyTorch sparse tensors.
### Regular
The Example.ipynb file contains an example of an E-I split architecture network trained for a short amount of time. This network has the same number of E and I neurons. The post-training accuracy in time, before, during and after the training period is shown.

### SpeedUp
The Example.ipynb file contains an example of a network trained for a short amount of time. The readout changes after training to separate the two classes, and the accuracy decays outside the training period. All other code files in this folder are the basis of the Sparse RNN model and training.

## DGPercolation
Here is the code providing the basis for the empirical percolation results on directed graphs (DG).

## HebbianLearning
This contains the two methods, OS and OS+
### OS
Single simulation can be run from the test_OS.m file, which will produce the accuracy achieved as a variable. 
### OS_Plus
Single simulation can be run from the test_OSplus.m file, which will produce a plot with the accuracy at every epoch. Above capacity, the accuracy will start decaying due to catastrophic forgetting after some epochs.

# Dependencies
The Python code was ran in a virtual environment created with Anaconda version 4.10.1, Python version 3.8.5 and PyTorch version 1.8.1.
The Matlab code was ran in Matlab version 9.8 (R2020a) or later.
