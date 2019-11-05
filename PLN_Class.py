import numpy as np
import sklearn as sk

class PLN():
    
    # Initialises all parameters for each layer of PLN/SSFN
    def __init__(self, Q: int, X, layer_no, n_hidden: int, W_ls = None):
        self.input = X
        self.n_l = no_layers
        self.V_Q = np.concatenate((np.eye(Q), -1*np.eye(Q)), axis=0)
        self.R_l = self.initialise_random_matrix((n_hidden - 2*Q), X.shape[1])
        self.O_l = self.initialise_random_matrix(n_hidden, Q)# Needs to be optimized by SGD/ADMM
        self.Y_l = np.zeros(n_hidden,1)

        return None

    # Computes element wise activation 
    def activation_function(self):
        return 

    # Performs the optimization process
    def optimize(self):
        return 

    def initialise_random_matrix(self,M,N):
        return 
        