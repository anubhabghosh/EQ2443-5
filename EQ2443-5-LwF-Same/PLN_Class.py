########################################################################
# Project Name: Decentralised Deep Learning without Forgetting 
# Creators: Anubhab Ghosh (anubhabg@kth.se)
#           Peng Liu (peliu@kth.se)
#           Yichen Yang (yyichen@kth.se)
#           Tingyi Li (tingyi@kth.se)
# Project Owners: Alireza Mahdavi Javid (almj@kth.se),
#                Xinyue Liang (xinyuel@kth.se)
# December 2019
#########################################################################
import numpy as np
import sklearn as sk

class PLN():
    
    # Initialises all parameters for each layer of PLN/SSFN
    def __init__(self, Q: int, X, layer_no, n_hidden: int, W_ls = None):
        self.input = X
        self.n_l = layer_no
        self.V_Q = np.concatenate((np.eye(Q), -1*np.eye(Q)), axis=0)
        self.R_l = self.initialise_random_matrix(n_hidden - 2*Q, X.shape[0])
        self.O_l = self.initialise_random_matrix(Q, n_hidden)# Needs to be optimized by SGD/ADMM
        self.Y_l = np.zeros((n_hidden,1))
        #if W_ls.all() == None:
        #    pass
        #else:
        #    self.W_ls = W_ls
        #return None

    # Computes element wise activation 
    def activation_function(self, Z):
        return np.where(Z > 0, Z, 0)

    def initialise_random_matrix(self,M,N):
        #A =  np.random.uniform(low = -1, high = 1, size = (M,N)) 
        #return A - np.mean(A)
        return np.random.normal(0, 1, size=(M, N))
        
    def normalization(self, A):
        return A / np.sum(A**2, axis=0, keepdims=True)**(1/2)