import numpy as np
import sklearn as sk
from Admm import admm_sameset, optimize_admm
from func_set import compute_NME, compute_accuracy

##################################################################
# Class for initialising the PLN objects in every layer

# Attributes:
# Q : no. of output classes
# X : Input data (can be raw input or hidden activation)
# layer_no :  Layer No. of the current PLN object
# n_hidden : No. of hidden nodes for that layer

# Functions:
# activation_function: Implements ReLU activation
# initialise_random_matrix : Implements Random matrix for every 
# layer
# Normalisation :  Magnitude normalisation for Random Matrix
##################################################################

class PLN():
    
    # Initialises all parameters for each layer of PLN/SSFN
    def __init__(self, Q: int, X, layer_no, n_hidden: int):
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
        #return np.where(Z > 0, Z, 0.1*Z) #modify to Leaky ReLU
        return np.where(Z > 0, Z, 0) #ReLU
        #return np.maximum(Z*0.1, Z)

    def initialise_random_matrix(self,M,N):
        #A =  np.random.uniform(low = -1, high = 1, size = (M,N)) 
        #return A - np.mean(A)
        return np.random.normal(0, 1, size=(M, N))
        
    def normalization(self, A):
        return A / np.sum(A**2, axis=0, keepdims=True)**(1/2)

################################################################
# pln: a list containing all pln layers. Or: an empty list
# W: the corresponding W matrix, or en empty np.array

# Structure: 
#   pln: a list containing all the PLN layers.
#   num_layer: number of layers
#   W_ls : linear mapping matrix in layer 0
#   mu : ADMM learning rate relevant
#   lamba : wls training factors avoiding overfitting
#   max_iterations : Maximum number of iteration in ADMM
################################################################

class PLN_network():

    # Contsructor function to initialise a PLN object
    def __init__(self, pln: np.array=None,W_ls: np.array=None,num_layer: int=20, mu=0.1, maxit=30, lamba=1e2):    
        
        if pln is not None:
            self.num_layer = len(pln)
            for i in range(self.num_layer):
                if type(pln[i]) != PLN: # If the list contains garbage values, it is initialised properly
                    self.pln=None
                    print("Construct Error!")
                    return
                self.pln[i] = pln[i]    # self.pln[i].O_l, Assigning the list a PLN object if datatype matches
        else:
            # If no PLN network is already constructed, assigns new variables
            self.num_layer = num_layer # Assigns the number of layers
            self.pln = np.array([None]*self.num_layer) # Assigns an empty list

        # Assigning the value of W_ls (or O* for Layer 0)
        if W_ls is not None:
            self.W_ls=W_ls # If the value is already present, no needt to calculate 
        else:
            self.W_ls=None # Assigned None, will be created later on

        self.mu = mu # ADMM multiplier for the given dataset
        self.max_iterations = maxit # No. of iterations for the ADMM Algorithm
        self.lam = lamba # Lagrangian parameter Lambda

    def construct_W(self, X_train, Y_train):
        
        # lam: Given regularization parameter as used in the paper for the used Dataset
        if self.W_ls is None:
            inv_matrix = np.linalg.inv(np.dot(X_train, X_train.T)+self.lam*np.eye(X_train.shape[0]))
            self.W_ls = np.dot(np.dot(Y_train, X_train.T), inv_matrix).astype(np.float32)
        else:
            return

    # Constructs a single layer of the network
    def construct_one_layer(self,  Y_train, Q, X_train = None, calculate_O = True, dec_flag = False, R_i = None):
        
        # Input arguments
        # Y_train : Training set targets
        # Q : number of output classes
        # X_train : Training set inputs, if None, it is using hidden activation inputs
        # calculate_O : Flag to check whether O is required to be calculated or not

        num_class = Y_train.shape[0] # Number of classes in the given network
        num_node = 2*num_class + 100 # Number of nodes in every layer (fixed in this case)
            
        if ((self.pln[0] is None) and (X_train is not None)):    
            # Construct the first layer
            layer_no = 0 # Layer Number/Index (0 to L-1)
            # Construct the first layer.
            self.pln[0] = PLN(Q, X_train, layer_no, num_node) 
            # Compute the top part of the Composite Weight Matrix
            W_top = np.dot(np.dot(self.pln[0].V_Q, self.W_ls), X_train)
            # Compute the Bottom part of the Composite Weight Matrix and inner product with input, along with including normalization
            if dec_flag == True:
                self.pln[0].R_l = R_i
            W_bottom = self.pln[0].normalization(np.dot(self.pln[0].R_l, X_train)) # Normalization performed is for the random matrix
            # Concatenating the outputs to form W*X
            pln_l1_Z_l = np.concatenate((W_top, W_bottom), axis=0)
            # Then applying the activation function g(.)
            self.pln[0].Y_l = self.pln[0].activation_function(pln_l1_Z_l)
            # Computing the Output Matrix by using 100 iterations of ADMM
            #print("ADMM for Layer No:{}".format(1))
            if calculate_O:
                self.pln[0].O_l = compute_ol(self.pln[0].Y_l, Y_train, self.mu, self.max_iterations, [], False)
        else:
            flag = True # mark whether all layers has been constructed
            for i in range(1,self.num_layer):
                if (self.pln[i] is None):
                    num_layers = i
                    flag = False
                    break
        # if we have 3 layers already existed self.pln[0:2], this index is 3, self.pln[3] is the 4th layer we wanna construct.
            if flag:
                print("All layers already constructed")
                return

            X = self.pln[num_layers-1].Y_l # Input is the Output g(WX) for the previous layer
            num_node = 2*Q + 100 # No. of nodes fixed for every layer
            self.pln[num_layers] = PLN(Q, X, num_layers, num_node) # Creating the PLN Object for the new layer
            # Compute the top part of the Composite Weight Matrix
            W_top = np.dot(np.dot(self.pln[num_layers].V_Q,self.pln[num_layers-1].O_l), X)
            # Compute the bottom part of the Composite Weight Matrix
            if dec_flag == True:
                self.pln[num_layers].R_l = R_i
            W_bottom = self.pln[num_layers].normalization(np.dot(self.pln[num_layers].R_l, X))
            # Concatenate the top and bottom part of the matrix
            pln_Z_l = np.concatenate((W_top, W_bottom), axis=0)
            # Apply the activation function
            self.pln[num_layers].Y_l = self.pln[num_layers].activation_function(pln_Z_l)
            # Compute the output matrix using ADMM for specified no. of iterations
            if calculate_O:
                self.pln[num_layers].O_l = compute_ol(self.pln[num_layers].Y_l, Y_train, self.mu, self.max_iterations, [], False)
        
    # Q is the dimension of the target variable.
    def construct_all_layers(self, X_train, Y_train, Q):   
        for _ in range(0, self.num_layer):
            #TODO: Some chabge might be needed
            # Construct the layers one by one, starting from the layer - 1 to the final layer
            #self.construct_one_layer(X_train, Y_train, Q)
            self.construct_one_layer(Y_train, Q, X_train=X_train)

    # This function is used to compute the test outputs for a given test set    
    def compute_test_outputs(self, X_test):
        
        num_layers = self.num_layer
        W_ls = self.W_ls
        PLN_1 = self.pln[0]
        if PLN_1 is not None:
            # Computes the network output for the first layer
            W_initial_top = np.dot(np.dot(PLN_1.V_Q, W_ls), X_test)
            W_initial_bottom = PLN_1.normalization(np.dot(PLN_1.R_l, X_test))
            Z = np.concatenate((W_initial_top, W_initial_bottom), axis=0)
            y = PLN_1.activation_function(Z)
        elif W_ls is not None:
            # In case only the layer 0 is present, then only the test output after least sqaures is computed
            return  np.dot(W_ls, X_test) 
        else:
            # In case no layer is computed
            print("W_ls not calculated!")
            return None
    
        # Computes the network output for each layer after the first layer
        # modify
        for i in range(1, num_layers):
            PLN_1 = self.pln[i] # Gets the PLN object for the zeroth layer
            if PLN_1 is not None:
                #TODO: y is not changed, so performance won't improve
                W_top = np.dot(np.dot(PLN_1.V_Q, self.pln[i-1].O_l), y)
                W_bottom = PLN_1.normalization(np.dot(PLN_1.R_l, y))
                Z = np.concatenate((W_top, W_bottom), axis=0)
                y = PLN_1.activation_function(Z)
            else:
                return np.dot(self.pln[i - 1].O_l, y)
        #print("I terminate at ", i)
        # Returns network output for the last layer
        return np.dot(self.pln[num_layers - 1].O_l, y)

# flag: whether implementing LwF or Not.
def compute_ol(Y,T,mu, max_iterations, O_prev, flag):

    # Computes the Output matrix by calling the ADMM Algorithm function with given parameters
    if flag:
        # rho is ADMM multiplier for LwF constraint
        rho = 100
        ol = admm_sameset(T, Y, mu, max_iterations, O_prev, rho)
        return ol
    # If LwF is not required, it is only sufficient to optimize a simple ADMM
    ol = optimize_admm(T, Y, mu, max_iterations)
    return ol

