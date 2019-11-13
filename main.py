import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
import scipy.io as sio
from PLN_Class import PLN
from Admm import optimize_admm
from LoadDataFromMat import importData
import numpy as np

# Import the dataset and calculate related parameters

def importDummyExample():
    # Dummy example for checking PLN
    mean = np.array([0,0,0,0,0,0,0,0,0,0])
    mean = mean.T
    #print(mean.shape)
    var = np.eye(10)
    X = np.random.multivariate_normal(mean,var,10)
    print(X.shape)
    #T = np.zeros((10,5))
    T = np.array([[1,1,0,0,0,0,0,0,0,0],
                  [0,0,1,1,0,0,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,0,0,1,1,0,0],
                  [0,0,0,0,0,0,0,0,1,1]])
    print(T.shape)
    Q = 0
    return X, T, Q

# Compute the W_ls by solving a Least Squares Regularization problem
def compute_Wls(X,T,lam):

    # the X are in n*p form, n sample, each sample has p dims. T is n*Q matrix, each sample is a row vector
    inv_matrix = np.linalg.inv(np.dot(X, X.T)+lam*np.eye(X.shape[0]))
    W_ls = np.dot(np.dot(T, X.T), inv_matrix).astype(np.float32)
    return W_ls

def compute_ol(Y,T,mu, max_iterations):

    # Computes the Output matrix by calling the ADMM Algorithm function with given parameters
    ol = optimize_admm(T, Y, mu, max_iterations)
    return ol

def compute_accuracy(predicted_lbl, true_lbl):

    # Computes a Classification Accuracy between true label
    acc = 100.*np.mean(np.argmax(predicted_lbl,axis=0)==np.argmax(true_lbl,axis=0))
    return acc

def compute_test_outputs(PLN_object_array, W_ls, num_layers, X_test):

    # Computes the network output for the first layer
    PLN_1 = PLN_object_array[0]
    W_initial_top = np.dot(np.dot(PLN_1.V_Q, W_ls), X_test)
    W_initial_bottom = PLN_1.normalization(np.dot(PLN_1.R_l, X_test))
    Z = np.concatenate((W_initial_top, W_initial_bottom), axis=0)
    y = PLN_1.activation_function(Z)

    # Computes the network output for each layer after the first layer
    for i in range(1, num_layers):
        PLN_l = PLN_object_array[i]
        W_top = np.dot(np.dot(PLN_l.V_Q, PLN_object_array[i-1].O_l), y)
        W_bottom = PLN_l.normalization(np.dot(PLN_l.R_l, y))
        Z = np.concatenate((W_top, W_bottom), axis=0)
        y = PLN_l.activation_function(Z)

    # Returns network output for the last layer
    return np.dot(PLN_object_array[num_layers - 1].O_l, y)

def compute_NME(predicted_lbl, actual_lbl):

    # This function computes the Normalized Mean Error (NME) in dB scale

    num = np.linalg.norm(actual_lbl - predicted_lbl, ord='fro') # Frobenius Norm of the difference between Predicted and True Label
    den = np.linalg.norm(actual_lbl, ord='fro') # Frobenius Norm of the True Label
    NME = 20 * np.log10(num / den)
    return NME

def main():

    ##########################################################################################
    # Dataset related parameters
    dataset_path = "../Datasets/" # Specify the dataset path in the Local without the name
    dataset_name = "Vowel" # Specify the Dataset name without extension (implicitly .mat extension is assumed)
    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path, dataset_name) # Imports the data with the no. of output classes
    
    mu = 1e3 # For the given dataset
    lambda_ls = 1e2 # Given regularization parameter as used in the paper for the used Dataset
    ##########################################################################################

    ##########################################################################################
    # Parameters related to ADMM optimization
    max_iterations = 100 # For the ADMM Algorithm
    ##########################################################################################

    ##########################################################################################
    # Compute Train and Test accuracy for a Regularized Least Squares Algorithm
    ##########################################################################################
    Wls = compute_Wls(X_train, Y_train, lambda_ls) 
    predict_train = np.dot(Wls, X_train)
    predict_test = np.dot(Wls, X_test)
    acc_train = compute_accuracy(predict_train, Y_train)
    acc_test = compute_accuracy(predict_test, Y_test)
    nme_test = compute_NME(predict_test, Y_test)
    print("Train and test accuracies are: {} and {}".format(acc_train, acc_test))
    print("NME Test:{}".format(nme_test))

    ##########################################################################################
    # Creating a list of PLN Objects
    ##########################################################################################
    
    PLN_objects = [] # The network is to be stored as a list of objects, with each object 
                     # representing a network layer
    no_layers = 20 # Total number of layers to be used

    # Creating a 1 layer Network
    
    num_class = Y_train.shape[0] # Number of classes in the given network
    num_node = 2*num_class + 1000 # Number of nodes in every layer (fixed in this case)
    layer_no = 0 # Layer Number/Index (0 to L-1)

    # Create an object of PLN Class
    pln_l1 = PLN(Q, X_train, layer_no, num_node, W_ls = Wls) 
    
    # Compute the top part of the Composite Weight Matrix
    W_top = np.dot(np.dot(pln_l1 .V_Q, Wls), X_train)

    # Compute the Bottom part of the Composite Weight Matrix and inner product with input, along with including normalization
    W_bottom = pln_l1.normalization(np.dot(pln_l1.R_l, X_train)) # Normalization performed is for the random matrix

    # Concatenating the outputs to form W*X 
    pln_l1_Z_l = np.concatenate((W_top, W_bottom), axis=0)

    # Then applying the activation function g(.)
    pln_l1.Y_l = pln_l1.activation_function(pln_l1_Z_l) 

    # Computing the Output Matrix by using 100 iterations of ADMM
    print("ADMM for Layer No:{}".format(1))
    pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu, max_iterations)

    # Appending the creating object for every layer to a list, which constitutes the 'network' 
    PLN_objects.append(pln_l1)
    
    predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, 1, X_test)
    predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, 1, X_train)
    print("Training Accuracy:{}".format(compute_accuracy(predicted_lbl_train, Y_train)))
    print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test)))
    print("Traning NME:{}".format(compute_NME(predicted_lbl_train, Y_train)))
    print("Test NME:{}".format(compute_NME(predicted_lbl_test, Y_test)))

    # ADMM Training for all the remaining layers using the training Data
    # Also involves subsequent testing on the test data
    
    for i in range(1, no_layers):

        print("************** ADMM for Layer No:{} **************".format(i+1))

        X = PLN_objects[i-1].Y_l # Input is the Output g(WX) for the previous layer
        num_node = 2*Q + 1000 # No. of nodes fixed for every layer
        pln = PLN(Q, X, i, num_node, W_ls=None) # Creating the PLN Object for the new layer

        # Compute the top part of the Composite Weight Matrix
        W_top = np.dot(np.dot(pln.V_Q,PLN_objects[i-1].O_l), X)

        # Compute the bottom part of the Composite Weight Matrix
        W_bottom = pln.normalization(np.dot(pln.R_l, X))

        # Concatenate the top and bottom part of the matrix
        pln_Z_l = np.concatenate((W_top, W_bottom), axis=0)

        # Apply the activation function
        pln.Y_l = pln.activation_function(pln_Z_l) 

        # Compute the output matrix using ADMM for specified no. of iterations
        pln.O_l = compute_ol(pln.Y_l, Y_train, mu, max_iterations)

        # Add the new layer to the 'Network' list
        PLN_objects.append(pln)

        # Compute training, test accuracy, NME for new appended networks
        predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, i+1, X_test)
        predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, i+1, X_train)
        print("Training Accuracy:{}\n".format(compute_accuracy(predicted_lbl_train, Y_train)))
        print("Test Accuracy:{}\n".format(compute_accuracy(predicted_lbl_test, Y_test)))
        print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
        print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test)))
        
    #predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test)
    #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))
    return None

if __name__ == "__main__":
    main() 