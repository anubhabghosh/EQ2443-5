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
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
import scipy.io as sio
from PLN_Class import PLN
from Admm import optimize_admm, admm_sameset
from LoadDataFromMat import importData
import numpy as np
import random
import copy

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

def compute_ol(Y,T,mu, max_iterations, O_prev=None, flag=False):

    # Computes the Output matrix by calling the ADMM Algorithm function with given parameters
    if flag:
        rho = copy.deepcopy(mu)
        ol = admm_sameset(T, Y, mu, max_iterations, O_prev, rho)
        return ol
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
'''
def train_O1(dataset):
    # print(X_train)
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    #
    # # X_train1 = X_train[:, 1:264]
    # # Y_train1 = Y_train[:, 1:264]
    # # X_test1 = X_test[:, 1:231]
    # # Y_test1 = Y_test[:, 1:231]
    # X_train2 = X_train[:, 265::]
    # Y_train2 = Y_train[:, 265::]
    # X_test2 = X_test[:, 232::]
    # Y_test2 = Y_test[:, 232::]

    dataset_path = "/Users/litingyi/Desktop/EQ2443-5-master/" # Specify the dataset path in the Local without the name
    dataset_name = "Vowel" # Specify the Dataset name without extension (implicitly .mat extension is assumed)
    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path, dataset_name) # Imports the data with the no. of output classes

    X_train = X_train[:, 1:264]
    Y_train = Y_train[:, 1:264]
    X_test = X_test[:, 1:231]
    Y_test = Y_test[:, 1:231]

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
    pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu, max_iterations, [], False)

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
        pln.O_l = compute_ol(pln.Y_l, Y_train, mu, max_iterations, [], False)

        # Add the new layer to the 'Network' list
        PLN_objects.append(pln)

        # Compute training, test accuracy, NME for new appended networks
        predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, i+1, X_test)
        predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, i+1, X_train)
        print("Training Accuracy:{}\n".format(compute_accuracy(predicted_lbl_train, Y_train)))
        print("Test Accuracy:{}\n".format(compute_accuracy(predicted_lbl_test, Y_test)))
        print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
        print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test)))

    predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test)
    print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))
    return PLN_objects


def train_O2(PLN_objects):
    # print(X_train)
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    #
    # # X_train1 = X_train[:, 1:264]
    # # Y_train1 = Y_train[:, 1:264]
    # # X_test1 = X_test[:, 1:231]
    # # Y_test1 = Y_test[:, 1:231]
    # X_train2 = X_train[:, 265::]
    # Y_train2 = Y_train[:, 265::]
    # X_test2 = X_test[:, 232::]
    # Y_test2 = Y_test[:, 232::]

    dataset_path = "/Users/litingyi/Desktop/EQ2443-5-master/" # Specify the dataset path in the Local without the name
    dataset_name = "Vowel" # Specify the Dataset name without extension (implicitly .mat extension is assumed)
    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path, dataset_name) # Imports the data with the no. of output classes

    X_train = X_train[:, 265::]
    Y_train = Y_train[:, 265::]
    X_test = X_test[:, 232::]
    Y_test = Y_test[:, 232::]

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

    # PLN_objects = [] # The network is to be stored as a list of objects, with each object
                     # representing a network layer
    no_layers = 20 # Total number of layers to be used

    # Creating a 1 layer Network

    num_class = Y_train.shape[0] # Number of classes in the given network
    num_node = 2*num_class + 1000 # Number of nodes in every layer (fixed in this case)
    layer_no = 0 # Layer Number/Index (0 to L-1)
    pln_l1 = PLN_objects[0]

    # # Create an object of PLN Class
    # pln_l1 = PLN(Q, X_train, layer_no, num_node, W_ls = Wls)
    #
    # # Compute the top part of the Composite Weight Matrix
    # W_top = np.dot(np.dot(pln_l1 .V_Q, Wls), X_train)
    #
    # # Compute the Bottom part of the Composite Weight Matrix and inner product with input, along with including normalization
    # W_bottom = pln_l1.normalization(np.dot(pln_l1.R_l, X_train)) # Normalization performed is for the random matrix
    #
    # # Concatenating the outputs to form W*X
    # pln_l1_Z_l = np.concatenate((W_top, W_bottom), axis=0)
    #
    # # Then applying the activation function g(.)
    # pln_l1.Y_l = pln_l1.activation_function(pln_l1_Z_l)
    #
    # # Computing the Output Matrix by using 100 iterations of ADMM
    # print("ADMM for Layer No:{}".format(1))
    # pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu, max_iterations)
    #
    # # Appending the creating object for every layer to a list, which constitutes the 'network'
    # PLN_objects.append(pln_l1)

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
        pln = PLN_objects[i] # Creating the PLN Object for the new layer

        # Compute the top part of the Composite Weight Matrix
        W_top = np.dot(np.dot(pln.V_Q,PLN_objects[i-1].O_l), X)

        # Compute the bottom part of the Composite Weight Matrix
        W_bottom = pln.normalization(np.dot(pln.R_l, X))

        # Concatenate the top and bottom part of the matrix
        pln_Z_l = np.concatenate((W_top, W_bottom), axis=0)

        # Apply the activation function
        PLN_objects[i].Y_l = pln.activation_function(pln_Z_l)

        # Compute the output matrix using ADMM for specified no. of iterations
        PLN_objects[i].O_l = compute_ol(pln.Y_l, Y_train, mu, max_iterations, PLN_objects[i-1].O_l, True)

        # Compute training, test accuracy, NME for new appended networks
        predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, i+1, X_test)
        predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, i+1, X_train)
        print("Training Accuracy:{}\n".format(compute_accuracy(predicted_lbl_train, Y_train)))
        print("Test Accuracy:{}\n".format(compute_accuracy(predicted_lbl_test, Y_test)))
        print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
        print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test)))

    predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test)
    print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))
'''

#######################################################################################################
# Function to split the data randomly into two disjoint datasets (both still belong to the same class)
#######################################################################################################
def splitdatarandom(Data, Labels, split_percent=0.5):

    no_of_samples = Data.shape[1]
    list_of_samples = np.arange(no_of_samples) # List of indexes for each sample
    list_of_samples_1 = np.sort(random.sample(list(list_of_samples), int(split_percent*no_of_samples))) # Choose a specific no. of sample indexes for one data
    list_of_samples_2 = list_of_samples[~np.in1d(list_of_samples, list_of_samples_1)] # Choose the other indexes for the second dataset
    
    Data_1 = Data[:,list_of_samples_1]
    Data_2 = Data[:,list_of_samples_2]
    assert Data_1.shape[0] == Data_2.shape[0]

    Labels_1 = Labels[:,list_of_samples_1]
    Labels_2 = Labels[:,list_of_samples_2]
    assert Labels_1.shape[0] == Labels_2.shape[0]

    return Data_1, Data_2, Labels_1, Labels_2

#######################################################################################################
# Function to Plot variation of test accuracy versus hyperparameter
#######################################################################################################
def plot_acc_vs_hyperparam(hyperparam_vec, test_acc_vec, title):
    
    plt.figure()
    plt.plot(hyperparam_vec, test_acc_vec, 'b--', linewidth=1.5, markersize=3)
    plt.xlabel('Hyperparameter Value')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.show()
    return None

def PLN_with_ADMM(X_train, Y_train, X_test, Y_test,  no_layers, max_iterations, lambda_ls, mu, O_prev_array = None, flag = False, W_ls_prev = None, Sweep=False):

    Q = Y_train.shape[0] # No. of classes in the data
    ##########################################################################################
    # Compute Train and Test accuracy for a Regularized Least Squares Algorithm
    ##########################################################################################
    
    if flag == False:
        # No need for LwF
        Wls = compute_Wls(X_train, Y_train, lambda_ls)
        predict_train = np.dot(Wls, X_train)
        predict_test = np.dot(Wls, X_test)
        acc_train = compute_accuracy(predict_train, Y_train)
        acc_test = compute_accuracy(predict_test, Y_test)
        nme_test = compute_NME(predict_test, Y_test)
        print("Train and test accuracies for LS are: {} and {}".format(acc_train, acc_test))
        print("NME Test for LS:{}".format(nme_test))
    
    else:
        # Incorporate LwF into the Least Squares
        if Sweep==True:
            test_acc_vec = []
            mu_vec = np.geomspace(1e-14, 1e14, 29)
            for mu in mu_vec:
                Wls = compute_ol(X_train, Y_train, mu, max_iterations, W_ls_prev, flag)
                predict_train = np.dot(Wls, X_train)
                predict_test = np.dot(Wls, X_test)
                acc_train = compute_accuracy(predict_train, Y_train)
                acc_test = compute_accuracy(predict_test, Y_test)
                nme_test = compute_NME(predict_test, Y_test)
                #print("Train and test accuracies for LS, for mu = {}, are: {} and {}".format(mu, acc_train, acc_test))
                #print("NME Test for LS:{}".format(nme_test))
                test_acc_vec.append(acc_test)
            title = "Plotting test accuracy for Least Squares in Lwf (same dataset)"
            plot_acc_vs_hyperparam(np.log10(mu_vec), test_acc_vec, title)
            mu_Wls_LwF = mu_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
            print("Chosen value of mu and rho for W_ls:{}".format(mu_Wls_LwF))
            Wls = compute_ol(X_train, Y_train, mu_Wls_LwF, max_iterations, W_ls_prev, flag)
    
    ##########################################################################################
    # Creating a list of PLN Objects
    ##########################################################################################
    
    PLN_objects = [] # The network is to be stored as a list of objects, with each object
                      # representing a network layer
    
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
    if flag:
        if Sweep==True:
            test_acc_vec = []
            mu_vec = np.geomspace(1e-14, 1e14, 29)
            for mu in mu_vec:
                pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu, max_iterations, O_prev_array[0].O_l, flag)
                PLN_objects.append(pln_l1) 
                predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, 1, X_test)
                predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, 1, X_train)
                #print("Training Accuracy:{}".format(compute_accuracy(predicted_lbl_train, Y_train)))
                #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test)))
                #print("Traning NME:{}".format(compute_NME(predicted_lbl_train, Y_train)))
                #print("Test NME:{}".format(compute_NME(predicted_lbl_test, Y_test)))
                test_acc_vec.append(compute_accuracy(predicted_lbl_test, Y_test))
                _ = PLN_objects.pop()

            title = "Plotting test accuracy for 1st Layer in Lwf (same dataset)"
            plot_acc_vs_hyperparam(np.log10(mu_vec), test_acc_vec, title)
            mu_Layer1_LwF = mu_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
            print("Chosen value of mu and rho for 1st Layer:{}".format(mu_Layer1_LwF))
            pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu_Layer1_LwF, max_iterations, O_prev_array[0].O_l, flag)
            PLN_objects.append(pln_l1) 
    else:
        pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu, max_iterations)
        # Appending the creating object for every layer to a list, which constitutes the 'network'
        PLN_objects.append(pln_l1) 
        predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, 1, X_test)
        predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, 1, X_train)
        #print("Training Accuracy:{}".format(compute_accuracy(predicted_lbl_train, Y_train)))
        #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test)))
        #print("Traning NME:{}".format(compute_NME(predicted_lbl_train, Y_train)))
        #print("Test NME:{}".format(compute_NME(predicted_lbl_test, Y_test)))
    
    # ADMM Training for all the remaining layers using the training Data
    # Also involves subsequent testing on the test data
    
    if flag:
        if Sweep==True:
            test_acc_vec = []
            mu_vec = np.geomspace(1e-14, 1e14, 29)
            for mu in mu_vec:
                for i in range(1, no_layers):
            
                    print("************** ADMM for Layer No:{}, mu:{} **************".format(i+1, mu))
    
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
                    pln.O_l = compute_ol(pln.Y_l, Y_train, mu, max_iterations, O_prev_array[i].O_l, flag)

                    # Add the new layer to the 'Network' list
                    PLN_objects.append(pln)
    
                    # Compute training, test accuracy, NME for new appended networks
                    #predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, i+1, X_test)
                    #predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, i+1, X_train)
                    #print("Training Accuracy:{}\n".format(compute_accuracy(predicted_lbl_train, Y_train)))
                    #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test)))
                    #print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
                    #print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test)))

                predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test) # For the entire network
                #print("Final Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))

                final_test_acc = compute_accuracy(predicted_lbl, Y_test) # Final Test Accuracy
                final_test_nme = compute_NME(predicted_lbl, Y_test)
                test_acc_vec.append(final_test_acc)

                # Remove the layers upto first layer
                for i in range(1, no_layers):
                    _ = PLN_objects.pop()


            title = "Plotting test accuracy for total network in Lwf (same dataset)"
            plot_acc_vs_hyperparam(np.log10(mu_vec), test_acc_vec, title)
            mu_Layers_LwF = mu_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
            print("Chosen value of mu and rho for total network:{}".format(mu_Layers_LwF))
            
            # Use this value of mu to reform the network
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
                pln.O_l = compute_ol(pln.Y_l, Y_train, mu, max_iterations, O_prev_array[i].O_l, flag)

                # Add the new layer to the 'Network' list
                PLN_objects.append(pln)
    
                # Compute training, test accuracy, NME for new appended networks
                #predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, i+1, X_test)
                #predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, i+1, X_train)
                #print("Training Accuracy:{}\n".format(compute_accuracy(predicted_lbl_train, Y_train)))
                #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test)))
                #print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
                #print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test)))

            predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test) # For the entire network
            #print("Final Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))

            final_test_acc = compute_accuracy(predicted_lbl, Y_test) # Final Test Accuracy
            final_test_nme = compute_NME(predicted_lbl, Y_test)
        
    else:

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
            #predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, i+1, X_test)
            #predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, i+1, X_train)
            #print("Training Accuracy:{}\n".format(compute_accuracy(predicted_lbl_train, Y_train)))
            #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test)))
            #print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
            #print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test)))

        predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test) # For the entire network
        #print("Final Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))

        final_test_acc = compute_accuracy(predicted_lbl, Y_test) # Final Test Accuracy
        final_test_nme = compute_NME(predicted_lbl, Y_test)
    
    return PLN_objects, final_test_acc, final_test_nme, no_layers, Wls

def main():

    ##########################################################################################
    # Dataset related parameters
    dataset_path = "../../Datasets/" # Specify the dataset path in the Local without the name
    dataset_name = "Vowel" # Specify the Dataset name without extension (implicitly .mat extension is assumed)
    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path, dataset_name) # Imports the data with the no. of output classes
    
    mu = 1e3 # For the given dataset
    lambda_ls = 1e2 # Given regularization parameter as used in the paper for the used Dataset
    no_layers = 5 # Total number of layers to be used (fixed by default)
    print("Dataset Used: {}, No. of layers:{}".format(dataset_name, no_layers))
    
    ##########################################################################################
    # Splits the data randomly into two disjoint datasets (by default random_split% = 0.5)
    X_train_1, X_train_2, Y_train_1, Y_train_2 = splitdatarandom(X_train, Y_train) # Dataset T_old: X_train_1, Y_train_1, X_test_1, Y_test_1
    X_test_1, X_test_2, Y_test_1, Y_test_2 = splitdatarandom(X_test, Y_test) # Dataset T_new: X_train_2, Y_train_2, X_test_2, Y_test_2
    print("Data has been split")

    ##########################################################################################
    # Parameters related to ADMM optimization
    max_iterations = 100 # For the ADMM Algorithm
    ##########################################################################################
    
    ##########################################################################################
    # Run ADMM Optimization for the whole dataset to get a baseline
    print("Baseline : Total Dataset")
    PLN_total_dataset, final_test_acc_bl, final_test_nme_bl, PLN_no_layers, Wls_total_dataset = PLN_with_ADMM(X_train, Y_train, X_test, \
                                                                                           Y_test,  no_layers, max_iterations, lambda_ls, mu) 
    predicted_lbl = compute_test_outputs(PLN_total_dataset, Wls_total_dataset, no_layers, X_test) # For the entire network
    print("Final Test Accuracy on T1 + T2:{}".format(compute_accuracy(predicted_lbl, Y_test)))
    
    #for _ in range(10):
    # Run ADMM Optimization for the first half dataset (no LwF)
    print("First Dataset, no LWF")
    PLN_first_dataset, final_test_acc_1, final_test_nme_1, PLN_no_layers_1, Wls_1 = PLN_with_ADMM(X_train_1, Y_train_1, X_test_1, \
                                                                                           Y_test_1, no_layers, max_iterations, lambda_ls, mu) 

    predicted_lbl = compute_test_outputs(PLN_first_dataset, Wls_1, no_layers, X_test_1) # For the 1st half
    print("Final Test Accuracy for T1:{}".format(compute_accuracy(predicted_lbl, Y_test_1)))

    # Run ADMM Optimization for the second half dataset (no LwF)
    print("Second Dataset, no LWF")
    PLN_second_dataset, final_test_acc_2, final_test_nme_2, PLN_no_layers_2, Wls_2 = PLN_with_ADMM(X_train_2, Y_train_2, X_test_2, \
                                                                                        Y_test_2, no_layers, max_iterations, lambda_ls, mu) 
    
    predicted_lbl = compute_test_outputs(PLN_second_dataset, Wls_2, no_layers, X_test_2) # For the 2nd half
    print("Final Test Accuracy for T2:{}".format(compute_accuracy(predicted_lbl, Y_test_2)))

    # Run ADMM Optimization for the second half dataset (no LwF)
    print("Implemeting LWF on Second Dataset, learned First Dataset")
    PLN_LwF_dataset, final_test_acc_LwF, final_test_nme_LwF, PLN_no_layers_LwF, Wls_LWF = PLN_with_ADMM(X_train_2, Y_train_2, X_test_2, Y_test_2,\
                                                            no_layers, max_iterations, lambda_ls, mu, O_prev_array=PLN_first_dataset,\
                                                            flag=True, W_ls_prev=Wls_1, Sweep=True)
    
    predicted_lbl_2 = compute_test_outputs(PLN_LwF_dataset, Wls_LWF, no_layers, X_test_2) # For the 2nd half
    print("Final Test Accuracy for T2:{}".format(compute_accuracy(predicted_lbl_2, Y_test_2)))

    predicted_lbl_1 = compute_test_outputs(PLN_LwF_dataset, Wls_LWF, no_layers, X_test_1) # For the 1st half
    print("Final Test Accuracy for T1:{}".format(compute_accuracy(predicted_lbl_1, Y_test_1)))

    predicted_lbl = compute_test_outputs(PLN_LwF_dataset, Wls_LWF, no_layers, X_test) # For the Complete Dataset
    print("Final Test Accuracy on T1 + T2:{}".format(compute_accuracy(predicted_lbl, Y_test)))

    return None

if __name__ == "__main__":
    main()
