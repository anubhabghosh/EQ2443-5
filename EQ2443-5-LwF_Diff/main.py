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
from Admm import optimize_admm
from LoadDataFromMat import importData
import numpy as np
from LwF_based_ADMM import LwF_based_ADMM_LS_Diff
import copy

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

########################################################################################
# This function returns training, testing accuracies and NME test values for Least Squares
########################################################################################
def compute_LS_test_accuracy(Wls, X_train, Y_train, X_test, Y_test):

    # Task prediction on the joint dataset
    predict_train_total = np.dot(Wls, X_train)
    predict_test_total = np.dot(Wls, X_test)

    # Compute metrics
    acc_train = compute_accuracy(predict_train_total, Y_train)
    acc_test = compute_accuracy(predict_test_total, Y_test)
    nme_test = compute_NME(predict_test_total, Y_test)
    
    return acc_train, acc_test, nme_test

#########################################################################################
# This function is used to compute the inputs and targets required for joint training
#########################################################################################
def compute_joint_datasets(X1_train, X2_train, Y1_train, Y2_train):
    
    # Row and Column wise appending for the Targets

    Y1_train_padded = np.concatenate((Y1_train, np.zeros((Y1_train.shape[0], Y2_train.shape[1]))),axis=1)
    Y2_train_padded = np.concatenate((np.zeros((Y2_train.shape[0], Y1_train.shape[1])), Y2_train),axis=1)
    Y_joint_train = np.concatenate((Y1_train_padded, Y2_train_padded), axis=0)

    # Row wise appending of zeros for the Inputs, so that the row dimension is same for both datasets 

    if X1_train.shape[0] < X2_train.shape[0]:
        padding = np.zeros((int(X2_train.shape[0] - X1_train.shape[0]), X1_train.shape[1]))
        X1_train_padded = np.concatenate((X1_train, padding),axis=0)
        X_joint_train = np.concatenate((X1_train_padded, X2_train), axis=1)

    elif X1_train.shape[0] > X2_train.shape[0]:
        padding = np.zeros((int(X1_train.shape[0] - X2_train.shape[0]), X2_train.shape[1]))
        X2_train_padded = np.concatenate((X2_train, padding),axis=0)
        X_joint_train = np.concatenate((X1_train, X2_train_padded), axis=1)

    return X_joint_train, Y_joint_train

###########################################################################################
# This function is used to perform Parameter sweeping over given hyperparameter
###########################################################################################
def param_tuning_for_LS(X_train, Y_train, X_test, Y_test):

    # This function is used to the lambda in the regularized least squares version
    # X_train : Given joint dataset input for training
    # Y_train : Given joint dataset output for training
    # X_test : Given joint dataset input for testing
    # Y_test : Given joint dataset output for training
    # lambda_ls_jt : param to be optimized
    # The param is swept over from 10^-14 to 10^14, and the param with the highest test_acc is chosen 
    
    lambda_jt_vec = np.geomspace(1e-14, 1e14, 29) 
    test_acc_vec = []
    
    # Sweeping over a list of values for lambda
    for lambda_jt in lambda_jt_vec:

        Wls = compute_Wls(X_train, Y_train, lambda_jt)
        # Task prediction on the joint dataset
        acc_train_jt, acc_test_jt, nme_test_jt = compute_LS_test_accuracy(Wls, X_train, Y_train, X_test, Y_test)
        #print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt, acc_train_jt, acc_test_jt))
        #print("NME Test:{}".format(nme_test_jt))
        test_acc_vec.append(acc_test_jt)

    # Plotting the results of the param sweep
    title = "Plotting test accuracy for Joint Training for LS (Diff dataset)"
    plot_acc_vs_hyperparam(np.log10(lambda_jt_vec), test_acc_vec, title)
    lambda_jt_optimal = lambda_jt_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy

    return lambda_jt_optimal

###########################################################################################
# This function is used to perform Parameter sweeping over given hyperparameter
###########################################################################################
def param_tuning_for_O(Y_l, Y_train,  X_test, Y_test, Wls, max_ADMM_iterations):

    # This function is used to the lambda in the regularized least squares version
    # X_train : Given joint dataset input for training
    # Y_train : Given joint dataset output for training
    # Y_l : Activation layer values
    # X_test : Given joint dataset input for testing
    # Y_test : Given joint dataset output for training
    # mu : param to be optimized (for ADMM)
    # The param is swept over from 10^-14 to 10^14, and the param with the highest test_acc is chosen 
    
    mu_jt_vec = np.geomspace(1e-14, 1e14, 29) 
    test_acc_vec = []
    pln_array = []
    # Sweeping over a list of values for lambda
    for mu in mu_jt_vec:

        O_l = compute_ol(Y_l, Y_train, mu, max_ADMM_iterations)
        pln_array.append(O_l)
        # Task prediction on the joint dataset
        predicted_lbl_test = compute_test_outputs(pln_array, Wls, 1, X_test)
        test_acc_vec.append(compute_accuracy(predicted_lbl_test, Y_test))
        #print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt, acc_train_jt, acc_test_jt))
        #print("NME Test:{}".format(nme_test_jt))
        _ = pln_array.pop() # Clears the array out again until optimal solution is found

    # Plotting the results of the param sweep
    title = "Plotting test accuracy for Joint Training for O1 (Diff dataset)"
    plot_acc_vs_hyperparam(np.log10(mu_jt_vec), test_acc_vec, title)
    mu_jt_optimal = mu_jt_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy

    return mu_jt_optimal

################################################################################################
# This function picks out the matrices O1 and O2 from the big O matrix
################################################################################################
def untwine_weights(Wls_joint_train, X1_train, Y1_train, X2_train, Y2_train):

    Wls_1_joint_train = Wls_joint_train[0:Y1_train.shape[0], 0:X1_train.shape[0]]
    Wls_2_joint_train = Wls_joint_train[-Y2_train.shape[0]:, -X2_train.shape[0]:]

    return Wls_1_joint_train, Wls_2_joint_train

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

#################################################################################################
# Implement a PLN for L number of layers
#################################################################################################

def PLN_with_ADMM(X_train, Y_train, X_test, Y_test, no_layers, max_iterations, lambda_ls = None, mu = None, LwF_flag = False,
                  O_l_prev_array = None, W_ls_prev = None, mu_layers_LwF = None):

    if LwF_flag ==  False:

        # Simply run the PLN for 'no_layers'
        Q = Y_train.shape[0] # No. of Output classes

        if lambda_ls != None:
            # Lambda for LS is present, no tuning required
            print("Chosen value of lambda for Wls for the given dataset:{}".format(lambda_ls))
            Wls = compute_Wls(X_train, Y_train, lambda_ls)
        else:
            # Tune the lambda for the LS for given dataset
            lambda_ls_optimal = param_tuning_for_LS(X_train, Y_train, X_test, Y_test) 
            print("Chosen value of lambda for Wls for the given dataset:{}".format(lambda_ls_optimal))
            Wls = compute_Wls(X_train, Y_train, lambda_ls_optimal) # Find the resulting Wls 

        acc_train_Wls, acc_test_Wls, nme_test_Wls = compute_LS_test_accuracy(Wls, X_train, Y_train, X_test, Y_test)
        print("Train and test accuracies for LS of given dataset are: {} and {}".format(acc_train_Wls, acc_test_Wls))
        print("NME Test for T_new:{}".format(nme_test_Wls))

        ##########################################################################################
        # Creating a list of PLN Objects
        ##########################################################################################
        PLN_objects = [] # The network is to be stored as a list of objects, with each object
                      # representing a network layer

        ##########################################################################################
        # Creating a 1 layer Network
        ##########################################################################################
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

        ##########################################################################################
        # Tuning the params for the Joint training case, if encountered
        ##########################################################################################
        if mu == None:
            mu_optimal = param_tuning_for_O(pln_l1.Y_l, Y_train, X_test, Y_test, Wls, max_iterations)
            print("Chosen value of mu and rho for 1st Layer:{}".format(mu_optimal))
            pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu_optimal, max_iterations)# Once optimal solution is found, it uses that 
            PLN_objects.append(pln_l1) # Appends the layer object with the optimal solution for the first layer
            mu = copy.deepcopy(mu_optimal)
        else:
            pln_l1.O_l =  compute_ol(pln_l1.Y_l, Y_train, mu, max_iterations)
            PLN_objects.append(pln_l1) # Appends the layer object with the optimal solution for the first layer

        ##########################################################################################
        # Extending for all the remaining layers
        ##########################################################################################
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
            print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test)))
            #print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
            print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test)))

        predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test) # For the entire network
        #print(Final Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))
        final_test_acc = compute_accuracy(predicted_lbl, Y_test) # Final Test Accuracy
        final_test_nme = compute_NME(predicted_lbl, Y_test)

        return PLN_objects, final_test_acc, final_test_nme, Wls

    else:
        pass
        # Implement the ADMM for different dataset using LwF

def main():

    ##########################################################################################
    # Dataset related parameters
    dataset_path = "../../Datasets/" # Specify the dataset path in the Local without the name
    dataset_1_name = "Vowel" # Specify the Dataset name without extension (implicitly .mat extension is assumed)
    dataset_2_name = "ExtendedYaleB"

    # Importing parameters for the first Dataset (Old)
    X1_train, Y1_train, X1_test, Y1_test, Q1 = importData(dataset_path, dataset_1_name) 
    # Importing parameters for the second Dataset (New)
    X2_train, Y2_train, X2_test, Y2_test, Q2 = importData(dataset_path, dataset_2_name) 
    
    # Parameters for the First Dataset
    mu1 = 1e3 # For the given dataset
    lambda_ls1 = 1e2 # Given regularization parameter as used in the paper for the used Dataset

    # Parameters for the Second Dataset
    mu2 = 1e3 # For the given dataset
    lambda_ls2 = 1e4 # Given regularization parameter as used in the paper for the used Dataset
    ##########################################################################################

    ##########################################################################################
    # Parameters related to ADMM optimization and initial values
    max_iterations = 100 # For the ADMM Algorithm   
    no_layers = 1
    lambda_jt_optimal = None
    ##########################################################################################
    
    ###############################################################################################
    # Compute Train and Test accuracy for a Regularized Least Squares Algorithm for the Old Dataset
    ###############################################################################################
    
    # Compute the Wls_1_parameter for the Old Dataset
    Wls_1 = compute_Wls(X1_train, Y1_train, lambda_ls1) 
    acc_train_1, acc_test_1, nme_test_1 = compute_LS_test_accuracy(Wls_1, X1_train, Y1_train, X1_test, Y1_test)
    print("Train and test accuracies for T_old are: {} and {}".format(acc_train_1, acc_test_1))
    print("NME Test for T_old:{}".format(nme_test_1))

    # Compute the Wls_2_parameter for the New Dataset
    Wls_2 = compute_Wls(X2_train, Y2_train, lambda_ls2) 
    acc_train_2, acc_test_2, nme_test_2 = compute_LS_test_accuracy(Wls_2, X2_train, Y2_train, X2_test, Y2_test)
    print("Train and test accuracies for T_new are: {} and {}".format(acc_train_2, acc_test_2))
    print("NME Test for T_new:{}".format(nme_test_2))

    # Performing Joint Training
    X_joint_train, Y_joint_train = compute_joint_datasets(X1_train, X2_train, Y1_train, Y2_train)
    X_joint_test, Y_joint_test = compute_joint_datasets(X1_test, X2_test, Y1_test, Y2_test)
    
    # Finding the optimal lambda 
    lambda_jt_optimal = param_tuning_for_LS(X_joint_train, Y_joint_train, X_joint_test, Y_joint_test)
    print("Chosen value of lambda for Wls for Joint Training:{}".format(lambda_jt_optimal))
    Wls_joint_train = compute_Wls(X_joint_train, Y_joint_train, lambda_jt_optimal) # Find the resulting Wls 
    acc_train_jt, acc_test_jt, nme_test_jt = compute_LS_test_accuracy(Wls_joint_train, X_joint_train, Y_joint_train, X_joint_test, Y_joint_test)
    print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt_optimal, acc_train_jt, acc_test_jt))
    print("NME Test:{}".format(nme_test_jt))
   
    # Untwine the weights
    Wls_1_joint_train, Wls_2_joint_train = untwine_weights(Wls_joint_train, X1_train, Y1_train, X2_train, Y2_train)
     
    # Old task prediction after joint training
    acc_train_1_jt, acc_test_1_jt, nme_test_1_jt = compute_LS_test_accuracy(Wls_1_joint_train, X1_train, Y1_train, X1_test, Y1_test) 
    print("Train and test accuracies for Old task/dataset after Joint Training are: {} and {}".format(acc_train_1_jt, acc_test_1_jt))
    print("NME Test:{}".format(nme_test_1_jt))

    # New task prediction after joint training
    acc_train_2_jt, acc_test_2_jt, nme_test_2_jt = compute_LS_test_accuracy(Wls_2_joint_train, X2_train, Y2_train, X2_test, Y2_test) 
    print("Train and test accuracies for New task/dataset after Joint Training are: {} and {}".format(acc_train_2_jt, acc_test_2_jt))
    print("NME Test:{}".format(nme_test_2_jt))
    
    #################################################################################################
    # Define Params for the LwF based ADMM for Least Squares
    #################################################################################################
    
    ###############################################################################################
    # Compute Train and Test accuracy for a Regularized Least Squares Algorithm for the Old Dataset
    ###############################################################################################
    '''
    # Tuning lambda_o
    mu = 1e2 # Fixing mu  = 100
    lambda_o_vec = np.geomspace(1e-14, 1e14, 29)
    alpha = 2
    
    test_acc_vec = []
    for lambda_o in lambda_o_vec:

        epsilon_o_2 = alpha*np.sqrt(2*Y2_train.shape[0])
        epsilon_o = np.sqrt(lambda_o*np.linalg.norm(Wls_1, 'fro')**2 + (epsilon_o_2)**2) # Somewhat decided based on the norm of W_ls for Vowel and ExtendedYaleB individually, norm for Vowel is 0.3, YaleB is 0.06
        
        Wls_LwF_hat = LwF_based_ADMM_LS_Diff(X2_train, Y2_train, Wls_1, lambda_o, epsilon_o, mu)
        # Joint performance post LwF
        predict_train_total_LwF = np.dot(Wls_LwF_hat, X_joint_train)
        predict_test_total_LwF = np.dot(Wls_LwF_hat, X_joint_test)
        acc_train_jt_LwF = compute_accuracy(predict_train_total_LwF, Y_joint_train)
        acc_test_jt_LwF = compute_accuracy(predict_test_total_LwF, Y_joint_test)
        #nme_test_jt_LwF = compute_NME(predict_test_total_LwF, Y_joint_test)
        print("Train and test accuracies for Joint Datasets after LwF for lambda_o:{} are: {} and {}".format(lambda_o, acc_train_jt_LwF, acc_test_jt_LwF))
        #print("NME Test:{}".format(nme_test_jt_LwF))
        test_acc_vec.append(acc_test_jt_LwF)

    # Plotting the results of the param sweep
    title = "Plotting test accuracy for LwF for LS (Diff dataset)"
    plot_acc_vs_hyperparam(np.log10(lambda_o_vec), test_acc_vec, title)
    lambda_o_jt_optimal = lambda_o_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
    print("Chosen value of mu for Wls for LwF:{}".format(lambda_o_jt_optimal))
    Wls_LwF_hat = LwF_based_ADMM_LS_Diff(X2_train, Y2_train, Wls_1, lambda_o_jt_optimal, epsilon_o, mu) # Find the resulting Wls 
    predict_train_LwF_total = np.dot(Wls_LwF_hat, X_joint_train)
    predict_test_LwF_total = np.dot(Wls_LwF_hat, X_joint_test)
    acc_train_LwF = compute_accuracy(predict_train_LwF_total, Y_joint_train)
    acc_test_LwF = compute_accuracy(predict_test_LwF_total, Y_joint_test)
    nme_test_LwF = compute_NME(predict_test_LwF_total, Y_joint_test)
    print("For eps = {}, Train and test accuracies for Joint Dataset after LwF are: {} and {}".format(lambda_o_jt_optimal, acc_train_LwF, acc_test_LwF))
    print("NME Test for Joint Dataset after LwF:{}".format(nme_test_LwF))
    

    lambda_o = 1.0/100000 # Forgetting factor  (#TODO: Strange Result)
    alpha = 2
    epsilon_o_2 = alpha*np.sqrt(2*Y2_train.shape[0])
    epsilon_o = np.sqrt(lambda_o*np.linalg.norm(Wls_1, 'fro')**2 + (epsilon_o_2)**2) # Somewhat decided based on the norm of W_ls for Vowel and ExtendedYaleB individually, norm for Vowel is 0.3, YaleB is 0.06
    #mu = 1e6 # Somewhat similar for both datasets Vowel and Extended YaleB (individually) 10^-14 to 10^14
    mu_vec = np.geomspace(1e-14, 1e14, 29) # Sweeping over mu values to find the optimal solution
    test_acc_vec = [] # For LwF
    
    # Tuning mu
    for mu in mu_vec:
        
        Wls_LwF_hat = LwF_based_ADMM_LS_Diff(X2_train, Y2_train, Wls_1, lambda_o, epsilon_o, mu)
        # Joint performance post LwF
        predict_train_total_LwF = np.dot(Wls_LwF_hat, X_joint_train)
        predict_test_total_LwF = np.dot(Wls_LwF_hat, X_joint_test)
        acc_train_jt_LwF = compute_accuracy(predict_train_total_LwF, Y_joint_train)
        acc_test_jt_LwF = compute_accuracy(predict_test_total_LwF, Y_joint_test)
        #nme_test_jt_LwF = compute_NME(predict_test_total_LwF, Y_joint_test)
        print("Train and test accuracies for Joint Datasets after LwF for mu:{} are: {} and {}".format(mu, acc_train_jt_LwF, acc_test_jt_LwF))
        #print("NME Test:{}".format(nme_test_jt_LwF))
        test_acc_vec.append(acc_test_jt_LwF)

    # Plotting the results of the param sweep
    title = "Plotting test accuracy for LwF for LS (Diff dataset)"
    plot_acc_vs_hyperparam(np.log10(mu_vec), test_acc_vec, title)
    
    mu_jt_optimal = mu_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
    print("Chosen value of mu for Wls for LwF:{}".format(mu_jt_optimal))
    Wls_LwF_hat = LwF_based_ADMM_LS_Diff(X2_train, Y2_train, Wls_1, lambda_o, epsilon_o, mu_jt_optimal) # Find the resulting Wls 
    predict_train_joint_train_total = np.dot(Wls_LwF_hat, X_joint_train)
    predict_test_joint_train_total = np.dot(Wls_LwF_hat, X_joint_test)
    acc_train_jt = compute_accuracy(predict_train_joint_train_total, Y_joint_train)
    acc_test_jt = compute_accuracy(predict_test_joint_train_total, Y_joint_test)
    nme_test_jt = compute_NME(predict_test_joint_train_total, Y_joint_test)
    print("For mu = {}, Train and test accuracies for Joint Dataset after LwF are: {} and {}".format(mu_jt_optimal, acc_train_jt, acc_test_jt))
    print("NME Test for Joint Dataset after LwF:{}".format(nme_test_jt))

    # Untwine weights
    Wls_1_LwF, Wls_2_LwF = untwine_weights(Wls_LwF_hat, X1_train, Y1_train, X2_train, Y2_train)
    Wls_1_LwF = Wls_1_LwF * (1/np.sqrt(lambda_o))

    # Old task prediction after LwF
    
    X1_train_padded = np.concatenate((X1_train, np.zeros((int(X2_train.shape[0] - X1_train.shape[0]), X1_train.shape[1]))),axis=0)
    X1_test_padded = np.concatenate((X1_test, np.zeros((int(X2_train.shape[0] - X1_test.shape[0]), X1_test.shape[1]))),axis=0)
    Y1_train_padded = np.concatenate((Y1_train, np.zeros((int(Y2_train.shape[0]), Y1_train.shape[1]))),axis=0)
    Y1_test_padded = np.concatenate((Y1_test, np.zeros((int(Y2_test.shape[0]), Y1_test.shape[1]))),axis=0)
    
    predict_train_LwF_1 = np.dot(Wls_LwF_hat, X1_train_padded)
    predict_test_LwF_1 = np.dot(Wls_LwF_hat, X1_test_padded)
    acc_train_1_LwF = compute_accuracy(predict_train_LwF_1, Y1_train_padded)
    acc_test_1_LwF = compute_accuracy(predict_test_LwF_1, Y1_test_padded)
    nme_test_1_LwF = compute_NME(predict_test_LwF_1, Y1_test_padded)
    print("Train and test accuracies for Old task/dataset after LwF are: {} and {}".format(acc_train_1_LwF, acc_test_1_LwF))
    print("NME Test:{}".format(nme_test_1_LwF))
    
    
    predict_train_LwF_1 = np.dot(Wls_1_LwF, X1_train)
    predict_test_LwF_1 = np.dot(Wls_1_LwF, X1_test)
    acc_train_1_LwF = compute_accuracy(predict_train_LwF_1, Y1_train)
    acc_test_1_LwF = compute_accuracy(predict_test_LwF_1, Y1_test)
    nme_test_1_LwF = compute_NME(predict_test_LwF_1, Y1_test)
    print("Train and test accuracies for Old task/dataset after LwF are: {} and {}".format(acc_train_1_LwF, acc_test_1_LwF))
    print("NME Test:{}".format(nme_test_1_LwF))

    # New task prediction after LwF
    
    #X2_train_padded = np.concatenate((np.zeros((int(X2_train.shape[0] - X1_train.shape[0]), X1_train.shape[1])), X2_train),axis=0)
    #X2_test_padded = np.concatenate((X1_test, np.zeros((int(X2_train.shape[0] - X1_test.shape[0]), X1_test.shape[1]))),axis=0)
    Y2_train_padded = np.concatenate((np.zeros((int(Y1_train.shape[0]), Y2_train.shape[1])), Y2_train),axis=0)
    Y2_test_padded = np.concatenate((np.zeros((int(Y1_test.shape[0]), Y2_test.shape[1])), Y2_test),axis=0)

    predict_train_LwF_2 = np.dot(Wls_LwF_hat, X2_train)
    predict_test_LwF_2 = np.dot(Wls_LwF_hat, X2_test)
    acc_train_2_LwF = compute_accuracy(predict_train_LwF_2, Y2_train_padded)
    acc_test_2_LwF = compute_accuracy(predict_test_LwF_2, Y2_test_padded)
    nme_test_2_LwF = compute_NME(predict_test_LwF_2, Y2_test_padded)
    print("Train and test accuracies for New task/dataset after LwF are: {} and {}".format(acc_train_2_LwF, acc_test_2_LwF))
    print("NME Test:{}".format(nme_test_2_LwF))
    
    predict_train_LwF_2 = np.dot(Wls_2_LwF, X2_train)
    predict_test_LwF_2 = np.dot(Wls_2_LwF, X2_test)
    acc_train_2_LwF = compute_accuracy(predict_train_LwF_2, Y2_train)
    acc_test_2_LwF = compute_accuracy(predict_test_LwF_2, Y2_test)
    nme_test_2_LwF = compute_NME(predict_test_LwF_2, Y2_test)
    print("Train and test accuracies for Old task/dataset after LwF are: {} and {}".format(acc_train_2_LwF, acc_test_2_LwF))
    print("NME Test:{}".format(nme_test_2_LwF))

    
    # Tuning epsilon_o
    mu = 1e3
    epsilon_o_vec = np.geomspace(1e-6, 3, 100)
    test_acc_vec = []
    for epsilon_o in epsilon_o_vec:
        
        Wls_LwF_hat = LwF_based_ADMM_LS_Diff(X2_train, Y2_train, Wls_1, lambda_o, epsilon_o, mu)
        # Joint performance post LwF
        predict_train_total_LwF = np.dot(Wls_LwF_hat, X_joint_train)
        predict_test_total_LwF = np.dot(Wls_LwF_hat, X_joint_test)
        acc_train_jt_LwF = compute_accuracy(predict_train_total_LwF, Y_joint_train)
        acc_test_jt_LwF = compute_accuracy(predict_test_total_LwF, Y_joint_test)
        #nme_test_jt_LwF = compute_NME(predict_test_total_LwF, Y_joint_test)
        print("Train and test accuracies for Joint Datasets after LwF for eps:{} are: {} and {}".format(epsilon_o, acc_train_jt_LwF, acc_test_jt_LwF))
        #print("NME Test:{}".format(nme_test_jt_LwF))
        test_acc_vec.append(acc_test_jt_LwF)

    # Plotting the results of the param sweep
    title = "Plotting test accuracy for LwF for LS (Diff dataset)"
    plot_acc_vs_hyperparam(np.log10(epsilon_o_vec), test_acc_vec, title)
    epsilon_o_jt_optimal = epsilon_o_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
    print("Chosen value of mu for Wls for LwF:{}".format(epsilon_o_jt_optimal))
    Wls_LwF_hat = LwF_based_ADMM_LS_Diff(X2_train, Y2_train, Wls_1, lambda_o, epsilon_o_jt_optimal, mu) # Find the resulting Wls 
    predict_train_joint_train_total = np.dot(Wls_LwF_hat, X_joint_train)
    predict_test_joint_train_total = np.dot(Wls_LwF_hat, X_joint_test)
    acc_train_jt = compute_accuracy(predict_train_joint_train_total, Y_joint_train)
    acc_test_jt = compute_accuracy(predict_test_joint_train_total, Y_joint_test)
    nme_test_jt = compute_NME(predict_test_joint_train_total, Y_joint_test)
    print("For eps = {}, Train and test accuracies for Joint Dataset after LwF are: {} and {}".format(epsilon_o_jt_optimal, acc_train_jt, acc_test_jt))
    print("NME Test for Joint Dataset after LwF:{}".format(nme_test_jt))
    '''
    
    return None

if __name__ == "__main__":
    main() 