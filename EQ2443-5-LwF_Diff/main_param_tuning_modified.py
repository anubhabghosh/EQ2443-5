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
from LwF_based_ADMM import LwF_based_ADMM_Diff_LS, LwF_based_ADMM_Diff_O
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


def appendWls(Wls, X2):

    # This function appends Wls to make it consistent in shape
    if Wls.shape[1] < X2.shape[0]:
        Wls_append = np.concatenate((Wls, np.zeros((Wls.shape[0], X2.shape[0] - Wls.shape[1]))), axis=1)
    return Wls_append

def appendX(X1, X2):

    # This function appends Wls to make it consistent in shape
    if X1.shape[1] < X2.shape[1]:
        X1_append = np.concatenate((X1, np.zeros((X1.shape[0], X2.shape[1] - X1.shape[1]))), axis=1)
    return X1_append

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

    if X1_train.shape[0] <= X2_train.shape[0]:
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
def param_tuning_for_LS(X_train, Y_train, X_test, Y_test, LwF_flag = None, Wls_prev = None, epsilon_o = None, mu = None, eps_2 = None):

    # This function is used to the lambda in the regularized least squares version
    # X_train : Given joint dataset input for training
    # Y_train : Given joint dataset output for training
    # X_test : Given joint dataset input for testing
    # Y_test : Given joint dataset output for training
    # lambda_ls_jt : param to be optimized
    # The param is swept over from 10^-14 to 10^14, and the param with the highest test_acc is chosen 
    
    if LwF_flag == False:
        
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

    else:
        
        # This is the case when we need to tune lambda as well as mu for the ADMM for calculating Wls (for the composite PLN)
        lambda_o_vec = np.geomspace(1e-10, 1e10, 21) 
        mu = 1e2
        test_acc_vec = []
        
        # Sweeping over a list of values for lambda
        for lambda_o in lambda_o_vec:
            
            epsilon_o = lambda_o * np.linalg.norm(Wls_prev)**2 + eps_2
            Wls = LwF_based_ADMM_Diff_LS(X_train, Y_train, Wls_prev, lambda_o, epsilon_o, mu)
            # Task prediction on the joint dataset
            acc_train_Wls_LwF, acc_test_Wls_LwF, nme_test_Wls_LwF = compute_LS_test_accuracy(Wls, X_joint_train, Y_joint_train, X_joint_test, Y_joint_test)            
            #acc_train_Wls_LwF, acc_test_Wls_LwF, nme_test_Wls_LwF = compute_LS_test_accuracy(Wls, X_joint_train, Y_joint_train, X1_test_append, Y1_test_append)            
            
            #print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt, acc_train_jt, acc_test_jt))
            #print("NME Test:{}".format(nme_test_jt))
            test_acc_vec.append(acc_test_Wls_LwF)

        # Plotting the results of the param sweep
        title = "Plotting test accuracy for LwF for LS versus Lambda_o (Diff dataset)"
        plot_acc_vs_hyperparam(np.log10(lambda_o_vec), test_acc_vec, title)
        lambda_o_optimal = lambda_o_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
        
        #lambda_o_optimal = 1e-4
        # Tuning lambda and then tuning mu
        test_acc_vec = []
        mu_vec = np.geomspace(1e-5, 1e10, 16) 
        # Sweeping over a list of values for lambda
        for mu in mu_vec:

            epsilon_o = lambda_o_optimal * np.linalg.norm(Wls_prev)**2 + eps_2
            Wls = LwF_based_ADMM_Diff_LS(X_train, Y_train, Wls_prev, lambda_o_optimal, epsilon_o, mu)
            # Task prediction on the joint dataset
            acc_train_Wls_LwF, acc_test_Wls_LwF, nme_test_Wls_LwF = compute_LS_test_accuracy(Wls, X_joint_train, Y_joint_train, X_joint_test, Y_joint_test)
            #acc_train_Wls_LwF, acc_test_Wls_LwF, nme_test_Wls_LwF = compute_LS_test_accuracy(Wls, X_joint_train, Y_joint_train, X1_test_append, Y1_test_append)
            #print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt, acc_train_jt, acc_test_jt))
            #print("NME Test:{}".format(nme_test_jt))
            test_acc_vec.append(acc_test_Wls_LwF)

        # Plotting the results of the param sweep
        title = "Plotting test accuracy for LwF for LS vs mu (Diff dataset)"
        plot_acc_vs_hyperparam(np.log10(mu_vec), test_acc_vec, title)
        mu_optimal = mu_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy

        return lambda_o_optimal, mu_optimal


###########################################################################################
# This function is used to perform Parameter sweeping over given hyperparameter:mu
###########################################################################################
def param_tuning_for_O(pln_object, Y_train,  X_test, Y_test, Wls, max_ADMM_iterations):

    # This function is used to tune the 'mu' param in ADMM algorithm
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

        pln_object.O_l = compute_ol(pln_object.Y_l, Y_train, mu, max_ADMM_iterations)
        pln_array.append(pln_object)
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

###########################################################################################
# This function is used to perform Parameter sweeping over given hyperparameter:lambda
###########################################################################################
def param_tuning_for_O_LwF(pln_object, T2_train, X2_test, T2_test, O1_star, Wls, epsilon_o, mu, eps2=None, eps1=None):

# This function is used to tune the 'lambda_o' param  (forgetting parameter)
# X2_train : New Dataset Input 
# T2_train :  New Dataset Label
# O1_star : Can be the past O matrix
# X2_test : Given joint dataset input for testing
# T2_test : Given joint dataset output for testing
# lambda_o : param to be optimized (for ADMM)
# mu: to be optimized after fixing lambda
# The param is swept over from 10^-10 to 10^10, and the param with the highest test_acc is chosen (Currently optimisation over only the 1st layer)

    # tuning lambda
    lambda_o_vec = np.geomspace(1e-10, 1e10, 21) 
    test_acc_vec = []
    pln_array = []
    mu = 1e3
    
    # Sweeping over a list of values for lambda
    for lambda_o in lambda_o_vec:
        
        epsilon_o = lambda_o * eps1 + eps2
        pln_object.O_l = LwF_based_ADMM_Diff_O(pln_object.Y_l, T2_train, O1_star, lambda_o, epsilon_o, mu)
        pln_array.append(pln_object)
        # Task prediction on the joint dataset
        predicted_lbl_test = compute_test_outputs(pln_array, Wls, 1, X_joint_test)   
        test_acc_vec.append(compute_accuracy(predicted_lbl_test, Y_joint_test))
        #predicted_lbl_test = compute_test_outputs(pln_array, Wls, 1, X1_test_append)   
        #test_acc_vec.append(compute_accuracy(predicted_lbl_test, Y1_test_append))
        
        #print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt, acc_train_jt, acc_test_jt))
        #print("NME Test:{}".format(nme_test_jt))
        _ = pln_array.pop() # Clears the array out again until optimal solution is found

    # Plotting the results of the param sweep
    title = "Plotting test accuracy for LwF for O1 vs lambda_o (Diff dataset)"
    plot_acc_vs_hyperparam(np.log10(lambda_o_vec), test_acc_vec, title)
    lambda_o_optimal = lambda_o_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
    
    #lambda_o_optimal = 1
    # Tuning lambda and then tuning mu
    test_acc_vec = []
    pln_array = []
    mu_vec = np.geomspace(1e-10, 1e10, 21) 
    # Sweeping over a list of values for lambda
    for mu in mu_vec:

        epsilon_o = lambda_o_optimal * eps1 + eps2
        #epsilon_o = 1
        pln_object.O_l = LwF_based_ADMM_Diff_O(pln_object.Y_l, T2_train, O1_star, lambda_o_optimal, epsilon_o, mu)
        pln_array.append(pln_object)
        # Task prediction on the joint dataset
        predicted_lbl_test = compute_test_outputs(pln_array, Wls, 1, X_joint_test) 
        test_acc_vec.append(compute_accuracy(predicted_lbl_test, Y_joint_test))
        #predicted_lbl_test = compute_test_outputs(pln_array, Wls, 1, X1_test_append)   
        #test_acc_vec.append(compute_accuracy(predicted_lbl_test, Y1_test_append))
        
        #print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt, acc_train_jt, acc_test_jt))
        #print("NME Test:{}".format(nme_test_jt))
        _ = pln_array.pop() # Clears the array out again until optimal solution is found

    # Plotting the results of the param sweep
    title = "Plotting test accuracy for LwF for O1 vs mu (Diff dataset)"
    plot_acc_vs_hyperparam(np.log10(mu_vec), test_acc_vec, title)
    mu_optimal = mu_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy

    return lambda_o_optimal, mu_optimal

################################################################################################
# This function picks out the matrices O1 and O2 from the big O matrix
################################################################################################
def untwine_weights(Weights, X1_train, Y1_train, X2_train, Y2_train, case):

    if case == "LS": # partitioning a big LS matrix

        Q1 = Y1_train.shape[0]
        Q2 = Y2_train.shape[0]
        P1 = X1_train.shape[0]
        P2 = X2_train.shape[0]
        
        Weights_1 = Weights[0:Y1_train.shape[0], 0:X1_train.shape[0]]
        Weights_2 = Weights[-Y2_train.shape[0]:, -X2_train.shape[0]:]
        return Weights_1, Weights_2

    else: # partitioning a big O matrix

        #print("Shape of the Big O matrix:{}".format(Weights.shape))
        Q1 = Y1_train.shape[0]
        Q2 = Y2_train.shape[0]
        P1 = X1_train.shape[0]
        P2 = X2_train.shape[0]

        O1_appended = Weights[0:Q1, 0:Weights.shape[1]]
        O2_appended = Weights[Q1:Q1+Q2, 0:Weights.shape[1]]

        O1_1 = O1_appended[:,0:Q1]
        O1_2 = O1_appended[:,Q1 + Q2 : 2*Q1 + Q2]
        O1_1_and_2 = np.concatenate((O1_1, O1_2), axis=1)
        O1_random = O1_appended[:,-1000:]

        O1 = np.concatenate((O1_1_and_2, O1_random), axis=1)

        O2_1 = O2_appended[:,Q1:Q1 + Q2]
        O2_2 = O2_appended[:,2*Q1 + Q2 : 2*Q1 + 2*Q2]
        O2_1_and_2 = np.concatenate((O2_1, O2_2), axis=1)
        O2_random = O2_appended[:,-1000:]

        O2 = np.concatenate((O2_1_and_2, O2_random), axis=1)

        return O1, O2        

#################################################################################################
# Creates two PLN objecqt arrays from the joint array output
#################################################################################################
def create_indiv_obj_arrays(PLN_total_datasets, PLN_first_dataset, PLN_second_dataset, Wls_jt,\
                            X1_train, Y1_train, X2_train, Y2_train, lambda_ls = 1, lambda_o = 1):
    
    assert len(PLN_total_datasets) == len(PLN_first_dataset)
    assert len(PLN_total_datasets) == len(PLN_second_dataset)

    Wls_1_jt, Wls_2_jt = untwine_weights(Wls_jt, X1_train, Y1_train, X2_train, Y2_train, "LS")
    Wls_1_jt = Wls_1_jt / np.sqrt(lambda_ls)

    PLN_first_dataset_jt = copy.deepcopy(PLN_first_dataset) 
    PLN_second_dataset_jt = copy.deepcopy(PLN_second_dataset)

    for i in range(len(PLN_total_datasets)):
        PLN_first_dataset_jt[i].O_l, PLN_second_dataset_jt[i].O_l = untwine_weights(PLN_total_datasets[i].O_l, X1_train, Y1_train, X2_train, Y2_train, "O")
        PLN_first_dataset_jt[i].O_l = PLN_first_dataset_jt[i].O_l / np.sqrt(lambda_o)

    return PLN_first_dataset_jt, PLN_second_dataset_jt, Wls_1_jt, Wls_2_jt

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
                  O_l_prev_array = None, W_ls_prev = None, mu_layers_LwF = None, eps_Wls = None, eps_o = None,
                  eps_o_1 = None, eps_o_2 = None, eps_Wls_1 = None, eps_Wls_2 = None):

    if LwF_flag ==  False:

        # Simply run the PLN for 'no_layers'
        Q = Y_train.shape[0] # No. of Output classes

        if lambda_ls != None:
            # Lambda for LS is present, no tuning required
            print("Chosen value of lambda for Wls for the given dataset :{}".format(lambda_ls))
            Wls = compute_Wls(X_train, Y_train, lambda_ls)
        else:
            # Tune the lambda for the LS for given dataset
            lambda_ls_optimal = param_tuning_for_LS(X_train, Y_train, X_test, Y_test, LwF_flag=False) 
            print("Chosen value of lambda for Wls for the given dataset after tuning:{}".format(lambda_ls_optimal))
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
        #print("ADMM for Layer No:{}".format(1))

        ##########################################################################################
        # Tuning the params for the Joint training case, if encountered
        ##########################################################################################
        if mu == None:
            mu_optimal = param_tuning_for_O(pln_l1, Y_train, X_test, Y_test, Wls, max_iterations)
            print("Chosen value of mu and rho for 1st Layer after tuning:{}".format(mu_optimal))
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
            
            #print("************** ADMM for Layer No:{} **************".format(i+1))
    
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
            #print("Training Accuracy:{}\n".format(compute_accuracy(predicted_lbl_train, Y_train)))
            #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test)))
            #print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
            #print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test)))

        predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test) # For the entire network
        #print(Final Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))
        final_test_acc = compute_accuracy(predicted_lbl, Y_test) # Final Test Accuracy
        final_test_nme = compute_NME(predicted_lbl, Y_test)

        return PLN_objects, final_test_acc, final_test_nme, Wls

    else:
        
        ####### Implement LwF for Diff Datasets #######
        # Simply run the PLN for 'no_layers'
        if lambda_ls != None:
            # Lambda for LS is present, no tuning required
            print("Chosen value of lambda for Wls for the given dataset :{}".format(lambda_ls))
            Wls = compute_Wls(X_train, Y_train, lambda_ls)
        else:
            # Tune the lambda for the LS for given dataset
            lambda_ls_optimal, mu_ls_optimal = param_tuning_for_LS(X_train, Y_train, X_test, Y_test, LwF_flag=True, Wls_prev=W_ls_prev, epsilon_o=eps_Wls, mu=None, eps_2=eps_Wls_2 ) 
            eps_Wls = lambda_ls_optimal * eps_Wls_1 + eps_Wls_2
            print("Chosen value of lambda for Wls (LwF) for T1 + T2 after tuning:{}".format(lambda_ls_optimal))
            Wls = LwF_based_ADMM_Diff_LS(X_train, Y_train, W_ls_prev, lambda_ls_optimal, eps_Wls, mu_ls_optimal) # Find the resulting Wls 

        #acc_train_Wls, acc_test_Wls, nme_test_Wls = compute_LS_test_accuracy(Wls, X_joint_train, Y_joint_train, X_joint_test, Y_joint_test)
        acc_train_Wls, acc_test_Wls, nme_test_Wls = compute_LS_test_accuracy(Wls, X_joint_train, Y_joint_train, X_joint_test, Y_joint_test)
        print("Train and test accuracies for LS for T1 + T2 after LwF are: {} and {}".format(acc_train_Wls, acc_test_Wls))
        print("NME Test for T1 + T2:{}".format(nme_test_Wls))

        ##########################################################################################
        # Creating a list of PLN Objects
        ##########################################################################################
        PLN_objects = [] # The network is to be stored as a list of objects, with each object
                      # representing a network layer

        ##########################################################################################
        # Creating a 1 layer Network
        ##########################################################################################
        
        layer_no = 0 # Layer Number/Index (0 to L-1)
        Q = W_ls_prev.shape[0] + Y_train.shape[0] # Number of classes in the given network
        num_node = 2*Q + 1000 # Number of nodes in every layer (fixed in this case)
        # Create an object of PLN Class
        pln_l1 = PLN(Q, X_train, layer_no, num_node, W_ls = Wls)
        # Compute the top part of the Composite Weight Matrix
        W_top = np.dot(np.dot(pln_l1.V_Q, Wls), X_train)
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
            lambda_o_optimal, mu_optimal = param_tuning_for_O_LwF(pln_l1, Y_train, X_test, Y_test, O_l_prev_array[0].O_l, Wls, eps_o, mu, eps2=eps_o_2, eps1=eps_o_1)
            eps_o = lambda_o_optimal * eps_o_1 + eps_o_2
            #eps_o = 1
            print("Chosen value of mu and rho for 1st Layer (LwF) after tuning:{}".format(mu_optimal))
            pln_l1.O_l = LwF_based_ADMM_Diff_O(pln_l1.Y_l, Y_train, O_l_prev_array[0].O_l, lambda_o_optimal, eps_o, mu_optimal)# Once optimal solution is found, it uses that 
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
            pln.O_l = LwF_based_ADMM_Diff_O(pln.Y_l, Y_train, O_l_prev_array[i].O_l, lambda_o_optimal, eps_o, mu)
            # Add the new layer to the 'Network' list
            PLN_objects.append(pln)
            
            # Compute training, test accuracy, NME for new appended networks
            '''
            predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, i+1, X_test)
            predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, i+1, X_train)
            Y_train_hat = np.concatenate((np.dot(O_l_prev_array[i].O_l, X_train) * np.sqrt(lambda_o_optimal),Y_train), axis=0)
            Y_test_hat = np.concatenate((np.dot(O_l_prev_array[i].O_l, X_test) * np.sqrt(lambda_o_optimal),Y_test), axis=0)
            print("Training Accuracy:{}".format(compute_accuracy(predicted_lbl_train, Y_train_hat)))
            print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_test_hat)))
            #print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_train)))
            print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test_hat)))
            '''
            predicted_lbl_test = compute_test_outputs(PLN_objects, Wls, i+1, X_joint_test)
            predicted_lbl_train = compute_test_outputs(PLN_objects, Wls, i+1, X_joint_train)
            #Y_train_hat = np.concatenate((np.dot(O_l_prev_array[i].O_l, X_joint_train) * np.sqrt(lambda_o_optimal),Y_joint_train), axis=0)
            #Y_test_hat = np.concatenate((np.dot(O_l_prev_array[i].O_l, X_joint_test) * np.sqrt(lambda_o_optimal),Y_joint_test), axis=0)
            print("Training Accuracy:{}".format(compute_accuracy(predicted_lbl_train, Y_joint_train)))
            print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test, Y_joint_test)))
            #print("Traning NME:{}\n".format(compute_NME(predicted_lbl_train, Y_joint_train)))
            print("Test NME:{}\n".format(compute_NME(predicted_lbl_test, Y_joint_test)))
            
        '''
        predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test) # For the entire network
        #print(Final Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))
        final_test_acc = compute_accuracy(predicted_lbl, Y_test_hat) # Final Test Accuracy
        final_test_nme = compute_NME(predicted_lbl, Y_test_hat)
        '''
        predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_joint_test) # For the entire network
        #print(Final Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))
        final_test_acc = compute_accuracy(predicted_lbl, Y_joint_test) # Final Test Accuracy
        final_test_nme = compute_NME(predicted_lbl, Y_joint_test)
        
        return PLN_objects, final_test_acc, final_test_nme, Wls, lambda_o_optimal, lambda_ls_optimal


def main():

    ##########################################################################################
    # Dataset related parameters
    dataset_path = "../../Datasets/" # Specify the dataset path in the Local without the name
    dataset_1_name = "Vowel" # Specify the Dataset name without extension (implicitly .mat extension is assumed)
    dataset_2_name = "ExtendedYaleB"

    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path, dataset_1_name) 

    X1_Y1_train_indices = np.argmax(Y_train, axis=0) < int(Y_train.shape[0]/2)
    X2_Y2_train_indices = ~X1_Y1_train_indices

    X1_train = X_train[:, X1_Y1_train_indices]
    Y1_train = Y_train[:, X1_Y1_train_indices]

    X2_train = X_train[:, X2_Y2_train_indices]
    Y2_train = Y_train[:, X2_Y2_train_indices]

    X1_Y1_test_indices = np.argmax(Y_test, axis=0) < int(Y_test.shape[0]/2)
    X2_Y2_test_indices = ~X1_Y1_test_indices

    X1_test = X_test[:, X1_Y1_test_indices]
    Y1_test = Y_test[:, X1_Y1_test_indices]

    X2_test = X_test[:, X2_Y2_test_indices]
    Y2_test = Y_test[:, X2_Y2_test_indices]

    Q1 = Y1_train.shape[0]
    Q2 = Y2_train.shape[0]


    # Importing parameters for the first Dataset (Old)
    #X1_train, Y1_train, X1_test, Y1_test, Q1 = importData(dataset_path, dataset_1_name) 
    # Importing parameters for the second Dataset (New)
    #X2_train, Y2_train, X2_test, Y2_test, Q2 = importData(dataset_path, dataset_2_name) 
    
    # Parameters for the First Dataset
    #mu1 = 1e3 # For the given dataset
    #lambda_ls1 = 1e2 # Given regularization parameter as used in the paper for the used Dataset

    # Parameters for the Second Dataset
    #mu2 = 1e3 # For the given dataset
    #lambda_ls2 = 1e4 # Given regularization parameter as used in the paper for the used Dataset

    mu1 = 1e3 # For the given dataset
    lambda_ls1 = 1e2 # Given regularization parameter as used in the paper for the used Dataset
    mu2 = 1e3 # For the given dataset
    lambda_ls2 = 1e2 # Given regularization parameter as used in the paper for the used Dataset
    ##########################################################################################

    ##########################################################################################
    # Parameters related to ADMM optimization and initial values
    max_iterations = 100 # For the ADMM Algorithm   
    no_layers = 20
    alpha = 2
    ###############################################################################################
    # Compute Train and Test accuracy for an ADMM based Algorithm for the Old Dataset
    ###############################################################################################
    
    # Run ADMM Optimization for the first dataset (no LwF)
    print("First Dataset, no LWF")
    PLN_first_dataset, final_test_acc_1, final_test_nme_1, Wls_1 = PLN_with_ADMM(X1_train, Y1_train, X1_test, Y1_test, \
                                                                                                  no_layers, max_iterations, lambda_ls1, mu1, LwF_flag=False,\
                                                                                                  O_l_prev_array=None, W_ls_prev=None, mu_layers_LwF=None)
    eps1_Wls = np.linalg.norm(Wls_1, ord='fro')**2
    print("Eps_1 for Least Squares for first dataset: {}".format(eps1_Wls))
    #eps1_o = np.linalg.norm(PLN_first_dataset[0].O_l, ord='fro')**2
    eps1_o = (alpha*np.sqrt(2*Q1))**2 # Actually squared here, because Square root is applied in ADMM projection for LwF
    print("Eps_1 for Layers for first dataset: {}".format(eps1_o))
    predicted_lbl = compute_test_outputs(PLN_first_dataset, Wls_1, no_layers, X1_test) # For the 1st dataset
    print("Final Test Accuracy for T1:{}".format(compute_accuracy(predicted_lbl, Y1_test)))

    # Run ADMM Optimization for the second half dataset (no LwF)
    print("Second Dataset, no LWF")
    PLN_second_dataset, final_test_acc_2, final_test_nme_2, Wls_2 = PLN_with_ADMM(X2_train, Y2_train, X2_test, Y2_test, \
                                                                                                  no_layers, max_iterations, lambda_ls2, mu2, LwF_flag=False,\
                                                                                                  O_l_prev_array=None, W_ls_prev=None, mu_layers_LwF=None)
    eps2_Wls = np.linalg.norm(Wls_2, ord='fro')**2
    print("Eps_2 for Least Squares for second dataset: {}".format(eps2_Wls))
    #eps2_o = np.linalg.norm(PLN_second_dataset[0].O_l, ord='fro')**2
    eps2_o = (alpha*np.sqrt(2*Q2))**2 # Actually squared here, because Square root is applied in ADMM projection for LwF
    print("Eps_2 for Layers for second dataset: {}".format(eps2_o))
    predicted_lbl = compute_test_outputs(PLN_second_dataset, Wls_2, no_layers, X2_test) # For the 1st dataset
    print("Final Test Accuracy for T2:{}".format(compute_accuracy(predicted_lbl, Y2_test)))

    # Run ADMM optimizatin for the joint datasets (no LwF)
    
    global X_joint_train, X_joint_test, Y_joint_train, Y_joint_test

    X_joint_train, Y_joint_train = compute_joint_datasets(X1_train, X2_train, Y1_train, Y2_train)
    X_joint_test, Y_joint_test = compute_joint_datasets(X1_test, X2_test, Y1_test, Y2_test)
    
    mu_jt = None 
    lambda_jt = None
    print("Joint Training for Datasets T1 and T2, without LwF")
    PLN_total_datasets, final_test_acc_jt, final_test_nme_jt, Wls_jt = PLN_with_ADMM(X_joint_train, Y_joint_train, X_joint_test, \
                                                                                        Y_joint_test, no_layers, max_iterations, lambda_jt, mu_jt, \
                                                                                        LwF_flag=False, O_l_prev_array=None, W_ls_prev=None, mu_layers_LwF=None) 
    
    predicted_lbl = compute_test_outputs(PLN_total_datasets, Wls_jt, no_layers, X_joint_test) # For the 2nd half
    print("Final Test Accuracy for T1 + T2 (joint):{}".format(compute_accuracy(predicted_lbl, Y_joint_test)))
    
    # Run ADMM Optimization for the first dataset (no LwF)
    print("First Dataset, no LWF, after joint training")
    P = max(X1_test.shape[0], X2_test.shape[0])
    Q = predicted_lbl.shape[0]

    #global X1_test_append, Y1_test_append, X2_test_append, Y2_test_append

    if X1_test.shape[0] < P:
        X1_test_append = np.concatenate((X1_test, np.zeros((P - X1_test.shape[0],X1_test.shape[1]))), axis=0)
        X1_test_append = np.concatenate((X1_test_append, np.zeros((X1_test_append.shape[0], X2_test.shape[1]))), axis=1)
    else:
        X1_test_append = copy.deepcopy(X1_test)
        X1_test_append = np.concatenate((X1_test_append, np.zeros((X1_test_append.shape[0], X2_test.shape[1]))), axis=1)

    predicted_lbl_1 = compute_test_outputs(PLN_total_datasets, Wls_jt, no_layers, X1_test_append) # For the Old dataset

    if Y1_test.shape[0] <= (Q//2):
        Y1_test_append = np.concatenate((Y1_test, np.zeros((Q - Y1_test.shape[0], Y1_test.shape[1]))), axis=0)
        Y1_test_append = np.concatenate((Y1_test_append, np.zeros((Y1_test_append.shape[0], Y2_test.shape[1]))), axis=1)
    else:
        Y1_test_append = np.concatenate((np.zeros((Q - Y1_test.shape[0], Y1_test.shape[1])),Y1_test), axis=0)
        Y1_test_append = np.concatenate((np.zeros((Y1_test_append.shape[0], Y2_test.shape[1])), Y1_test_append), axis=1)
    
    print("Final Test Accuracy for T1 (after joint):{}".format(compute_accuracy(predicted_lbl_1, Y1_test_append)))

    # Run ADMM Optimization for the second half dataset (no LwF)
    print("Second Dataset, no LWF, after joint training")

    if X2_test.shape[0] < P:
        X2_test_append = np.concatenate((X2_test, np.zeros((P - X2_test.shape[0],X2_test.shape[1]))), axis=0)
        X2_test_append = np.concatenate((np.zeros((X2_test_append.shape[0], X1_test.shape[1])), X2_test_append), axis=1)
    else:
        X2_test_append = copy.deepcopy(X2_test)
        X2_test_append = np.concatenate((np.zeros((X2_test_append.shape[0], X1_test.shape[1])), X2_test_append), axis=1)

    predicted_lbl_2 = compute_test_outputs(PLN_total_datasets, Wls_jt, no_layers, X2_test_append) # For the New dataset

    if Y2_test.shape[0] < (Q//2):
        Y2_test_append = np.concatenate((Y2_test, np.zeros((Q - Y2_test.shape[0], Y2_test.shape[1]))), axis=0)
        Y2_test_append = np.concatenate((Y2_test_append, np.zeros((Y2_test_append.shape[0], Y1_test.shape[1]))), axis=1)
    else:
        Y2_test_append = np.concatenate((np.zeros((Q - Y2_test.shape[0], Y2_test.shape[1])),Y2_test), axis=0)
        Y2_test_append = np.concatenate((np.zeros((Y2_test_append.shape[0], Y1_test.shape[1])), Y2_test_append), axis=1)

    print("Final Test Accuracy for T2 (after joint):{}".format(compute_accuracy(predicted_lbl_2, Y2_test_append)))
    
    #################################### Applying LwF ######################################

    print("Joint Training for Datasets T1 and T2, after LwF")
    lambda_o = 1 # Initial Guess for Lambda
    lambda_jt_LwF = None
    mu_jt_LwF = None
    epsilon_o_Wls = lambda_o * eps1_Wls + eps2_Wls # Epsilon value for LS
    print("Eps for LS for T1+T2:{}".format(epsilon_o_Wls))
    epsilon_o = lambda_o * eps1_o + eps2_o # Epsilon value for O1
    print("Eps for O_l for T1+T2:{}".format(epsilon_o))
    PLN_total_datasets_LwF, final_test_acc_jt_LwF, final_test_nme_jt_LwF, Wls_jt_LwF, lambda_o, lambda_ls =  PLN_with_ADMM(X2_train, Y2_train, X2_test, Y2_test, no_layers,\
                                                                                                      max_iterations, lambda_jt_LwF, mu_jt_LwF, LwF_flag=True,\
                                                                                                      O_l_prev_array=PLN_first_dataset, W_ls_prev=Wls_1, \
                                                                                                      mu_layers_LwF=None, eps_Wls=epsilon_o_Wls, eps_o=epsilon_o, \
                                                                                                      eps_o_1 = eps1_o, eps_o_2 = eps2_o, eps_Wls_1 = eps1_Wls, eps_Wls_2 = eps2_Wls)
    # Check the Joint test accuracy after LwF (baseline : Joint training + joint testing)
    predicted_lbl = compute_test_outputs(PLN_total_datasets_LwF, Wls_jt_LwF, no_layers, X_joint_test) # For the 2nd half
    print("Final Test Accuracy for T1 + T2 (after LwF):{}".format(compute_accuracy(predicted_lbl, Y_joint_test)))
    
    # Check the individual accuracy after LwF of Old Dataset (baseline: Individual testing without LwF?)
    print("First Dataset, after LwF")
    predicted_lbl_1_LwF = compute_test_outputs(PLN_total_datasets_LwF, Wls_jt_LwF, no_layers, X1_test_append) # For the Old dataset
    print("Final Test Accuracy for T1 (after joint):{}".format(compute_accuracy(predicted_lbl_1_LwF, Y1_test_append)))

    # Check the individual accuracy after LwF of Second Dataset (baseline: Individual testing without LwF?)
    print("Second Dataset, no LWF, after joint training")
    predicted_lbl_2_LwF = compute_test_outputs(PLN_total_datasets_LwF, Wls_jt_LwF, no_layers, X2_test_append) # For the New dataset
    print("Final Test Accuracy for T2 (after joint):{}".format(compute_accuracy(predicted_lbl_2_LwF, Y2_test_append)))

    return None

if __name__ == "__main__":
    main() 