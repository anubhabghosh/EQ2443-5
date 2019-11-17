import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
import scipy.io as sio
from PLN_Class import PLN
from Admm import optimize_admm
from LoadDataFromMat import importData
import numpy as np
from LwF_based_ADMM import LwF_based_ADMM_LS_Diff

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

def compute_joint_datasets(X1_train, X2_train, Y1_train, Y2_train):
    
    # Row and Column wise appending

    X1_train_padded = np.concatenate((X1_train, np.zeros((int(X2_train.shape[0] - X1_train.shape[0]), X1_train.shape[1]))),axis=0)
    Y1_train_padded = np.concatenate((Y1_train, np.zeros((Y1_train.shape[0], Y2_train.shape[1]))),axis=1)
    Y2_train_padded = np.concatenate((np.zeros((Y2_train.shape[0], Y1_train.shape[1])), Y2_train),axis=1)
    X_joint_train = np.concatenate((X1_train_padded, X2_train), axis=1)
    Y_joint_train = np.concatenate((Y1_train_padded, Y2_train_padded), axis=0)

    return X_joint_train, Y_joint_train

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
    # Parameters related to ADMM optimization
    max_iterations = 100 # For the ADMM Algorithm
    ##########################################################################################

    ###############################################################################################
    # Compute Train and Test accuracy for a Regularized Least Squares Algorithm for the Old Dataset
    ###############################################################################################
    
    # Compute the Wls_1_parameter for the Old Dataset
    Wls_1 = compute_Wls(X1_train, Y1_train, lambda_ls1) 
    predict_train_1 = np.dot(Wls_1, X1_train)
    predict_test_1 = np.dot(Wls_1, X1_test)
    acc_train_1 = compute_accuracy(predict_train_1, Y1_train)
    acc_test_1 = compute_accuracy(predict_test_1, Y1_test)
    nme_test_1 = compute_NME(predict_test_1, Y1_test)
    print("Train and test accuracies for T_old are: {} and {}".format(acc_train_1, acc_test_1))
    print("NME Test for T_old:{}".format(nme_test_1))

    # Compute the Wls_2_parameter for the New Dataset
    Wls_2 = compute_Wls(X2_train, Y2_train, lambda_ls2) 
    predict_train_2 = np.dot(Wls_2, X2_train)
    predict_test_2 = np.dot(Wls_2, X2_test)
    acc_train_2 = compute_accuracy(predict_train_2, Y2_train)
    acc_test_2 = compute_accuracy(predict_test_2, Y2_test)
    nme_test_2 = compute_NME(predict_test_2, Y2_test)
    print("Train and test accuracies for T_new are: {} and {}".format(acc_train_2, acc_test_2))
    print("NME Test for T_new:{}".format(nme_test_2))

    # Performing Joint Training
    X_joint_train, Y_joint_train = compute_joint_datasets(X1_train, X2_train, Y1_train, Y2_train)
    X_joint_test, Y_joint_test = compute_joint_datasets(X1_test, X2_test, Y1_test, Y2_test)
    '''
    lambda_jt_vec = np.geomspace(1e-14, 1e14, 29)
    
    test_acc_vec = []
    
    # Sweeping over a list of values for lambda
    for lambda_jt in lambda_jt_vec:

        Wls_joint_train = compute_Wls(X_joint_train, Y_joint_train, lambda_jt)
        # Task prediction on the joint dataset
        predict_train_joint_train_total = np.dot(Wls_joint_train, X_joint_train)
        predict_test_joint_train_total = np.dot(Wls_joint_train, X_joint_test)
        acc_train_jt = compute_accuracy(predict_train_joint_train_total, Y_joint_train)
        acc_test_jt = compute_accuracy(predict_test_joint_train_total, Y_joint_test)
        nme_test_jt = compute_NME(predict_test_joint_train_total, Y_joint_test)
        #print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt, acc_train_jt, acc_test_jt))
        #print("NME Test:{}".format(nme_test_jt))
        test_acc_vec.append(acc_test_jt)

    # Plotting the results of the param sweep
    title = "Plotting test accuracy for Joint Training for LS (Diff dataset)"
    plot_acc_vs_hyperparam(np.log10(lambda_jt_vec), test_acc_vec, title)
    
    lambda_jt_optimal = lambda_jt_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
    '''
    lambda_jt_optimal = 1000
    print("Chosen value of lambda for Wls:{}".format(lambda_jt_optimal))
    Wls_joint_train = compute_Wls(X_joint_train, Y_joint_train, lambda_jt_optimal) # Find the resulting Wls 
    predict_train_joint_train_total = np.dot(Wls_joint_train, X_joint_train)
    predict_test_joint_train_total = np.dot(Wls_joint_train, X_joint_test)
    acc_train_jt = compute_accuracy(predict_train_joint_train_total, Y_joint_train)
    acc_test_jt = compute_accuracy(predict_test_joint_train_total, Y_joint_test)
    nme_test_jt = compute_NME(predict_test_joint_train_total, Y_joint_test)
    print("For lambda = {}, Train and test accuracies for Joint Datasets are: {} and {}".format(lambda_jt_optimal, acc_train_jt, acc_test_jt))
    print("NME Test:{}".format(nme_test_jt))

    Wls_1_joint_train, Wls_2_joint_train = untwine_weights(Wls_joint_train, X1_train, Y1_train, X2_train, Y2_train)
    
    
    # Old task prediction after joint training
    
    predict_train_joint_train_1 = np.dot(Wls_1_joint_train, X1_train)
    predict_test_joint_train_1 = np.dot(Wls_1_joint_train, X1_test)
    acc_train_1_jt = compute_accuracy(predict_train_joint_train_1, Y1_train)
    acc_test_1_jt = compute_accuracy(predict_test_joint_train_1, Y1_test)
    nme_test_1_jt = compute_NME(predict_test_joint_train_1, Y1_test)
    print("Train and test accuracies for Old task/dataset after Joint Training are: {} and {}".format(acc_train_1_jt, acc_test_1_jt))
    print("NME Test:{}".format(nme_test_1_jt))

    # New task prediction after joint training
    
    predict_train_joint_train_2 = np.dot(Wls_2_joint_train, X2_train)
    predict_test_joint_train_2 = np.dot(Wls_2_joint_train, X2_test)
    acc_train_2_jt = compute_accuracy(predict_train_joint_train_2, Y2_train)
    acc_test_2_jt = compute_accuracy(predict_test_joint_train_2, Y2_test)
    nme_test_2_jt = compute_NME(predict_test_joint_train_2, Y2_test)
    print("Train and test accuracies for New task/dataset after Joint Training are: {} and {}".format(acc_train_2_jt, acc_test_2_jt))
    print("NME Test:{}".format(nme_test_2_jt))
    
    #################################################################################################
    # Define Params for the LwF based ADMM for Least Squares
    #################################################################################################
    
    lambda_o = 0.0001 # Forgetting factor  (#TODO: Strange Result)
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
    
    predict_train_LwF_1 = np.dot(Wls_1_LwF, X1_train)
    predict_test_LwF_1 = np.dot(Wls_1_LwF, X1_test)
    acc_train_1_LwF = compute_accuracy(predict_train_LwF_1, Y1_train)
    acc_test_1_LwF = compute_accuracy(predict_test_LwF_1, Y1_test)
    nme_test_1_LwF = compute_NME(predict_test_LwF_1, Y1_test)
    print("Train and test accuracies for Old task/dataset after Joint Training are: {} and {}".format(acc_train_1_LwF, acc_test_1_LwF))
    print("NME Test:{}".format(nme_test_1_LwF))

    # New task prediction after LwF
    
    predict_train_LwF_2 = np.dot(Wls_2_LwF, X2_train)
    predict_test_LwF_2 = np.dot(Wls_2_LwF, X2_test)
    acc_train_2_LwF = compute_accuracy(predict_train_LwF_2, Y2_train)
    acc_test_2_LwF = compute_accuracy(predict_test_LwF_2, Y2_test)
    nme_test_2_LwF = compute_NME(predict_test_LwF_2, Y2_test)
    print("Train and test accuracies for New task/dataset after Joint Training are: {} and {}".format(acc_train_2_LwF, acc_test_2_LwF))
    print("NME Test:{}".format(nme_test_2_LwF))
    '''
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