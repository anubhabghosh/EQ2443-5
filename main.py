import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
import scipy.io as sio
from PLN_Class import PLN
from Admm import optimize_admm
import numpy as np

# Import the dataset and calculate related parameters
def importData(dataset_path):

    Data_dicts = sio.loadmat(dataset_path)
    X_train = Data_dicts['train_x'].astype(np.float32) # The input training data
    Y_train = Data_dicts['train_y'].astype(np.float32) # The input trainign target
    X_test = Data_dicts['test_x'].astype(np.float32) # The input test data
    Y_test = Data_dicts['test_y'].astype(np.float32) # The input test labels

    Q = Y_train.shape[0] # No.of classes in the target spaces

    return X_train, Y_train, X_test, Y_test, Q

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
    #for i in range(10)
    #    T[i] =
    return X, T
    X = None
    T = None
    Q = 0
    return X, T, Q

# Compute the W_ls by solving a Least Squares Regularization problem
def compute_Wls(X,T,lam):

    # the X are in n*p form, n sample, each sample has p dims. T is n*Q matrix, each sample is a row vector
    #print(X.shape)
    #print(T.shape)
    inv_matrix = np.linalg.inv(np.dot(X, X.T)+lam*np.eye(X.shape[0]))
    W_ls = np.dot(np.dot(T, X.T), inv_matrix).astype(np.float32)
    #W_ls = np.dot(T, inv_matrix).dot(X.T)
    #print(W_ls.shape)
    #Q = T.shape[0]
    return W_ls

def compute_ol(Y,T,mu, max_iterations):

    ol = optimize_admm(T, Y, mu, max_iterations)
    return ol

def compute_accuracy(predicted_lbl, true_lbl):

    acc = 100.*np.mean(np.argmax(predicted_lbl,axis=0)==np.argmax(true_lbl,axis=0))
    return acc

def compute_test_outputs(PLN_object_array, W_ls, num_layers, X_test):

    PLN_1 = PLN_object_array[0]
    W_initial_top = np.dot(np.dot(PLN_1.V_Q, W_ls), X_test)
    W_initial_bottom = PLN_1.normalization(np.dot(PLN_1.R_l, X_test))
    Z = np.concatenate((W_initial_top, W_initial_bottom), axis=0)
    y = PLN_1.activation_function(Z)

    for i in range(1, num_layers-1):
        PLN_l = PLN_object_array[i]
        W_top = np.dot(np.dot(PLN_l.V_Q, PLN_object_array[i-1].O_l), y)
        W_bottom = PLN_l.normalization(np.dot(PLN_l.R_l, y))
        Z = np.concatenate((W_top, W_bottom), axis=0)
        y = PLN_l.activation_function(Z)

    return np.dot(PLN_object_array[num_layers - 1].O_l, y)

def main():

    #X, T = importDummyExample()
    dataset_path = "./Satimage.mat"
    mu = 1e5 # For the given dataset
    max_iterations = 100 # For the ADMM Algorithm
    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path)

    lambda_ls = 1e6 # Given regularization parameter as used in the paper
    Wls = compute_Wls(X_train, Y_train, lambda_ls)
    print(Q)

    # Compute Train and Test accuracy
    
    predict_train = np.dot(Wls, X_train)
    predict_test = np.dot(Wls, X_test)
    acc_train = compute_accuracy(predict_train, Y_train)
    acc_test = compute_accuracy(predict_test, Y_test)
    print("Train and test accuracies are: {} and {}".format(acc_train, acc_test))
    
    # Creating a list of PLN Objects
    
    PLN_objects = []
    no_layers = 20

    # Creating a 1 layer Network
    num_class = Y_train.shape[0]
    num_node = 2*num_class + 1000
    layer_no = 1
    pln_l1 = PLN(Q, X_train, layer_no, num_node, W_ls = Wls)
    W_top = np.dot(np.dot(pln_l1 .V_Q, Wls), X_train)
    W_bottom = pln_l1.normalization(np.dot(pln_l1.R_l, X_train))
    pln_l1_Z_l = np.concatenate((W_top, W_bottom), axis=0)
    pln_l1.Y_l = pln_l1.activation_function(pln_l1_Z_l) #TODO: To be computed as g(WX)
    
    #pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu, max_iterations)
    pln_l1.O_l = compute_ol(pln_l1.Y_l, Y_train, mu, max_iterations)
    PLN_objects.append(pln_l1)
    predicted_lbl = compute_test_outputs(PLN_objects, Wls, 1, X_test)
    print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))

    # ADMM Training for all the layers using the training Data
    
    for i in range(1, no_layers):

        print("ADMM for Layer No:{}".format(i))
        X = PLN_objects[i-1].Y_l # Input for the previous layer
        num_node = 2*Q + 1000 # No. of nodes fixed for every layer
        pln = PLN(Q, X, i, num_node, W_ls=None)
        pln_W = np.concatenate((np.dot(pln.V_Q,PLN_objects[i-1].O_l), pln.R_l),axis=0)
        W_top = np.dot(np.dot(pln.V_Q,PLN_objects[i-1].O_l), X)
        W_bottom = pln.normalization(np.dot(pln.R_l, X))
        pln_Z_l = np.concatenate((W_top, W_bottom), axis=0)
        pln.Y_l = pln.activation_function(pln_Z_l) #TODO: To be computed as g(WX)
        pln.O_l = compute_ol(pln.Y_l, Y_train, mu, max_iterations)
        PLN_objects.append(pln)
        predicted_lbl = compute_test_outputs(PLN_objects, Wls, i, X_test)
        print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))

    #predicted_lbl = compute_test_outputs(PLN_objects, Wls, no_layers, X_test)
    #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl, Y_test)))
    
    return None

if __name__ == "__main__":
    main() 