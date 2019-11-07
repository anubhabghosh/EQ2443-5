import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
import scipy.io as sio

# Import the dataset and calculate related parameters
def importData(dataset_path):

    Data_dicts = sio.loadmat(dataset_path)
    X_train = Data_dicts['train_x'] # The input training data
    Y_train = Data_dicts['train_y'] # The input trainign target
    X_test = Data_dicts['test_x'] # The input test data
    Y_test = Data_dicts['test_y'] # The input test labels

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

# Compute the W_ls by solving a Least Squares Regularization problem
def compute_Wls(X,T,lam):
    
    # the X are in n*p form, n sample, each sample has p dims. T is n*Q matrix, each sample is a row vector

    print(X.shape)
    print(T.shape)
    W_ls = np.dot(np.dot(T, X.T), np.linalg.inv(np.dot(X, X.T)+lam*np.eye(X.shape[0])))
    print(W_ls.shape)
    Q = T.shape[0]
    return W_ls

def comput_ol(Y,T,lam):
    #reg = sklearn.linear_model.Ridge(alpha = lam)
    #reg.fit(Y,T)
    num_class = T.shape[1]
    num_node = 2*num_class+1000
    #ol = np.zeros([num_class,num_node])
    ol = np.dot(np.linalg.inv(np.dot(Y.T, Y) + lam * np.eye(Y.shape[1])), np.dot(Y.T, T))
    return ol

def main():

    #X, T = importDummyExample()
    dataset_path = "./EQ2443-5/Satimage.mat"
    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path)
    lamda_ls = 10**6 # Given regularization parameter as used in the paper
    Wls = compute_Wls(X_train,Y_train,lamda_ls)
    print(Q)

    # Sklearn Ridge Regression
    #clf = sklearn.linear_model.Ridge(alpha=1.0)
    #clf.fit(X, T.T)
    #A = clf.get_params(deep = True)
    #print(A)
    return None



if __name__ == "__main__":
    main()