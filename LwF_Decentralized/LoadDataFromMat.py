# This file contains functions that generate the data from the .mat files

# Import necessary packages
import numpy as np 
import scipy.io as sio 

def importData(dataset_path, dataset_name):

    # Input: Dataset_Path - Contains the full link to the dataset and is the input argument to the function

    if dataset_name == "Vowel" or dataset_name == "ExtendedYaleB" or dataset_name == "AR" or \
       dataset_name == "Scene15" or dataset_name == "Caltech101" or dataset_name == "Letter":
        
        data_dict = sio.loadmat(dataset_path + dataset_name + ".mat")
        N_train, N_test = get_no_samples(dataset_name)
        X = data_dict['featureMat'].astype(np.float32)
        Y = data_dict['labelMat'].astype(np.float32)

        if dataset_name == "Vowel":
            # Vowel has No random partitioning

            X_train = X[:,:N_train]
            Y_train = Y[:,:N_train]
            X_test = X[:,-N_test:]
            Y_test = Y[:,-N_test:]
            Q = Y.shape[0]
        else:
            # Other datasets have random partitioning

            choice = np.random.choice(np.arange(X.shape[1]), size=(N_train,), replace=False)
            ind = np.zeros(X.shape[1], dtype=bool)
            ind[choice] = True
            rest = ~ind
            X_train = X[:,ind]
            Y_train = Y[:,ind]
            X_test = X[:,rest]
            Y_test = Y[:,rest]
            Q = Y.shape[0]


    elif dataset_name == "Satimage" or dataset_name == "NORB" or dataset_name == "Shuttle" or \
         dataset_name == "MNIST":

        data_dict = sio.loadmat(dataset_path + dataset_name + ".mat")
        N_train, N_test = get_no_samples(dataset_name)

        X_train = data_dict['train_x'].astype(np.float32) # The input training data
        Y_train = data_dict['train_y'].astype(np.float32) # The input trainign target
        X_test = data_dict['test_x'].astype(np.float32) # The input test data
        Y_test = data_dict['test_y'].astype(np.float32) # The input test labels
        Q = Y_train.shape[0] # No.of classes in the target spaces
       

    if X_train.shape[0] > X_train.shape[1]: # In case data is not in form of row vectors
        X_train = np.transpose(X_train)
        Y_train = np.transpose(Y_train)
        X_test = np.transpose(X_test)
        Y_test = np.transpose(Y_test)
        Q = Y_train.shape[0] # No.of classes in the target spaces
    

    return X_train, Y_train, X_test, Y_test, Q

def get_no_samples(name):

    # This defines a dictionary containing #training and #test examples
    no_samples_dict = {"Vowel":(528, 462),\
                       "ExtendedYaleB":(1600, 800),\
                       "AR":(1800, 800),\
                       "Satimage":(4435, 2000),\
                       "Scene15":(3000, 1400),\
                       "Caltech101":(6000, 3000),\
                       "Letter":(13333, 6667),\
                       "NORB":(24300, 24300),\
                       "Shuttle":(43500, 14500),\
                       "MNIST":(60000, 10000)}

    N_train = no_samples_dict[name][0] # returns the number of training and test examples based on the dataset_name
    N_test = no_samples_dict[name][1]
    return N_train, N_test
