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
from PLN_Class import PLN,PLN_network
from Admm import *
from LoadDataFromMat import importData
import numpy as np
from sklearn.model_selection import train_test_split
from func_set import *
import random

# Import the dataset and calculate related parameters
# Compute the W_ls by solving a Least Squares Regularization problem
#def compute_Wls(X,T,lam):

    # the X are in n*p form, n sample, each sample has p dims. T is n*Q matrix, each sample is a row vector
    #inv_matrix = np.linalg.inv(np.dot(X, X.T)+lam*np.eye(X.shape[0]))
    #W_ls = np.dot(np.dot(T, X.T), inv_matrix).astype(np.float32)
    #return W_ls


def split_training_set(X_train, Y_train, Section_num): # =4
    #X_train_Part_1, X_train_Part_2, Y_train_Part_1, Y_train_Part_2 = train_test_split(X_train.transpose(), Y_train.transpose(), train_size=0.5)
    X_train_whole=[]
    Y_train_whole=[]
    X_train_Part_1, X_train_Part_2, Y_train_Part_1, Y_train_Part_2 = train_test_split(X_train.transpose(), Y_train.transpose(), train_size=1.0/Section_num)
    X_train_whole.append(X_train_Part_1.transpose())
    Y_train_whole.append(Y_train_Part_1.transpose())

    for i in range(1,Section_num-1): #(1 2 3 )
        X_train_Part_1, X_train_Part_2, Y_train_Part_1, Y_train_Part_2 = train_test_split(X_train_Part_2, Y_train_Part_2, train_size=1.0/(Section_num-i))
        X_train_whole.append(X_train_Part_1.transpose())
        Y_train_whole.append(Y_train_Part_1.transpose())
    
    X_train_whole.append(X_train_Part_2.transpose())
    Y_train_whole.append(Y_train_Part_2.transpose())

    return X_train_whole, Y_train_whole


def optimize_O_steps(pln_net:PLN_network, X_train_whole, Y_train_whole, X_test,Y_test, Q, mu: np.float32=1e-1, alpha = 2): #Signal_matrix, O_matrix, Lambda_k, Z_k,
    
    num_nodes = len(pln_net)
    #Z_k = np.array([None]*num_nodes)
    #cnvge_flag = False
    O_matrix = np.array([None]*num_nodes) # Ouptut matrix for every node
    Lambda_k = np.array([None]*num_nodes) # ADMM helping / auxillary variable
    Signal_matrix = np.array([None]*num_nodes) # Represents the hidden activation

    eps0 = alpha*np.sqrt(2*Q)
    Th = eps0*1e-4
    
    for num_layer in range(pln_net[0].num_layer):
        #converge_flag = np.array([False]*num_nodes)
        
        for i in range(num_nodes):
            if (num_layer==0):
                #pln_net[i].construct_one_layer(Y_train_whole[i], Q, X_train = X_train_whole[i])
                
                if i > 0:
                    pln_net[i].construct_one_layer(Y_train_whole[i], Q, X_train = X_train_whole[i], calculate_O=True, dec_flag=True, R_i = pln_net[0].pln[num_layer].R_l)
                else:
                    pln_net[i].construct_one_layer(Y_train_whole[i], Q, X_train = X_train_whole[i])
                
            else:
                #pln_net[i].construct_one_layer(Y_train_whole[i], Q, calculate_O=False)
                
                if i > 0:
                    pln_net[i].construct_one_layer(Y_train_whole[i], Q, X_train = None, calculate_O=False, dec_flag = True, R_i = pln_net[0].pln[num_layer].R_l)
                else:
                    pln_net[i].construct_one_layer(Y_train_whole[i], Q, X_train = None, calculate_O=False)
                
            Signal_matrix[i] = pln_net[i].pln[num_layer].Y_l
            #O_matrix[i] = pln_net[i].pln[num_layer].O_l
            # Extract signal vector and O matrix
            
            # Initialising these ADMM arrays
            Z_k = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            O_matrix[i] = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            Lambda_k[i] = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            #Z_k[i] = O_matrix[i] + Lambda_k[i]
        
        for _ in range(pln_net[0].max_iterations):
            
            error = np.array([0.0]*num_nodes)
            
            for i in range(num_nodes):
                # Optimize O matrix first
                O_matrix[i] = admm_decent_Only_O_Onetime(Signal_matrix[i], Y_train_whole[i], Lambda_k[i], Z_k)
                pln_net[i].pln[num_layer].O_l = O_matrix[i]
        
            Z_k = admm_decent_Only_Z_Onetime(Lambda_k, O_matrix, Q, proj_coef=10*num_nodes) #TODO: Motivating choice of epsilon
                #update lambda
            for i in range(num_nodes):
                Lambda_k[i] = Lambda_k[i] + O_matrix[i] - Z_k
                error[i] += np.linalg.norm(O_matrix[i] - Z_k)
            # Z is the average! all the O matrix should converge to Z!

            if np.mean(error) < Th:
                #cnvge_flag = True
                #print("Converge!")
                break
                        
        #print("Threshold:", Th)
        #print("Residual error difference of O matrix:{}".format(error))

        #print("One round iteration")
        
        print("Layer:", num_layer+1, "Done")
        #print(Z_k)
       
        #print("Number of layer{}: O After joint optimization:".format(num_layer+1))
        for i in range(num_nodes):
            predicted_lbl_test = pln_net[i].compute_test_outputs(X_test)
            print("Node No. : {}, Testing Accuracy:{}\n".format(i+1, compute_accuracy(predicted_lbl_test, Y_test)))
            print("Node No. : {}, Testing NME:{}\n".format(i+1, compute_NME(predicted_lbl_test, Y_test))) 

    return pln_net

def optimize_W_steps(pln_net:PLN_network, X_train_whole, Y_train_whole, rho: np.float32=1e-1, lambda_rate=1e2, alpha=2): # , W_matrix, Lambda_k, Z_k
    num_nodes = len(X_train_whole)
    
    W_matrix = np.array([None]*num_nodes)
    Lambda_k = np.array([None]*num_nodes)
    
    for i in range(num_nodes):
        W_matrix[i] = pln_net[i].W_ls
        Lambda_k[i] = np.zeros( (Y_train_whole[i].shape[0], len(X_train_whole[i])) )
    
    Z_k = np.mean(W_matrix)

    num = Y_train_whole[0].shape[0]
    eps0 = alpha*np.sqrt(2*num)
    Th = eps0*1e-4 #TODO: Motivating choice of eps0??
    
    for _ in range(pln_net[0].max_iterations*1):
        error = np.array([0.0]*num_nodes)
        for i in range(num_nodes):
            # Optimize O matrix first
            W_matrix[i] = admm_decent_Only_W_Onetime(X_train_whole[i], Y_train_whole[i], Lambda_k[i], Z_k)
            pln_net[i].W_ls = W_matrix[i]
        
        Z_k = admm_decent_Only_Z0_Onetime(Lambda_k, W_matrix, num_nodes)
            #update lambda
        
        for i in range(num_nodes):
            Lambda_k[i] = Lambda_k[i] + W_matrix[i] - Z_k
            error[i] += np.linalg.norm(W_matrix[i] - Z_k)
        # Z is the average! all the O matrix should converge to Z!

        if np.mean(error) < Th:
            #cnvge_flag = True
            print("Converge!")
            break

    #print("Threshold:", Th)
    #print("Residual error difference of O matrix:{}".format(error))
    #print("Residual error difference of O matrix:{}".format(error))

    return pln_net

def train_decentralized_networks(X_train_whole, Y_train_whole, X_test, Y_test, Q):
    
    num_nodes = len(X_train_whole)
    if (num_nodes != len(Y_train_whole)):
        print("Input parameter Error!")
        return None 

    pln_all=[] # returned value, bunches of PLN network
    for i in range(num_nodes):
        pln_all.append(PLN_network(num_layer = 10, mu=0.1, maxit=100))
        # deflult: num_layer = 20, mu=1e3, maxit=30, lamba=1e2
        pln_all[i].construct_W(X_train_whole[i], Y_train_whole[i])


    #print("W_ls Before joint optimization:")

    for i in range(num_nodes):
        predicted_lbl_test = pln_all[i].compute_test_outputs(X_test)
        #print("Node No. : {}, Testing Accuracy:{}\n".format(i+1, compute_accuracy(predicted_lbl_test, Y_test)))
        #print("Node No. : {}, Testing NME:{}\n".format(i+1, compute_NME(predicted_lbl_test, Y_test))) 
        
    pln_all = optimize_W_steps(pln_all, X_train_whole, Y_train_whole)

    print("W_ls After joint optimization:")
    for i in range(num_nodes):
        predicted_lbl_test = pln_all[i].compute_test_outputs(X_test)
        print("Node No. : {}, Testing Accuracy:{}\n".format(i+1, compute_accuracy(predicted_lbl_test, Y_test)))
        print("Node No. : {}, Testing NME:{}\n".format(i+1, compute_NME(predicted_lbl_test, Y_test))) 
    # ADMM implenmentation for W matrix.

    pln_all = optimize_O_steps(pln_all, X_train_whole, Y_train_whole, X_test, Y_test, Q)
        
    return pln_all

def train_centralized_networks(X_train, Y_train, X_test, Y_test, Q):  # X_test, Y_test,
    
    pln_net= PLN_network(num_layer = 10, mu=0.1, maxit=100)
    
    pln_net.construct_W(X_train, Y_train)
    
    predicted_lbl_test = pln_net.compute_test_outputs(X_test)
    print("Testing Accuracy:{}\n".format(compute_accuracy(predicted_lbl_test, Y_test)))
    print("Testing NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test))) 
    
    for num_layer in range(pln_net.num_layer):
        pln_net.construct_one_layer(Y_train, Q, X_train=X_train)    
        predicted_lbl_test = pln_net.compute_test_outputs(X_test)
        print("Layer: num {}. Testing Accuracy:{}\n".format(num_layer+1, compute_accuracy(predicted_lbl_test, Y_test)))
        print("Layer: num {}. Testing NME:{}\n".format(num_layer+1, compute_NME(predicted_lbl_test, Y_test))) 

        #pln_net.construct_all_layers(X_train, Y_train, Q)  
    return pln_net

def split_data_uniform(X_train, Y_train, num_nodes):
    
    # Splitting the data uniformly such that each node has roughly equal number of samples from each class
    # Within the class the number of samples might differs
    num_classes = Y_train.shape[0] # no. of classes
    indices_array = [] # List for calculating indices for every class index
    
    # Indices corresponding to every class
    for i in range(num_classes):
        indices = np.argwhere((Y_train[i,:] == 1)==True)
        indices_array.append(indices)

    # Getting the data on a per class basis
    X_train_per_class = []
    Y_train_per_class = []

    for i in range(len(indices_array)):

        indices_class_i = indices_array[i] # Getting the list of indices for class i
        # Getting the training and testing data for class i
        x_train_i = X_train[:,indices_class_i]
        y_train_i = Y_train[:,indices_class_i] 
        X_train_per_class.append(x_train_i.reshape((X_train.shape[0], len(indices_class_i))))
        Y_train_per_class.append(y_train_i.reshape((Y_train.shape[0], len(indices_class_i))))
    
    # Getting the data for every node
    indices_per_node = []

    for i in range(len(indices_array)):
        # List of list of indices for every class
        indices_class_i = indices_array[i].reshape(-1,) # Getting the list of indices for class i
        random.shuffle(indices_class_i)
        indices_per_node_class_i = np.array_split(indices_class_i, num_nodes)
        indices_per_node.append(indices_per_node_class_i) 

    X_train_whole = []
    Y_train_whole = []

    for i in range(num_nodes):
        
        X_train_per_node = []
        Y_train_per_node = []
        
        for j in range(len(indices_per_node)):
            
            indices_for_node_i_per_class = indices_per_node[j]
            X_train_per_node.append(X_train[:,indices_for_node_i_per_class[i]])
            Y_train_per_node.append(Y_train[:,indices_for_node_i_per_class[i]])
        
        X_train_whole.append(np.column_stack(X_train_per_node))
        Y_train_whole.append(np.column_stack(Y_train_per_node))

    return X_train_whole, Y_train_whole

def main():

    dataset_path = "../../Datasets/" # Specify the dataset path in the Local without the name
    dataset_name = "Vowel" # Specify the Dataset name without extension (implicitly .mat extension is assumed)
    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path, dataset_name) # Imports the data with the no. of output classes
    num_nodes=5
    #X_train_whole, Y_train_whole = split_training_set(X_train, Y_train, num_nodes)
    X_train_whole, Y_train_whole = split_data_uniform(X_train, Y_train, num_nodes)
    # return    X_train_whole[0:4]  and  Y_train_whole[0:4]


    print("***************** Centralized Version *****************")
    pln_cent = train_centralized_networks(X_train, Y_train, X_test, Y_test, Q)
    predicted_lbl_test_cent = pln_cent.compute_test_outputs(X_test)
    
    print("***************** Decentralized Version *****************")
    pln_decent = train_decentralized_networks(X_train_whole, Y_train_whole, X_test, Y_test, Q)
    
    test_acc_consensus = []
    test_nme_consensus = []
    for i in range(num_nodes):
        predicted_lbl_test = pln_decent[i].compute_test_outputs(X_test)
        print("Network No.{}. Test Accuracy:{}\n".format(i, compute_accuracy(predicted_lbl_test, Y_test)))
        print("Network No.{}. Test NME:{}\n".format(i, compute_NME(predicted_lbl_test, Y_test))) 
        test_acc_consensus.append(compute_accuracy(predicted_lbl_test, Y_test))
        test_nme_consensus.append(compute_NME(predicted_lbl_test, Y_test))
    
    print("**************** Final Results ************************")
    print("Test Accuracy:{}\n".format(compute_accuracy(predicted_lbl_test_cent, Y_test)))
    print("Test NME:{}\n".format(compute_NME(predicted_lbl_test_cent, Y_test))) 
    
    print("**************** Final Results ************************")
    print("Mean Test Accuracy (Decentralised):{}".format(np.mean(np.array(test_acc_consensus))))
    print("Mean Test NME (Decentralised):{}".format(np.mean(np.array(test_nme_consensus))))
    
    return None

if __name__ == "__main__":
    main()


"""     for _ in range(pln_all[0].max_iterations):
        for i in range(num_nodes):
            # Here we have 2 optimizing methods, one is to apply the W optimization algorithm
            # and the other is to optimize W using the same framework as we do to O, both working.
            if (~converge_flag[i]):
                #W_matrix[i], Lambda_k[i], Z_k[i], converge_flag[i] = admm_decent_O_Onetime(X_train_whole[i], Y_train_whole[i], W_matrix, Lambda_k, i, Z_k[i], converge_flag[i]) # mu=1e3 by default.
                W_matrix[i], Lambda_k[i], Z_k[i], converge_flag[i] = admm_decent_W_Onetime(X_train_whole[i], Y_train_whole[i], W_matrix, Lambda_k, i, Z_k[i], converge_flag[i]) # rho=1e-1, lambda_rate=1e2
            else:
                print("ADMM W_ls, nodes{}, converged.".format(i))
            # Till now, by comparing the result, we found that the O optimizaing framework seems better!
            pln_all[i].W_ls = W_matrix[i]
    del W_matrix """