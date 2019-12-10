import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
import scipy.io as sio
from PLN_Class import PLN, PLN_network
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


def optimize_O_steps(pln_net:PLN_network, X_train_whole, Y_train_whole, X_test,Y_test, Q, mu: np.float32=1e-1, \
                     alpha = 2, H_fault=None, H_non_fault=None, fault_flag=False, B = None): #Signal_matrix, O_matrix, Lambda_k, Z_k,
    
    num_nodes = len(pln_net)
    #Z_k = np.array([None]*num_nodes)
    #cnvge_flag = False
    O_matrix = np.array([None]*num_nodes) # Ouptut matrix for every node
    Lambda_k = np.array([None]*num_nodes) # ADMM helping / auxillary variable
    Z_k = np.array([None]*num_nodes) # ADMM helping / auxillary variable
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
            #Z_k = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            Z_k[i] = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            O_matrix[i] = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            Lambda_k[i] = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            #Z_k[i] = O_matrix[i] + Lambda_k[i]
        
        for k in range(pln_net[0].max_iterations):
            
            error = np.array([0.0]*num_nodes)

            # Optimize O matrix first            
            for i in range(num_nodes):

                O_matrix[i] = admm_decent_Only_O_Onetime(Signal_matrix[i], Y_train_whole[i], Lambda_k[i], Z_k[i], mu)
                pln_net[i].pln[num_layer].O_l = O_matrix[i]

            # Calculating the Z matrix for every node (effectively same thing is calculated by every node)    
            #for i in range(num_nodes):
            #    Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k, O_matrix, Q, proj_coef=10*num_nodes) #TODO: Motivating choice of epsilon
            
            if k <= (pln_net[0].max_iterations*1 // 2):
            
                for i in range(num_nodes): 

                    node_list_i = np.array(np.nonzero(H_non_fault[i,:]))
                    node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                    num_node = len(node_list_i)
                    Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k, O_matrix, Q, num_node, alpha = 2, proj_coef=10*num_nodes) #TODO: Motivating choice of epsilon
        
            else:
                
                if fault_flag == False:
                    for i in range(num_nodes): 
                        node_list_i = np.array(np.nonzero(H_non_fault[i,:]))
                        node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                        num_node = len(node_list_i)
                        Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k, O_matrix, Q, num_node, alpha = 2, proj_coef=10*num_nodes, fault_flag=fault_flag) #TODO: Motivating choice of epsilon
            
                elif fault_flag == True:
                
                    for i in range(num_nodes): 
                        
                        node_list_i = np.array(np.nonzero(H_fault[i,:]))
                        node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                        Lambda_k_update = Lambda_k[node_list_i]
                        O_matrix_update = O_matrix[node_list_i]
                        num_node = len(node_list_i)
                        Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k_update, O_matrix_update, Q, num_node, alpha = 2, proj_coef=10*num_nodes, fault_flag=fault_flag) #TODO: Motivating choice of epsilon

                    for b in range(B + 5):
                        #print("Convergence of Z in progress, Iteration : {}". format(b))
                        #if b == 0: 
                            
                        #    for i in range(num_nodes):
                                
                        #        node_list_i = np.array(np.nonzero(H_fault[i,:]))
                        #        Lambda_k_update = Lambda_k[node_list_i]
                        #        O_matrix_update = O_matrix[node_list_i]
                        #        num_node = len(node_list_i)
                        #        Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k_update, O_matrix_update, Q, num_node, alpha = 2, proj_coef=10*num_nodes, fault_flag=True)

                        #else:
                        
                        for i in range(num_nodes):           
                            node_list_i = np.array(np.nonzero(H_fault[i,:]))
                            node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                            Z_k_update = Z_k[node_list_i]
                            Z_k[i] = np.sum(Z_k_update) * (1.0/len(node_list_i))
                    
                    
                    for i in range(num_nodes):
                        eps0 = alpha * np.sqrt(2*Q)            
                        Z_k[i] = project_fun(Z_k[i], eps0)
   
            #update lambda
            for i in range(num_nodes):
                Lambda_k[i] = Lambda_k[i] + O_matrix[i] - Z_k[i]
                error[i] += np.linalg.norm(O_matrix[i] - Z_k[i])

            #for i in range(num_nodes):
            #    Lambda_k[i] = Lambda_k[i] + O_matrix[i] - Z_k
            #    error[i] += np.linalg.norm(O_matrix[i] - Z_k)

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

def optimize_O_steps_Lwf(pln_net:PLN_network, X_train_whole, Y_train_whole, X_test,Y_test, Q, mu: np.float32=1e-1, alpha = 2,\
                         forgetting_factor = None, pln_all_new = None, H_fault=None, H_non_fault=None, fault_flag=False, B = None):

    num_nodes = len(pln_net)
    #Z_k = np.array([None]*num_nodes)
    #cnvge_flag = False
    O_matrix = np.array([None]*num_nodes) # Ouptut matrix for every node
    Lambda_k = np.array([None]*num_nodes) # ADMM helping / auxillary variable
    Z_k = np.array([None]*num_nodes) # ADMM helping / auxillary variable
    Signal_matrix = np.array([None]*num_nodes) # Represents the hidden activation
    O_matrix_new = np.array([None]*num_nodes)
    eps0 = alpha*np.sqrt(2*Q)
    Th = eps0*1e-4
    
    for num_layer in range(pln_net[0].num_layer):
        #converge_flag = np.array([False]*num_nodes)
        
        for i in range(num_nodes):
            if (num_layer==0):
                #pln_net[i].construct_one_layer(Y_train_whole[i], Q, X_train = X_train_whole[i])
                
                if i > 0:
                    pln_all_new[i].construct_one_layer(Y_train_whole[i], Q, X_train = X_train_whole[i], calculate_O=True, dec_flag=True, R_i = pln_net[0].pln[num_layer].R_l)
                else:
                    pln_all_new[i].construct_one_layer(Y_train_whole[i], Q, X_train = X_train_whole[i])
                
            else:
                #pln_net[i].construct_one_layer(Y_train_whole[i], Q, calculate_O=False)
                
                if i > 0:
                    pln_all_new[i].construct_one_layer(Y_train_whole[i], Q, X_train = None, calculate_O=False, dec_flag = True, R_i = pln_net[0].pln[num_layer].R_l)
                else:
                    pln_all_new[i].construct_one_layer(Y_train_whole[i], Q, X_train = None, calculate_O=False)
                
            Signal_matrix[i] = pln_all_new[i].pln[num_layer].Y_l
            #sample_num_previous = Signal_matrix[i].shape[1]
            #O_matrix[i] = pln_net[i].pln[num_layer].O_l
            # Extract signal vector and O matrix
            
            # Initialising these ADMM arrays
            #Z_k = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            Z_k[i] = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            O_matrix[i] = pln_net[i].pln[num_layer].O_l
            Lambda_k[i] = np.zeros( (Y_train_whole[i].shape[0], len(Signal_matrix[i])))
            #Z_k[i] = O_matrix[i] + Lambda_k[i]
        
        for k in range(pln_net[0].max_iterations):
            
            error = np.array([0.0]*num_nodes)

            # Optimize O matrix first            
            for i in range(num_nodes):

                O_matrix_new[i] = admm_decent_Only_O_Onetime_LwF(Signal_matrix[i], Y_train_whole[i], Lambda_k[i], Z_k[i], mu, O_matrix[i], forgetting_factor)
                pln_all_new[i].pln[num_layer].O_l = O_matrix_new[i]

            # Calculating the Z matrix for every node (effectively same thing is calculated by every node)    
            #for i in range(num_nodes):
            #    Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k, O_matrix, Q, proj_coef=10*num_nodes) #TODO: Motivating choice of epsilon
            
            if k <= (pln_net[0].max_iterations*1 // 2):
            
                for i in range(num_nodes): 

                    node_list_i = np.array(np.nonzero(H_non_fault[i,:]))
                    node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                    num_node = len(node_list_i)
                    Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k, O_matrix_new, Q, num_node, alpha = 2, proj_coef=10*num_nodes) #TODO: Motivating choice of epsilon
        
            else:
                
                if fault_flag == False:
                    for i in range(num_nodes): 
                        node_list_i = np.array(np.nonzero(H_non_fault[i,:]))
                        node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                        num_node = len(node_list_i)
                        Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k, O_matrix_new, Q, num_node, alpha = 2, proj_coef=10*num_nodes, fault_flag=fault_flag) #TODO: Motivating choice of epsilon
            
                elif fault_flag == True:
                
                    for i in range(num_nodes): 
                        
                        node_list_i = np.array(np.nonzero(H_fault[i,:]))
                        node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                        Lambda_k_update = Lambda_k[node_list_i]
                        O_matrix_update = O_matrix_new[node_list_i]
                        num_node = len(node_list_i)
                        Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k_update, O_matrix_update, Q, num_node, alpha = 2, proj_coef=10*num_nodes, fault_flag=fault_flag) #TODO: Motivating choice of epsilon

                    for b in range(B + 5):
                        #print("Convergence of Z in progress, Iteration : {}". format(b))
                        #if b == 0: 
                            
                        #    for i in range(num_nodes):
                                
                        #        node_list_i = np.array(np.nonzero(H_fault[i,:]))
                        #        Lambda_k_update = Lambda_k[node_list_i]
                        #        O_matrix_update = O_matrix[node_list_i]
                        #        num_node = len(node_list_i)
                        #        Z_k[i] = admm_decent_Only_Z_Onetime(Lambda_k_update, O_matrix_update, Q, num_node, alpha = 2, proj_coef=10*num_nodes, fault_flag=True)

                        #else:
                        
                        for i in range(num_nodes):           
                            node_list_i = np.array(np.nonzero(H_fault[i,:]))
                            node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                            Z_k_update = Z_k[node_list_i]
                            Z_k[i] = np.sum(Z_k_update) * (1.0/len(node_list_i))
                    
                    
                    for i in range(num_nodes):
                        eps0 = alpha * np.sqrt(2*Q)            
                        Z_k[i] = project_fun(Z_k[i], eps0)
   
            #update lambda
            for i in range(num_nodes):
                Lambda_k[i] = Lambda_k[i] + O_matrix_new[i] - Z_k[i]
                error[i] += np.linalg.norm(O_matrix_new[i] - Z_k[i])

            #for i in range(num_nodes):
            #    Lambda_k[i] = Lambda_k[i] + O_matrix[i] - Z_k
            #    error[i] += np.linalg.norm(O_matrix[i] - Z_k)

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
            predicted_lbl_test = pln_all_new[i].compute_test_outputs(X_test)
            print("Node No. : {}, Testing Accuracy:{}\n".format(i+1, compute_accuracy(predicted_lbl_test, Y_test)))
            print("Node No. : {}, Testing NME:{}\n".format(i+1, compute_NME(predicted_lbl_test, Y_test))) 
    
    return pln_all_new

def optimize_W_steps(pln_net:PLN_network, X_train_whole, Y_train_whole, rho: np.float32=1e-1, lambda_rate=1e2, \
                     alpha=2, H_fault=None, H_non_fault=None,fault_flag=False, B = None): # , W_matrix, Lambda_k, Z_k
    
    num_nodes = len(X_train_whole)
    
    W_matrix = np.array([None]*num_nodes)
    Lambda_k = np.array([None]*num_nodes)
    Z_k = np.array([None]*num_nodes)

    for i in range(num_nodes):
        W_matrix[i] = pln_net[i].W_ls
        Lambda_k[i] = np.zeros((Y_train_whole[i].shape[0], len(X_train_whole[i])) )
    
    #Z_k = np.mean(W_matrix)
    
    for i in range(num_nodes):
        Z_k[i] = np.mean(W_matrix)

    num = Y_train_whole[0].shape[0]
    eps0 = alpha*np.sqrt(2*num)
    Th = eps0*1e-4 #TODO: Motivating choice of eps0??
    
    for k in range(pln_net[0].max_iterations*1):
        
        error = np.array([0.0]*num_nodes)
             
        # Optimize O matrix first
        for i in range(num_nodes):
       
            W_matrix[i] = admm_decent_Only_W_Onetime(X_train_whole[i], Y_train_whole[i], Lambda_k[i], Z_k[i], rho)
            pln_net[i].W_ls = W_matrix[i]
        
        # Update the value of Z for every node
        if k <= (pln_net[0].max_iterations*1 // 2):
            print("ADMM_Iteration No.:{}, No Fault".format(k))
            for i in range(num_nodes): 
                Z_k[i] = admm_decent_Only_Z0_Onetime(Lambda_k, W_matrix, num_nodes, rho, lambda_rate)
        
        else:
            
            if fault_flag == False:
                for i in range(num_nodes): 
                    print("ADMM_Iteration No.:{}, No Fault".format(k))
                    num_node = int(1.0 / H_non_fault[i,i])
                    Z_k[i] = admm_decent_Only_Z0_Onetime(Lambda_k, W_matrix, num_node, rho, lambda_rate)
            
            elif fault_flag == True:
                
                print("ADMM_Iteration No.:{}, Activated Fault".format(k))
                for i in range(num_nodes): 
                    
                    node_list_i = np.array(np.nonzero(H_fault[i,:]))
                    node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                    Lambda_k_update = Lambda_k[node_list_i]
                    W_matrix_update = W_matrix[node_list_i]
                    num_node = int(1.0 / H_fault[i,i])
                    Z_k[i] = admm_decent_Only_Z0_Onetime(Lambda_k_update, W_matrix_update, num_node, rho, lambda_rate)

                for b in range(1, B+1):

                    #print("Convergence of Z in progress, Iteration number : {}".format(b))

                    #if b == 0: 
                    #    for i in range(num_nodes):
                            
                    #        node_list_i = np.array(np.nonzero(H_fault[i,:]))
                    #        Lambda_k_update = Lambda_k[node_list_i]
                    #        W_matrix_update = W_matrix[node_list_i]
                    #        num_node = len(node_list_i)
                    #        Z_k[i] = admm_decent_Only_Z0_Onetime(Lambda_k_update, W_matrix_update, num_node, rho, lambda_rate)

                    #else:

                    for i in range(num_nodes):           
                        node_list_i = np.array(np.nonzero(H_fault[i,:]))
                        Z_k_update = Z_k[node_list_i]
                        num_node = int(1.0 / H_fault[i,i])
                        Z_k[i] = np.sum(Z_k_update) * (1.0/num_node)
                
        #Z_k = admm_decent_Only_Z0_Onetime(Lambda_k, W_matrix, num_nodes, rho, lambda_rate)
            
        #update lambda
        for i in range(num_nodes):
            Lambda_k[i] = Lambda_k[i] + W_matrix[i] - Z_k[i]
            error[i] += np.linalg.norm(W_matrix[i] - Z_k[i])

        #for i in range(num_nodes):
        #    Lambda_k[i] = Lambda_k[i] + W_matrix[i] - Z_k
        #    error[i] += np.linalg.norm(W_matrix[i] - Z_k)

        # Z is the average! all the O matrix should converge to Z!

        if np.mean(error) < Th:
            #cnvge_flag = True
            print("Converge!")
            break

    #print("Threshold:", Th)
    #print("Residual error difference of O matrix:{}".format(error))
    #print("Residual error difference of O matrix:{}".format(error))

    return pln_net

def optimize_W_steps_Lwf(pln_net:PLN_network, X_train_whole, Y_train_whole, pln_all_new, \
                         rho: np.float32=1e-1, lambda_rate=1e2, alpha=2, forgetting_factor = None, \
                         H_fault=None, H_non_fault=None,fault_flag=False, B = None): # , W_matrix, Lambda_k, Z_k
    ### the second input is previous network, to extract the W_ls
    num_nodes = len(X_train_whole)
    
    W_matrix = np.array([None]*num_nodes)
    Lambda_k = np.array([None]*num_nodes)
    W_matrix_new = np.array([None]*num_nodes)
    Z_k = np.array([None]*num_nodes)

    for i in range(num_nodes):
        W_matrix[i] = pln_net[i].W_ls
        Lambda_k[i] = np.zeros((Y_train_whole[i].shape[0], len(X_train_whole[i])) )
    
    #Z_k = np.mean(W_matrix)
    for i in range(num_nodes):
        Z_k[i] = np.mean(W_matrix)

    num = Y_train_whole[0].shape[0]
    eps0 = alpha*np.sqrt(2*num)
    Th = eps0*1e-4 # Motivating choice of eps0??
    #forgetting_factor = 1

    #for _ in range(pln_net[0].max_iterations*1):
    #    error = np.array([0.0]*num_nodes)
    #    for i in range(num_nodes):
            # Optimize O matrix first
    #        W_matrix_new[i] = admm_decent_Only_W_Onetime_LwF(X_train_whole[i], Y_train_whole[i], Lambda_k[i], Z_k, W_matrix[i], forgetting_factor, )
    #        pln_all_new[i].W_ls = W_matrix_new[i]
        
    #    Z_k = admm_decent_Only_Z0_Onetime(Lambda_k, W_matrix_new, num_nodes)
            #update lambda
        
    #    for i in range(num_nodes):
    #        Lambda_k[i] = Lambda_k[i] + W_matrix_new[i] - Z_k
    #        error[i] += np.linalg.norm(W_matrix[i] - Z_k)
        # Z is the average! all the O matrix should converge to Z!

    for k in range(pln_net[0].max_iterations*1):
        
        error = np.array([0.0]*num_nodes)
             
        # Optimize O matrix first
        for i in range(num_nodes):
       
            W_matrix_new[i] = admm_decent_Only_W_Onetime_LwF(X_train_whole[i], Y_train_whole[i], Lambda_k[i], Z_k[i], rho,  W_matrix[i], forgetting_factor)
            pln_all_new[i].W_ls = W_matrix_new[i]
        
        # Update the value of Z for every node
        if k <= (pln_net[0].max_iterations*1 // 2):
            print("ADMM_Iteration No.:{}, No Fault".format(k))
            for i in range(num_nodes): 
                Z_k[i] = admm_decent_Only_Z0_Onetime(Lambda_k, W_matrix_new, num_nodes, rho, lambda_rate)
        
        else:
            
            if fault_flag == False:
                for i in range(num_nodes): 
                    print("ADMM_Iteration No.:{}, No Fault".format(k))
                    num_node = int(1.0 / H_non_fault[i,i])
                    Z_k[i] = admm_decent_Only_Z0_Onetime(Lambda_k, W_matrix_new, num_node, rho, lambda_rate)
            
            elif fault_flag == True:
                
                print("ADMM_Iteration No.:{}, Activated Fault".format(k))
                for i in range(num_nodes): 
                    
                    node_list_i = np.array(np.nonzero(H_fault[i,:]))
                    node_list_i = node_list_i.reshape((node_list_i.shape[1],))
                    Lambda_k_update = Lambda_k[node_list_i]
                    W_matrix_update = W_matrix_new[node_list_i]
                    num_node = int(1.0 / H_fault[i,i])
                    Z_k[i] = admm_decent_Only_Z0_Onetime(Lambda_k_update, W_matrix_update, num_node, rho, lambda_rate)

                for b in range(1, B+1):

                    #print("Convergence of Z in progress, Iteration number : {}".format(b))

                    #if b == 0: 
                    #    for i in range(num_nodes):
                            
                    #        node_list_i = np.array(np.nonzero(H_fault[i,:]))
                    #        Lambda_k_update = Lambda_k[node_list_i]
                    #        W_matrix_update = W_matrix[node_list_i]
                    #        num_node = len(node_list_i)
                    #        Z_k[i] = admm_decent_Only_Z0_Onetime(Lambda_k_update, W_matrix_update, num_node, rho, lambda_rate)

                    #else:

                    for i in range(num_nodes):           
                        node_list_i = np.array(np.nonzero(H_fault[i,:]))
                        Z_k_update = Z_k[node_list_i]
                        num_node = int(1.0 / H_fault[i,i])
                        Z_k[i] = np.sum(Z_k_update) * (1.0/num_node)
                
        #Z_k = admm_decent_Only_Z0_Onetime(Lambda_k, W_matrix, num_nodes, rho, lambda_rate)
            
        #update lambda
        for i in range(num_nodes):
            Lambda_k[i] = Lambda_k[i] + W_matrix_new[i] - Z_k[i]
            error[i] += np.linalg.norm(W_matrix_new[i] - Z_k[i])

        if np.mean(error) < Th:
            #cnvge_flag = True
            print("Converge!")
            break

    #print("Threshold:", Th)
    #print("Residual error difference of O matrix:{}".format(error))
    #print("Residual error difference of O matrix:{}".format(error))
    return pln_all_new


def train_decentralized_networks(X_train_whole, Y_train_whole, X_test, Y_test, Q, mu,\
                                 lambda_admm, rho, alpha, num_layers, H_fault=None, \
                                 H_non_fault=None, fault_flag=False, B = None):
    
    num_nodes = len(X_train_whole)
    if (num_nodes != len(Y_train_whole)):
        print("Input parameter Error!")
        return None 

    pln_all=[] # returned value, bunches of PLN network
    for i in range(num_nodes):
        pln_all.append(PLN_network(num_layer = num_layers, mu=mu, maxit=100, lamba=lambda_admm, rho=rho))
        # deflult: num_layer = 20, mu=1e3, maxit=30, lamba=1e2
        pln_all[i].construct_W(X_train_whole[i], Y_train_whole[i])

    #print("W_ls Before joint optimization:")

    for i in range(num_nodes):
        predicted_lbl_test = pln_all[i].compute_test_outputs(X_test)
        #print("Node No. : {}, Testing Accuracy:{}\n".format(i+1, compute_accuracy(predicted_lbl_test, Y_test)))
        #print("Node No. : {}, Testing NME:{}\n".format(i+1, compute_NME(predicted_lbl_test, Y_test))) 
        
    pln_all = optimize_W_steps(pln_all, X_train_whole, Y_train_whole, rho, lambda_admm, alpha, H_fault, H_non_fault, fault_flag, B = B)

    print("W_ls After joint optimization:")
    for i in range(num_nodes):
        predicted_lbl_test = pln_all[i].compute_test_outputs(X_test)
        print("Node No. : {}, Testing Accuracy:{}\n".format(i+1, compute_accuracy(predicted_lbl_test, Y_test)))
        print("Node No. : {}, Testing NME:{}\n".format(i+1, compute_NME(predicted_lbl_test, Y_test))) 
    # ADMM implenmentation for W matrix.

    pln_all = optimize_O_steps(pln_all, X_train_whole, Y_train_whole, X_test, Y_test, Q, mu=mu, alpha=alpha, H_fault=H_fault, H_non_fault=H_non_fault, fault_flag=fault_flag, B = B)
        
    return pln_all

def train_centralized_networks(X_train, Y_train, X_test, Y_test, Q, mu, num_layers):  # X_test, Y_test,
    
    pln_net=PLN_network(num_layer = num_layers, mu=mu, maxit=100)
    
    pln_net.construct_W(X_train, Y_train, mu)
    
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

def train_centralized_networks_LwF_same(X_train, Y_train, X_test, Y_test, Q, mu, num_layers, pln_old, LwF_flag):  # X_test, Y_test,
    
    # Implement Parameter tuning in the Wls layer here, not in PLN_Class
    pln_net=PLN_network(num_layer = num_layers, mu=mu, maxit=100)
    
    lambda_o = None # Initially no choice for forgetting factor
    lambda_o_optimal_ls, mu_optimal_ls = param_tuning_for_LS_centralised(pln_net, X_train, Y_train, X_test, Y_test, lambda_o, mu, pln_net.max_iterations, pln_old.W_ls)
    pln_net.construct_W(X_train, Y_train, mu_optimal_ls, lambda_o_optimal_ls, LwF_flag, pln_old.W_ls, max_iterations=100) # Implelement LwF in compute_ol part

    predicted_lbl_test = pln_net.compute_test_outputs(X_test)
    print("Testing Accuracy:{}\n".format(compute_accuracy(predicted_lbl_test, Y_test)))
    print("Testing NME:{}\n".format(compute_NME(predicted_lbl_test, Y_test))) 
    
    # Implement a function which can be used for optimising the other layers of the network
    lambda_o = None
    lambda_o_optimal, mu_optimal = param_tuning_for_O_centralised(pln_net, X_train, Y_train, X_test, Y_test, lambda_o, mu, pln_net.max_iterations, pln_old)
    pln_net.mu = mu_optimal

    '''
    for num_layer in range(pln_net.num_layer):

        pln_net.construct_one_layer(Y_train, Q, X_train=X_train) #TODO: Implement LwF in the compute_ol function 
        predicted_lbl_test = pln_net.compute_test_outputs(X_test)
        #TODO: Implement LwF Modification here 
        print("Layer: num {}. Testing Accuracy:{}\n".format(num_layer+1, compute_accuracy(predicted_lbl_test, Y_test)))
        print("Layer: num {}. Testing NME:{}\n".format(num_layer+1, compute_NME(predicted_lbl_test, Y_test))) 
    '''

    for num_layer in range(pln_net.num_layer):

        pln_net.construct_one_layer(Y_train, Q, X_train=X_train, calculate_O = True, dec_flag = False, \
                                            R_i = None, LwF_flag = True, pln_Old = pln_old, lambda_o = lambda_o_optimal, mu = mu_optimal)  
            
        predicted_lbl_test = pln_net.compute_test_outputs(X_test)
        acc_test_O = compute_accuracy(predicted_lbl_test, Y_test)
        nme_test_O = compute_NME(predicted_lbl_test, Y_test)
        print("Layer: num {}. Testing Accuracy:{}\n".format(num_layer+1, acc_test_O))
        print("Layer: num {}. Testing NME:{}\n".format(num_layer+1, nme_test_O))
        #pln_net.construct_all_layers(X_train, Y_train, Q)  
    
    return pln_net

def train_decentralized_networks_LwF_same(X_train_new_whole, Y_train_new_whole, X_test, Y_test, Q, \
                                          mu, num_layers, pln_old, forgetting_factor, \
                                          lambda_admm, rho, alpha, H_fault=None, H_non_fault=None,
                                          fault_flag=None, B=None):  # X_test, Y_test,
    
    num_nodes = len(X_train_new_whole)
    if (num_nodes != len(Y_train_new_whole)):
        print("Input parameter Error!")
        return None 

    pln_all_new =[] # returned value, bunches of PLN network
    for i in range(num_nodes):
        pln_all_new.append(PLN_network(num_layer = num_layers, mu=mu, maxit=100, lamba=lambda_admm, rho=rho))
        # deflult: num_layer = 20, mu=1e3, maxit=30, lamba=1e2
    #    pln_all[i].construct_W(X_train_new_whole[i], Y_train_new_whole[i])

    #print("W_ls Before joint optimization:")

    #for i in range(num_nodes):
    #    predicted_lbl_test = pln_all_new[i].compute_test_outputs(X_test)
        #print("Node No. : {}, Testing Accuracy:{}\n".format(i+1, compute_accuracy(predicted_lbl_test, Y_test)))
        #print("Node No. : {}, Testing NME:{}\n".format(i+1, compute_NME(predicted_lbl_test, Y_test))) 
        
    pln_all_new = optimize_W_steps_Lwf(pln_old, X_train_new_whole, Y_train_new_whole, pln_all_new, rho, lambda_admm, alpha, forgetting_factor, H_fault, H_non_fault, fault_flag, B = B)

    print("W_ls After joint optimization:")
    for i in range(num_nodes):
        predicted_lbl_test = pln_all_new[i].compute_test_outputs(X_test)
        print("Node No. : {}, Testing Accuracy:{}\n".format(i+1, compute_accuracy(predicted_lbl_test, Y_test)))
        print("Node No. : {}, Testing NME:{}\n".format(i+1, compute_NME(predicted_lbl_test, Y_test))) 
    # ADMM implenmentation for W matrix.

    pln_all_new = optimize_O_steps_Lwf(pln_old, X_train_new_whole, Y_train_new_whole, X_test, Y_test, Q, mu=mu, alpha=alpha,\
                                   forgetting_factor=forgetting_factor,pln_all_new=pln_all_new, H_fault=H_fault, \
                                   H_non_fault=H_non_fault, fault_flag=fault_flag, B = B)
        
    return pln_all_new  

##########################################################################################
# This function is used to define the mixing matrix for Fault Tolerant framework
##########################################################################################
def compute_mixing_matrix(num_nodes, hidden_nodes = None):

    # Decide the ideal matrix for the number of nodes presents
    H_no_fault = (1.0/num_nodes)*np.ones((num_nodes, num_nodes)) # This is what is supposed to be obtained in case network is fully connected
    
    # Decide the actual matrix
    if hidden_nodes != None: 
        
        # Assuming hidden_nodes is a list of tuples, where each tuple represents a link that is broken (fault has occured)
        J = copy.deepcopy(H_no_fault) * num_nodes
        #I = np.eye(num_nodes, num_nodes)
        #T = np.flipud(I)
        for i in range(len(hidden_nodes)):
            h_ij = hidden_nodes[i]  # get the co-ordinate,which indicates the deleted branch
            x = h_ij[0]
            y = h_ij[1]
            J[x, y] = 0 # Setting those values to be zero
            J[y, x] = 0 # Setting those values to be zero

        #J = (J + J.T)/2
        #Num_of_links_actual = num_nodes*2 + 2 - len(hidden_nodes)
        #eig_val, eig_vec = np.linalg.eig(J)
        norm_factor = 1 / np.sum(J, axis=1)
        for i in range(len(norm_factor)):
            J[i,:] = J[i,:] * norm_factor[i] 
        
        H_fault = copy.deepcopy(J)
        # Check doubly stochastic:
        if np.all(np.sum(H_fault, axis=1) == np.ones((num_nodes,))):
            print("H is Right Stochastic!!")

        if np.all(np.sum(H_fault, axis=0) == np.ones((num_nodes,))):
            print("H is Left Stochastic!!")

        # Check symmetry:
        if np.all(H_fault.T == H_fault):
            print("H is Symmetric")

        print("H:\n{}".format(H_fault))

        # Using the actual value of H during fault, trying to find out when H converges to H_no_fault
        H = copy.deepcopy(H_fault)
        b = 1
        error = np.linalg.norm(H - H_no_fault, 'fro')
        th = 1e-6
        while error > th:
            H = np.matmul(H, H)
            print(H)
            error = np.linalg.norm(H - H_no_fault, 'fro')
            print("Iteration for b:{}, and error:{}".format(b, error))
            b = b + 1

        print("Optimal value of B (no. of iterations for convergence of Z) : {}".format(b))

    else:

        # No fault in network
        H_fault = (1.0/num_nodes)*np.ones((num_nodes, num_nodes))
        b = None
        print("H:{}".format(H_fault))
        
    return H_fault, H_no_fault, b

##########################################################################################
# This function performs parameter tuning for the case of same datasets in 
# case of centralised scenario for Least Sqaures
##########################################################################################

def param_tuning_for_LS_centralised(pln_net, X_train, Y_train, X_test, Y_test, lambda_o, mu, max_iterations, W_ls_prev):
    
    # Input arguments:
    # pln_net : New network object
    # X_train :  Dataset Input
    # Y_train : Dataset target
    # X_test : Test set input
    # Y_test : Test set label
    # lambda_o : Forgetting factor present if any
    # max_iterations :  No. of max iterations
    # W_ls_prev : The Least Square Matrix as found for the 'Old Task'

    # Output:
    # W_ls : The least square matrix for the 'New Task'

    # Tuning the value of lambda_o
    if lambda_o == None:

        lambda_o_vec = np.geomspace(1e-6, 1e9, 16)
        test_acc_vec = []

        for lambda_o in lambda_o_vec:

            pln_net.construct_W(X_train, Y_train, mu, lambda_o, True, W_ls_prev, max_iterations=100) # Implelement LwF in compute_ol part
            predicted_lbl_test = pln_net.compute_test_outputs(X_test)
            acc_test_Wls = compute_accuracy(predicted_lbl_test, Y_test)
            nme_test_Wls = compute_NME(predicted_lbl_test, Y_test)
            print("Testing Accuracy:{}\n".format(acc_test_Wls))
            print("Testing NME:{}\n".format(nme_test_Wls)) 
            test_acc_vec.append(acc_test_Wls)
        
         # Plotting the results of the param sweep
        title = "Plotting test accuracy for LwF for LS versus lambda_o (Same dataset)"
        plot_acc_vs_hyperparam(np.log10(lambda_o_vec), test_acc_vec, title)
        lambda_o_optimal = lambda_o_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy

        # Tuning the value of mu
        mu_vec = np.geomspace(1e-14, 1e14, 29)
        test_acc_vec = []

        for mu in mu_vec:

            pln_net.construct_W(X_train, Y_train, mu, lambda_o_optimal, True, W_ls_prev, max_iterations=100) # Implelement LwF in compute_ol part
            predicted_lbl_test = pln_net.compute_test_outputs(X_test)
            acc_test_Wls = compute_accuracy(predicted_lbl_test, Y_test)
            nme_test_Wls = compute_NME(predicted_lbl_test, Y_test)
            print("Testing Accuracy:{}\n".format(acc_test_Wls))
            print("Testing NME:{}\n".format(nme_test_Wls)) 
            test_acc_vec.append(acc_test_Wls)
        
         # Plotting the results of the param sweep
        title = "Plotting test accuracy for LwF for LS versus mu (Same dataset)"
        plot_acc_vs_hyperparam(np.log10(mu_vec), test_acc_vec, title)
        mu_optimal = mu_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
    
    return  lambda_o_optimal, mu_optimal

##########################################################################################
# This function performs parameter tuning for the case of same datasets in 
# case of centralised scenario for Least Sqaures
##########################################################################################

def param_tuning_for_O_centralised(pln_net, X, T, X_test, Y_test, lambda_o, mu, max_iterations, pln_old):
    
    # Input arguments:
    # pln_net : New network object
    # Y :  Hidden activation
    # T : Dataset target
    # X_test : Test set input
    # Y_test : Test set label
    # lambda_o : Forgetting factor present if any
    # max_iterations :  No. of max iterations
    # pln_old : The first dataset output network

    # Output:
    # W_ls : The least square matrix for the 'New Task'

    Q = T.shape[0] # No. of classes in target set
    # Tuning the value of lambda_o
    if lambda_o == None:

        lambda_o_vec = np.geomspace(1e-6, 1e9, 16)
        test_acc_vec = []
        mu = 1e3
        for lambda_o in lambda_o_vec:
            
            for num_layer in range(pln_net.num_layer):
                pln_net.construct_one_layer(T, Q, X_train=X, calculate_O = True, dec_flag = False, \
                                            R_i = None, LwF_flag = True, pln_Old = pln_old, \
                                            lambda_o = lambda_o, mu= mu)  
            
            predicted_lbl_test = pln_net.compute_test_outputs(X_test)
            acc_test_O = compute_accuracy(predicted_lbl_test, Y_test)
            nme_test_O = compute_NME(predicted_lbl_test, Y_test)
            print("Layer: num {}. Testing Accuracy:{}\n".format(num_layer+1, acc_test_O))
            print("Layer: num {}. Testing NME:{}\n".format(num_layer+1, nme_test_O))
            test_acc_vec.append(acc_test_O)

            # Clean up the PLN_network
            for num_layer in range(pln_net.num_layer):
                pln_net.pln[num_layer] = None

         # Plotting the results of the param sweep
        title = "Plotting test accuracy for LwF for O versus lambda_o (Same dataset)"
        plot_acc_vs_hyperparam(np.log10(lambda_o_vec), test_acc_vec, title)
        lambda_o_optimal = lambda_o_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy

        # Tuning the value of mu
        mu_vec = np.geomspace(1e-14, 1e14, 29)
        test_acc_vec = []

        for mu in mu_vec:
            
            for num_layer in range(pln_net.num_layer):
                pln_net.construct_one_layer(T, Q, X_train=X, calculate_O = True, dec_flag = False, \
                                            R_i = None, LwF_flag = True, pln_Old = pln_old, lambda_o = lambda_o_optimal, mu= mu)  
            
            predicted_lbl_test = pln_net.compute_test_outputs(X_test)
            acc_test_O = compute_accuracy(predicted_lbl_test, Y_test)
            nme_test_O = compute_NME(predicted_lbl_test, Y_test)
            print("Layer: num {}. Testing Accuracy:{}\n".format(num_layer+1, acc_test_O))
            print("Layer: num {}. Testing NME:{}\n".format(num_layer+1, nme_test_O))
            test_acc_vec.append(acc_test_O)
        
            # Clean up the PLN_network
            for num_layer in range(pln_net.num_layer):
                pln_net.pln[num_layer] = None
            
         # Plotting the results of the param sweep
        title = "Plotting test accuracy for LwF for O versus mu (Same dataset)"
        plot_acc_vs_hyperparam(np.log10(mu_vec), test_acc_vec, title)
        mu_optimal = mu_vec[np.argmax(np.array(test_acc_vec))] # Choosing the mu value that gives maximum accuracy
    
    return lambda_o_optimal, mu_optimal

#######################################################################################################
# Function to Plot variation of test accuracy versus hyperparameter
#######################################################################################################
def plot_acc_vs_hyperparam(hyperparam_vec, test_acc_vec, title):
    
    plt.figure(1)
    plt.plot(hyperparam_vec, test_acc_vec, 'b--', linewidth=2.0, markersize=3)
    #plt.plot(hyperparam_vec, test_acc_vec_1, 'r+-', linewidth=1.5, markersize=2)
    #plt.plot(hyperparam_vec, test_acc_vec_2, 'go-', linewidth=1.5, markersize=2)
    plt.xlabel('Hyperparameter Value')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    plt.legend(["Test_Accuracy"])
    #plt.legend(["Joint Testing", "Old Task testing", "New Task testing"])
    plt.show()
    return None

def main():

    dataset_path = "../../Datasets/" # Specify the dataset path in the Local without the name
    dataset_name = "Vowel" # Specify the Dataset name without extension (implicitly .mat extension is assumed)
    X_train, Y_train, X_test, Y_test, Q = importData(dataset_path, dataset_name) # Imports the data with the no. of output classes
    num_nodes=5
    num_layers = 10
    #X_train_whole, Y_train_whole = split_training_set(X_train, Y_train, num_nodes)
    X_train_whole, Y_train_whole = split_data_uniform(X_train, Y_train, num_nodes)
    # return    X_train_whole[0:4]  and  Y_train_whole[0:4]
    
    # Formulate the mixing matrix
    hidden_node_list = [(0,2), (0,3), (1, 4), (1, 3), (2,4)]
    #nbd_list = num_nodes * np.ones((num_nodes,1))
    H_fault, H_no_fault, B_iterations = compute_mixing_matrix(num_nodes, hidden_nodes=hidden_node_list)

    # Dataset related parameters to be set manually
    lambda_admm = 1e2 
    mu_centralised = 1e3
    mu_decentralised = 1e-1 
    rho = 1e-1
    alpha = 2

    print("***************** Centralized Version *****************")
    pln_cent = train_centralized_networks(X_train, Y_train, X_test, Y_test, Q, mu_centralised, num_layers)
    predicted_lbl_test_cent = pln_cent.compute_test_outputs(X_test)
    
    print("***************** Decentralized Version *****************")
    pln_decent = train_decentralized_networks(X_train_whole, Y_train_whole, X_test, \
                                              Y_test, Q, mu_decentralised, lambda_admm, \
                                              rho, alpha, num_layers, H_fault=H_fault, H_non_fault=H_no_fault,\
                                              fault_flag=False, B = None)
    
    test_acc_consensus = []
    test_nme_consensus = []
    for i in range(num_nodes):
        predicted_lbl_test = pln_decent[i].compute_test_outputs(X_test)
        print("Network No.{}. Test Accuracy:{}\n".format(i, compute_accuracy(predicted_lbl_test, Y_test)))
        print("Network No.{}. Test NME:{}\n".format(i, compute_NME(predicted_lbl_test, Y_test))) 
        test_acc_consensus.append(compute_accuracy(predicted_lbl_test, Y_test))
        test_nme_consensus.append(compute_NME(predicted_lbl_test, Y_test))
    
    print("**************** Final Results (Centralised) ************************")
    print("Test Accuracy:{}\n".format(compute_accuracy(predicted_lbl_test_cent, Y_test)))
    print("Test NME:{}\n".format(compute_NME(predicted_lbl_test_cent, Y_test))) 
    
    print("**************** Final Results (Decentralised) ************************")
    print("Mean Test Accuracy (Decentralised):{}".format(np.mean(np.array(test_acc_consensus))))
    print("Mean Test NME (Decentralised):{}".format(np.mean(np.array(test_nme_consensus))))
    
    print("***************** Decentralized Version with Fault Tolerance *****************")
    pln_decent_ft = train_decentralized_networks(X_train_whole, Y_train_whole, X_test, \
                                              Y_test, Q, mu_decentralised, lambda_admm, \
                                              rho, alpha, num_layers, H_fault=H_fault, H_non_fault=H_no_fault,\
                                              fault_flag=True, B = B_iterations)
    
    test_acc_consensus_ft = []
    test_nme_consensus_ft = []
    for i in range(num_nodes):
        predicted_lbl_test = pln_decent_ft[i].compute_test_outputs(X_test)
        print("Network No.{}. Test Accuracy:{}\n".format(i, compute_accuracy(predicted_lbl_test, Y_test)))
        print("Network No.{}. Test NME:{}\n".format(i, compute_NME(predicted_lbl_test, Y_test))) 
        test_acc_consensus_ft.append(compute_accuracy(predicted_lbl_test, Y_test))
        test_nme_consensus_ft.append(compute_NME(predicted_lbl_test, Y_test))
    '''
    print("**************** Final Results (Centralised) ************************")
    print("Test Accuracy:{}\n".format(compute_accuracy(predicted_lbl_test_cent, Y_test)))
    print("Test NME:{}\n".format(compute_NME(predicted_lbl_test_cent, Y_test))) 
    
    print("**************** Final Results (Decentralised) ************************")
    print("Mean Test Accuracy (Decentralised):{}".format(np.mean(np.array(test_acc_consensus))))
    print("Mean Test NME (Decentralised):{}".format(np.mean(np.array(test_nme_consensus))))
    
    print("**************** Final Results (Decentralised with FT) ************************")
    print("Mean Test Accuracy (Decentralised with FT):{}".format(np.mean(np.array(test_acc_consensus_ft))))
    print("Mean Test NME (Decentralised with FT):{}".format(np.mean(np.array(test_nme_consensus_ft))))
    '''
    
    ##########################################################################################################
    # Implementation of Centralised Scenario with LwF for Same Datasets
    ##########################################################################################################
    X_old_train, X_new_train, Y_old_train, Y_new_train = splitdatarandom(X_train, Y_train, split_percent=0.5) # Randomly splitting training 
    
    print("***************** Centralized Version (using LwF) *****************")
    
    pln_cent_old = train_centralized_networks(X_old_train, Y_old_train, X_test, Y_test, Q, mu_centralised, num_layers)
    predicted_lbl_test_cent_old = pln_cent_old.compute_test_outputs(X_test)
    
    pln_cent_new = train_centralized_networks_LwF_same(X_new_train, Y_new_train, X_test, Y_test, Q, mu_centralised, num_layers, pln_cent_old, LwF_flag=True)
    predicted_lbl_test_cent_new = pln_cent_new.compute_test_outputs(X_test) 
    
    print("**************** Final Results (Centralised) ************************")
    print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test_cent, Y_test)))
    print("Test NME:{}\n".format(compute_NME(predicted_lbl_test_cent, Y_test))) 

    #print("**************** Final Results (Centralised - Old ) ************************")
    #print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test_cent_old, Y_test)))
    #print("Test NME:{}\n".format(compute_NME(predicted_lbl_test_cent_old, Y_test))) 

    print("**************** Final Results (Centralised - New, after LwF) ************************")
    print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test_cent_new, Y_test)))
    print("Test NME:{}\n".format(compute_NME(predicted_lbl_test_cent_new, Y_test)))                                                                               # set into old and new task
    
    ##########################################################################################################
    # Implementation of Decentralised Scenario with LwF for Same Datasets
    ##########################################################################################################
    # Use LwF for decentralised scenario for Old and New Datasets

    X_old_train_whole, Y_old_train_whole = split_data_uniform(X_old_train, Y_old_train, num_nodes)
    X_new_train_whole, Y_new_train_whole = split_data_uniform(X_new_train, Y_new_train, num_nodes)
    pln_decent_lwf_old = train_decentralized_networks(X_old_train_whole, Y_old_train_whole, X_test, Y_test, Q, \
                                                      mu_decentralised, lambda_admm, rho, alpha, num_layers,\
                                                      H_fault=H_fault, H_non_fault=H_no_fault, fault_flag=False, B=None)


    mu_optimal = 10
    forgetting_factor_optimal = 1.0

    pln_decent_lwf_new = train_decentralized_networks_LwF_same(X_new_train_whole, Y_new_train_whole, X_test, Y_test,\
                                                               Q, mu_optimal, num_layers, pln_decent_lwf_old,\
                                                               forgetting_factor_optimal, lambda_admm, rho, alpha, H_fault=H_fault,\
                                                               H_non_fault=H_no_fault, fault_flag=False, B=None)
    test_acc_consensus_LwF = []
    test_nme_consensus_LwF = []
    for i in range(num_nodes):
        predicted_lbl_test_LwF = pln_decent_lwf_new[i].compute_test_outputs(X_test)
        print("Network No.{}. Test Accuracy:{}\n".format(i, compute_accuracy(predicted_lbl_test_LwF, Y_test)))
        print("Network No.{}. Test NME:{}\n".format(i, compute_NME(predicted_lbl_test_LwF, Y_test))) 
        test_acc_consensus_LwF.append(compute_accuracy(predicted_lbl_test_LwF, Y_test))
        test_nme_consensus_LwF.append(compute_NME(predicted_lbl_test_LwF, Y_test))
    
    ########################################################################################
    # Implement the LwF with Fault Tolerance
    ########################################################################################
    
    pln_decent_lwf_old_ft = train_decentralized_networks(X_old_train_whole, Y_old_train_whole, X_test, Y_test, Q, \
                                                      mu_decentralised, lambda_admm, rho, alpha, num_layers,\
                                                      H_fault=H_fault, H_non_fault=H_no_fault, fault_flag=True, B=B_iterations)


    mu_optimal = 10
    forgetting_factor_optimal = 1.0

    pln_decent_lwf_new_ft = train_decentralized_networks_LwF_same(X_new_train_whole, Y_new_train_whole, X_test, Y_test,\
                                                               Q, mu_optimal, num_layers, pln_decent_lwf_old_ft,\
                                                               forgetting_factor_optimal, lambda_admm, rho, alpha, H_fault=H_fault,\
                                                               H_non_fault=H_no_fault, fault_flag=True, B=B_iterations)
    test_acc_consensus_LwF_ft = []
    test_nme_consensus_LwF_ft = []
    for i in range(num_nodes):
        predicted_lbl_test_LwF_ft = pln_decent_lwf_new_ft[i].compute_test_outputs(X_test)
        print("Network No.{}. Test Accuracy:{}\n".format(i, compute_accuracy(predicted_lbl_test_LwF_ft, Y_test)))
        print("Network No.{}. Test NME:{}\n".format(i, compute_NME(predicted_lbl_test_LwF_ft, Y_test))) 
        test_acc_consensus_LwF_ft.append(compute_accuracy(predicted_lbl_test_LwF_ft, Y_test))
        test_nme_consensus_LwF_ft.append(compute_NME(predicted_lbl_test_LwF_ft, Y_test))

    print("**************** Final Results (Centralised) ************************")
    print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test_cent, Y_test)))
    print("Test NME:{}".format(compute_NME(predicted_lbl_test_cent, Y_test))) 

    print("**************** Final Results (Centralised - New, after LwF) ************************")
    print("Test Accuracy:{}".format(compute_accuracy(predicted_lbl_test_cent_new, Y_test)))
    print("Test NME:{}\n".format(compute_NME(predicted_lbl_test_cent_new, Y_test)))   
    
    print("**************** Final Results (Decentralised) ************************")
    print("Mean Test Accuracy (Decentralised):{}".format(np.mean(np.array(test_acc_consensus))))
    print("Mean Test NME (Decentralised):{}".format(np.mean(np.array(test_nme_consensus))))

    print("**************** Final Results (Decentralised with FT) ************************")
    print("Mean Test Accuracy (Decentralised with FT):{}".format(np.mean(np.array(test_acc_consensus_ft))))
    print("Mean Test NME (Decentralised with FT):{}".format(np.mean(np.array(test_nme_consensus_ft))))

    print("**************** Final Results (Decentralised after LwF) ************************")
    print("Mean Test Accuracy (Decentralised-LwF):{}".format(np.mean(np.array(test_acc_consensus_LwF))))
    print("Mean Test NME (Decentralised-LwF):{}".format(np.mean(np.array(test_nme_consensus_LwF))))

    print("**************** Final Results (Decentralised after LwF with FT) ************************")
    print("Mean Test Accuracy (Decentralised-LwF-FT):{}".format(np.mean(np.array(test_acc_consensus_LwF_ft))))
    print("Mean Test NME (Decentralised-LwF-FT):{}".format(np.mean(np.array(test_nme_consensus_LwF_ft))))

    return None

if __name__ == "__main__":
    main()
