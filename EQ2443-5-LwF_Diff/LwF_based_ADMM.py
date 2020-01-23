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
import numpy as np 
import copy

def LwF_based_ADMM_Diff_LS(X2, T2, Wls1_star, lambda_o, epsilon_ls, mu_ls):

    # Input params :
    # X2 : New Dataset Input
    # Y2 : New Dataset Output
    # O1_star : Output matrix for the Training on the Old Dataset
    # lambda_o : Forgetting factor for the LS problem (using ADMM)
    # epsilon_ls : Projection_Factor for the LS problem (using ADMM)
    # mu_ls : ADMM multiplier which is being optimized / already optimized

    # Output params:
    # W_tilde : Composite matrix containing jointly optimized W1 and W2 (Layer 0)
    
    #print("ADMM for LS (for LwF)")
    #print("Forgetting factor:{}".format(lambda_o))
    #print("ADMM Multiplier(mu):{}".format(mu_ls))

    # ADMM Iteration Related parameters
    num_iterations_admm = 100 # No. of ADMM Iterations
    
    Q1 = Wls1_star.shape[0] # Output Dimension in the first dataset (Old)
    P1 = Wls1_star.shape[1] # Input dimension in the first dataset (Old)
    Q2 = T2.shape[0] # Output dimension in the second dataset (New)
    P2 = X2.shape[0] # Input dimension in the second dataset (New)

    # Check whether Wls1_star (from Old dataset can be multiplied or not)
    if Wls1_star.shape[1] <= X2.shape[0]:
        # Wls1_star needs more columns to perform dot with X2
        Wls1_star = np.concatenate((Wls1_star, np.zeros((Q1, P2 - P1))), axis=1)
    else:
        # X2 needs more columns (appended towards the end)
        X2 = np.concatenate((np.zeros((P1 - P2, X2.shape[1])), X2), axis=0)
    
    Wls2 = np.zeros((Q2, P2)) # Initially no idea about the optimal Wls2, so set to zeros
    #X2 = np.concatenate((X2, np.zeros((P1, X2.shape[1]))), axis=0)

    T2_hat_top = np.dot(Wls1_star, X2) * np.sqrt(lambda_o) # Upper part of the composite T_tilde / T_hat
    T2_hat_bottom = T2 # Lower part kept as it is
    T2_hat = np.concatenate((T2_hat_top, T2_hat_bottom), axis=0) # Forms the composite T matrix

    q_k = np.zeros((Q1 + Q2, max(P1, P2)))
    #q_k = np.concatenate((Wls1_star * np.sqrt(lambda_o), Wls2), axis=0) # Forms the structure of ADMM q_k Matrix as [[Wls_1 | O][O]]
    lambda_k = np.zeros(q_k.shape) # Lambda_k refers to the helping variable which is set to zeros initially

    inv_matrix = np.linalg.inv(np.dot(X2, X2.T) + (1/mu_ls)*np.eye(X2.shape[0])) # Compute the inverse in the formula for O_k (k = iteration no.)
    
    for it in range(num_iterations_admm):

        lambda_prev = copy.deepcopy(lambda_k)
        O_k = np.dot((np.dot(T2_hat, X2.T) + (1/mu_ls)*(q_k + lambda_k)), inv_matrix) # Compute the O_k (for LS)
        project_item = O_k - lambda_k
        #print("Norm:{}".format(np.linalg.norm(project_item, 'fro')))
        if np.linalg.norm(project_item, 'fro') > np.sqrt(epsilon_ls): # Projecting the matrix into the region specified by root of epsilon_ls
            proj_matrix = np.sqrt(epsilon_ls)/np.linalg.norm(project_item, 'fro')
            q_k = np.dot(proj_matrix, project_item)
            #print("Norm:{}".format(np.linalg.norm(q_k, 'fro')))
        else:
            proj_matrix = 1 # No need to project, because already within the boundary
            q_k = np.dot(proj_matrix, project_item)
        lambda_k = lambda_k + q_k - O_k # Updating the helping variable
        error = np.linalg.norm(lambda_k - lambda_prev)
        #print("Residual error difference:{} for Iteration:{}".format(error, it))

    W_tilde = copy.deepcopy(O_k)
    return W_tilde # Returns the composite Wls containing Wls1, Wls2

def LwF_based_ADMM_Diff_O(Y, T, O1_star, lambda_o, epsilon_o, mu):

    # Input params :
    # Y : Activation layer output of the augmented PLN network
    # T : Targets in the augmented PLN network
    # O1_star : Output matrix for the Training on the Old Dataset (this is obtained for each layer)
    # lambda_o : Forgetting factor (for the O)
    # epsilon_o : Projection_Factor (for the O)
    # mu : ADMM multiplier for the O being optimized / already optimized
    
    # Output params:
    # O_tilde : Composite matrix containing jointly optimized O1 and O2 (for each layer L)

    #print("ADMM for O (for LwF)")
    #print("Forgetting factor:{}".format(lambda_o))
    #print("ADMM Multiplier(mu):{}".format(mu))

    # ADMM Iteration Related parameters
    num_iterations_admm = 100 # No. of ADMM Iterations
    
    Q1 = O1_star.shape[0] # Output dimension of the O1 matrix :  No. of classes in Old dataset
    P1 = O1_star.shape[1] # Input dimension of the O1 matrix : 2 * No. of classes in Old Dataset + no. of nodes for random layer
    Q2 = T.shape[0] # Output dimension: Total no. of classes for the New Dataset
    P = Y.shape[0] # Input dimension of the O_tilde matrix: 2 * No. of classes in Old + New Dataset + no. of nodes for random layer
    

    # Define Y_2 composite 
    if P1 < P: 
        # In case the O1_star matrix has less columns than the rows of Y
        # O1_star needs to be made into the structure [O1 | O | O1 | O | Random]

        # Extracting the individual parts
        O1_1 = O1_star[:,0:Q1]
        O1_2 = O1_star[:,Q1:2*Q1]
        O1_random  = O1_star[:,2*Q1:]

        # Append zeros and form the structure
        O1_zeros = np.zeros((Q1, Q2))
        O1_star = np.concatenate((O1_1, O1_zeros, O1_2, O1_zeros, O1_random), axis=1)
    
    O2 = np.zeros((Q2, P)) # Bottom part of the O_tilde matrix initialised with zeros (maybe a modification is needed with appending some random part instead of zeros)

    T_hat_top = np.dot(O1_star, Y) * np.sqrt(lambda_o)
    T_hat_bottom = T
    T_hat = np.concatenate((T_hat_top, T_hat_bottom), axis=0)  # Forming the composite T matrix

    #q_k = np.zeros((Q1 + Q2, P))
    q_k = np.concatenate((O1_star * np.sqrt(lambda_o), O2), axis=0) # Forms the structure of q_k = [[O1 | O | O1 | O | random] [O]]
    lambda_k = np.zeros(q_k.shape) # Lambda_k refers to the helping variable which is set to zeros initially

    inv_matrix = np.linalg.inv(np.dot(Y, Y.T) + (1/mu)*np.eye(Y.shape[0])) # Forming the inverse matrix which is needed for creating O
    
    for it in range(num_iterations_admm):

        lambda_prev = copy.deepcopy(lambda_k)
        O_k = np.dot((np.dot(T_hat, Y.T) + (1/mu)*(q_k + lambda_k)), inv_matrix) # Forms the O matrix
        project_item = O_k - lambda_k
        #print("Norm:{}".format(np.linalg.norm(project_item, 'fro')))
        if np.linalg.norm(project_item, 'fro') > epsilon_o: # Projection into the region defined by sqrt (epsilon)
            proj_matrix = epsilon_o/np.linalg.norm(project_item, 'fro')
            q_k = np.dot(proj_matrix, project_item)
        else:
            proj_matrix = 1
            q_k = np.dot(proj_matrix, project_item)
        lambda_k = lambda_k + q_k - O_k # Updting the helping variable
        error = np.linalg.norm(lambda_k - lambda_prev)
        #print("Residual error difference:{} for Iteration:{}".format(error, it))

    O_tilde = copy.deepcopy(O_k) # Obtaining the composite matrix conatining both o1 and o2
    return O_tilde