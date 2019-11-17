import numpy as np 
import copy

def LwF_based_ADMM_LS_Diff(X2, Y2, Wls1_star, lambda_o, epsilon_o, mu):

    # Input params :
    # X2 : New Dataset Input
    # Y2 : New Dataset Output
    # O1_star : Output matrix for the Training on the Old Dataset
    # lambda_o : Forgetting factor
    # epsilon_o : Projection_Factor

    # ADMM Iteration Related parameters
    num_iterations_admm = 100 # No. of ADMM Iterations
    
    Q1 = Wls1_star.shape[0]
    P1 = Wls1_star.shape[1]
    Q2 = Y2.shape[0]
    P2 = X2.shape[0]

    # Define Y_2 composite 
    if Wls1_star.shape[1] < X2.shape[0]:
        Wls1_star = np.concatenate((Wls1_star, np.zeros((Q1, P2 - P1))), axis=1)
    
    Wls2 = np.zeros((Q2, P2))
    #X2 = np.concatenate((X2, np.zeros((P1, X2.shape[1]))), axis=0)

    Y2_hat_top = np.dot(Wls1_star, X2) * np.sqrt(lambda_o)
    Y2_hat_bottom = Y2
    Y2_hat = np.concatenate((Y2_hat_top, Y2_hat_bottom), axis=0)

    #q_k = np.zeros((Q1 + Q2, P2 + P1))
    q_k = np.concatenate((Wls1_star * np.sqrt(lambda_o), Wls2), axis=0)
    lambda_k = np.zeros(q_k.shape)

    # Define W_ls composite
    #Wls_hat =np.concatenate((Wls1_star * np.sqrt(lambda_o), Wls2), axis=0)

    inv_matrix = np.linalg.inv(np.dot(X2, X2.T) + (1/mu)*np.eye(X2.shape[0]))
    
    for it in range(num_iterations_admm):

        lambda_prev = copy.deepcopy(lambda_k)
        O_k = np.dot((np.dot(Y2_hat, X2.T) + (1/mu)*(q_k + lambda_k)), inv_matrix)
        project_item = O_k - lambda_k
        #print("Norm:{}".format(np.linalg.norm(project_item, 'fro')))
        if np.linalg.norm(project_item, 'fro') > epsilon_o:
            proj_matrix = epsilon_o/np.linalg.norm(project_item, 'fro')
            q_k = np.dot(proj_matrix, project_item)
            #print("Norm:{}".format(np.linalg.norm(q_k, 'fro')))
        else:
            proj_matrix = 1
            q_k = np.dot(proj_matrix, project_item)
        lambda_k = lambda_k + q_k - O_k
        error = np.linalg.norm(lambda_k - lambda_prev)
        #print("Residual error difference:{} for Iteration:{}".format(error, it))

    return O_k