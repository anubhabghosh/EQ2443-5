import numpy as np 
import copy 
def optimize_admm(T, Y, mu, max_iterations):

    # THis function implements ADMM using :
    # Input parameters: Target (T), mu (reg. param) and Hidden actiavtion Y(g(WX))

    # Initialise Values 
    q_k = np.zeros((T.shape[0], len(Y)))
    #O_k = np.zeros(T.shape[0], len(Y))
    lambda_k = np.zeros((T.shape[0], len(Y)))
    alpha = 2
    eps = alpha*np.sqrt(2*T.shape[0])
    inv_matrix = np.linalg.inv(np.dot(Y, Y.T) + (1/mu)*np.eye(Y.shape[0]))

    for it in range(max_iterations):

        lambda_prev = copy.deepcopy(lambda_k)
        O_k = np.dot((np.dot(T, Y.T) + (1/mu)*(q_k + lambda_k)), inv_matrix)
        project_item = O_k - lambda_k
        
        if np.linalg.norm(project_item, 'fro') > eps:
            proj_matrix = eps/np.linalg.norm(project_item, 'fro')
            q_k = np.dot(proj_matrix, project_item)
        else:
            proj_matrix = 1
            q_k = np.dot(proj_matrix, project_item)
        
        lambda_k = lambda_k + q_k - O_k
        error = np.linalg.norm(lambda_k - lambda_prev)
        #print("Residual error difference:{} for Iteration:{}".format(error, it))

    return O_k