import numpy as np 
import copy 
def optimize_admm(T, Y, mu, max_iterations):

    # THis function implements ADMM using :
    # Input parameters: Target (T), mu (reg. param) and Hidden actiavtion Y(g(WX))

    # Initialise Values 
    q_k = np.random.rand(T.shape[0], len(Y))
    O_k = np.random.rand(T.shape[0], len(Y))
    lambda_k = np.random.rand(T.shape[0], len(Y))

    for it in range(max_iterations):

        lambda_prev = copy.deepcopy(lambda_k)

        O_k = np.dot((np.dot(T, Y.T) + (1/mu)*(q_k + lambda_k)), \
              np.linalg.inv(np.dot(Y, Y.T) + (1/mu)*np.eye(Y.shape[0])))

        if np.linalg.norm(q_k, ord='fro') != 0:
            proj_matrix = np.dot(q_k, np.dot(np.linalg.inv(np.dot(q_k.T, q_k)), q_k.T))
            q_k = np.dot(proj_matrix, (O_k - lambda_k))
        
        lambda_k = lambda_k + q_k - O_k
        error = np.linalg.norm(lambda_k - lambda_prev)
        print("Residual error difference:{} for Iteration:{}".format(error, it))

    return O_k