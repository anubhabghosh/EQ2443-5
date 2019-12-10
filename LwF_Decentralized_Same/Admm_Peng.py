import numpy as np
import copy

def optimize_admm(T, Y, mu, max_iterations):

    # THis function implements ADMM using :
    # Input parameters: Target (T), mu (reg. param) and Hidden actiavtion Y(g(WX))

    # Initialise Values
    q_k = np.zeros((T.shape[0], len(Y)))
    #print(q_k.shape)
    #O_k = np.zeros(T.shape[0], len(Y))
    lambda_k = np.zeros((T.shape[0], len(Y)))
    alpha = 2
    eps = alpha*np.sqrt(alpha*T.shape[0])
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

def admm_sameset(T, Y, mu, max_iterations, O_prev, rho):

    # THis function implements ADMM using :
    # Input parameters: Target (T), mu (reg. param) and Hidden actiavtion Y(g(WX))

    # Initialise Values
    #q_k = np.zeros((T.shape[0], len(Y)))
    Z_k = np.zeros((T.shape[0], len(Y)))
    W_k = np.zeros(T.shape)
    S_k = np.zeros(T.shape)
    O_k = np.zeros((T.shape[0], len(Y)))
    #W_k = np.dot(O_prev, Y) - np.dot(O_k, Y)

    lambda_k = np.zeros((T.shape[0], len(Y)))
    alpha = 2
    eps0 = alpha*np.sqrt(2*T.shape[0])
    eps1 = eps0/100
    #inv_matrix1 = np.linalg.inv(np.dot(Y, Y.T) + (1/mu)*np.eye(Y.shape[0]))
    inv_matrix = np.linalg.inv((1+1/rho)*np.dot(Y, Y.T) + (1/mu)*np.eye(Y.shape[0]))


    for _ in range(max_iterations):
        lambda_prev = copy.deepcopy(lambda_k)
        #np.dot(T, Y.T) # Here comes problem!
        #np.dot((W_k - np.dot(O_prev, Y) + S_k), Y.T)
        O_k = np.dot((np.dot(T, Y.T) + (1/mu)*(Z_k + lambda_k) - 1/rho * np.dot((W_k - np.dot(O_prev, Y) + S_k), Y.T)), inv_matrix)
        #project function
        project_item = O_k - lambda_k
        Z_k = project_fun(project_item, eps0)

        project_item = np.dot(O_prev, Y) - np.dot(O_k, Y) - S_k
        W_k = project_fun(project_item, eps1)

        lambda_k = lambda_k + Z_k - O_k

        S_k = S_k + W_k - np.dot(O_prev, Y) + np.dot(O_k, Y)

        #error = np.linalg.norm(lambda_k - lambda_prev)
        #print("Residual error difference:{} for Iteration:{}".format(error, it))

    return O_k

'''
def admm_decent_W_Onetime(X, Y, W_matrix, lambda_k, lambda_index, z_k, cnvge_flag, rho: np.float32=1e-1, lambda_rate=1e2):
    # W is a list of W_ls matrices of all PLNs
    num_pln = len(W_matrix)
    #Th= 1e-1
    #lambda_prev = copy.deepcopy(lambda_k[lambda_index])
    Th = 2*np.sqrt(2*Y.shape[0])*0.001
    inv_matrix_xxt = np.linalg.inv(np.dot(X, X.T) + (1.0/rho)*np.eye(X.shape[0]))
    W_ret = np.dot(np.dot(Y, X.T) + (1.0/rho)*(z_k - lambda_k[lambda_index]), inv_matrix_xxt)
    z_k=np.sum(W_matrix + lambda_k)/(rho*lambda_rate+ num_pln)
    #for i in range(num_pln):
    #    z_k=
    lambda_k[lambda_index] = lambda_k[lambda_index] + W_ret - z_k
    error = np.linalg.norm(W_ret - z_k)
    print("Residual error difference:{}".format(error))
    if error < Th:
        cnvge_flag = True
        print("Converge!")
    return W_ret, lambda_k[lambda_index], z_k, cnvge_flag
'''
def admm_decent_Only_W_Onetime(X, Y, lambda_k, z_k, rho: np.float32=1e-1 ):
    #Th = 2*np.sqrt(2*Y.shape[0])*0.001 
    inv_matrix_xxt = np.linalg.inv(np.dot(X, X.T) + (1.0/rho)*np.eye(X.shape[0]))
    W_ret = np.dot(np.dot(Y, X.T) + (1.0/rho)*(z_k - lambda_k), inv_matrix_xxt)
    return W_ret
def admm_decent_Only_W_Onetime_lwf(W_ls_prev, forgetting_factor, X, Y, lambda_k, z_k, rho: np.float32=1e-1 ):
    #Th = 2*np.sqrt(2*Y.shape[0])*0.001 
    inv_matrix_xxt = np.linalg.inv(np.dot(X, X.T) + forgetting_factor*np.dot(X,X.T)+ (1.0/rho)*np.eye(X.shape[0]))
    W_ret = np.dot(np.dot(Y, X.T) + forgetting_factor*np.dot(W_ls_prev,np.dot(X,X.T)) + (1.0/rho)*(z_k - lambda_k), inv_matrix_xxt)
    return W_ret
def admm_decent_Only_Z0_Onetime(lambda_k, O_matrix,num_pln, rho: np.float32=1e-1, lambda_rate=1e2):
    #eps0 = alpha*np.sqrt(2*dim)
    #Th= eps0*0.1  
    z_k=np.sum(O_matrix + lambda_k)/(rho*lambda_rate+ num_pln)
    #z_k = project_fun(z_k, eps0)
    return z_k
    '''
def admm_decent_Only_Z0_Onetime_lwf(lambda_k, O_matrix,num_pln, rho: np.float32=1e-1, lambda_rate=1e2):
    #eps0 = alpha*np.sqrt(2*dim)
    #Th= eps0*0.1  
    z_k=np.sum(O_matrix + lambda_k)/(rho*lambda_rate+ num_pln)
    #z_k = project_fun(z_k, eps0)
    return z_k
    '''
def admm_decent_Only_O_Onetime(X, Y, lambda_k, z_k, mu: np.float32=1e-1):
    inv_matrix_xxt = np.linalg.inv(np.dot(X, X.T) + (1.0/mu)*np.eye(X.shape[0]))
    O_ret = np.dot(np.dot(Y, X.T) + (1.0/mu)*(z_k - lambda_k), inv_matrix_xxt)
    
    return O_ret
def admm_decent_Only_O_Onetime_lwf(O_prev,forgetting_factor,X, Y, lambda_k, z_k, mu: np.float32=1e-1):

    inv_matrix_xxt = np.linalg.inv(np.dot(X, X.T) + forgetting_factor*np.dot(X,X.T) + (1.0/mu)*np.eye(X.shape[0]))
    O_ret = np.dot(np.dot(Y, X.T) + forgetting_factor*np.dot(O_prev,np.dot(X,X.T)) + (1.0/mu)*(z_k - lambda_k), inv_matrix_xxt)
    return O_ret
def admm_decent_Only_Z_Onetime(lambda_k, O_matrix, dim, alpha = 2, proj_coef=10):
    #eps0 = alpha*np.sqrt(2*dim) * proj_coef 
    eps0 = alpha*np.sqrt(2*dim) # Overfitting contsraint
    z_k = np.mean(O_matrix + lambda_k)
    z_k = project_fun(z_k, eps0)
    return z_k
'''
def admm_decent_O_Onetime(X, Y, O_matrix, lambda_k, lambda_index, z_k, cnvge_flag, mu: np.float32=1e-1, alpha = 2 ):
    # mu is the ratio weighting the other nodes to the local dataset.
    # O is a list of W_ls matrices of all PLNs
    #num_pln = len(O_matrix)

    #if eps0 is None:
    eps0 = alpha*np.sqrt(2*Y.shape[0])
    Th= eps0*0.1
    #lambda_prev = copy.deepcopy(lambda_k[lambda_index])

    inv_matrix_xxt = np.linalg.inv(np.dot(X, X.T) + (1.0/mu)*np.eye(X.shape[0]))
 
    O_ret = np.dot(np.dot(Y, X.T) + (1.0/mu)*(z_k - lambda_k[lambda_index]), inv_matrix_xxt)

    z_k = np.mean(O_matrix + lambda_k)
    
    z_k = project_fun(z_k, eps0)
    
    lambda_k[lambda_index] = lambda_k[lambda_index] + O_ret - z_k

    error = np.linalg.norm(z_k - O_ret)
    print("Residual error difference of O matrix:{}".format(error))
    if error < Th:
        #cnvge_flag = True
        print("Converge!")
    return O_ret, lambda_k[lambda_index], z_k, cnvge_flag 
'''
def project_fun(project_item, eps):
    if np.linalg.norm(project_item, 'fro') > eps:
        proj_matrix = eps/np.linalg.norm(project_item, 'fro')
        q_k = np.dot(proj_matrix, project_item)
        return q_k
    else:
        return project_item

 