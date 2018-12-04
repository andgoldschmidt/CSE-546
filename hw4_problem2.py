import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from itertools import compress, combinations

# -----------------------------------------------------------------------------

def function(x):
    '''
    Array-friendly 4-step function on [0,1.]
    '''
    return 10*np.sum([x >= k/5 for k in [1,2,3,4]], axis=0)

def k_rbf(x,z,gamma):
    '''
    rbf (gaussian kernel)
    '''
    return np.exp(-gamma*np.square(x-z))

def ls_loss_fn(residuals, params=None):
    return cp.pnorm(residuals, p=2)**2

def huber_loss_fn(residuals, params):
    M = params
    return np.sum([i for i in cp.huber(residuals, M)])

def regularizer(alpha, K):
    return cp.quad_form(alpha, K)


# -----------------------------------------------------------------------------
# Lazy 3 functions that do basically the same thing: 
# Solve the optimization problem
# -----------------------------------------------------------------------------

def cvx_kernel_1d(x, y, loss_fn, loss_params, kernel, k_params, lambd):
    '''
    Return function that computes sum over all alpha-weighted kernel data vectors 
    for each new x (each x_p in possible many x_predict).
    
    Notes on cvxpy:
    (1) The optimal objective value is returned by `cp.Problem.solve()`.
    (2) After solve, optimal value for alpha is stored in `alpha.value`.
    
    @requires loss_fn take args ([K,alpha,y],*loss_params)
    Expect that x_predict is an iterable of x_p data
    '''
    xi, xj = np.meshgrid(x, x)
    K = kernel(xi, xj, k_params)
    
    alpha = cp.Variable(len(x))
    _lambd = cp.Parameter(nonneg=True)
    _lambd.value = lambd
    
    residuals = cp.matmul(K, alpha) - y
    objective = cp.Minimize(loss_fn(residuals, loss_params) + _lambd*regularizer(alpha, K))
    prob = cp.Problem(objective)
    prob.solve()
    
    return lambda x_predict: np.array([np.sum(alpha.value*kernel(x, x_p, k_params))
                                       for x_p in x_predict])


def cvx_kernel_1d_type2(x, y, loss_fn, loss_params, kernel, k_params, lambd1, lambd2):
    xi, xj = np.meshgrid(x, x)
    K = kernel(xi, xj, k_params)
    
    n = len(x)
    alpha = cp.Variable(n)
    D = -1*np.identity(n)[:-1] + np.identity(n)[1:]
    _lambd1 = cp.Parameter(nonneg=True)
    _lambd2 = cp.Parameter(nonneg=True)
    _lambd1.value = lambd1
    _lambd2.value = lambd2
    
    residuals = cp.matmul(K, alpha) - y
    objective = cp.Minimize(loss_fn(residuals, loss_params) 
                            + _lambd1 * cp.norm1(cp.matmul(D, cp.matmul(K, alpha)))
                            + _lambd2 * regularizer(alpha, K))
    prob = cp.Problem(objective)
    prob.solve()
    
    return lambda x_predict: np.array([np.sum(alpha.value*kernel(x, x_p, k_params))
                                       for x_p in x_predict])


def cvx_kernel_1d_type3(x, y, loss_fn, loss_params, kernel, k_params, lambd):
    xi, xj = np.meshgrid(x, x)
    K = kernel(xi, xj, k_params)
    
    n = len(x)
    D = -1*np.identity(n)[:-1] + np.identity(n)[1:]
    alpha = cp.Variable(n)
    _lambd = cp.Parameter(nonneg=True)
    _lambd.value = lambd
    
    residuals = cp.matmul(K, alpha) - y
    objective = cp.Minimize(loss_fn(residuals, loss_params) + _lambd*regularizer(alpha, K))
    constraints = [cp.matmul(D, cp.matmul(K, alpha)) >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return lambda x_predict: np.array([np.sum(alpha.value*kernel(x, x_p, k_params))
                                       for x_p in x_predict])

# -----------------------------------------------------------------------------
# k_fold for each function
# -----------------------------------------------------------------------------
def k_fold_cv(k, x, y, loss_fn, loss_params, kernel, k_params, lambd):
    '''
    Returns the average loss of the k-fold cross validation.
    @requires len(x)/k in integers (creates len(x)/k folds)
    '''
    n = len(x)
    indices = np.arange(n).astype(int)
    k_folds = np.random.permutation(indices).reshape(int(n/k), k) # Each row is a k-fold. 
    k_loss = np.zeros(int(n/k))
    for i, k_validation in enumerate(k_folds):
        select_train = np.ones(n).astype(int)
        select_train[k_validation] = 0
        x_train = np.array([xi for xi in compress(x, select_train)])
        y_train = np.array([yi for yi in compress(y, select_train)])
        
        f_hat_k = cvx_kernel_1d(x_train, y_train, loss_fn, loss_params, kernel, k_params, lambd)
        
        residuals = y[k_validation]-f_hat_k(x[k_validation])
        k_loss[i] = (loss_fn(residuals, loss_params)/len(k_validation)).value
    return np.mean(k_loss)

def k_fold_cv_type2(k, x, y, loss_fn, loss_params, kernel, k_params, lambd1, lambd2):
    '''
    Returns the average loss of the k-fold cross validation.
    @requires len(x)/k in integers (creates len(x)/k folds)
    '''
    n = len(x)
    indices = np.arange(n).astype(int)
    k_folds = np.random.permutation(indices).reshape(int(n/k), k) # Each row is a k-fold. 
    k_loss = np.zeros(int(n/k))
    for i, k_validation in enumerate(k_folds):
        select_train = np.ones(n).astype(int)
        select_train[k_validation] = 0
        x_train = np.array([xi for xi in compress(x, select_train)])
        y_train = np.array([yi for yi in compress(y, select_train)])
        
        f_hat_k = cvx_kernel_1d_type2(x_train, y_train, loss_fn, loss_params, kernel, k_params, lambd1, lambd2)
        
        residuals = y[k_validation]-f_hat_k(x[k_validation])
        k_loss[i] = (loss_fn(residuals, loss_params)/len(k_validation)).value
    return np.mean(k_loss)

def k_fold_cv_type3(k, x, y, loss_fn, loss_params, kernel, k_params, lambd):
    n = len(x)
    indices = np.arange(n).astype(int)
    k_folds = np.random.permutation(indices).reshape(int(n/k), k) # Each row is a k-fold. 
    k_loss = np.zeros(int(n/k))
    for i, k_validation in enumerate(k_folds):
        select_train = np.ones(n).astype(int)
        select_train[k_validation] = 0
        x_train = np.array([xi for xi in compress(x, select_train)])
        y_train = np.array([yi for yi in compress(y, select_train)])
        
        f_hat_k = cvx_kernel_1d_type3(x_train, y_train, loss_fn, loss_params, kernel, k_params, lambd)
        
        residuals = y[k_validation]-f_hat_k(x[k_validation])
        k_loss[i] = (loss_fn(residuals, loss_params)/len(k_validation)).value
    return np.mean(k_loss)


# =============================================================================
# Main functions: need one plot for each (a) - (d)
# We'll hardcode 100 iterations for each hyper-parameter search
# =============================================================================
if __name__ == '__main__':
    # prelims
    n=50
    x_data = np.array([(i-1)/(n-1) for i in range(1,n+1)])
    np.random.seed(1)
    y_data = np.array([0 if i==25 else 1 for i in range(1,n+1)])*(function(x_data) + np.random.randn(n))


    # --- part a --------------------------------------------------------------
    g_max = 1000
    g_min = 1

    lambd_max = 5
    lambd_min = .005

    best_error_ls = np.inf
    best_ls = []

    for i in range(100):
        gamma0 = np.random.rand()*(g_max-g_min) + g_min
        lambd0 = np.random.rand()*(lambd_max-lambd_min) + lambd_min
        error = k_fold_cv(1, x_data, y_data, ls_loss_fn, None, k_rbf, gamma0, lambd0)
        if error < best_error_ls:
            best_error_ls = error
            best_ls = [gamma0, lambd0]

    f_hat = cvx_kernel_1d(x_data, y_data, 
                          loss_fn=ls_loss_fn, loss_params=None,
                          kernel=k_rbf, k_params=best_ls[0], lambd=best_ls[1])

    fig, ax = plt.subplots(1)
    ax.plot(x, function(x))
    ax.scatter(x_data, y_data)
    ax.plot(x, f_hat(x))
    ax.legend(['f(x)', '$\widehat{\ f\ }(x)}$', 'Data'], fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    fig.savefig('figs/hw4_2a_ls_plt.pdf')
    np.savetxt(X=best_ls,fname='figs/hw4_2a_ls.txt')

    # --- part b --------------------------------------------------------------
    g_max = 1000
    g_min = 1

    lambd_max = 5
    lambd_min = .005

    M_max = 10
    M_min = .1

    best_huber = []
    best_error_huber = np.inf

    for i in range(100):
        gamma0 = np.random.rand()*(g_max-g_min) + g_min
        lambd0 = np.random.rand()*(lambd_max-lambd_min) + lambd_min
        M0 = np.random.rand()*(M_max-M_min) + M_min
        error = k_fold_cv(1, x_data, y_data, huber_loss_fn, M0, k_rbf, gamma0, lambd0)
        if error < best_error_huber:
            best_error_huber = error
            best_huber = [gamma0, lambd0, M0]

    f_hat = cvx_kernel_1d(x_data, y_data, 
                          loss_fn=huber_loss_fn, loss_params=best_huber[2],
                          kernel=k_rbf, k_params=best_huber[0], lambd=best_huber[1])

    fig, ax = plt.subplots(1)
    ax.plot(x, function(x))
    ax.scatter(x_data, y_data)
    ax.plot(x, f_hat(x))
    ax.legend(['f(x)', '$\widehat{\ f\ }(x)}$', 'Data'], fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    fig.savefig('figs/hw4_2b_huber_plt.pdf')
    np.savetxt(X=best_huber,fname='figs/hw4_2b_huber.txt')

    # --- part c --------------------------------------------------------------
    g_max = 1000
    g_min = 1

    lambd1_max = 5
    lambd1_min = .005

    lambd2_max = 5
    lambd2_min = .005

    best_tv = []
    best_error_tv = np.inf

    for i in range(100):
        gamma0 = np.random.rand()*(g_max-g_min) + g_min
        lambd1 = np.random.rand()*(lambd1_max-lambd1_min) + lambd1_min
        lambd2 = np.random.rand()*(lambd2_max-lambd2_min) + lambd2_min
        error = k_fold_cv_type2(1, x_data, y_data, ls_loss_fn, None, k_rbf, gamma0, lambd1, lambd2)
        if error < best_error_tv:
            best_error_tv = error
            best_tv = [gamma0, lambd1, lambd2]

    f_hat = cvx_kernel_1d_type2(x_data, y_data, 
                          loss_fn=ls_loss_fn, loss_params=None,
                          kernel=k_rbf, k_params=best_tv[0],
                          lambd1=best_tv[1], lambd2=best_tv[2])

    fig, ax = plt.subplots(1)
    ax.plot(x, function(x))
    ax.scatter(x_data, y_data)
    ax.plot(x, f_hat(x))
    ax.legend(['f(x)', '$\widehat{\ f\ }(x)}$', 'Data'], fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    fig.savefig('figs/hw4_2c_tv_plt.pdf')
    np.savetxt(X=best_tv,fname='figs/hw4_2c_tv.txt')

    # --- part d --------------------------------------------------------------
    g_max = 1000
    g_min = 1

    lambd_max = 5
    lambd_min = .005

    best_error_qp = np.inf
    best_qp = []

    for i in range(100):
        gamma0 = np.random.rand()*(g_max-g_min) + g_min
        lambd0 = np.random.rand()*(lambd_max-lambd_min) + lambd_min
        error = k_fold_cv_type3(1, x_data, y_data, ls_loss_fn, None, k_rbf, gamma0, lambd0)
        if error < best_error_qp:
            best_error_qp = error
            best_qp = [gamma0, lambd0]

    f_hat = cvx_kernel_1d_type3(x_data, y_data, loss_fn=ls_loss_fn, loss_params=None,
                                kernel=k_rbf, k_params=best_qp[0], lambd=best_qp[1])

    fig, ax = plt.subplots(1)
    ax.plot(x, function(x))
    ax.scatter(x_data, y_data)
    ax.plot(x, f_hat(x))
    ax.legend(['f(x)', '$\widehat{\ f\ }(x)}$', 'Data'], fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    fig.savefig('figs/hw4_2d_qp_plt.pdf')
    np.savetxt(X=best_tv,fname='figs/hw4_2d_qp.txt')