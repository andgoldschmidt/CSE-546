import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import compress, combinations

def f(x):
    return 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)

def k_poly(x, z, d):
    '''
    x, z in R^n; d in R
    '''
    return np.power(1+ x*z, d)

def k_rbf(x, z, gamma):
    '''
    x, z in R^n; gamma in R
    '''
    return np.exp(-gamma*np.square(x-z))

def kernel_ridge_regression_1d(x, y, kernel, k_params, regularization):
    '''
    Find alpha_hat analytically. Return function that computes sum over all alpha-weighted 
    kernel data vectors for each new x (each x_p in possible many x_predict).
    
    @requires x_predict is an iterable of x_p data
    '''
    xi, xj = np.meshgrid(x,x)
    K = kernel(xi,xj,k_params)
    alpha_hat,_,_,_ = np.linalg.lstsq(K.T@K+regularization*K, K.T@y, rcond=None) # Solve, but not nec. full-rank
    return lambda x_predict: np.array([np.sum(alpha_hat*kernel(x, x_p, k_params)) for x_p in x_predict])

def k_fold_cv(k, x, y, kernel, k_params, regularization):
    '''
    Returns the average error of the k-fold cross validation.
    @requires len(x)/k in integers (creates len(x)/k folds)
    '''
    n = len(x)
    indices = np.arange(n).astype(int)
    k_folds = np.random.permutation(indices).reshape(int(n/k), k) # Each row is a k-fold.
    k_error = np.zeros(int(n/k))
    for i, k_validation in enumerate(k_folds):
        k_train = np.ones(n).astype(int)
        k_train[k_validation] = 0
        x_train = np.array([xi for xi in compress(x, k_train)])
        y_train = np.array([yi for yi in compress(y, k_train)])
        f_hat_k = kernel_ridge_regression_1d(x_train, y_train, kernel, k_params, regularization)
        k_error[i] = np.sum(np.power(y[k_validation]-f_hat_k(x[k_validation]), 2))/len(k_validation)
    return np.mean(k_error)

def run(N, K, B, saveloc=''):
  np.random.seed(1)
  x = np.random.rand(N)
  y = f(x) + np.random.randn(N)

  # ---- Part 1: Polynomial kernel ----------------------------------------------------------------
  search_reg = np.power(10., np.linspace(-3,3,10))
  search_d = np.linspace(0,N,20)

  search_poly = np.zeros([len(search_reg)*len(search_d),3])
  index = 0
  for d_i in search_d:
      for reg_i in search_reg:
          search_poly[index] = np.array([d_i, reg_i, k_fold_cv(K, x, y, k_poly, d_i, reg_i)])
          index = index + 1
  np.savetxt(os.path.join(saveloc, 'poly_log_K{}_B{}_N{}.txt'.format(K,B,N)), search_poly)

  x_real = np.linspace(0,1,100)
  d,reg,_ = search_poly[np.argmin(search_poly[:,2])]
  f_hat_poly = kernel_ridge_regression_1d(x, y, k_poly, d, reg)

  B = 300
  indices = np.arange(N).astype(int)
  f_hats = []
  for i in range(B):
      bootstrap = np.random.choice(indices, N, replace=True)
      f_hats.append(kernel_ridge_regression_1d(x[bootstrap], y[bootstrap], k_poly, d, reg))

  # Form bootstrap confidence intervals using percentiles at each x
  low, high = np.sort(np.array([f_hat(x_real) for f_hat in f_hats]), axis=0)[[int(0.025*B), int(0.975*B)],:]

  fig = plt.figure()
  ax = plt.gca()
  ax.grid(True)
  ax.plot(x_real, f(x_real), lw=4)
  ax.plot(x_real, f_hat_poly(x_real), c='darkred', lw=4)
  ax.fill_between(x_real, low, high, color='red', alpha=0.5)
  ax.scatter(x, y)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  fig.savefig(os.path.join(saveloc,'kpoly_K_{}_B{}_N{}.pdf'.format(K,B,N)))

  # ---- Part 2: Gaussian kernel ------------------------------------------------------------------
  search_reg = np.power(10., np.linspace(-3,3,10))
  search_gamma = np.linspace(0,N,20)

  search_rbf = np.zeros([len(search_reg)*len(search_gamma),3])
  index = 0
  for g_i in search_gamma:
      for reg_i in search_reg:
          search_rbf[index] = np.array([g_i, reg_i, k_fold_cv(K, x, y, k_rbf, g_i, reg_i)])
          index += 1
  np.savetxt(os.path.join(saveloc, 'rbf_log_K{}_B{}_N{}.txt'.format(K,B,N)), search_rbf)

  x_real = np.linspace(0,1,250)
  gamma,reg,_ = search_rbf[np.argmin(search_rbf[:,2])]
  gamma = 1/np.median([(i[0]-i[1])**2 for i in combinations(x,2)]) # Use the heuristic instead
  f_hat_rbf = kernel_ridge_regression_1d(x, y, k_rbf, gamma, reg)

  indices = np.arange(N).astype(int)
  f_hats = []
  for i in range(B):
      bootstrap = np.unique(np.random.choice(indices, N, replace=True))
      f_hats.append(kernel_ridge_regression_1d(x[bootstrap], y[bootstrap], k_rbf, gamma, reg))

  # Form bootstrap confidence intervals using percentiles at each x
  low, high = np.sort(np.array([f_hat(x_real) for f_hat in f_hats]), axis=0)[[int(0.025*B), int(0.975*B)],:]

  fig = plt.figure()
  ax = plt.gca()
  ax.grid(True)
  ax.plot(x_real, f(x_real), lw=4)
  ax.plot(x_real, f_hat_rbf(x_real), c='darkred', lw=4)
  ax.fill_between(x_real, low, high, color='red', alpha=0.5)
  ax.scatter(x, y)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  fig.savefig(os.path.join(saveloc, 'krbf_K{}_B{}_N{}.pdf'.format(K,B,N)))

# =================================================================================================
if __name__ == "__main__":
  figdir = r'/home/andy/Documents/class/cse/cse546/CSE-546-HW-3/figs'
  run(30, 1, 300, figdir)
  run(300, 10, 300, figdir)