import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Note:
# X: d,n 
# y: n,
# w: d,

def mu(w,b,X,y,penalty):
  ''' n, '''
  return 1/(1+np.exp(-y*(b+X.T@w)))

def J(w,b,X,y,penalty):
  _,n=X.shape    
  return -np.sum(np.log(mu(w,b,X,y,penalty)))/n + penalty*np.sum(np.square(w))

def grad_w_J(w,b,X,y,penalty):
  ''' d, '''
  d,n = X.shape
  mu_i = mu(w,b,X,y,penalty)
  return (1/n)*np.einsum('n,n,dn->d',(1-mu_i),-y, X) + 2*penalty*w

def hess_w_J(w,b,X,y,penalty):
  ''' d,d '''
  d,n = X.shape
  mu_i = mu(w,b,X,y,penalty)
  return (1/n)*np.einsum('n,n,kn,ln->kl',-mu_i*(1-mu_i), np.square(y), X, X) + 2*penalty*np.identity(d)

def gradient_descent(w0,b0,X,y,penalty,learn_rate,max_iterations):
  results = np.zeros(int(max_iterations))
  wi = np.zeros([int(max_iterations),len(w0)])
  bi = np.zeros(int(max_iterations))
  wi[0] = w0
  bi[0] = b0
  results[0] = J(w0,b0,X,y,penalty)
  i = 0
  while i < max_iterations-1:
      i = i+1
      wi[i] = wi[i-1] - learn_rate*grad_w_J(wi[i-1],bi[i-1],X,y,penalty)
      bi[i] = bi[i-1] - learn_rate*grad_b_J(wi[i-1],bi[i-1],X,y,penalty)
      results[i] = J(wi[i],bi[i],X,y,penalty)
      # We can add various exit conditions...
  return wi,bi,results,i

def class_error(w,b,X,y):
  d,n = X.shape
  classified_x = b+X.T@w > 0
  known_labels = y == 1
  return np.sum(~(classified_x==known_labels)/n)

def make_q5_plots(train_J, test_J, test_error, train_error, arbitrary_xmax):
  fig,ax = plt.subplots(1,2,figsize=[12,5])

  ax[0].plot(np.arange(len(train_J)), train_J)
  ax[0].plot(np.arange(len(test_J)),test_J)
  ax[0].legend(['Test J','Train J'], fontsize=18)
  ax[0].set_xlim([-1,arbitrary_xmax]);
  ax[0].set_ylabel('J(w,b)',fontsize=18)
  ax[0].set_xlabel('Iteration',fontsize=18)

  ax[1].plot(np.arange(len(test_error)), test_error)
  ax[1].plot(np.arange(len(train_error)),train_error)
  ax[1].legend(['Test J','Train J'], fontsize=18)
  ax[1].set_xlim([-1,arbitrary_xmax]);
  ax[1].set_ylabel('Misclass. error',fontsize=18)
  ax[1].set_xlabel('Iteration',fontsize=18)
  return fig,ax

def stochastic(function,w,b,x,y,penalty,batch):
  subsetX = X[:,batch]
  subsetY = y[batch]
  return function(w,b,subsetX,subsetY,penalty)

def stochastic_gradient_descent(w0,b0,X,y,penalty,learn_rate,max_iterations,batch_size):
  results = np.zeros(int(max_iterations))
  wi = np.zeros([int(max_iterations),len(w0)])
  bi = np.zeros(int(max_iterations))
  wi[0] = w0
  bi[0] = b0
  results[0] = J(w0,b0,X,y,penalty)
  i = 0
  while i < max_iterations-1:
      i = i+1
      batch = np.random.randint(0,len(y),batch_size)
      wi[i] = wi[i-1] - learn_rate*stochastic(grad_w_J,wi[i-1],bi[i-1],X,y,penalty,batch)
      bi[i] = bi[i-1] - learn_rate*stochastic(grad_b_J,wi[i-1],bi[i-1],X,y,penalty,batch)
      results[i] = J(wi[i],bi[i],X,y,penalty)
  return wi,bi,results

def newton_gradient_descent(w0,b0,X,y,penalty,learn_rate,max_iterations):
  results = np.zeros(int(max_iterations))
  wi = np.zeros([int(max_iterations),len(w0)])
  bi = np.zeros(int(max_iterations))
  wi[0] = w0
  bi[0] = b0
  results[0] = J(w0,b0,X,y,penalty)
  i = 0
  while i < max_iterations-1:
      i = i+1
      mv_w = np.linalg.solve(a=hess_w_J(wi[i-1],bi[i-1],X,y,penalty),
                             b=-grad_w_J(wi[i-1],bi[i-1],X,y,penalty))
      mv_b = -hess_b_J(wi[i-1],bi[i-1],X,y,penalty)/hess_b_J(wi[i-1],bi[i-1],X,y,penalty)
      wi[i] = wi[i-1] + learn_rate*mv_w
      bi[i] = bi[i-1] + learn_rate*mv_b
      results[i] = J(wi[i],bi[i],X,y,penalty)
  return wi,bi,results

if __name__ == '__main__':
  from mnist import MNIST
  mndata = MNIST(r'./data/')
  X_train, labels_train = map(np.array, mndata.load_training())
  X_test, labels_test = map(np.array, mndata.load_testing())
  X_train = X_train/255.0
  X_test = X_test/255.0

  keep_train = np.bitwise_or(labels_train==2, labels_train==7)
  keep_test = np.bitwise_or(labels_test==2, labels_test==7)

  X_train_27 = X_train[keep_train]
  labels_train_27 = labels_train[keep_train].astype(int)
  X_test_27 = X_test[keep_test]
  labels_test_27 = labels_test[keep_test].astype(int)

  labels_train_27[labels_train_27==2]=-1
  labels_train_27[labels_train_27==7]=1
  labels_test_27[labels_test_27==2]=(-1)
  labels_test_27[labels_test_27==7]=1

  # Part 1: Gradient Descent

  # Make some short variable names 
  X = X_train_27.T # d,n
  d,n = X.shape
  y = labels_train_27
  w0 = np.zeros(d)
  b0 = 0
  penalty0 = 0.1

  # Do part (b)
  learn_rate = 0.05
  max_iter = 100

  train_w,train_b,train_J,_ = gradient_descent(w0,b0,X,y,penalty0,learn_rate,max_iter)
  test_J = np.array([J(w,b,X_test_27.T,labels_test_27,penalty0) for w,b in zip(train_w,train_b)])

  test_error = [class_error(w, b, X_test_27.T, labels_test_27) for w,b in zip(train_w,train_b)]
  train_error = [class_error(w, b, X_train_27.T, labels_train_27) for w,b in zip(train_w,train_b)]

  fig, ax = make_q5_plots(train_J, test_J, test_error, train_error, 100):
  fig.savefig(r'./figs/gd.png')


  # Part 2: Stochastic Gradient Descent

  # Do part (c)
  learn_rate = 0.05
  max_iter = 100
  size = 1

  sg_train_w1,sg_train_b1,sg1_train_J = stochastic_gradient_descent(w0,b0,X,y,penalty0,learn_rate,max_iter,size)
  sg1_test_J = np.array([J(w,b,X_test_27.T,labels_test_27,penalty0) for w,b in zip(sg_train_w1,sg_train_b1)])

  sg1_test_error = [class_error(w, b, X_test_27.T, labels_test_27) for w,b in zip(sg_train_w1,sg_train_b1)]
  sg1_train_error = [class_error(w, b, X_train_27.T, labels_train_27) for w,b in zip(sg_train_w1,sg_train_b1)]

  fig, ax = make_q5_plots(sg1_train_J, sg1_test_J, sg1_test_error, sg1_train_error, 100):
  fig.savefig(r'./figs/sgd_bs1.png')

  # Do part (d)
  learn_rate = 0.05
  max_iter = 100
  size = 100

  sg_train_w100,sg_train_b100,sg100_train_J = stochastic_gradient_descent(w0,b0,X,y,penalty0,learn_rate,max_iter,size)
  sg100_test_J = np.array([J(w,b,X_test_27.T,labels_test_27,penalty0) for w,b in zip(sg_train_w100,sg_train_b100)])

  sg100_test_error = [class_error(w, b, X_test_27.T, labels_test_27) for w,b in zip(sg_train_w100,sg_train_b100)]
  sg100_train_error = [class_error(w, b, X_train_27.T, labels_train_27) for w,b in zip(sg_train_w100,sg_train_b100)]

  fig, ax = make_q5_plots(sg100_train_J, sg100_test_J, sg100_test_error, sg100_train_error, 100):
  fig.savefig(r'./figs/sgd_bs1.png')

  # Part 3: Newtons Method

  # Do part (e)
  learn_rate = 0.05
  max_iter = 5 # my computer will not be able to do more

  n_train_w,n_train_b,n_train_J = newton_gradient_descent(w0,b0,X,y,penalty0,learn_rate,max_iter)
  n_test_J = np.array([J(w,b,X_test_27.T,labels_test_27,penalty0) for w,b in zip(n_train_w,n_train_b)])

  n_test_error = [class_error(w, b, X_test_27.T, labels_test_27) for w,b in zip(n_train_w,n_train_b)]
  n_train_error = [class_error(w, b, X_train_27.T, labels_train_27) for w,b in zip(n_train_w,n_train_b)]

  fig,ax = make_q5_plots(n_train_J, n_test_J, n_test_error, n_train_error, arbitrary_xmax)
  fig.savefig(r'./figs/ngd.png')