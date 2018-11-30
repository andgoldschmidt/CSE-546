import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

def classify_kmeans(x, k, epsilon, seed=None, init='k++'):
    '''
    Classify x in R^(n,d) into k groups according to Lloyd's algorithm.
    Exit condition is that new centers are maximally displaced by epsilon.
    @init: allowed options are 'k++', 'box'
    @k: must be >=1
    
    '''
    n,d = x.shape
    if seed:
        np.random.seed(seed)
        
    if init == 'box':
        bbox = np.array([np.min(x, axis=0), np.max(x, axis=0)]) # Initialize within data range.
        centers = np.random.rand(k,d)*(bbox[1]-bbox[0])+bbox[0]
    elif init == 'k++':
        indices = np.arange(n)
        centers = x[np.random.choice(indices),:].reshape(1,d)
        for i in range(k-1):
            distances = np.array([np.sum(np.square(xi-centers), axis=1) for xi in x])
            weights = np.min(distances, axis=1)
            weights = weights/np.sum(weights)
            centers = np.append(centers, x[np.random.choice(indices, p=weights),:].reshape(1,d), axis=0)
    else:
        print('Unknown initialization scheme {}'.format(init))
        return None, None, None

    objective = []
    condition = True
    while condition:
        distances = np.array([np.sum(np.square(xi-centers), axis=1)  for xi in x])
        classify = np.argmin(distances, axis=1)
        objective.append(np.sum([np.sum(distances[classify==c, c])/np.sum(classify==c)
                                 if np.any(classify==c) else 0 for c in range(k)]))
        # If no points in class, reuse old center
        new_centers = np.array([np.mean(x[classify==c], axis=0) if np.any(classify==c)
                                else centers[c] for c in range(k)])
        if np.max(np.sum(np.square(new_centers - centers), axis=1)) < epsilon:
            condition = False
        centers = np.copy(new_centers)
    return classify, centers, objective

def clean_ax(ax):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

# =================================================================================================
if __name__ == '__main__':
    from mnist import MNIST
    mndata = MNIST(r'../CSE-546-HW-1/data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0

    # 1. Plot centers for uniform (bounding box) initialization
    all_classes = []
    all_centers = []
    all_objectives = []
    for k in [5,10,20]:
        kclasses, kcenters, kobj = classify_kmeans(X_train, k, epsilon=1e-1, init='box')
        # Save data
        all_classes.append(kclasses)
        all_centers.append(kcenters)
        all_objectives.append(kobj)
        # Plot center images
        fig, ax = plt.subplots(1, k, figsize=[k,10])
        for c, anAx in zip(kcenters, ax):
            anAx.imshow(c.reshape([28,28]))
            clean_ax(anAx)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(os.path.join(r'./figs','imshow_box_{}.pdf'.format(k)))

    # 2. Plot centers for k-means++ initialization
    all_classes_2 = []
    all_centers_2 = []
    all_objectives_2 = []
    for k in [5,10,20]:
        kclasses, kcenters, kobj = classify_kmeans(X_train, k, epsilon=1e-1, init='k++')
        # Save data
        all_classes_2.append(kclasses)
        all_centers_2.append(kcenters)
        all_objectives_2.append(kobj)
        # Plot center images
        fig, ax = plt.subplots(1, k, figsize=[k,10])
        for c, anAx in zip(kcenters, ax):
            anAx.imshow(c.reshape([28,28]))
            clean_ax(anAx)
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(os.path.join(r'./figs','imshow_kpp_{}.pdf'.format(k)))

    # 3. Plot k-means objective as a function of iteration
    fig, ax = plt.subplots(1,2, figsize=[10,4])

    for obj in all_objectives:
        ax[0].plot(np.arange(len(obj)), obj, lw=5, alpha=0.75)
        ax[0].set_xlabel('Iteration (box)', fontsize = 12)
        ax[0].set_ylabel('k-means Objective', fontsize=12)
    ax[0].legend(['k=5','k=10','k=20'],fontsize=12)
    ax[0].grid(True)

    for obj in all_objectives_2:
        ax[1].plot(np.arange(len(obj)), obj, lw=5, alpha=0.75)
        ax[1].set_xlabel('Iteration (k++)', fontsize = 12)
        ax[1].set_yticklabels([])
    ax[1].legend(['k=5','k=10','k=20'],fontsize=12)
    ax[1].grid(True)

    # Fix shared axis size using (experimental) limits 
    for iax in ax:
        iax.set_ylim([0,3000])
        iax.set_xlim([-1,15])

    fig.subplots_adjust(wspace=0)
    fig.savefig(os.path.join(r'./figs','kmeans_combo_objs.pdf'))