import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

# Set-up: Iterations and random seed
n = 40000
seed = 1
np.random.seed(1)

# Define a very literal FnHat.  
# n.b. could also have done this by sorting x-data and labelling, as was suggested.
Fn_Hat = lambda Z,X: [np.sum(Z <= xi)/len(Z) for xi in X]

# Create plot resolution (x values) and figure environment
x = np.linspace(-3,3,1000)
fig = plt.figure()
ax = fig.gca()
legend = []

# (a) Normally-distributed data
aZ = np.random.normal(loc=0, scale=1, size=n)

# (b) Central limit theorem data: Generate and plot
for k in [1,8,64,512]:
    # Creates a vector of n versions of Y_k. The Y_k are made from k summed variables B_i
    vecYk = np.sum(np.random.choice([-1,1],size=(k,n)), axis=0)/np.sqrt(k)
    ax.plot(x, Fn_Hat(vecYk, x))
    legend.append('k={}'.format(k))

# Plot the part (a) data last.
ax.plot(x,Fn_Hat(aZ,x))
legend.append('N(0,1)')

# Plot asthetics
ax.legend(legend, fontsize=12)
ax.grid(alpha=0.33)
plt.xlabel('Observations', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.savefig('./plot_hw0p9.png')