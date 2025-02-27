# plot functions
# usefull functions to generate plots
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino'], 'size':13})
rc('text', usetex=True)



plt.ion()
plt.show()

def plot_bi_function(pp, theta_grid1, theta_grid2, ax=None) :
    j_min, j_max = np.min(pp), np.max(pp)
    levels = np.linspace(j_min, j_max, 20)

    if ax is None :
        fig = plt.figure(figsize=(4.5, 2.5))
        ax = fig.add_subplot(111)

    ax.contourf(theta_grid1, theta_grid2, pp.T, cmap='viridis', levels=levels)
    # ax.set_title(r'Objective prior via simpson')
    ax.axis([theta_grid1.min(), theta_grid1.max(), theta_grid2.min(), theta_grid2.max()])
    # plt.colorbar()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    ax.figure.tight_layout()
    return ax



