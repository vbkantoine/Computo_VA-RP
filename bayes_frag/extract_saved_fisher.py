# extract_saved_fisher script
# from a save fisher approx for probit model (from save_fisher)
# express approximation functions
import os
import pickle
from numba import jit, prange
import numpy as np
import matplotlib.pyplot as plt

from config import data_path, IM_dict

##
def get_fisher_IM(IM) :
    approx_class = Approx_fisher(IM_dict[IM]['save_fisher_path'], IM_dict[IM]['save_fisher_arr'])
    return approx_class.fisher_approx

def get_jeffreys_IM(IM) :
    approx_class = Approx_fisher(IM_dict[IM]['save_fisher_path'], IM_dict[IM]['save_fisher_arr'])
    return approx_class.jeffreys_approx

class Approx_fisher() :
    """Compute approximation function of Fisher information matrix from saved computations"""
    def __init__(self, fisher_file, save_fisher_arr) :
        self.fisher_file = fisher_file
        file = open(os.path.join(data_path, self.fisher_file), 'rb')
        self.I = pickle.load(file)
        file.close()

        self.J = (self.I[:,:,0,0]*self.I[:,:,1,1] - self.I[:,:,0,1]**2)**0.5

        self.almin = save_fisher_arr.alpha_min
        self.almax = save_fisher_arr.alpha_max
        self.bemin = save_fisher_arr.beta_min
        self.bemax = save_fisher_arr.beta_max
        self.theta_tab1 = save_fisher_arr.alpha_tab
        self.theta_tab2 = save_fisher_arr.beta_tab

        self.fisher_approx = self._fisher_approx_mth()
        self.jeffreys_approx = self._jeffreys_approx_mth()

    def _fisher_approx_mth(self) :
        almax = self.almax
        bemax = self.bemax
        almin = self.almin
        bemin = self.bemin
        theta_tab1 = self.theta_tab1
        theta_tab2 = self.theta_tab1
        I = self.I
        @jit(nopython=True, parallel=True, cache=True) #verify if parallel do not cause any error here
        def fisher_approx(theta_array, cut=True) :
            l = theta_array.shape[0]
            Fis = np.zeros((l,2,2))
            tmax = np.array([almax, bemax])
            tmin = np.array([almin, bemin])
            # for k, theta in enumerate(theta_array) :
            for k in prange(l) :
                theta = theta_array[k]
                if np.any(theta>tmax) or np.any(theta<tmin) :
                    Fis[k] = 0
                else :
                    i = np.argmin(np.abs(theta_tab1-theta[0]))
                    j = np.argmin(np.abs(theta_tab2-theta[1]))
                    Fis[k] = I[i,j]+0
            return Fis
        return fisher_approx

    def _jeffreys_approx_mth(self) :
        almax = self.almax
        bemax = self.bemax
        almin = self.almin
        bemin = self.bemin
        theta_tab1 = self.theta_tab1
        theta_tab2 = self.theta_tab1
        J = self.J
        @jit(nopython=True, parallel=True)
        def jeffreys_approx(theta_array, cut=True) :
            l = theta_array.shape[0]
            jeff = np.zeros(l)
            tmax = np.array([almax, bemax])
            tmin = np.array([almin, bemin])
            # for k, theta in enumerate(theta_array) :
            for k in prange(l) :
                theta = theta_array[k]
                if np.any(theta>tmax) or np.any(theta<tmin) :
                    jeff[k] = 0
                else :
                    i = np.argmin(np.abs(theta_tab1-theta[0]))
                    j = np.argmin(np.abs(theta_tab2-theta[1]))
                    jeff[k] = J[i,j]+0
            return jeff
        return jeffreys_approx



if __name__=="__main__":
    IM = 'PGA'
    fisher_file = os.path.join(data_path, 'Fisher_array_{}2'.format(IM))

    from config import thet_arrays
    dict_save_fisher = {'C': 0.8*10**-2, 'save_fisher_arr': thet_arrays(10**-5, 10, 10**-3, 2, 2000, 2000), 'save_fisher_path':'Fisher_array_PGA2'}
    # save_fisher_arr = IM_dict[IM]['save_fisher_arr']
    save_fisher_arr = dict_save_fisher['save_fisher_arr']
    fisher = Approx_fisher(fisher_file, save_fisher_arr)

    plt.ion()
    plt.show()

    num_theta = 200
    theta_tab = np.zeros((num_theta,2))
    theta_tab[:,0] = np.linspace(10**-5, 10, num=num_theta)
    theta_tab[:,1] = np.linspace(10**-1,1/2, num=num_theta)
    tmin = theta_tab.min()
    tmax = theta_tab.max()
    theta_grid1, theta_grid2 = np.meshgrid(theta_tab[:,0], theta_tab[:,1])

    II = np.zeros((num_theta,num_theta,2,2))
    # JJ = np.zeros((num_theta,num_theta))

    for i,alpha in enumerate(theta_tab[:,0]) :
        th = np.concatenate((alpha*np.ones((num_theta,1)),theta_tab[:,1].reshape(num_theta,1)), axis=1)
        II[i,:] = fisher.fisher_approx(th)
        # JJ[i,:] = jeffrey_approx(th)
    JJ = np.nan_to_num(np.abs(np.nan_to_num(II[:,:,0,0]*II[:,:,1,1] - II[:,:,0,1]**2)**0.5))

    # following: in a function in an other script
    plt.figure()
    axes = plt.axes(projection="3d")
    axes.plot_surface(theta_grid1, theta_grid2, JJ.T)

    plt.title('Jeffreys prior')
    axes.set_xlabel('alpha')
    axes.set_ylabel('beta')
    axes.set_zlabel('J')


    j_min, j_max = 0, np.max(JJ)
    levels = np.linspace(j_min, j_max, 15)

    plt.figure(figsize=(4.5, 3))
    plt.contourf(theta_grid1, theta_grid2, JJ.T, cmap='viridis', levels=levels)
    plt.title(r'Jeffreys prior')
    plt.axis([theta_grid1.min(), theta_grid1.max(), theta_grid2.min(), theta_grid2.max()])
    plt.colorbar()
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")
    plt.tight_layout()
    plt.show()




