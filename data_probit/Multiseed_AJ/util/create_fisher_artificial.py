import os
import inspect
directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])) # get script's path
# os.chdir(directory)
ch_directory = os.path.join(directory, r'./../')
os.chdir(ch_directory)
import sys
sys.path.append(ch_directory)
sys.path.append(directory)

import numpy as np
import scipy.stats as stat
import scipy.special as spc
try :
    from scipy.integrate import simpson
except :
    from scipy.integrate import simps as simpson


from bayes_frag.save_fisher import save_fisher
from bayes_frag.data import Data
from bayes_frag.config import thet_arrays
from bayes_frag import fisher

np.random.seed(1)

theta_vrai = np.array([3.37610525, 0.43304097])

name_file = 'fisher'
name_file2 = 'fisher_light'

save_path = os.path.join(directory, name_file)
save_path2 = os.path.join(directory, name_file2)

mu_a = 0 #0.008652035768450072
sigma_a = 1 #np.sqrt(1.0329485223341686)


class Data_simplified_big(Data) :
    """subclass of Data
        the goal is to simplify the original class to let it containing the artificially generated data
        # todo : implement such 'artficial data class' in data.py
    """
    def __init__(self, size=10**5):
        """
        Args:
            i (int): in the genrated array, id to take into account in this run
        """
        self.A = np.exp(mu_a + sigma_a*stat.norm.rvs(size=(size,1)))
        U = np.random.uniform(size=(size, 1))
        self.Z = U<=(1/2+1/2*spc.erf((np.log(self.A)-np.log(theta_vrai[0]))/theta_vrai[1]))
        self.Y = np.zeros_like(self.A)
        self.a_tab = None
        self.h_a = None
        self._set_a_tab()
        self.f_A = None
        self.f_A_tab = None
        self._compute_f_A()
        self.increasing_mode = True



dict_save_fisher = {'C': 0.8*10**-2, 'save_fisher_arr': thet_arrays(10**-5, 10, 10**-3, 2, 2000, 2000), 'save_fisher_path':name_file}

dict_save_fisher2 = {'C': 0.8*10**-2, 'save_fisher_arr': thet_arrays(10**-5, 10, 10**-3, 2, 500, 500), 'save_fisher_path':name_file2}

if __name__=="__main__" :
    dat = Data_simplified_big()

    def save_fisher_computation():
        # save_fisher_arr = dict_save_fisher['save_fisher_arr']
        save_fisher_arr = dict_save_fisher2['save_fisher_arr']

        alpha_min = save_fisher_arr.alpha_min
        alpha_max = save_fisher_arr.alpha_max
        beta_min = save_fisher_arr.beta_min
        beta_max = save_fisher_arr.beta_max
        num = save_fisher_arr.num_alpha

        theta_tab1 = save_fisher_arr.alpha_tab
        theta_tab2 = save_fisher_arr.beta_tab

        function = fisher.fisher_function("simpson", dat)
        I = save_fisher(save_path2, function, theta_tab1, theta_tab2)


    def compute_ratio_Kc(kappa, alpha, theta_array=thet_arrays(10**-5,10,10**-4,4,1000,1000)) :
        """
        compute ratio c/K where $K = \int_\Theta J a^{1/\alpha}; c = \int_\Theta J a^{1+1/\alpha}$ \n
        where $a(\theta) = \beta^\kappa$ \n
        theta_array delimits the integration domain, its size must be the same w.r.t. both dimensions
        """
        J_func = fisher.Jeffreys_function(data=dat)
        a_func = lambda beta_tab : beta_tab[np.newaxis]**kappa


        K = J_func(theta_array.alpha_tab, theta_array.beta_tab, dat.a_tab) * a_func(theta_array.beta_tab)**(1/alpha)
        K = simpson(K, theta_array.beta_tab, axis=-1)
        K = simpson(K, theta_array.alpha_tab)
        c = J_func(theta_array.alpha_tab, theta_array.beta_tab, dat.a_tab) * a_func(theta_array.beta_tab)**(1+1/alpha)
        c = simpson(c, theta_array.beta_tab, axis=-1)
        c = simpson(c, theta_array.alpha_tab)

        print("K={}".format(K))
        print("c={}".format(c))
        print("c/K={}".format(c/K))
        return c/K

    compute_ratio_Kc(1/8, 1/2)


