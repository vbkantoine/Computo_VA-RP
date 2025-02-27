# config file
# for permanent configurations
import numpy as np
from scipy import optimize
import os
import inspect

directory = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0])) # get script's path
#data_path = r'../data'
data_path = os.path.join(directory, r'../data')
#csv_path = r'KH_xi=2__sa.csv'
csv_path = r'KH_xi=2__sa.csv'

class thet_arrays():
    """a class to store a mesh of the domain of theta for the computation of a Fisher information matrix
    """
    def __init__(self, alpha_min, alpha_max, beta_min, beta_max, num_a, num_b) :
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_alpha = num_a
        self.num_beta = num_b
        al_t, h_a = np.linspace(alpha_min, alpha_max, num_a, retstep=True)
        be_t, h_b = np.linspace(beta_min, beta_max, num_b, retstep=True)
        self.alpha_tab = al_t
        self.h_alpha = h_a
        self.beta_tab = be_t
        self.h_beta = h_b

IM_dict = dict()
IM_dict['PGA'] = {'C': 0.8*10**-2, 'save_fisher_arr': thet_arrays(10**-5, 10, 10**-3, 2, 500, 500), 'save_fisher_path':'Fisher_array_PGA'}
# C: 0.01074
IM_dict['sa_5hz'] = {'C':0.8*10**-2, 'save_fisher_arr': thet_arrays(10**-5, 50, 10**-5, 2, 2000, 2000), 'save_fisher_path':'Fisher_array_sa_5hz'}


