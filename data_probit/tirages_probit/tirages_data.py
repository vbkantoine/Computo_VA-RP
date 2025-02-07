import os
import numpy as np
import scipy.stats as stat
import scipy.special as spc
import pickle

np.random.seed(1)

theta_vrai = np.array([3.37610525, 0.43304097])

N_max = 200 # valeur de N=nombre de 'vraies' données maximum pour les tirages a posteriori
M = 200     # nombre de jeux de données différents

# hyperparamètres de la loi de l'IM
mu_a = 0     #0.008652035768450072
sigma_a = 1   #1.0329485223341686

A = np.exp(mu_a + sigma_a*stat.norm.rvs(size=(N_max, M)))
U = np.random.uniform(size=(N_max, M))
Z = (U <= 1/2+1/2*spc.erf((np.log(A)-np.log(theta_vrai[0]))/theta_vrai[1])).astype(int)

path_save = os.getcwd()   # saves in the working directory, ie in the same folder as this file
if not os.path.exists(path_save) :
    os.mkdir(path_save)

pickle.dump([A, Z], open(os.path.join(path_save, 'tirages_data'), mode='wb'))
















