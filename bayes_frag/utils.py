# utils script
# contains usefull generic function
from numba import jit, prange
import numpy as np
import math
try :
    from scipy.integrate import simpson
except :
    from scipy.integrate import simps as simpson




## array manipulation usefull functions :

def rep0(array, N) :
    """create an axis 0 on the array, and repeat it N times over axis 0"""
    return array[np.newaxis].repeat(N, axis=0)

def rep1(array, N) :
    """create an axis -1 on the array, and repeat it N times over axis -1"""
    return array[...,np.newaxis].repeat(N, axis=-1)



@jit(nopython=True, cache=True)
def jrep0(array, N) :
    """same than rep0 but in numba mode"""
    art = np.expand_dims(array, axis=0)
    arr = np.expand_dims(array, axis=0)
    for k in range(N-1) :
        arr = np.append(arr, art, axis=0)
    return arr

@jit(nopython=True, cache=True)
def jrep1(array, N) :
    """same than rep1 but in numba mode"""
    art = np.expand_dims(array, axis=-1)
    arr = np.expand_dims(array, axis=-1)
    for k in range(N-1) :
        arr = np.append(arr, art, axis=-1)
    return arr


## for integral calculations

@jit(nopython=True, parallel=True, cache=True)
def simpson_numb(y, x) :
    n = y.shape[0]
    lim_id = (n+1)//2
    id_05 = (2*np.arange(n)+1)[:lim_id-1]
    id_1 = (2*np.arange(n))[:lim_id]
    y_05 = y[id_05]
    y_1 = y[id_1]
    h = x[1]-x[0]
    integ = h/3 * (y_1[:-1]+y_1[1:] + 4*y_05).sum()
    return integ
    # return y.sum()*h


@jit(nopython=True, parallel=True, cache=True)
def phi_numb(alpha_grid, beta_grid, a_tab) :
    alp_n = len(alpha_grid)
    bet_n = len(beta_grid)
    a_n = len(a_tab)
    phi_f_inv = np.zeros((alp_n, bet_n, a_n))
    phi_f_inv_op = np.zeros((alp_n, bet_n, a_n))
    for i in prange(alp_n) :
        for j in prange(bet_n) :
            for l in prange(a_n) :
                alpha = alpha_grid[i]+0.0
                beta = beta_grid[j]+0.0
                g = np.log(a_tab/alpha)/beta
                gamma = g[l]+0.0
                #
                er = math.erf(gamma)
                phi_f_inv[i,j,l] = (1/2 + 1/2*er + (er==-1))**(-1) * (er!=-1)
                phi_f_inv_op[i,j,l] = (1/2 - 1/2*er + (er==1))**(-1) * (er!=1)
    return phi_f_inv, phi_f_inv_op



## Curve comparison





















