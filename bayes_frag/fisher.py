# fisher script
# defnie functions for fisher inforation computation under probit model
import numpy as np
import scipy.special as spc
import math
from numba import jit, prange
from scipy.integrate import simps as simpson, quadrature
from utils import rep1, jrep1, simpson_numb, phi_numb


## First funcs

phi_f_inv = lambda g : (1/2 + 1/2*spc.erf(g) + (spc.erf(g)==-1))**(-1) * (spc.erf(g)!=-1)
phi_f_inv_op = lambda g : (1/2 - 1/2*spc.erf(g) + (spc.erf(g)==1))**(-1) * (spc.erf(g)!=1)



# Fisher MC. A values set by user after
@jit(nopython=True,parallel=True)
def Fisher_MC(alpha_grid, beta_grid, A_simul):
    alp_n = len(alpha_grid)
    bet_n = len(beta_grid)
    A_s = A_simul + 0
    A_num = len(A_s)
    I = np.zeros((alp_n, bet_n, 2, 2))
    #for i,alpha in enumerate(alpha_grid) :
    #    for j,beta in enumerate(beta_grid) :
    for i in prange(alp_n) :
        alpha = alpha_grid[i]
        for j in prange(bet_n) :
            beta = beta_grid[j]
            #
            theta = np.zeros(2)
            theta[0] = alpha+0
            theta[1] = beta+0
            A_11, A_12, A_21, A_22, A_31, A_32 = 0.0,0.0,0.0,0.0,0.0,0.0
            for k,a in enumerate(A_s) :
                gam = (np.log(a/theta[0])/theta[1])
                phi_prime2 = np.exp(-2*gam**2)/np.pi
                phi_inv = (1/2+1/2*math.erf(gam) + (math.erf(gam)==-1))**(-1) * (math.erf(gam)!=-1)
                phi_inv_op = (1/2-1/2*math.erf(gam) + (math.erf(gam)==1))**(-1) * (math.erf(gam)!=1)
                A_11 += np.log(a/theta[0])*phi_prime2 * phi_inv/A_num
                A_12 += np.log(a/theta[0])*phi_prime2 * phi_inv_op/A_num
                A_21 += np.log(a/theta[0])**2*phi_prime2 * phi_inv/A_num
                A_22 += np.log(a/theta[0])**2*phi_prime2 * phi_inv_op/A_num
                A_31 += phi_prime2 * phi_inv/A_num
                A_32 += phi_prime2 * phi_inv_op/A_num
            I[i,j,0,0] = 1/(theta[0]*theta[1])**2 * (A_31+A_32)
            I[i,j,0,1] = 1/(theta[0]*theta[1]**3) * (A_11+A_12)
            I[i,j,1,1] = 1/theta[1]**4 * (A_21+A_22)
            I[i,j,1,0] = I[i,j,0,1] + 0
    return I




# for fisher computation which necessists A
def fisher_function(name, data=None) :
    """Return the Fisher function corresponding to the pompted name
        As the computation needs the approximation of f_A, data must be prompted
        name={'MC','quadrature','simpson','rectangles_numba','simpson_numba', 'simpson_numba_paral'}"""
    if data is None :
        assert name=='MC'
        return Fisher_MC

    f_A = data.f_A

    @jit(nopython=True,parallel=True)
    def f_A_mult_numb(a_im, a_n) :
        return np.exp(-((jrep1(data.A, a_n)-a_im)/h)**2).sum(axis=0)/h/np.sqrt(np.pi)/n

    def func_integ(alpha, beta) :
        funcA11 = lambda a_im : np.log(a_im/alpha)*np.exp(-2*((np.log(a_im/alpha)/beta)**2))/np.pi * phi_f_inv(np.log(a_im/alpha)/beta) * f_A(a_im)
        funcA12 = lambda a_im : np.log(a_im/alpha)*np.exp(-2*((np.log(a_im/alpha)/beta)**2))/np.pi * phi_f_inv_op(np.log(a_im/alpha)/beta) * f_A(a_im)
        funcA21 = lambda a_im : np.log(a_im/alpha)**2*np.exp(-2*((np.log(a_im/alpha)/beta)**2))/np.pi * phi_f_inv(np.log(a_im/alpha)/beta) * f_A(a_im)
        funcA22 = lambda a_im : np.log(a_im/alpha)**2*np.exp(-2*((np.log(a_im/alpha)/beta)**2))/np.pi * phi_f_inv_op(np.log(a_im/alpha)/beta) * f_A(a_im)
        funcA31 = lambda a_im : np.exp(-2*((np.log(a_im/alpha)/beta)**2))/np.pi * phi_f_inv(np.log(a_im/alpha)/beta) * f_A(a_im)
        funcA32 = lambda a_im : np.exp(-2*((np.log(a_im/alpha)/beta)**2))/np.pi * phi_f_inv_op(np.log(a_im/alpha)/beta) * f_A(a_im)
        return funcA11, funcA12, funcA21, funcA22, funcA31, funcA32

    if name=='quadrature' :
        def Fisher_quadrature(alpha_grid, beta_grid):
            alp_n = len(alpha_grid)
            bet_n = len(beta_grid)
            I = np.zeros((alp_n,bet_n,2,2))
            for i,alpha in enumerate(theta_tab[:,0]) :
                for j, beta in enumerate(theta_tab[:,1]) :
                    funcA11, funcA12, funcA21, funcA22, funcA31, funcA32 = func_integ(alpha, beta)
                    A11 = quadrature(funcA11, tmin, tmax)[0]
                    A12 = quadrature(funcA12, tmin, tmax)[0]
                    A21 = quadrature(funcA21, tmin, tmax)[0]
                    A22 = quadrature(funcA22, tmin, tmax)[0]
                    A31 = quadrature(funcA31, tmin, tmax)[0]
                    A32 = quadrature(funcA32, tmin, tmax)[0]
                    I[i,j,0,0] = (A31+A32)/alpha**2/beta**2
                    I[i,j,1,0] = (A12+A11)/alpha/beta**3
                    I[i,j,0,1] = I[i,j,1,0]+0
                    I[i,j,1,1] = (A21 + A22)/beta**4
            return I
        return Fisher_quadrature

    if name=='simpson' :
        def Fisher_Simpson(alpha_grid, beta_grid, a_tab):
            alp_n = len(alpha_grid)
            bet_n = len(beta_grid)
            a_n = len(a_tab)
            I = np.zeros((alp_n,bet_n,2,2))
            for i,alpha in enumerate(alpha_grid) :
                for j, beta in enumerate(beta_grid) :
                    if alpha<=0 or beta<=0 :
                        I[i,j] = 0
                    else :
                        g = np.log(a_tab/alpha)/beta
                        phi_f_inv = (1/2 + 1/2*spc.erf(g) + (spc.erf(g)==-1))**(-1) * (spc.erf(g)!=-1)
                        phi_f_inv_op = (1/2 - 1/2*spc.erf(g) + (spc.erf(g)==1))**(-1) * (spc.erf(g)!=1)
                        # f_A_a = f_A_mult(a_tab, a_n)
                        f_A_a = data.f_A(a_tab)
                        funcA11 = np.log(a_tab/alpha)*np.exp(-2*(g**2))/np.pi * phi_f_inv *f_A_a
                        funcA12 = np.log(a_tab/alpha)*np.exp(-2*(g**2))/np.pi * phi_f_inv_op *f_A_a
                        funcA21 = np.log(a_tab/alpha)**2*np.exp(-2*(g**2))/np.pi * phi_f_inv *f_A_a
                        funcA22 = np.log(a_tab/alpha)**2*np.exp(-2*(g**2))/np.pi * phi_f_inv_op *f_A_a
                        funcA31 = np.exp(-2*(g**2))/np.pi * phi_f_inv *f_A_a
                        funcA32 = np.exp(-2*(g**2))/np.pi * phi_f_inv_op *f_A_a
                        A11 = simpson(funcA11, a_tab)
                        A12 = simpson(funcA12, a_tab)
                        A21 = simpson(funcA21, a_tab)
                        A22 = simpson(funcA22, a_tab)
                        A31 = simpson(funcA31, a_tab)
                        A32 = simpson(funcA32, a_tab)
                        I[i,j,0,0] = (A31+A32)/alpha**2/beta**2
                        I[i,j,1,0] = (A12+A11)/alpha/beta**3
                        I[i,j,0,1] = I[i,j,1,0]+0
                        I[i,j,1,1] = (A21 + A22)/beta**4
                    #J[i,j] = np.sqrt(I[i,j,0,0]*I[i,j,1,1] - I[i,j,0,1]**2)
            return I
        return Fisher_Simpson

    if name=='rectangles_numba' :
        @jit(nopython=True,parallel=True, cache=True)
        def Fisher_rectangles(alpha_grid, beta_grid, a_tab, h_a):
            alp_n = len(alpha_grid)
            bet_n = len(beta_grid)
            I = np.zeros((alp_n, bet_n, 2,2))
            # for i,alpha in enumerate(alpha_grid) :
            #     for j,beta in enumerate(beta_grid) :
            for i in prange(alp_n) :
                alpha = alpha_grid[i]
                for j in prange(bet_n) :
                    beta = beta_grid[j]
                    #
                    theta = np.zeros(2)
                    theta[0] = alpha+0
                    theta[1] = beta+0
                    A_11, A_12, A_21, A_22, A_31, A_32 = 0.0,0.0,0.0,0.0,0.0,0.0
                    for k,a in enumerate(a_tab) :
                        f_A_a = np.exp(-((A-a)/h)**2).mean()/h/np.sqrt(np.pi)
                        gam = np.log(a/theta[0])/theta[1]
                        phi_prime2 = np.exp(-2*gam**2)/np.pi
                        phi_inv = (1/2+1/2*math.erf(gam) + (math.erf(gam)==-1))**(-1) * (math.erf(gam)!=-1)
                        phi_inv_op = (1/2-1/2*math.erf(gam) + (math.erf(gam)==1))**(-1) * (math.erf(gam)!=1)
                        A_11 += np.log(a/theta[0])*phi_prime2 * phi_inv * f_A_a * h_a
                        A_12 += np.log(a/theta[0])*phi_prime2 * phi_inv_op * f_A_a * h_a
                        A_21 += np.log(a/theta[0])**2*phi_prime2 * phi_inv * f_A_a * h_a
                        A_22 += np.log(a/theta[0])**2*phi_prime2 * phi_inv_op * f_A_a * h_a
                        A_31 += phi_prime2 * phi_inv * f_A_a * h_a
                        A_32 += phi_prime2 * phi_inv_op * f_A_a * h_a
                    I[i,j,0,0] = 1/(theta[0]*theta[1])**2 * (A_31+A_32)
                    I[i,j,0,1] = 1/(theta[0]*theta[1]**3) * (A_11+A_12)
                    I[i,j,1,1] = 1/theta[1]**4 * (A_21+A_22)
                    I[i,j,1,0] = I[i,j,0,1] + 0
            return I
        return Fisher_rectangles

    if name=='simpson_numba' :
        @jit(nopython=True, cache=True, parallel=True)
        def Fisher_Simpson_Numb(alpha_grid, beta_grid, a_tab) :
            alp_n = len(alpha_grid)
            bet_n = len(beta_grid)
            a_n = len(a_tab)
            I = np.zeros((alp_n,bet_n,2,2,1))
            # for i,alpha in enumerate(alpha_grid) :
            #     for j, beta in enumerate(beta_grid) :
            for i in range(alp_n) :
                alpha = alpha_grid[i]+0.0
                for j in range(bet_n) :
                    beta = beta_grid[j]+0.0
                    if alpha<=0 or beta<=0 :
                        I[i,j] = 0
                    else :
                        #
                        g = np.log(a_tab/alpha)/beta
                        phi_f_inv = np.zeros(a_n)
                        phi_f_inv_op = np.zeros(a_n)
                        #for l,gamma in enumerate(g) :
                        for l in prange(a_n) :
                            gamma = g[l]+0.0
                            #
                            er = math.erf(gamma)
                            phi_f_inv[l] = (1/2 + 1/2*er + (er==-1))**(-1) * (er!=-1)
                            phi_f_inv_op[l] = (1/2 - 1/2*er + (er==1))**(-1) * (er!=1)
                        #f_A_a = f_A_mult_numb(a_tab, a_n)
                        f_A_a = np.exp(-((jrep1(A, a_n)-a_tab)/h)**2).sum(axis=0)/(n*h)/np.sqrt(np.pi)
                        #
                        funcA11 = np.log(a_tab/alpha)*np.exp(-2*(g**2))/np.pi * phi_f_inv *f_A_a
                        funcA12 = np.log(a_tab/alpha)*np.exp(-2*(g**2))/np.pi * phi_f_inv_op *f_A_a
                        funcA21 = np.log(a_tab/alpha)**2*np.exp(-2*(g**2))/np.pi * phi_f_inv *f_A_a
                        funcA22 = np.log(a_tab/alpha)**2*np.exp(-2*(g**2))/np.pi * phi_f_inv_op *f_A_a
                        funcA31 = np.exp(-2*(g**2))/np.pi * phi_f_inv *f_A_a
                        funcA32 = np.exp(-2*(g**2))/np.pi * phi_f_inv_op *f_A_a
                        A11 = simpson_numb(funcA11, a_tab)
                        A12 = simpson_numb(funcA12, a_tab)
                        A21 = simpson_numb(funcA21, a_tab)
                        A22 = simpson_numb(funcA22, a_tab)
                        A31 = simpson_numb(funcA31, a_tab)
                        A32 = simpson_numb(funcA32, a_tab)
                        I[i,j,0,0] = (A31+A32)/alpha**2/beta**2
                        I[i,j,1,0] = (A12+A11)/alpha/beta**3
                        I[i,j,0,1] = I[i,j,1,0]+0
                        I[i,j,1,1] = (A21 + A22)/beta**4
                        #J[i,j] = np.sqrt(I[i,j,0,0]*I[i,j,1,1] - I[i,j,0,1]**2)
            return I[:,:,:,:,0]
        return Fisher_Simpson_Numb

    if name=='simpson_numba_paral' :
        @jit(nopython=True, parallel=True, cache=True)
        def Fisher_Simpson_Numb_paral(alpha_grid, beta_grid, a_tab) :
            alp_n = len(alpha_grid)
            bet_n = len(beta_grid)
            a_n = len(a_tab)
            I = np.zeros((alp_n,bet_n,2,2))
            phi_f_inv = np.zeros((alp_n, bet_n, a_n))
            phi_f_inv_op = np.zeros((alp_n, bet_n, a_n))
            f_A_a = np.exp(-((jrep1(A, a_n)-a_tab)/h)**2).sum(axis=0)/(n*h)/np.sqrt(np.pi)
            phi_f_inv, phi_f_inv_op = phi_numb(alpha_grid, beta_grid, a_tab)
                    #f_A_a = f_A_mult_numb(a_tab, a_n)

            gg = np.zeros(a_n)
            for i2 in prange(alp_n):
                for j2 in prange(bet_n) :
                    alpha2 = alpha_grid[i2]+0.0
                    beta2 = beta_grid[j2]+0.0
                    gg[:] = np.log(a_tab/alpha2)/beta2

                    funcA11 = np.log(a_tab/alpha2)*np.exp(-2*(gg**2))/np.pi * phi_f_inv[i2,j2] *f_A_a
                    funcA12 = np.log(a_tab/alpha2)*np.exp(-2*(gg**2))/np.pi * phi_f_inv_op[i2,j2] *f_A_a
                    funcA21 = np.log(a_tab/alpha2)**2*np.exp(-2*(gg**2))/np.pi * phi_f_inv[i2,j2] *f_A_a
                    funcA22 = np.log(a_tab/alpha2)**2*np.exp(-2*(gg**2))/np.pi * phi_f_inv_op[i2,j2] *f_A_a
                    funcA31 = np.exp(-2*(gg**2))/np.pi * phi_f_inv[i2,j2] *f_A_a
                    funcA32 = np.exp(-2*(gg**2))/np.pi * phi_f_inv_op[i2,j2] *f_A_a
                    A11 = simpson_numb(funcA11, a_tab)
                    A12 = simpson_numb(funcA12, a_tab)
                    A21 = simpson_numb(funcA21, a_tab)
                    A22 = simpson_numb(funcA22, a_tab)
                    A31 = simpson_numb(funcA31, a_tab)
                    A32 = simpson_numb(funcA32, a_tab)
                    I[i2,j2,0,0] = (A31+A32)/alpha2**2/beta2**2
                    I[i2,j2,1,0] = (A12+A11)/alpha2/beta2**3
                    I[i2,j2,0,1] = I[i2,j2,1,0]+0
                    I[i2,j2,1,1] = (A21 + A22)/beta2**4
            return I
        return Fisher_Simpson_Numb_paral



## Jeffreys func
##             .

@jit(nopython=True,parallel=True)
def Jeffreys_MC(alpha_grid, beta_grid, A_simul) :
    I = Fisher_MC(alpha_grid, beta_grid, A_simul)
    return np.sqrt(I[:,:,0,0]*I[:,:,1,1] - I[:,:,1,0]**2)

def Jeffreys_function(name='simpson', data=None) :
    """Return the Jeffreys function corresponding to the prompted name
        As the computation needs the approximation of f_A, data must be prompted
        name={'MC','quadrature','simpson','rectangles_numba','simpson_numba', 'simpson_numba_paral'}"""
    if data is None :
        assert name=='MC'

    if name=='MC' :
        return Jeffreys_MC

    I_func = fisher_function(name, data)

    if name=='quadrature' :
        def Jeffreys_quadrature(alpha_grid, beta_grid) :
            I = I_func(alpha_grid, beta_grid)
            return np.sqrt(I[:,:,0,0]*I[:,:,1,1] - I[:,:,1,0]**2)
        return Jeffreys_quadrature

    if name=="rectangles_numba" :
        @jit(nopython=True,parallel=True)
        def Jeffreys_rectangles(alpha_grid, beta_grid, a_tab, h_a) :
            I = I_func(alpha_grid, beta_grid, a_tab, h_a)
            return np.sqrt(I[:,:,0,0]*I[:,:,1,1] - I[:,:,1,0]**2)
        return Jeffreys_rectangles

    if name=='simpson' :
        def Jeffreys_simpson(alpha_grid, beta_grid, a_tab) :
            I = I_func(alpha_grid, beta_grid, a_tab)
            return np.sqrt(I[:,:,0,0]*I[:,:,1,1] - I[:,:,1,0]**2)
        return Jeffreys_simpson

    if name=='simpson_numba' :
        @jit(nopython=True,parallel=True)
        def Jeffreys_simpson_numba(alpha_grid, beta_grid, a_tab) :
            I = I_func(alpha_grid, beta_grid, a_tab)
            return np.sqrt(I[:,:,0,0]*I[:,:,1,1] - I[:,:,1,0]**2)
        return Jeffreys_simpson_numba

    if name=='simpson_numba_paral' :
        @jit(nopython=True,parallel=True, cache=True)
        def Jeffreys_simpson_numba_paral(alpha_grid, beta_grid, a_tab) :
            I = I_func(alpha_grid, beta_grid, a_tab)
            return np.sqrt(I[:,:,0,0]*I[:,:,1,1] - I[:,:,1,0]**2)
        return Jeffreys_simpson_numba_paral




#todo : if name==main, plots


