# distributions script
# contains numba dentisty functions
from numba import jit
import numpy as np
import numpy.random as rd
import scipy.stats as stat
import scipy.special as spc


## probability densities / generating functions in numba for HM algs
# densities calculated up to a constant

@jit(nopython=True)
def gauss_density(x, mu, sigma) :
    return np.exp(-(x-mu)**2/2/sigma**2)

def mult_gauss_density() :
    return True

@jit(nopython=True)
def chi2_density(x, k) :
    return x**(k/2-1) * np.exp(-x/2)


@jit(nopython=True, parallel=True, cache=True)
def log_norm_density(x, mu, sigma) : #compatibility with multivariate while there are mutual independance
    return 1/x * np.exp(-np.sum((np.log(x)-mu)**2, axis=0)/2/sigma**2)


@jit(nopython=True, cache=True)
def gamma_normal_density(mu, tau, m, a, b, lamb) :

    return tau**(a-1/2) * np.exp(-b*tau) * np.exp(-lamb*tau*(mu-m)**2/2)

@jit(nopython=True, cache=True)
def log_gamma_normal_pdf(mu, tau, m, a, b, lamb) :
    l = mu.shape[0]
    tab = np.zeros(l)
    for i in range(l):
        if np.isnan(mu[i]) or tau[i]<=0 :
            tab[i] = -10**5
        else :
            tab[i] = np.log(gamma_normal_density(mu[i], tau[i], m, a, b, lamb))
    return tab

#conditionnal usage for HM

@jit(nopython=True)
def chi2_conditionnal(z1, z2) :
    k = z1.shape[1]
    lamb = z2/k
    return chi2_density(z1/lamb, k)/lamb

@jit(nopython=True)
def gauss_conditionnal(z1, z2, sigma_prop) :
    return gauss_density(z1, 2, sigma_prop)

@jit(nopython=True, parallel=True, cache=True)
def log_norm_conditionnal(z1, z2, sigma_prop) :
    d = z2.shape[1]
    return log_norm_density(z1, np.log(z2), sigma_prop).sum(axis=1)



## generators

@jit(nopython=True)
def gen_gauss() :
    return True

@jit(nopython=True)
def gen_chi2(z1, k) :
    return True

def gen_cond_chi2(z1, z2) :
    return True

@jit(nopython=True, parallel=True, cache=True)
def gen_log_norm(mu, sigma) :
    d = mu.shape[0]
    return np.exp(mu + sigma * rd.randn(d))

@jit(nopython=True, parallel=True, cache=True)
def gen_log_norm_cond(z2, sigma_prop) :
    return gen_log_norm(np.log(z2), sigma_prop)


if __name__=="__main__" :
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()
    ## lois résultante
    # vérification que si log alph, 1/beta2 ~ GN(lamb, mu, c, d) alors
    # Z = np.sqrt(lamb)*(np.log(alpha)-np.log(a))/beta - np.sqrt(lambda)*(np.log(a)-mu)/beta ~N(0,1) indep beta et
    # Z*beta*np.sqrt(c/d) ~ T(2c)
    #
    # loga = 1
    # mu = 1.11
    # lamb = 1.5
    # c = 3
    # d = 4

    mu, lamb, c, d = 1.170048843014274, 20.492226945500363, 5.110591647090938, 3.859056696751078
    loga = np.exp(mu + 0.1)


    num_est = 100000

    ## ### 1. histo de Z:
    tau = 1/d*stat.gamma.rvs(a=c, size=num_est)
    logalph = mu + np.sqrt((lamb*tau)**-1) * stat.norm.rvs(size=num_est)
    beta = 1/np.sqrt(tau)

    Z = np.sqrt(lamb) * (logalph - loga)/beta + np.sqrt(lamb) * (loga - mu)/beta

    z_arr = np.linspace(-5, 5, 100)
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(Z, 100, density=True, label='Z')
    ax.plot(z_arr, stat.norm.pdf(z_arr), label=r'$\mathcal{N}(0,1)$')
    ax.set_title(r'$Z=\sqrt{\lambda}\beta^{-1}(\log\alpha-\mu)$')
    ax.legend()

    ## 2. ### histo de Zbeta(c/d)**.5
    Y = Z*beta*np.sqrt(c/d)
    fig = plt.figure(2)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(Y, 100, density=True, label=r'Y')
    ax.plot(z_arr, stat.t.pdf(z_arr,2*c), label=r'$\mathcal{T}(2c)$')
    ax.set_title(r'$Y=Z\beta\sqrt{c/d}$')
    ax.legend()


    # Ensuite,
    ## ### histo de ( log(alph)-loga )/(bet/sqrt(2d)
    U = np.sqrt(lamb)* (logalph-loga)/(beta)
    # U = Z - np.sqrt(lamb) * (loga - mu)/beta
    # U_verif = stat.norm.rvs(size=num_est) + np.sqrt(lamb)*(mu-loga)/(np.sqrt(2*d)) * stat.chi.rvs(2*c, size=num_est)
    fig = plt.figure(3)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(U, 100, density=True, label=r'U')
    # ax.hist(U_verif, 100, density=True, label=r'Uverif')
    z_arr2 = np.linspace(-50, 5, 200)
    mgf_chi = lambda x : spc.hyp1f1(c, 1/2, x**2/2) + x*np.sqrt(2)*(spc.gamma((2*c+1)/2)/spc.gamma(c)) * spc.hyp1f1(c+1/2, 3/2, x**2/2)
    f = (mu-loga)/np.sqrt(2*d)*np.sqrt(lamb)
    cste = 1/np.sqrt(2*np.pi) * np.sqrt(f**2+1)**(-2*c) #/ 300
    ax.plot(z_arr2, np.exp(-z_arr2**2/2) * mgf_chi(z_arr2*f/np.sqrt(f**2+1))*cste, label=r'$\mathcal{N}(\sqrt{\frac{\lambda}{2d}}(\mu-\log a)\chi(2c), 1)$')
    ax.set_title(r'$U=\sqrt{\lambda }\beta^{-1}(\log\alpha - \log a)$')
    ax.legend()

    ## ###Tests autour de beta~chi
    fig = plt.figure(5)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(np.sqrt(2*d)/beta, 100, density=True, label=r'$\sqrt{2d}/\beta$')
    ax.plot(z_arr, stat.chi.pdf(z_arr, 2*c), label=r'$\chi(2c)$')
    ax.legend()

    ## quantiles
    # # empirical quantiles
    num_a = 500
    a_tab = np.linspace(10**-5, 10, num=num_a)
    conf = 0.05

    curves_q1 = np.zeros_like(a_tab)
    curves_q2 = np.zeros_like(a_tab)
    # curves_q1 = np.zeros_like(a_tab)
    # th = np.concatenate((logalph[np.newaxis], beta[np.newaxis]), 0)
    curves = 1/2+1/2*spc.erf((np.log(a_tab[np.newaxis]) - logalph[:,np.newaxis])/beta[:,np.newaxis])
    q1_, q2_ = np.quantile(curves, 1-conf/2, axis=0), np.quantile(curves, conf/2, axis=0)
    for l in range(num_a) :
        curves_q1[l] = curves[np.abs(curves[:,l] - q1_[l]).argmin()][l] #usefulness not certain
        curves_q2[l] = curves[np.abs(curves[:,l] - q2_[l]).argmin()][l]
    curves_med = np.median(curves, axis=0)
    # curves_med = np.mean(curves, axis=0)
    # err_conf = simpson(np.abs(curves_q1 - curves_q2)**2, self.data.a_tab)**(1/2)
    # err_med = simpson(np.abs(curves_med - self.ref.curve_MLE)**2, self.data.a_tab)**(1/2)

    #theo quantiles chernoff U = sqrt(lamb)(logalph - loga)/beta
    # U-> -U donc f -> -f
    f = -(mu - np.log(a_tab))*np.sqrt(lamb)/np.sqrt(2*d) # E[U|chi]= f*chi
    m = f*np.sqrt(2)*spc.gamma(c+1/2)/spc.gamma(c) # E[U] = m
    # q = np.sqrt((12*f**2 +2)*np.log(2/conf)) - m
    # q = np.sqrt(2*(1+f**2)*np.log(2/conf/spc.beta(c,1/2))) - m + np.abs(f)*np.sqrt(2)*spc.gamma(c+1/2)/spc.gamma(c) * spc.beta(c,1/2)/spc.beta(c+1/2, 3/2)
    q1 = stat.norm.ppf(conf/2) + (f>0)*(stat.chi.ppf(conf/2, 2*c, scale=np.abs(f))) - (f<0)*(stat.chi.ppf(1-conf/2, 2*c, scale=np.abs(f)))
    q2 = stat.norm.ppf(1-conf/2) - (f<0)*(stat.chi.ppf(conf/2, 2*c, scale=np.abs(f))) + (f>0)*(stat.chi.ppf(1-conf/2, 2*c, scale=np.abs(f)))
    cq1 = 1/2 + 1/2 * spc.erf(q1/np.sqrt(lamb))
    cq2 = 1/2 + 1/2 * spc.erf(q2/np.sqrt(lamb))
    # cq1 = 1/2+1/2*spc.erf((m+q)/np.sqrt(lamb))
    # cq2 = 1/2+1/2*spc.erf((m-q)/np.sqrt(lamb))

    fig = plt.figure(4)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(a_tab, cq1, '-b', label='$q_{\mathcal{N}}+q_{r\chi}$')
    ax.plot(a_tab, cq2, '-b')
    ax.plot(a_tab, 1/2+1/2*spc.erf(m/np.sqrt(lamb)), '--b')
    ax.plot(a_tab, curves_q1, '-r', label='empirical')
    ax.plot(a_tab, curves_q2, '-r')
    ax.plot(a_tab, curves_med, '--r')
    ax.legend()
    ax.set_title('confidence intervals comparison')






















##
