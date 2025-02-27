#  model script
# define a statistical model
import numpy as np
from scipy import optimize
import scipy.special as spc
from scipy.integrate import simps as simpson  #from scipy.integrate import simpson #for new python versions
import matplotlib.pyplot as plt

import plot_functions
from reference_curves import Reference_curve


def probit_fargility_curve(a, theta) :
    return 1/2+1/2*spc.erf(np.log(a[np.newaxis]/theta[:,0,np.newaxis])/theta[:,1,np.newaxis])


class Model() :
    """statistical model class
    gather data, ditributions, and simulation methods"""
    def __init__(self, prior, likelihood, data, post_simul, log=True, numba=True, linear=False, ref=None, bounds_bst=None, option_bst=None, fragility_curve_func=probit_fargility_curve) :
        """prior = 1-entry function
            lideklihood = 3-entry function (s,a,theta)
            data = data object with observations dataset
            post_simul = 2- entry executable (a,s) which returns a 
                1-entry (num_estimation) function for posterior simulations
            log = True if the distribution densities are log-evaluations
            numba = True if the distribution functions are compiled with numba
            linear = False if the probit model is considered
            ref = reference_curves.Reference_curve object. set from data autamatically if None"""
        self.prior = prior
        self.likelihood = likelihood
        self.data = data
        self.log = log
        self.numba = numba
        self.predictions = None
        self.post_simul = post_simul
        self.linear = linear
        self.fragility_curve_func = fragility_curve_func
        if ref is None :
            self.ref = Reference_curve(data)
        else :
            self.ref = ref
        self.logs = {'post':dict(), 'MLE':dict()}
        self.A, self.S = None, None
        if linear :
            opp_log_vraiss = lambda theta : -likelihood(data.Y.flatten(), data.A.flatten(), theta)
            if ref is None :
                self.ref._compute_MLE_curve(opp_log_vraiss)
        if bounds_bst is None :
            self.bounds_bst = [(self.data.A.min(), self.data.A.max()),(self.data.A.var()/50,self.data.A.var()*50)]
        else :
            self.bounds_bst = bounds_bst
        if option_bst is None :
            self.option_bst = {'maxiter':50}
        else :
            self.option_bst = option_bst

    def _update_data(self, k) :
        if self.linear :
            self.A,self.S,_ = self.data.default_draw(k)
        else :
            self.A,_,self.S = self.data.default_draw(k)

    def posterior_estimation(self, k, n_est) : 
        """Compute a posteriori samples
            k = sub-dataset-sample-size for the estimation
            n_est = number of estimations"""
        # if particular_update is None:
        self._update_data(k)
        # else :
        #     self.A, self.S = particular_update(k)
        simulation_func = self.post_simul(self.A,self.S)
        th_post = simulation_func(n_est)
        self.logs['post'][k] = th_post
        return th_post

    def bootstrap(self, k, n_est, l_to_min=None) :
        """Compute MLE estimations via bootstrapping
           Bootstrapping drawn within the updtaed k data
            k = samples dataset-size for estimation
            n_est = number of estimations
            (to remove)l_to_min = loglikelihood to minimize (if None -log(self.likelihood))"""
        self._update_data(k)
        def func_log_vr(z, a) : # express a theta-function that must be minimized
            if l_to_min is None :
                if self.log :
                    def log_vr(theta) :
                        return -self.likelihood(z, a, np.array(theta).reshape(1,2))
                else :
                    def log_vr(theta) :
                        return -np.log(self.likelihood(z,a,theta))
            else :
                return l_to_min(z,a)
            return log_vr
        th_MLE = np.zeros((n_est,2))
        t0 = self.ref.theta_MLE
        bounds = self.bounds_bst
        options = self.option_bst
        for n in range(n_est) :
            draw_ids = np.random.randint(k, size=k)
            S = self.S[draw_ids]
            A = self.A[draw_ids]
            log_vr = func_log_vr(S,A)
            th_MLE[n] = optimize.minimize(log_vr, t0, bounds=bounds, options=options).x
        self.logs['MLE'][k] = th_MLE
        return th_MLE

    def bootstrap_MAP(self, k, n_est, l_to_min=None) :
        """Generate MAP simulations via bootstrap
        Args:
            k (int): datasample_size
            n_est (int): number of estimated items
            l_to_min (function, optional): function to minimize equivalently to -log(post). Defaults to None.
        Returns: (numpy.array)
        """
        self._update_data(k)
        def func_log_post(z, a) : # express a theta-function that must be minimized
            if l_to_min is None :
                if self.log :
                    def log_post(theta) :
                        return -self.likelihood(z, a, np.array(theta).reshape(1,2)) - self.prior(np.array(theta).reshape(1,2))
                else :
                    def log_post(theta) :
                        return -np.log(self.likelihood(z,a,theta.reshape(1,2))) - np.log(self.prior(np.array(theta).reshape(1,2)))
            else :
                return l_to_min(z,a)
            return log_post
        th_MAP = np.zeros((n_est,2))
        t0 = self.ref.theta_MLE
        bounds = [(self.data.A.min(), self.data.A.max()),(self.data.A.var()/50,self.data.A.var()*50)]
        for n in range(n_est) :
            draw_ids = np.random.randint(k, size=k)
            S = self.S[draw_ids]
            A = self.A[draw_ids]
            log_post = func_log_post(S,A)
            th_MAP[n] = optimize.minimize(log_post, t0, bounds=bounds, options={'maxiter':50}).x
        if not 'MAP' in self.logs.keys() :
            self.logs['MAP'] = dict()
        self.logs['MAP'][k] = th_MAP
        return th_MAP

    def clear_logs(self) :
        """Clear the simulation logs 
        and if they exist the logs of the simulation function"""
        try :
            self.post_simul.clear_logs()
        except :
            pass
        self.logs = {'post':{}, 'MLE':{}}

    def run_simulations(self, ks, n_est, sim={'MLE','post'}, print_fo=False, clear_cache=False) :
        """run simulations for k in ks=list[ints] for key in sim=sublist {'MLE','post','MAP'} 
            set print_fo to True to print k in ks for every simulation
            et clear_cache to True to clear logs before running
        """
        if clear_cache :
            self.clear_logs()
        for i,k in enumerate(ks) :
            if print_fo :
                print('Simulation {}/{}, dataset_size={}, simulating {}'.format(i+1,len(ks),k,sim))
            for key in sim :
                if key=='MLE' :
                    self.bootstrap(k,n_est)
                elif key=='post' :
                    self.posterior_estimation(k, n_est)
                elif key=='MAP' :
                    self.bootstrap_MAP(k, n_est)
        return self.logs

    def compute_meds_qt_err(self, klogs_, sim=['post', 'MLE'], conf=0.05) :
        """Compute median quantiles and some errors for computed simulations from samples of sizes k in klogs_
        Args:
            klogs_ (iterable): sample size to consider.
            sim (list in ['post', 'MLE', 'MAP'], optional): kind of simulation to consider. Defaults to ['post', 'MLE'].
            conf (float, optional): confidence probability. Defaults to 0.05.
        Returns: tuple(curves_q1, curves_q2, curves_med, err_conf, err_med, err_quad)
            curves_q1 (dict[key in sim: numpy array len(klogs_)*len(self.data.a_tab)]): quantiles 1-conf/2 scaled.
            curves_q2 (dict[key in sim: numpy array len(klogs_)*len(self.data.a_tab)]): quantiles conf/2 scaled.
            curves_med (dict[key in sim: numpy array len(klogs_)*len(self.data.a_tab)]): medians.
            err_conf (dict[key in sim: numpy array len(klogs_)]): confidence scales |q1-q2|_(L2(A))^2.
            err_med (dict[key in sim: numpy array len(klogs_)]): median errors |Pmed-Pref|_(L2(A))^2.
            err_quad (dict[key in sim: numpy array len(klogs_)]): quadratic errors E[|P-Pref|_(L2(A))^2 |s,a].
        """
        # assert np.all(np.array([np.abs(np.asarray(klogs_)).max()<=len(self.logs[key]) for key in sim]))
        assertion_tab = np.array([[k in self.logs[key] for k in klogs_] for key in sim], dtype=bool)
        assert np.all(assertion_tab), "no existing simulation for {}".format({key: klogs_[assertion_tab[i]] for (i,key) in enumerate(sim)})
        # klogs = np.mod(klogs_, min([len(self.logs[key]) for key in sim]))
        klogs = np.array(klogs_)
        num_c = len(klogs)
        num_a = self.data.a_tab.shape[0]
        curves_q1 = {key:np.zeros((num_c, num_a)) for key in sim}
        curves_q2 = {key:np.zeros((num_c, num_a)) for key in sim}
        curves_med = {key:np.zeros((num_c, num_a)) for key in sim}

        err_conf = {key:np.zeros(num_c) for key in sim}
        err_med = {key:np.zeros(num_c) for key in sim}
        err_quad = {key:np.zeros(num_c) for key in sim}

        for i,k in enumerate(klogs) :
            for key in sim :
                th = self.logs[key][k]
                curves = self.fragility_curve_func(self.data.a_tab, th)
                # curves = 1/2+1/2*spc.erf(np.log(self.data.a_tab[np.newaxis]/th[:,0,np.newaxis])/th[:,1,np.newaxis])
                q1_, q2_ = np.quantile(curves, 1-conf/2, axis=0), np.quantile(curves, conf/2, axis=0)
                for l in range(num_a) :
                    curves_q1[key][i,l] = curves[np.abs(curves[:,l] - q1_[l]).argmin()][l] #usefulness not certain
                    curves_q2[key][i,l] = curves[np.abs(curves[:,l] - q2_[l]).argmin()][l] 
                # curves_q1[key][i] = q1_[i]+0 
                # curves_q2[key][i] = q2_[i]+0 
                curves_med[key][i] = np.median(curves, axis=0)
                err_conf[key][i] = simpson(np.abs(curves_q1[key][i] - curves_q2[key][i])**2 *self.data.f_A_tab[np.newaxis], self.data.a_tab)**(1/2)
                err_med[key][i] = simpson(np.abs(curves_med[key][i] - self.ref.curve_MLE)**2 *self.data.f_A_tab[np.newaxis], self.data.a_tab)**(1/2)
                err_quad[key][i] = simpson((curves - self.ref.curve_MLE[np.newaxis])**2 *self.data.f_A_tab[np.newaxis], self.data.a_tab, axis=1).mean()

        return curves_q1, curves_q2, curves_med, err_conf, err_med, err_quad

    def average_errors(self, ks_, sim=['post', 'MLE'], n_est=200, num_mean=200, conf=0.05, print_fo=False) :
        """Compute average errors |q1-q2|_(L2(A))^2 (confidence zone scale), |Pmed-Pref|_(L2(A))^2 (median distance) 
        and E[|P-Pref|_(L2(A))^2] (quadratic error). Execute new simulations.
        Args:
            ks_ (iterable): dataset-sizes onto which compute.
            sim (list in ['post', 'MLE', 'MAP'], optional): simulation methods onto which compute. Defaults to ['post', 'MLE'].
            n_est (int, optional): Number of simulations to run with one data sample.
            num_mean (int, optional): Number of times data must be re-sampled. Defaults to 20.
            conf (float, optional): confidence interval probability. Defaults to 0.05.
            print_fo (bool, optional): whether or not printing updates at each iterations. Defaults to False. 
        Returns: tuple (err_conf, err_med, err_quad_avg)
            err_conf (dict[key in sim: numpy array(len(ks_))]): confidence zone scale.
            err_med (dict[key in sim: numpy array(len(ks_))]): median error.
            err_quad_avg (dict[key in sim: numpy array(len(ks_))]): quadradic error.
        """
        self.data.set_not_increasing_mode() # draw randomly data in the dataset for simulations
        theta = {key: np.zeros((len(ks_), num_mean, n_est, 2)) for key in sim}
        curves_post = {key: np.zeros((len(ks_), len(self.data.a_tab), num_mean, n_est)) for key in sim}
        err_quad_avg = {key: np.zeros((len(ks_))) for key in sim}
        err_conf = {key: np.zeros((len(ks_))) for key in sim}
        err_med = {key: np.zeros((len(ks_))) for key in sim}
        for key in sim :
            for i,k in enumerate(ks_) :
                for n in range(num_mean):
                    self.run_simulations([k], n_est, sim)
                    theta[key][i,n] = self.logs[key][k]
                    curves_post[key][i,:,n,:] = self.fragility_curve_func(self.data.a_tab, theta[key][i,n]).T
                    # curves_post[key][i,:,n,:] = 1/2 + 1/2*spc.erf(np.log(self.data.a_tab[...,np.newaxis]/theta[key][i,n,np.newaxis,:,0])/theta[key][i,n,np.newaxis,:,1])
            curves_post[key] = curves_post[key].reshape(len(ks_), len(self.data.a_tab), -1)
            err_conf[key] = simpson((np.quantile(curves_post[key], 1-conf/2, axis=-1)- np.quantile(curves_post[key], conf/2, axis=-1))**2 *self.data.f_A_tab[np.newaxis,:], self.data.a_tab, axis=1)
            err_med[key] = simpson((np.median(curves_post[key], axis=-1)-self.ref.curve_MLE[np.newaxis,:])**2 *self.data.f_A_tab[np.newaxis,:], self.data.a_tab, axis=1)
            err_quad_avg[key] = simpson((curves_post[key]-self.ref.curve_MLE[np.newaxis,:,np.newaxis])**2 *self.data.f_A_tab[np.newaxis,:,np.newaxis], self.data.a_tab, axis=1).mean(axis=-1)
        self.data.set_increasing_mode() # stop random draw
        return err_conf, err_med, err_quad_avg


    def plot_simul(self, kplot_, sim=['post', 'MLE'], axes=None, conf=0.05, colors_tab=['blue', 'orange', 'green', 'red', 'magenta', 'grey', 'yellow'], ref=True, alpha=1, keep_points=1) :
        """sim= sub list of {'post', 'MLE', 'MAP'} decide which simulation to plot
        axes = array of axes: size len(sim)*2. If None a new figure is created
        kplot_ = logs id to plot (array of ids)
        conf = probability of confidence interval
        colors_tab = list of colors for the plots
        return axes
        """
        # 1. fig raw th estimations
        # 2. fig curves
        curves_q1, curves_q2, curves_med, err_conf, err_med, err_quad = self.compute_meds_qt_err(kplot_, sim, conf)
        # kplot = np.mod(kplot_, min([len(self.logs[key]) for key in sim]))
        kplot = np.array(kplot_)
        if axes is None :
            fig = plt.figure()
            axes = np.zeros((len(sim),2), dtype='object')
            for j in range(len(sim)) :
                axes[j][0] = fig.add_subplot(2,len(sim), j+1)
                axes[j][1] = fig.add_subplot(2,len(sim), j+3 - len(sim)%2)
        for i,k in enumerate(kplot) :
            for j,key in enumerate(sim) :
                len_sim = self.logs[key][k].shape[0]
                id_keeps = np.random.choice(np.arange(len_sim), size=int(keep_points*len_sim), replace=False)
                axes[j][0].plot(self.logs[key][k][:,0][id_keeps], self.logs[key][k][:,1][id_keeps], 'o', color=colors_tab[i], label='{}'.format(k), alpha=alpha)
                if i==len(kplot)-1 and ref:
                    axes[j][0].plot(self.ref.theta_MLE[0], self.ref.theta_MLE[1], 'x', color='magenta', label=r'ref')
                axes[j][0].set_xlabel(r'$\alpha$')
                axes[j][0].set_ylabel(r'$\beta$')
                axes[j][0].set_title(r'{} simulations of $\theta$'.format(key))
                axes[j][0].legend()
                axes[j][1].fill_between(self.data.a_tab, curves_q1[key][i], curves_q2[key][i], facecolor=colors_tab[i], alpha=alpha)
                axes[j][1].plot(self.data.a_tab, curves_q1[key][i], label=r'{}'.format(k), color=colors_tab[i], alpha=alpha)
                if i==len(kplot)-1 and ref:
                    axes[j][1].plot(self.data.a_tab, self.ref.curve_MLE, color='magenta', label=r'ref')
                axes[j][1].set_xlabel(r'$a=$'+self.data.IM)
                axes[j][1].set_ylabel(r'$P_f(a)$')
                axes[j][1].set_title('Estimations from {}'.format(key))
                axes[j][1].legend()
        return axes

    def plot_errors(self, kplot_, sim=['post', 'MLE'], axes=None, conf=0.05, name_post={'MLE':'MLE', 'post':'post'}):
        """plot error figures for simulations from items in sim
            sim = sublist of {'MLE', 'post', 'MAP'}
            axes = list of 3 axes. If None a new figure is created    
            conf = probability for confidence intervals
            return axes
        """
        curves_q1, curves_q2, curves_med, err_conf, err_med, err_quad = self.compute_meds_qt_err(kplot_, sim, conf)
        # kplot = np.mod(kplot_, min([len(self.logs[key]) for key in sim]))
        kplot = np.array(kplot_)
        if axes is None :
            fig = plt.figure()
            axes = [fig.add_subplot(1,3,j+1) for j in range(3)]
        for key in sim :
            axes[0].plot(kplot, err_conf[key], label=name_post[key])
            axes[1].plot(kplot, err_med[key], label=name_post[key])
            axes[2].plot(kplot, err_quad[key], label=name_post[key])
        axes[0].set_xlabel('Number of data')
        axes[0].set_ylabel("error")
        axes[0].set_title(r'Confidence scale: $||q_{r/2}^{\alpha,\beta|s,a}-q_{1-r/2}^{\alpha,\beta|s,a}||_{L^2(\mathbb{P}_A)}$, '+r'$r={}$'.format(conf))
        axes[0].legend()
        axes[1].set_xlabel('Number of data') 
        axes[1].set_ylabel("error")
        axes[1].set_title(r'Median error $||q_{med}^{\alpha,\beta|s,a} - P_{ref}||_{L^2(\mathbb{P}_A)}$')
        axes[1].legend()
        axes[2].set_xlabel('Number of data')
        axes[2].set_ylabel("error")
        axes[2].set_title(r"quadratic error $\mathbb{E}[||P_{\alpha,\beta}-P_{ref}||_{L^2(\mathbb{P}_A)}^2|s,a]$")
        axes[2].legend()
        return axes

    def plot_average_errors(self, kplot_, sim=['post', 'MLE'], n_est=200, num_mean=200, conf=0.05, print_fo=False, axes=None, name_post={'MLE':'MLE', 'post':'post'}) :
        """plot average erros from several random data samples.
        Args:
            kplot_ (iterable): data sample sizes onto which to simulate and compute
            sim (list in ['post','MLE','MAP'], optional): simulation ind to compute. Defaults to ['post', 'MLE'].
            n_est (int, optional): number of theta estimation to perform for each data sample. Defaults to 200.
            num_mean (int, optional): number of different data sample to consider for avery k. Defaults to 200.
            conf (float, optional): confidence probability. Defaults to 0.05.
            print_fo (bool, optional): whether or not to plot updates about the iterations. Defaults to False.
            axes (list[pyplot axes], optional): list of 3 axes to plot errors. If None a new figure is created. Defaults to None.
            name_post (dict, optional): dict to change key in sim label on figure. Defaults to {'MLE':'MLE', 'post':'post'}.
        Returns:
            axes (list[pyplot axes]): list of 3 pyplot axes onto which are plotted the errors
        """
        kplot = np.array(kplot_)
        if axes is None :
            fig = plt.figure()
            axes = [fig.add_subplot(1,3,j+1) for j in range(3)]
        err_conf, err_med, err_quad = self.average_errors(kplot_, sim, n_est, num_mean, conf, print_fo)
        for key in sim :
            axes[0].plot(kplot, err_conf[key], label=name_post[key])
            axes[1].plot(kplot, err_med[key], label=name_post[key])
            axes[2].plot(kplot, err_quad[key], label=name_post[key])
        axes[0].set_xlabel('Number of data')
        axes[0].set_ylabel("error")
        axes[0].set_title(r'Confidence scale: $||q_{r/2}^{\alpha,\beta}-q_{1-r/2}^{\alpha,\beta}||_{L^2(\mathbb{P}_A)}$, '+r'$r={}$'.format(conf))
        axes[0].legend()
        axes[1].set_xlabel('Number of data') 
        axes[1].set_ylabel("error")
        axes[1].set_title(r'Median error $||q_{med}^{\alpha,\beta} - P_{ref}||_{L^2(\mathbb{P}_A)}$')
        axes[1].legend()
        axes[2].set_xlabel('Number of data')
        axes[2].set_ylabel("error")
        axes[2].set_title(r"quadratic error $\mathbb{E}[||P_{\alpha,\beta}-P_{ref}||_{L^2(\mathbb{P}_A)}^2]$")
        axes[2].legend()
        return axes


    def plot_prior(self, th_array, const=0, ax=None, ret_arrays=False) :
        """Plot the prior over th_array = n*2 array of theta values
            const = float to add(if log)/mult(else) to the density
            if ax=None a new figure is created
            if ret_arrays=True return pp(log_density/raw density else), ppe(density/mult density else) on top of the ax
        """
        th_grid1, th_grid2 = np.meshgrid(th_array[:,0], th_array[:,1])
        n = len(th_array)
        pp = np.zeros((n,n))
        for i in range(n) :
            for j in range(n) :
                pp[i,j] = (self.prior(np.array([[th_array[i,0], th_array[j,1]]])) + self.log*const).flatten()
        if self.log :
            ppe = np.exp(pp - pp.max())
        else :
            ppe = pp*(const+1)
        if not ret_arrays :
            return plot_functions.plot_bi_function(ppe, th_grid1, th_grid2, ax)
        else :
            return plot_functions.plot_bi_function(ppe, th_grid1, th_grid2, ax), pp, ppe

    def plot_post(self, k, th_array, const=0, ax=None, ret_arrays=False) :
        """Plot the posterior (from k samples) over th_array = n*2 array of theta values
            const = float to add(if log)/mult(else) to the density
            if ax=None a new figure is created
            if ret_arrays=True return pp(log_density/raw density else), ppe(density/mult density else) on top of the ax
        """
        self._update_data(k)
        th_grid1, th_grid2 = np.meshgrid(th_array[:,0], th_array[:,1])
        n = len(th_array)
        pp = np.zeros((n,n))
        for i in range(n) :
            for j in range(n) :
                pp[i,j] = (self.likelihood(self.S, self.A, np.array([[th_array[i,0], th_array[j,1]]])) + self.log*const + self.prior(np.array([[th_array[i,0], th_array[j,1]]]))).flatten()
        if self.log :
            ppe = np.exp(pp - pp.max()/2)
        else :
            ppe = pp*(const+1)
        if not ret_arrays :
            return plot_functions.plot_bi_function(ppe, th_grid1, th_grid2, ax)
        else :
            return plot_functions.plot_bi_function(ppe, th_grid1, th_grid2, ax), pp, ppe

    def plot_likelihood(self, k, th_array, const=0, ax=None, ret_arrays=False) :
        """Plot the likelihood (from k samples) over th_array = n*2 array of theta values
            const = float to add(if log)/mult(else) to the density
            if ax=None a new figure is created
            if ret_arrays=True return pp(log_density/raw density else), ppe(density/mult density else) on top of the ax
        """
        self._update_data(k)
        th_grid1, th_grid2 = np.meshgrid(th_array[:,0], th_array[:,1])
        n = len(th_array)
        pp = np.zeros((n,n))
        for i in range(n) :
            for j in range(n) :
                pp[i,j] = (self.likelihood(self.S, self.A, np.array([[th_array[i,0], th_array[j,1]]])) + self.log*const ).flatten()
        if self.log :
            ppe = np.exp(pp - pp.max())
        else :
            ppe = pp*(const+1)
        if not ret_arrays :
            return plot_functions.plot_bi_function(ppe, th_grid1, th_grid2, ax)
        else :
            return plot_functions.plot_bi_function(ppe, th_grid1, th_grid2, ax), pp, ppe



## specific subclass

class Model_math(Model) :
    """Model for custom computation of estimation curves, not from empirical estimations
    """
    def __init__(self, prior, likelihood, data, post_simul, compute_quantile, log=True, numba=True, linear=False):
        """compute_quantile= function(A,S,a_tab, conf)
            computule quantile curves for a confidence interval of conf proba, and the median"""
        super(Model_math, self).__init__(prior, likelihood, data, post_simul, log, numba, linear)
        self.compute_quantile = compute_quantile

    def compute_meds_qt_math(self, kids, conf=0.05, draw_meth=None) :
        """draw_method for explicit the way to draw k data.
            data.icreasing_draw if None"""
        num_c = len(kids)
        num_a = self.data.a_tab.shape[0]
        q1_curves = np.zeros((num_c, num_a))
        q2_curves = np.zeros((num_c, num_a))
        med_curves = np.zeros((num_c, num_a))
        err_conf = np.zeros(num_c)
        err_med = np.zeros(num_c)
        for i,k in enumerate(kids) :
            if draw_meth is None :
                if self.linear :
                    self.A,self.S,_ = self.data.increasing_draw(k)
                else :
                    self.A,_,self.S = self.data.increasing_draw(k)
            q1_curves[i], q2_curves[i], med_curves[i] =  self.compute_quantile(self.A, self.S, self.data.a_tab, conf)
            
            err_conf[i] = simpson(np.abs(q1_curves[i] - q2_curves[i])**2, self.data.a_tab)**(1/2)
            err_med[i] = simpson(np.abs(med_curves[i] - self.ref.curve_MLE)**2, self.data.a_tab)**(1/2)
        return q1_curves, q2_curves, med_curves, err_conf, err_med

    def plot_confidence_math(self, kids_conf, kids_err=None, axes=None, conf=0.05, colors_tab=['blue', 'orange', 'green', 'red', 'magenta', 'grey', 'yellow']):
        """plot confidence curves mathematically for k in kids_conf, and the error for k in kids_err if not None
            axes = list(1 ax) if kids_err=None ; = list(3 axes) otherwise
            if = None create a new figure
            kids_conf must be a sublist of kids_err if the former is not None 
        """
        if axes is None :
            fig = plt.figure()
            axes = list()
            axes.append(fig.add_subplot(111))
            if not kids_err is None : 
                fig2 = plt.figure()
                axes.append(fig2.add_subplot(121))
                axes.append(fig2.add_subplot(122))
        if kids_err is None :
            k_compute = kids_conf
        else : 
            k_compute = kids_err 
        q1_curves, q2_curves, med_curves, err_conf, err_med = self.compute_meds_qt_math(k_compute, conf)

        for i,k in enumerate(k_compute) :
            if k in kids_conf :
                axes[0].fill_between(self.data.a_tab, q1_curves[i], q2_curves[i], facecolor=colors_tab[i])
                axes[0].plot(self.data.a_tab, q1_curves[i], label=r'{}'.format(k), color=colors_tab[i])
                axes[0].plot(self.data.a_tab, self.ref.curve_MLE, color='magenta', label=r'ref')
                axes[0].set_xlabel(r'$a=$'+self.data.IM)
                axes[0].set_ylabel(r'$P_f(a)$')
                axes[0].set_title('Confidence posterior interval')
                axes[0].legend()
        
        if not kids_err is None :
            axes[1].plot(kids_err, err_conf)
            axes[2].plot(kids_err, err_med)
            axes[1].set_xlabel('Number of data') ## simulations are supposed to have been done with step 1 increasing number of data
            axes[1].set_ylabel("error")
            axes[1].set_title(r'Confidence scale: $|q_{r/2}-q_{1-r/2}|_2$, '+r'$r={}$'.format(conf))
            axes[1].legend()
            axes[2].set_xlabel('Number of data') ## simulations are supposed to have been done with step 1 increasing number of data
            axes[2].set_ylabel("error")
            axes[2].set_title(r'Median error $|q_{med} - P_{ref}|_2$')
            axes[2].legend()

        return axes



class Model_pickle(Model) :
    def __init__(self, prior, likelihood, data, model_file, log=True, numba=True, linear=False, ref=None, bounds_bst=None, option_bst=None, fragility_curve_func=probit_fargility_curve):
        super().__init__(prior, likelihood, data, lambda x:x, log, numba, linear, ref, bounds_bst, option_bst, fragility_curve_func)
        
        self.A = model_file['A']
        self.S = model_file['S']
        self.logs = model_file['logs']

    def _update_data(self, k) :
        if len(self.A.squeeze())>k :
            pass
        else :
            super()._update_data(k)
    
        





