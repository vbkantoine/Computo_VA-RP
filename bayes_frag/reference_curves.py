# reference_curves script
# compute reference curves estimation from a full dataset
import numpy as np
from scipy.special import erf
from scipy import stats as stat
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

from data import Probit_curve
from stat_functions import get_opp_log_vr_probit

##
#
class Ref_main():
    def __init__(self, data) :
        """data = data object containing the dataset"""
        self.data = data
        self.reglin = LinearRegression().fit(np.log(data.A), np.log(data.Y))
        self.curve_MC = None
        self.curve_MC_var = None
        self.a_tab_MC = None
        self.probit_ref = None
        self._compute_empirical_curves()

    def _compute_empirical_curves(self, Nc=35) :
        kmeans_IM = KMeans(n_clusters=Nc).fit(self.data.A)
        a_m_tab = kmeans_IM.cluster_centers_.flatten()
        pMC = np.zeros_like(a_m_tab)
        sMC = np.zeros_like(a_m_tab)

        for i in range(Nc) :
            z_ci = self.data.Z[kmeans_IM.labels_==i]
            pMC[i] = np.mean(z_ci)
            sMC[i] = np.sqrt(pMC[i]*(1-pMC[i])/z_ci.shape[0])*1.96 #todo: re-think the 1.96 constant w.r.t a desired alpha-confidence
        id_sort = np.argsort(a_m_tab)
        self.a_tab_MC = a_m_tab[id_sort]
        self.curve_MC = pMC[id_sort]
        self.curve_MC_var = sMC[id_sort]
        return kmeans_IM


class Reference_curve(Ref_main) :
    """Reference_curve class
    gather 'reference' items for comparison, computed with the consideration of all the available data"""
    def __init__(self, data) :
        """data = data object containing the dataset"""
        super().__init__(data)
        self.theta_MLE = None
        self.curve_MLE = None
        self.curve_MC_data_tabs = None
        self._compute_MLE_curve()

    def _compute_MLE(self, opp_log_vraiss=None) :
        if opp_log_vraiss is None :
            opp_log_vraiss = get_opp_log_vr_probit(self.data.Z, self.data.A)
        self.theta_MLE = optimize.minimize(opp_log_vraiss, np.array([3,0.3]), options={'maxiter':50}).x
        self.probit_ref = Probit_curve(self.theta_MLE[0], self.theta_MLE[1])
        return self.theta_MLE

    def _compute_MLE_curve(self) :
        if self.theta_MLE is None :
            self._compute_MLE()
        self.curve_MLE = 1/2 + 1/2 * erf(np.log(self.data.a_tab/self.theta_MLE[0])/self.theta_MLE[1])

    def plot_ref_fig(self, ax=None, num_points=500) :
        """plot the reference fragility curves
            If ax is None a new figure is created"""
        if ax is None :
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # ax.plot(self.data.a_tab, self.curve_MC, label='MC')
        ax.plot(self.a_tab_MC, self.curve_MC, '--', label='MC', color='grey', alpha=0.35)
        ax.fill_between(self.a_tab_MC, self.curve_MC+self.curve_MC_var, self.curve_MC-self.curve_MC_var, color='grey', alpha=0.6)
        # ax.plot(self.data.a_tab, self.curve_MC_lis, label='Kernel')
        ax.plot(self.data.a_tab, self.curve_MLE, label=r'MLE', color='magenta') #, $\theta=({:4.2f},{:4.2f})$'.format(self.theta_MLE[0], self.theta_MLE[1]))
        id_points = np.random.choice(np.arange(len(self.data.A)), size=num_points)
        ax.plot(self.data.A[id_points], self.data.Z[id_points], 'x', color='red', markersize=3, label='obs.')
        ax.set_xlim((self.data.a_tab.min(), self.data.a_tab.max()))
        ax.set_xlabel('a={} (m/s$^2$)'.format(self.data.IM))
        ax.set_ylabel(r'$P_f(a)$')
        ax.set_title(r'Reference fragility curve')
        ax.legend(loc='lower right')
        ax.set_ylim((-0.01, 1.01))
        return ax

    def _compute_MC_data_tab(self) :
        if self.curve_MC is None :
            self._compute_empirical_curves()
        c_tab = np.zeros_like(self.data.a_tab)
        s_tab = np.zeros((self.data.a_tab.shape[0],2))
        for i,a in enumerate(self.data.a_tab) :
            if a <= self.a_tab_MC[0] :
                c_tab[i] = a* self.curve_MC[0]/self.a_tab_MC[0]
                smc_inf_0, smc_inf_1 = 0, self.curve_MC[0]-self.curve_MC_var[0]
                smc_max_0, smc_max_1 = 0, self.curve_MC[0]+self.curve_MC_var[0]
                amc_0 = 0
                amc_1 = self.a_tab_MC[0]
            elif a>self.a_tab_MC[-1] :
                c_tab[i] = (a-self.a_tab_MC[-1])*(1-self.curve_MC[-1])/(self.data.a_tab.max()-self.a_tab_MC[-1]) + self.curve_MC[-1]
                smc_inf_0, smc_inf_1 = self.curve_MC[-1]-self.curve_MC_var[-1], 1
                smc_max_0, smc_max_1 = self.curve_MC[-1]+self.curve_MC_var[-1], 1
                amc_0 = self.a_tab_MC[-1]
                amc_1 = self.data.a_tab.max()
            else :
                i_mc = (self.a_tab_MC<=a).sum() -1
                c_tab[i] = (a-self.a_tab_MC[i_mc]) * (self.curve_MC[i_mc+1]-self.curve_MC[i_mc])/ (self.a_tab_MC[i_mc+1]-self.a_tab_MC[i_mc]) + self.curve_MC[i_mc]
                smc_inf_0, smc_inf_1 = self.curve_MC[i_mc]-self.curve_MC_var[i_mc], self.curve_MC[i_mc+1]-self.curve_MC_var[i_mc+1]
                smc_max_0, smc_max_1 = self.curve_MC[i_mc]+self.curve_MC_var[i_mc], self.curve_MC[i_mc+1]+self.curve_MC_var[i_mc+1]
                amc_0 = self.a_tab_MC[i_mc]
                amc_1 = self.a_tab_MC[i_mc+1]
            s_tab[i,0] = np.maximum((a-amc_0) * (smc_inf_1-smc_inf_0 ) / (amc_1-amc_0 ) + smc_inf_0, 0)
            s_tab[i,1] = np.minimum((a-amc_0) * (smc_max_1-smc_max_0 ) / (amc_1-amc_0 ) + smc_max_0, 1)
        self.curve_MC_data_tabs = [c_tab, s_tab]
        return self.curve_MC_data_tabs



class Reference_saved_MLE(Reference_curve) :
    def __init__(self, data, path_MLE):
        self.path_MLE = path_MLE
        super().__init__(data)

    def _compute_MLE(self):
        self.theta_MLE = pickle.load(open(self.path_MLE, 'rb'))
        self.probit_ref = Probit_curve(self.theta_MLE[0], self.theta_MLE[1])


class Reference_known_MLE(Reference_curve) :
    def __init__(self, data, theta_MLE):
        """
        Args:
            data (Data): data
            theta_MLE (array (2)): MLE theta reference for Data
        """
        super().__init__(data)
        self.theta_MLE = theta_MLE
        self.probit_ref = Probit_curve(theta_MLE[0], theta_MLE[1])
        self._compute_MLE_curve()

    def _compute_MLE(self):
        pass

    def _compute_MLE_curve(self):
        if self.theta_MLE is None :
            pass
        else :
            return super()._compute_MLE_curve()



# def get_curve_opt_threshold(alpha, beta, q) :
#         x = stat.norm.ppf(q)
#         return np.exp(beta*x/np.sqrt(2) + np.log(alpha))



if __name__=="__main__":
    from data import Data
    import os
    import config
    plt.ion()
    plt.show()
    IM = 'sa_5hz'
    # dat = Data(IM)
    # dat = Data('PGA', csv_path='Res_ASG.csv', quantile_C=0.9, name_inte='rot_nlin', shuffle=True)
    # dat = Data(IM, csv_path='Res_ASG_Lin.csv', quantile_C=0.9, name_inte='rot')
    dat = Data(IM, csv_path='Res_ASG_SA_PGA_RL_RNL_80000.csv', quantile_C=0.9, name_inte='rot_nlin', shuffle=True)
    # ref = Reference_saved_MLE(dat, os.path.join(config.data_path, 'ref_MLE_ASG_{}'.format(IM)))
    ref = Reference_curve(dat)
    ref._compute_empirical_curves(20) #15 10^4, 25 10^5
    ref.plot_ref_fig()

    from config import data_path
    import os
    pickle.dump(ref.theta_MLE, open(os.path.join(data_path, 'ref_MLE_ASG_80000_{}'.format(IM)), 'wb'))



