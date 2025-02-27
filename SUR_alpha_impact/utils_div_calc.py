
import scipy.stats as stat
import numpy as np



def hats_data(A,Y) :
    return 2*Y-1, np.log(A)

def build_index_f_div_post(A,Y, sample, be=1/2) :
    
    alpha_prior = sample[:,0]
    beta_prior = sample[:,1]

    hat_y_k, hat_a_k = hats_data(A,Y) # shape k

    P_rec_s_arr = stat.norm.cdf(np.nan_to_num((hat_a_k[:,np.newaxis] -np.log(alpha_prior)[np.newaxis])/ beta_prior[np.newaxis] ) ) # shape (k, mc)
    P_rec_s_arr *= (2*(hat_y_k>0)-1)[:,np.newaxis] 
    P_rec_s_arr += 1*(hat_y_k<0)[:,np.newaxis]
    P_rec_s = np.prod(P_rec_s_arr, axis=0) # shape mc
    assert np.all(P_rec_s==1)
    Lkp = np.nan_to_num(P_rec_s.mean(), nan=1)
    def index_n(a,s) :
        if a==0 :
            return 0
        P1a_rec_s = P_rec_s*stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior)**(1-be) # shape mc
        P0a_rec_s = P_rec_s*(1-stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior))**(1-be) 

        Int1 = (P1a_rec_s*stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior)**be).mean()**(be) * np.mean(P1a_rec_s)
        Int0 = (P0a_rec_s*(1-stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior))**be).mean()**(be) * np.mean(P0a_rec_s)
        div0 = be*(be-1) *  (P0a_rec_s* (1-stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior))**be).mean()**(be-1) * np.mean(P0a_rec_s)
        div1 = be*(be-1) *  (P1a_rec_s* stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior)**be).mean()**(be-1) * np.mean(P1a_rec_s)
        return  be*(be-1) *(Int1+Int0) /Lkp**(be+1), div0*(s==0)/Lkp**be + div1*(s==1)/Lkp**be
    return index_n



def build_index_f_div(A,Y, mu_a_ln, sigma_a_ln, beta_invcdf, be=1/2) :
    num_MC = 2000
    alpha_prior = np.exp( mu_a_ln+sigma_a_ln*stat.norm.rvs(size=num_MC) )
    beta_prior = beta_invcdf(stat.uniform.rvs(size=num_MC))
    hat_y_k, hat_a_k = hats_data(A,Y) # shape k

    P_rec_s_arr = stat.norm.cdf(np.nan_to_num((hat_a_k[:,np.newaxis] -np.log(alpha_prior)[np.newaxis])/ beta_prior[np.newaxis] ) ) # shape (k, mc)
    P_rec_s_arr *= (2*(hat_y_k>0)-1)[:,np.newaxis] 
    P_rec_s_arr += 1*(hat_y_k<0)[:,np.newaxis]
    P_rec_s = np.prod(P_rec_s_arr, axis=0) # shape mc
    k = A.shape[0]
    Lkp = P_rec_s.mean()
    def index_n(a,s) :
        if a==0 :
            return 0
        P1a_rec_s = P_rec_s*stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior)**(1-be) # shape mc
        P0a_rec_s = P_rec_s*(1-stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior))**(1-be) 
        Int1 = (P1a_rec_s*stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior)**be).mean()**(be) * np.mean(P1a_rec_s)
        Int0 = (P0a_rec_s*(1-stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior))**be).mean()**(be) * np.mean(P0a_rec_s)
        
        div0 = be*(be-1) *  (P0a_rec_s* (1-stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior))**be).mean()**(be-1) * np.mean(P0a_rec_s)
        div1 = be*(be-1) *  (P1a_rec_s* stat.norm.cdf((np.log(a) -np.log(alpha_prior) )/beta_prior)**be).mean()**(be-1) * np.mean(P1a_rec_s)
        return be*(be-1) *(Int1+Int0) /Lkp**(be+1), div0*(s==0)/Lkp**be + div1*(s==1)/Lkp**be
    return index_n