# save_fisher script
# provide a saved approximation of fisher information matrix for probit model
import numpy as np
import pickle
import os

from config import data_path, IM_dict
import fisher



## Calcul et sauvegarde d'un maillage fin de Jeffrey


# function = fisher.Fisher_Simpson

# a_tab = np.linspace(10**-5, 10, )

# I = fisher.Fisher_Simpson(theta_tab1, theta_tab2, a_tab)

def save_fisher(save_path, function, theta_tab1, theta_tab2) :
    num = theta_tab1.shape[0]
    I = np.zeros((num,num,2,2))

    for i,alpha in enumerate(theta_tab1) :
        for j, beta in enumerate(theta_tab2) :
            a_tab = np.exp(np.linspace(np.log(alpha)-4*beta, np.log(alpha)+4*beta, 40))
            I[i,j] = function(np.array([alpha]).reshape(1,1), np.array([beta]).reshape(1,1), a_tab)
        if i%10 ==0 :
            print("i={}/{}".format(i,num))
            # print(alpha)
            # break
        # break

    file = open(save_path, mode='wb')
    pickle.dump(I, file)
    file.close()

    return I

if __name__=="__main__":
    from data import Data
    IM = "sa_5hz"
    dat = Data(IM)

    save_fisher_arr = IM_dict[IM]['save_fisher_arr']

    alpha_min = save_fisher_arr.alpha_min
    alpha_max = save_fisher_arr.alpha_max
    beta_min = save_fisher_arr.beta_min
    beta_max = save_fisher_arr.beta_max
    num = save_fisher_arr.num_alpha

    theta_tab1 = save_fisher_arr.alpha_tab
    theta_tab2 = save_fisher_arr.beta_tab

    save_path = os.path.join(data_path, r"Fisher_array_{}".format(IM))
    function = fisher.fisher_function("simpson", dat)
    I = save_fisher(save_path, function, theta_tab1, theta_tab2)








##
