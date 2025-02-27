import numpy as np
from scipy.optimize import minimize

from ..model import Model, probit_fargility_curve


class Minimizer() :
    def __init__(self, mini_func) :
        self.x = None
        self.minimizer = mini_func

    def __call__(self, func, x0, bounds=None) :
        self.x = self.minimizer(func, x0, bounds=bounds)
        return self


class Adaptative_Acquisition() :
    def __init__(self, index_builder, data, refill=False, minimizer=minimize, starting_points=2, full_logs=False) :
        self.logs = {'I':[], 'A':None, 'Y':None}
        self.index_builder = index_builder
        self.data = data
        self.minimizer = minimizer
        self.start = starting_points
        self.updatedA = data.A.flatten() +0
        self.refill = refill
        self.full_logs = full_logs

    def _clear_logs(self) :
        self.logs = {'I':[], 'A':None, 'Y':None}
        self.updatedA = self.data.A.flatten() +0

    def sample(self, k) :
        self._clear_logs()
        A = np.zeros((k,1))
        Y = np.zeros((k,1))
        # ids = np.random.choice(np.arange(self.data.A.shape[0])[self.data.A.flatten()>5], size=self.start, replace=False)
        # A[:self.start], Y[:self.start] = self.data.A[ids]+0, self.data.Y[ids]+0
        A[:self.start],Y[:self.start],_ = self.data.default_draw(self.start - 1)
        A,Y = A.flatten(), Y.flatten()
        p = self.start
        while p!=k :
            index = self.index_builder(A[:p],Y[:p])
            if self.full_logs :
                self.logs['I'].append(index)
            ne_a = self.minimizer(lambda a:-index(a), 2).x #, bounds=((self.data.a_tab[0], self.data.a_tab[-1]))).x
            # minimize()
            next_id = np.argmin(np.abs(ne_a-self.updatedA))
            A[p], Y[p] = self.data.A[next_id], self.data.Y[next_id]
            p += 1
            if not self.refill :
                self.updatedA[next_id] = -10**5
        self.logs['A'] = A+0
        self.logs['Y'] = Y+0
        print(self.logs)
        return A[:,np.newaxis], Y[:,np.newaxis]

    def sample_from_init(self, k) :
        A = np.zeros(k)
        Y = np.zeros(k)
        start_already = self.logs['A'].shape[0]
        A[:start_already] = self.logs['A'][:start_already]+0
        Y[:start_already] = self.logs['Y'][:start_already]+0
        p = start_already
        while p!=k :
            index = self.index_builder(A[:p],Y[:p])
            self.logs['I'].append(index)
            ne_a = self.minimizer(lambda a:-index(a), 2).x #, bounds=((self.data.a_tab[0], self.data.a_tab[-1]))).x
            # minimize()
            next_id = np.argmin(np.abs(ne_a-self.updatedA))
            A[p], Y[p] = self.data.A[next_id], self.data.Y[next_id]
            p += 1
            if not self.refill :
                self.updatedA[next_id] = -10**5
        self.logs['A'] = A+0
        self.logs['Y'] = Y+0
        return A[:,np.newaxis], Y[:,np.newaxis]
        

# class Adaptative_from_post(Adaptative_Acquisition) :
#     def __init__(self, index_builder, data, refill=False, minimizer=minimize, starting_points=2):
#         super().__init__(index_builder, data, refill, minimizer, starting_points) 
    
#     def sample(self, k):
#         return super().sample(k)
    
#     def sample_from_init(self, k):
#         return super().sample_from_init(k)




class Model_AS(Model) :
    def __init__(self, AA, *args, **kwargs) :
        super(Model_AS, self).__init__(*args, **kwargs)
        self.adaptative = AA

    def _update_data(self, k) :
        A, Y = self.adaptative.sample(k)
        if self.linear :
            self.A, self.S =  A, Y
        else :
            self.A, self.S = A, 1*(Y>=self.data.C)


class Model_AS_1run(Model_AS) :
    def __init__(self, *args, **kwargs) : #self, prior, likelihood, data, post_simul, log=True, numba=True, linear=False, ref=None, bounds_bst=None, option_bst=None, fragility_curve_func=...):
        super(Model_AS_1run, self).__init__(*args, **kwargs)
        #super().__init__(prior, likelihood, data, post_simul, log, numba, linear, ref, bounds_bst, option_bst, fragility_curve_func)

    def _update_data(self, k) :
        try :
            shape_A = len(self.A)
        except :
            shape_A = 0
        if shape_A<self.adaptative.start :
            # print('debug: k={}, shapeselfA={}, ')
            A, Y = self.adaptative.sample(k)
        elif shape_A>=k :
            A, Y = self.A[:k]+0, self.S[:k]+0
        else :
            A, Y = self.adaptative.sample_from_init(k)
            print('echo')
        if self.linear :
            self.A, self.S =  A, Y
        else :
            self.A, self.S = A, 1*(Y>=self.data.C)
    
    def _choose_A_S(self, A, Y) :
        if self.linear :
            self.A, self.S =  A, Y
        else :
            self.A, self.S = A, 1*(Y>=self.data.C)

    def clear_logs(self):
        super().clear_logs()
        self.adaptative._clear_logs()
        self.A, self.S = None, None


# class Model_AS_pers_curve(Model_AS) :
#     def __init__(self, curves_est, AA, *args, **kwargs) :
#         super(Model_AS_pers_curve, self).__init__(AA, *args, **kwargs)
#         self.curves_est = curves_est


class Model_AS_1run_posterior(Model_AS_1run) :
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert type(self.adaptative.index_builder)==type(Index_class())
        # index_class = self.adaptative.index_builder
        # if index_class.likelihood is None :
        #     index_class.likelihood = self.likelihood
        #     index_class.log = self.log

    def _update_data(self, k):
        if len(self.logs['post'].keys())>0 :
            n_last_post = list(self.logs['post'].keys())[-1]
            self.adaptative.index_builder.n_data = n_last_post
            self.adaptative.index_builder.posterior_sample = self.logs['post'][n_last_post]
        return super()._update_data(k)
    
    def clear_logs(self):
        super().clear_logs()
        self.adaptative.index_builder.n_data = 0


class Index_class() :
    def __init__(self, index_func_build, a_priori_points) : #, likelihood, log=False) :
        self.index_func = index_func_build
        # self.likelihood = likelihood
        # self.prior = None
        self.posterior_sample = a_priori_points
        self.n_data = 0

    def __call__(self, A, Y) :
        # if self.n_data == 0 :
        #     pass
        return self.index_func(A[self.n_data:], Y[self.n_data:], self.posterior_sample)
        





if __name__=="__main__" :
    from ..data import Data
    IM = 'PGA'
    data = Data(IM)
    def naive_min(func, x0, bounds=None) :
        return data.a_tab[np.nanargmin([func(a) for a in data.a_tab])]