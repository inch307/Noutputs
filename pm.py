import numpy as np
import math
import duchi

class PM():
    def __init__(self, d, eps):
        self.d = d
        self.eps = eps
        self.k = max(1, min(d, math.floor(eps / 2.5)))
        self.eps_k = self.eps / self.k
        self.A = (np.exp(eps/2/self.k) + 1) / (np.exp(eps/2/self.k) - 1)

        if self.eps > np.log( (-5 + 2*np.cbrt(6353-405*np.sqrt(241) + 2*np.cbrt(6353+405*np.sqrt(241))  )) / 27 ):
            self.alpha = 1 - np.exp(-eps/2/self.k)
        else:
            self.alpha = 0

    def PM_single(self, x):
        u = np.random.rand()

        l = x * (self.A + 1) / 2 - (self.A - 1) / 2
        r = l + self.A - 1

        if u < np.exp(self.eps_k/2) / (np.exp(self.eps_k/2)+1):
            y = np.random.uniform(l, r)
        else:
            dist_ratio = (l + self.A) / (self.A - r)
            if np.random.rand() > 1 / (1 + dist_ratio):
                y = np.random.uniform(-self.A, l)
            else:
                y = np.random.uniform(r, self.A)
        
        return y

    def HM_single(self, x):
        if np.random.rand() < self.alpha:
            y = self.PM_single(x)
        else:
            y = duchi.Duchi_single(x, self.eps_k)

        return y
    
    def HM_max_var(self):

        
    def HM_multi(self, x):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        js = np.random.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            y[j] = self.d * self.HM_single(x[j]) / self.k

        return y.reshape(original_shape)
