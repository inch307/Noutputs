import numpy as np
import random
import itertools
import math

def comb(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

class Duchi():
    def __init__(self, d, eps):
        self.d = d
        self.eps = eps

        # for Duchi multi
        self.T = np.array(list(itertools.product([-1, 1], repeat = self.d)))

        if self.d % 2 == 0:
            self.C_d = 2**(self.d - 1) / comb(self.d-1, (self.d-1) // 2)
        else:
            self.C_d = (2**(self.d - 1) + comb(self.d, self.d // 2) / 2) / comb(self.d - 1, self.d // 2)

        self.B = self.C_d * (np.exp(eps) + 1) / (np.exp(eps) - 1)

    def Duchi_single(self, x, eps):
        u = np.random.choice([-1, 1], 1, p = [-x * (np.exp(eps)-1) / (2*np.exp(eps) + 2)  + 0.5  , x * (np.exp(eps)-1) / (2*np.exp(eps) + 2)  + 0.5])    
        return u * (np.exp(eps) + 1) / (np.exp(eps) - 1)

    def Duchi_var(self, x, eps):
        return ((np.exp(eps) + 1) / (np.exp(eps) - 1))**2 - x**2
    
    def Duchi_max_var(self):
        return

    def Duchi_multi(self, x, eps):
        original_shape = x.shape
        x = x.reshape(self.d)
        v = []
        for i in range(self.d):
            v.append(np.random.choice([-1, 1], size=1, p = [0.5 - 0.5*x[i], 0.5 + 0.5*x[i]]))
        v = np.array(v)

        Tv = np.dot(self.T, v)

        T_plus = np.where(np.any(Tv >= 0, axis=1))[0]
        T_minus = np.where(np.any(Tv <= 0, axis=1))[0]

        u = np.random.choice([0, 1], 1, p = [1 / (np.exp(eps) + 1) , np.exp(eps) / (np.exp(eps) + 1)])

        if u == 0:
            R_T = self.T[np.random.choice(T_minus, 1)]
        else:
            R_T = self.T[np.random.choice(T_plus, 1)]

        return self.B * R_T.reshape(original_shape)

    
if __name__ == '__main__':
    t = np.array([[0.7, 0.3], [-0.7, -0.4]])
    t_star = Duchi_multi(t, 5)
    print(t_star)