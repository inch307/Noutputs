import math
import numpy as np

class TOPM():
    def __init__(self, d, eps, t=None):
        self.d = d
        self.eps = eps
        # eps_star for TO_single
        self.k = max(1, min(d, math.floor(eps / 2.5)))
        self.eps_k = self.eps / self.k
        if t is None:
            self.t = np.exp(self.eps_k / 3)
        else:
            self.t = t
        self.P0_0, self.C = self.get_Ps()

        self.Px_0, self.Px_1 = self.construct_prob()

        # for piecewise mech
        self.A = (np.exp(self.eps_k) + self.t) * (self.t + 1) / (self.t * (np.exp(self.eps_k)-1))

        # HM
        self.beta = self.get_beta()
        self.x_star = (self.beta - 1) * self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2 / ( 2 * (np.exp(self.eps_k) - self.P0_0)**2 * (self.beta * (np.exp(self.eps_k) + self.t) - np.exp(self.eps_k) + 1))

    def construct_prob(self):
        Px_1 = (np.exp(self.eps_k) - self.P0_0) / (np.exp(self.eps_k) + 1) - (1 - self.P0_0) / 2
        Px_0 = self.P0_0 / np.exp(self.eps_k) - self.P0_0

        return Px_0, Px_1

    def get_Ps(self):
        if self.eps_k < np.log(2):
            P0_0 = 0
        elif self.eps_k <= np.log((3 + np.sqrt(65)) / 2):
            delta_0 = np.exp(4*self.eps_k) + 14*np.exp(3*self.eps_k) + 50*np.exp(2*self.eps_k) - 2 * np.exp(self.eps_k) + 25
            delta_1 = -2*np.exp(6*self.eps_k) -42*np.exp(5*self.eps_k) -270*np.exp(4*self.eps_k) -404*np.exp(3*self.eps_k) -918*np.exp(2*self.eps_k) +30*np.exp(self.eps_k) -250
            
            P0_0 = (-1/6) * ( -np.exp(2*self.eps_k) - 4*np.exp(self.eps_k) - 5 + 2 * np.sqrt(delta_0) * np.cos(np.pi/3 + (1/3) * np.arccos(-delta_1 / (2*np.sqrt(delta_0**3))) ))
        else:
            P0_0 = np.exp(self.eps_k) / (np.exp(self.eps_k) + 2)

        C = (np.exp(self.eps_k) + 1) / ((np.exp(self.eps_k)-1) * (1 - P0_0 / np.exp(self.eps_k)))

        return P0_0, C

    def get_beta(self):
        if self.eps_k > 0 and self.eps_k < 0.610986:
            beta = 0
        elif self.eps_k < np.log(2):
            beta = (2*(np.exp(self.eps_k) - self.P0_0)**2 * (np.exp(self.eps_k) -1) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2) / (2 * (np.exp(self.eps_k)-self.P0_0)**2 * (np.exp(self.eps_k) + self.t) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2)
        else:
            c = np.exp(self.eps_k)
            A = self.P0_0**2 * c**2 * (c+1)**4 / (4 * (c + self.t)**2 * (c - self.P0_0)**4 * (c-1)) - self.P0_0**2 * c**2 * (c+1)**4 / (2 * (c+self.t)**2 * (c-1) * (c-self.P0_0)**4) + ( (self.t+1)**3 + c - 1) / (3 * self.t**2 * (c - 1)**2) - (1 - self.P0_0) * c**2 * (c+1)**2 / ((c+self.t)*(c-1)**2 * (c-self.P0_0)**2)
            B = (-(1 + self.t)**2 * self.P0_0**2 * c**2 * (c+1)**4) / (4* (c+self.t)**2 * (c-self.P0_0)**4 * (c-1))

            beta1 = (2*(np.exp(self.eps_k) - self.P0_0)**2 * (np.exp(self.eps_k) -1) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2) / (2 * (np.exp(self.eps_k)-self.P0_0)**2 * (np.exp(self.eps_k) + self.t) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2)

            beta = (-np.sqrt(B/A) + np.exp(self.eps_k) -1) / (np.exp(self.eps_k) + np.exp(self.eps_k/3))
        
        return beta

    def TO_single(self, x):
        P_0_x = self.P0_0 + self.Px_0 * np.abs(x)

        if x >= 0 and x <= 1:
            P_C_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(x)
            P_mC_x = 1 - P_C_x - P_0_x
        else:
            P_mC_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(x)
            P_C_x = 1 - P_0_x - P_mC_x

        u = np.random.choice([-1, 0, 1], 1, p = [P_mC_x, P_0_x, P_C_x])

        return u * self.C
    
    def TO_max_var(self):
        b = self.P0_0 * (1 - 1/np.exp(self.eps_k))

        if self.C**2*b < 2:
            V = (1 - self.P0_0) * self.C**2 + self.C**4 * b**2 / 4
        else:
            V = (1 - self.P0_0 + b) * self.C**2 - 1

        # if self.eps_k < np.log(2):
        #     return (np.exp(self.eps_k) + 1)**2 / (np.exp(self.eps_k) - 1)**2
        # elif self.eps_k <= np.log(5.53):
        #     return np.exp(2*self.eps_k) * (np.exp(self.eps_k) + 1)**2 / (np.exp(self.eps_k) - 1)**2 * ( (1-self.P0_0) / (np.exp(self.eps_k) - self.P0_0)**2 + (np.exp(self.eps_k) + 1)**2 * self.P0_0**2 / ( 4 * (np.exp(self.eps_k - self.P0_0))**4) )
        # else:
        #     return (np.exp(self.eps_k) + 2) * (np.exp(self.eps_k) + 10) / ( 4 * (np.exp(self.eps_k) -1)**2 )

        # if self.eps_k < np.log(2):
        #     V = (np.exp(self.eps_k) + 1)**2 / (np.exp(self.eps_k) - 1)**2
        # elif self.eps_k < np.log((3 + np.sqrt(65))/ 2):
        #     V = ((np.exp(self.eps_k) + 1)**2 * np.exp(self.eps_k * 2) / (np.exp(self.eps_k)-1)**2) * ( (1-self.P0_0) / (np.exp(self.eps_k) - self.P0_0)**2 + (np.exp(self.eps_k)+1)**2 * self.P0_0**2 / (4 * (np.exp(self.eps_k) - self.P0_0)**4)  )
        # else:
        #     print('aweaweqaweawe')
        #     V = (np.exp(self.eps_k) + 2) * (np.exp(self.eps_k) + 10) / (4 * np.exp(self.eps_k) - 1)**2

        return V
    
    def TO_multi(self, x):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        js = np.random.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            y[j] = self.d * self.TO_single(x[j]) / self.k

        return y.reshape(original_shape)
    
    def TO_var(self, x):
        b = self.P0_0*(1 - 1/np.exp(self.eps_k))
        C = (np.exp(self.eps_k) + 1) / ((np.exp(self.eps_k) -1) * (1 - self.P0_0 / np.exp(self.eps_k)))

        return C**2 * (1 - self.P0_0) + C**2 * b * np.abs(x) - x**2
    
    def PM_single(self, x):
        u = np.random.rand()

        l = (np.exp(self.eps_k) + self.t) * (x * self.t - 1) / (self.t * (np.exp(self.eps_k) - 1))
        r = (np.exp(self.eps_k) + self.t) * (x * self.t + 1) / (self.t * (np.exp(self.eps_k) - 1))

        if u < np.exp(self.eps_k) / (self.t + np.exp(self.eps_k)):
            y = np.random.uniform(l, r)
        else:
            if self.A - r > 1e-8:
                dist_ratio = (l + self.A) / (self.A - r)
            else:
                dist_ratio = np.inf
            if np.random.rand() > 1 / (1 + dist_ratio):
                y = np.random.uniform(-self.A, l)
            else:
                y = np.random.uniform(r, self.A)
        
        return y
    
    def PM_multi(self, x):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        js = np.random.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            y[j] = self.d * self.PM_single(x[j]) / self.k

        return y.reshape(original_shape)
    
    def HM_single(self, x):
        if np.random.rand() > self.beta:
            y = self.TO_single(x)
        else:
            y = self.PM_single(x)

        return y
    
    def HM_var(self, x):
        a = self.P0_0
        b = self.P0_0 * (1 - 1/np.exp(self.eps_k))
        return self.beta * ((self.t+1) * x**2 / (np.exp(self.eps_k)-1) + (self.t +np.exp(self.eps_k)) * (((self.t+1)**3) + np.exp(self.eps_k) - 1) / (3*self.t**2 * (np.exp(self.eps_k) -1)**2)) + (1 - self.beta) *( ( (1 - a) * np.exp(2*self.eps_k) * (np.exp(self.eps_k) + 1 )**2 / ((np.exp(self.eps_k) - 1)**2 * (np.exp(self.eps_k)-a)**2)) + b * np.abs(x) * np.exp(2 * self.eps_k) * (np.exp(self.eps_k) + 1 )**2 / ( (np.exp(self.eps_k) - 1)**2 * (np.exp(self.eps_k) - a) **2 ) - x**2 )
    
    # def HM_var(self, x):
    #     b = self.P0_0 * (1 - 1/np.exp(self.eps_k))
    #     C = np.exp(self.eps_k) * (np.exp(self.eps_k) +1) / ((np.exp(self.eps_k) -1) * np.exp(self.eps_k) - self.P0_0)
    #     return  self.beta * ((self.t+1) * x**2 / (np.exp(self.eps_k) - 1) + (self.t + np.exp(self.eps_k)) * ((self.t+ 1)**3 + np.exp(self.eps_k) -1) / (3 * self.t**2 * (np.exp(self.eps_k) -1))) + (1-self.beta) * ( (1-self.P0_0) * C**2 + C**2 * b * np.abs(x) - x**2 )
        
    
    def HM_max_var(self):
        # print(self.beta)

        b = self.beta

        
        # print(f'x_star is {x_star}')
        # print(f'V at x* is {self.HM_var(x_star)}')

        return self.HM_var(self.x_star)

        
        # print(f'V at 0 is {self.HM_var(0)}')
        # print(f'V at 1 is {self.HM_var(1)}')

    def HM_multi(self, x):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        js = np.random.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            y[j] = self.d * self.HM_single(x[j]) / self.k

        return y.reshape(original_shape)
    

if __name__ == '__main__':
    eps = 1.3
    topm = TOPM(1, eps, np.exp(eps/3))
    topm.TO_single(0.5)
    # print(topm.HM_max_var())