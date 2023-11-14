import cmath
import numpy as np
import math
import topm

# class Noutput():
#     def __init__(self, eps, N=None):
#         if N==None:
#             self.N = find_N(eps)
#         else:
#             self.N = N

    # def general_a()

class NOUTPUT():
    def __init__(self, d, eps):
        self.d = d
        self.eps = eps
        # print(eps)
        # TODO:
        self.k = max(1, min(d, math.floor(eps / 3)))
        self.eps_k = self.eps / self.k

        # min maxVar
        self.mechs = self.generate_mechanisms()
        self.best_mech = self.min_maxVar_mech()

        self.to = topm.TOPM(d, eps)

    def generate_mechanisms(self):
        mechs = []

        N_list, max_N = find_N(self.eps_k)

        # construct type 0
        for N in N_list:
            ais = []

            if N % 2 == 0:
                n = N // 2
            else:
                n = (N-1) // 2
            P = get_P_n(self.eps_k, N)
            a_n = get_a_n(self.eps_k, N)
            ais.append(a_n)
            a_n_1 = get_a_n_1(self.eps_k, N)
            ais.append(a_n_1)

            for i in range(n-2):
                ais.append(2*ais[-1]*( 2*(math.exp(self.eps_k)-1)*P - 1 ) - ais[-2])

            ais = make_a(ais, N)

            bias = np.sum(ais**2 * P)
            max_var = bias + (ais[-2] + ais[-1])**2/4 - ais[-2]

            mechs.append({'N': N, 'ais': ais, 'P':P, 'V': max_var, 'type': 0})

        # construct type 1
        for i in range(5):
            if max_N-i <= 3:
                break
            ais = []
            P = get_P_n(self.eps_k, max_N-i)
            a_n = get_a_n(self.eps_k, max_N-i)
            ais.append(a_n)
            t = (math.exp(self.eps_k)-1)*P

            if (max_N-i) % 2 == 0:
                coef_lst = [1/(4*t-1)]
            else:
                coef_lst = [1/(4*t-2)]

            if (max_N-i) % 2 == 0:
                n = (max_N-i) // 2
            else:
                n = (max_N-i-1) // 2

            for j in range(n-2):
                C = coef_lst[j]
                coef_lst.append((1-2*t + math.sqrt(C**2 + 2*C - 4*t*C +4*t**2 - 4*t + 1)) / (C**2 + 2*C - 4*t*C))

            for C in reversed(coef_lst):
                ais.append(ais[-1] * C)
            ais = make_a(ais, max_N-i)

            bias = np.sum(ais**2 * P)
            max_var = bias + (ais[-2] + ais[-1])**2/4 - ais[-2]

            mechs.append({'N': max_N - i, 'ais': ais, 'P':P, 'V': max_var, 'type': 1})

        return mechs
    
    def min_maxVar_mech(self):
        best_mech = {'V': np.inf}
        for m in self.mechs:
            if m['V'] < best_mech['V']:
                best_mech = m
        
        return best_mech

    def get_max_var(self):
        min_v = 100
        for m in self.mech:
            min_v = min(min_v, m['V'])
        return min_v

    def NO_single(self, x):
        prob = np.full(self.best_mech['N'], self.best_mech['P'])
        T = (np.exp(self.eps_k)-1) * self.best_mech['P']

        x_idx = 0
        for i in range(1, len(prob)):
            if x <= T * self.best_mech['ais'][i]:
                break
            else:
                x_idx += 1

        if x <= 0:
            a_i = self.best_mech['ais'][x_idx+1]
            a_i_1 = self.best_mech['ais'][x_idx]

            prob[x_idx] += (x - T*a_i) / (a_i_1 - a_i)
            prob[x_idx+1] += (T*a_i_1 - x) / (a_i_1 - a_i)
        else:
            a_i = self.best_mech['ais'][x_idx]
            a_i_1 = self.best_mech['ais'][x_idx+1]

            prob[x_idx] += (T*a_i_1 - x) / (a_i_1 - a_i)
            prob[x_idx+1] += (x - T*a_i) / (a_i_1 - a_i)

        u = np.random.choice([i for i in range(len(prob))], 1, p=prob)

        return self.best_mech['ais'][u]
    
    def NO_multi(self, x):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        js = np.random.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            y[j] = self.d * self.NO_single(x[j]) / self.k

        return y.reshape(original_shape)
    
    # 4-outputs + 3-outputs
    # TODO:
    def HM_single(self, x):
        if np.random.rand() > self.beta:
            y = self.to.TO_single(x)
        else:
            y = self.PM_single(x)

        return y
    
    def HM_multi(self, x):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        js = np.random.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            y[j] = self.d * self.HM_single(x[j]) / self.k

        return y.reshape(original_shape)
            
def make_a(ais, N):
    # for i in range(n-2):
        
    rev_ais = list(reversed(ais))
    minus_ais = []
    for i in ais:
        minus_ais.append(-i)
    
    if N % 2 == 0:
        return np.array(minus_ais + rev_ais)
    else:
        return np.array(minus_ais + [0] + rev_ais)

def find_N(eps):
    N_list = []
    cnt = 0
    max_N = 4

    for N in range(4, 1000):
        # print(f' checking {N}')
        if N % 2 == 0:
            n = N / 2
        else:
            n = (N-1) / 2
        P = get_P_n(eps, N)
        a_n = get_a_n(eps, N)
        a_n_1 = get_a_n_1(eps, N)
        T = get_T(eps, N)
        a_1 = general_a(T, a_n, a_n_1, n, 1).real

        # print(f'a_1 is {a_1} > 0')

        V_n = get_var_at_n(eps, a_n, a_n_1, P)
        V_0 = get_var_at_0(eps, N, P , a_1)
        # print(f'variance at a_n is {V_n}')
        # print(f'variance at 0 is {V_0}')
        # print(f'min condition is {V_n > V_0}')

        # for type1
        if a_1 > 0:
            max_N = N

        # for type0
        if V_n > V_0 and a_1 > 0:
            N_list.append(N)
        
        elif a_1 < 0:
            cnt += 1
        
        if cnt == 4:
            break

    return N_list, max_N

def general_a(T, a_n, a_n_1, n, k):
    a_k = ((a_n * (T+cmath.sqrt(T**2 - 1)) - a_n_1) * (T-cmath.sqrt(T**2-1))**(n-k) + (a_n_1 - a_n*(T-cmath.sqrt(T**2-1))) * (T+cmath.sqrt(T**2-1))**(n-k)) / (2*cmath.sqrt(T**2-1))
    return a_k

def get_P_n(eps, N):
    if N % 2 == 0:
        n = N / 2
        return  1 / (math.exp(eps) + 2*n - 1)
    else:
        n = (N-1) / 2
        return  1 / (math.exp(eps) + 2*n)

def get_a_n(eps, N):
    P = get_P_n(eps, N)
    return 1 / (P * (math.exp(eps)- 1))

def get_a_n_1(eps, N):
    P = get_P_n(eps, N)
    a_n = get_a_n(eps, N)
    return ((2*(math.exp(eps) -1)*P - 1) * a_n) / (8*P + 1)

def get_T(eps, N):
    P = get_P_n(eps, N)
    return 2*(math.exp(eps) -1)*P - 1

def get_var_at_n(eps, a_n, a_n_1, P):
    return (a_n + a_n_1)**2 / 4 - (math.exp(eps)-1) * P * a_n_1 * a_n

def get_var_at_0(eps, N, P, a_1):
    if N % 2 == 0:
        return (math.exp(eps)-1) * P * a_1**2
    else:
        return a_1**2 / 4




def struct_mechanism(eps, N=None):
    if N==None:
        N_list = find_N(eps)
    print(N_list)
    for N in N_list:
        if N % 2 == 0:
            n = N / 2
        else:
            n = (N-1) / 2

        P = get_P_n(eps, N)
        a_n = get_a_n(eps, N)
        a_n_1 = get_a_n_1(eps, N)
        T = get_T(eps, N)
        a_1 = general_a(T, a_n, a_n_1, n, 1).real

        V_n = get_var_at_n(eps, a_n, a_n_1, P)

        print(f'N is {N}, and V_n is {V_n}')
    
# noutput = NOUTPUT(10, 12)
# print(noutput.mech)
# print(math.exp(e))
# print(lst)