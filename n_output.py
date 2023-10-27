import cmath
import numpy as np
import math

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
        
        # TODO:
        self.k = max(1, min(d, math.floor(eps / 3)))
        self.eps_k = self.eps / self.k

        # min maxVar
        self.n_out_1 = find_N(self.eps_k)
        self.mech = []
        self.gen_mech()

        # find all same
        self.equal_gen()

    def gen_mech(self):
        for N in self.n_out_1:
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
                print('a_'+str(i))
                print(2*ais[-1]*( 2*(math.exp(self.eps_k)-1)*P - 1 ) - ais[-2])
                ais.append(2*ais[-1]*( 2*(math.exp(self.eps_k)-1)*P - 1 ) - ais[-2])

            ais = make_a(ais)

            bias = np.sum(ais**2 * P)
            max_var = 2 * bias + (ais[-2] + ais[-1])**2/4 - (math.exp(self.eps_k)-1) * P * ais[-1] * ais[-2]

            self.mech.append({'N': N, 'ais': ais, 'V': max_var, 'type': 0})

    def equal_gen(self):
        N = min(self.n_out_1)
        for i in range(1, 5):
            if N-i <= 3:
                break
            ais = []
            P = get_P_n(self.eps_k, N-i)
            a_n = get_a_n(self.eps_k, N-i)
            ais.append(a_n)
            t = (math.exp(self.eps_k)-1)*P

            coef_lst = [1/(4*t-1)]

            if N-i % 2 == 0:
                n = N-i // 2
            else:
                n = (N-i-1) // 2

            for j in range(n-2):
                C = coef_lst[j]
                coef_lst.append((1-2*t + math.sqrt(C**2 + 2*C - 4*t*C +4*t**2 - 4*t + 1)) / (C**2 + 2*C - 4*t*C))

            for C in reversed(coef_lst):
                ais.append(ais[-1] * C)
            ais = make_a(ais)

            bias = np.sum(ais**2 * P)
            max_var = 2 * bias + (ais[-2] + ais[-1])**2/4 - (math.exp(self.eps_k)-1) * P * ais[-1] * ais[-2]

            self.mech.append({'N': N - i, 'ais': ais, 'V': max_var, 'type': 1})

    def get_max_var(self):
        return
            
def make_a(ais):
    # for i in range(n-2):
        
    rev_ais = list(reversed(ais))
    minus_ais = []
    for i in ais:
        minus_ais.append(-i)
    
    return np.array(minus_ais + rev_ais)

def find_N(eps):
    N_list = []
    cnt = 0

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

        print(f'a_1 is {a_1} > 0')

        V_n = get_var_at_n(eps, a_n, a_n_1, P)
        V_0 = get_var_at_0(eps, N, P , a_1)
        print(f'variance at a_n is {V_n}')
        print(f'variance at 0 is {V_0}')
        print(f'min condition is {V_n > V_0}')

        if V_n > V_0 and a_1 > 0:
            N_list.append(N)
        
        elif a_1 < 0:
            cnt += 1
        
        if cnt == 4:
            break

    return N_list

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
    
noutput = NOUTPUT(10, 12)
print(noutput.mech)
# print(math.exp(e))
# print(lst)