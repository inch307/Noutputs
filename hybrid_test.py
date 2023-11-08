import topm
import n_output
import numpy as np
import matplotlib.pyplot as plt

eps = np.arange(0.2, 3.5, 0.01)

def get_dv(to, eps, beta = None):
    C = to.C
    a_n = (np.exp(eps) + 3) / (np.exp(eps) - 1)
    a_n_1 = (np.exp(eps) - 5) * (np.exp(eps) + 3) / ((np.exp(eps) + 11) * (np.exp(eps) - 1) )
    P_0_0 = np.exp(eps) / (np.exp(eps) + 2)
    P = 1 / (np.exp(eps) + 3)

    N = (a_n + a_n_1) / 2
    T = C / 2

    bias = 2 * P * (a_n**2 + a_n_1**2)

    beta = (-2*T*(N-T) + 1 - P_0_0 + a_n_1 - bias) / (2 * (N-T)**2)

    # print(N)
    # print(T)

    # return -1 + P_0_0 - 2*T*beta*(N-T) - 2*T*(T- T*beta + beta*N) + 2 *T*(N-T) + bias + 2 * N * beta*(N-T) + 2*N*(T-T*beta + beta*N) - a_n_1 + 2*T**2 - 2*N*T -2*beta*(N-T)**2

    return 2 * beta * (N - T)**2 + 2* T * (N - T) + P_0_0 - 1 - a_n_1 + 2 * P * (a_n**2 + a_n_1**2)

def var(to, eps, beta=None, x=None):
    C = to.C
    P = 1 / (np.exp(eps) + 3)
    a_n = (np.exp(eps) + 3) / (np.exp(eps) - 1)
    # min ver
    if eps <= 2.89:
        a_n_1 = (np.exp(eps) - 5) * (np.exp(eps) + 3) / ((np.exp(eps) + 11) * (np.exp(eps) - 1) )
    # equal ver
    else:
        coef_t = (np.exp(eps)-1)*P
        a_n_1 = a_n /(4*coef_t-1) 
    P_0_0 = np.exp(eps) / (np.exp(eps) + 2)
    

    N = (a_n + a_n_1) / 2
    T = C / 2

    bias = 2 * P * (a_n**2 + a_n_1**2)

    if beta == None:
        beta = (-2*T*(N-T) + 1 - P_0_0 + a_n_1 - bias) / (2 * (N-T)**2)
        # print(f'beta is {(-2*T*(N-T) + 1 - P_0_0 + a_n_1 - bias) / (2 * (N-T)**2)}')

    if x == None:
        x = (1 - beta) / 2 * C  + beta / 2 * (a_n + a_n_1)

    print(f'x is : {(1 - beta) / 2 * C  + beta / 2 * (a_n + a_n_1)}')
    # print(f'beta is { (-2*T*(N-T) + 1 - P_0_0 + a_n_1 - bias) / (2 * (N-T)**2) }')

    return (1 - beta) * ( (1 - P_0_0)*C**2 + C * x - x**2 ) + beta * ( (a_n + a_n_1) * x - a_n_1 + 2 * P * (a_n**2 + a_n_1**2) - x**2)

def duchi_N_var(eps, beta):
    a_n = (np.exp(eps) + 3) / (np.exp(eps) - 1)
    a_n_1 = (np.exp(eps) - 5) * (np.exp(eps) + 3) / ((np.exp(eps) + 11) * (np.exp(eps) - 1) )
    P = 1 / (np.exp(eps) + 3)

    x = beta * (a_n + a_n_1) / 2

    bias = 2 * P * (a_n**2 + a_n_1**2)

    print(f'beta is : {2 * (a_n_1 - bias + ((np.exp(eps) + 1) / (np.exp(eps) -1))**2) / (a_n + a_n_1)**2}')

    return (1 - beta) *  (((np.exp(eps) + 1) / (np.exp(eps) - 1))**2 - x**2) + beta * ( (a_n + a_n_1) * x - a_n_1 + 2 * P * (a_n**2 + a_n_1**2) - x**2)


V1 = []
V2 = []
V3 = []
V4 = []
# for e in eps:
#     to = topm.TOPM(1, e)
#     V1.append(get_dv(to, e, 0))
#     V2.append(get_dv(to, e, 1))

e = 8
to = topm.TOPM(1, e)
beta = np.arange(0, 1, 0.01)
xs = np.arange(0, 1, 0.0001)
# for b in beta:
#     V1.append(var(to, e, b))
    # V1.append(get_dv(to, e, b))
    # V1.append(duchi_N_var(e, b))

for x in xs:
    V1.append(var(to, e, x = x))
    
# print(var(to, e, beta=0))
# print(var(to, e, x = 0))
# print(var(to, e, x = 1))

# print(min(V1))

plt.plot(xs, V1, label='x')
# # plt.plot(eps, V2, label='beta = 1')
# # plt.plot(eps, V3, label='N_outputs')
# # plt.plot(eps, V4, label='HM_TP')

plt.legend()
plt.show()