import math
import numpy as np
import matplotlib.pyplot as plt
import topm
import n_output

def PM_max_var(eps):
    t = np.exp(eps/2)
    return (t + 1) / (np.exp(eps) - 1) + (t + np.exp(eps)) * ((t + 1)**3 + np.exp(eps) - 1) / (3*t**2*(np.exp(eps)-1)**2)

def PM_sub_max_var(eps):
    t = np.exp(eps/3)
    return (t + 1) / (np.exp(eps) - 1) + (t + np.exp(eps)) * ((t + 1)**3 + np.exp(eps) - 1) / (3*t**2*(np.exp(eps)-1)**2)

# def PM_max_var(eps):
#     return (t + 1) / (np.exp(eps) - 1) + (t + np.exp(eps)) * ((t + 1)**3 + np.exp(eps) - 1) / (3*t**2*(np.exp(eps)-1)**2)


def Duchi_max_var(eps):
    return ((np.exp(eps) + 1) / (np.exp(eps)-1))**2

def HM_Duchi_PM(eps):
    if eps < 0.61:
        return ((np.exp(eps) + 1) / (np.exp(eps)-1))**2
    else:
        return (np.exp(eps/2) + 3) / (3 * np.exp(eps/2) * (np.exp(eps/2) - 1)) + (np.exp(eps) +  1) ** 2 / (np.exp(eps/2) * (np.exp(eps)-1)**2)
    
def TO_NO_var(to, eps, beta=None, x=None):
    C = to.C
    P = 1 / (np.exp(eps) + 3)
    a_n = (np.exp(eps) + 3) / (np.exp(eps) - 1)
    # min ver
    if eps <= 2.89:
        a_n_1 = (np.exp(eps) - 5) * (np.exp(eps) + 3) / ((np.exp(eps) + 11) * (np.exp(eps) - 1) )
    # equal ver
    else:
        coef_t = (math.exp(eps)-1)*P
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

    # print(f'beta is { (-2*T*(N-T) + 1 - P_0_0 + a_n_1 - bias) / (2 * (N-T)**2) }')

    return (1 - beta) * ( (1 - P_0_0)*C**2 + C * x - x**2 ) + beta * ( (a_n + a_n_1) * x - a_n_1 + 2 * P * (a_n**2 + a_n_1**2) - x**2)


def NO_PM_var(eps, beta = None, x_star = None):
    P = 1 / (np.exp(eps) + 3)
    a_n = (np.exp(eps) + 3) / (np.exp(eps) - 1)
    # min ver
    if eps <= 2.89:
        a_n_1 = (np.exp(eps) - 5) * (np.exp(eps) + 3) / ((np.exp(eps) + 11) * (np.exp(eps) - 1) )
    # equal ver
    else:
        coef_t = (math.exp(eps)-1)*P
        a_n_1 = a_n /(4*coef_t-1) 
    # print(a_n_1)
    

    N = (a_n + a_n_1) / 2

    bias = 2 * P * (a_n**2 + a_n_1**2)

    t = np.exp(eps / 3)
    A =  (t+1) / (np.exp(eps) -1)
    B = (t + np.exp(eps)) * ((t + 1)**3 + np.exp(eps) - 1) / 3 / t**2 / (np.exp(eps)- 1)**2

    T_1 = ((a_n + a_n_1)**2 + 4*(1+A)*(bias - a_n_1 - B) ) / 4 / (1 + A)**2
    T_2 = A**2 * (a_n + a_n_1)**2 / 4 / (1 + A)**2
    
    if beta is None:
        beta = (A + np.sqrt(T_2 / T_1)) / (1 + A)
    # print(f'beta is {(A + np.sqrt(T_2 / T_1)) / (1 + A)}')

    if x_star is None:
        x_star = beta * (a_n + a_n_1) / 2 / (beta + A * beta - A)
    # print(f'x_star is {beta * (a_n + a_n_1) / 2 / (beta + A * beta - A)}')

    return (1-beta) * (A* x_star**2 + B) + beta * (bias + (a_n + a_n_1)*x_star - a_n_1 - x_star**2)
    
def N_max_var(eps):
    return

eps = np.arange(0.2, 8, 0.01)



# DUCHI
# PM
# DuCHI, PM
# PM sub
# TO
# TO PM

# Noutpus type 0, 1


# for e in eps:
plt.plot(eps, Duchi_max_var(eps), label='duchi')
# plt.plot(eps, PM_max_var(eps), label='PM')

plt.plot(eps, PM_sub_max_var(eps), label='PM_sub')
# print(PM_sub_max_var(2))

V1 = []
V2 = []
V3 = []
V4 = []
V5 = []
V6 = []
for e in eps:
    V1.append(HM_Duchi_PM(e))
    to = topm.TOPM(1, e)
    V2.append(to.TO_max_var())
    no = n_output.NOUTPUT(1, e)
    V3.append(no.get_max_var())
    V4.append(to.HM_max_var())
    V5.append(NO_PM_var(e))
    V6.append(TO_NO_var(to, e))


plt.plot(eps, V1, label='Duchi-Pm')
plt.plot(eps, V2, label='TO')
plt.plot(eps, V3, label='N_outputs')
plt.plot(eps, V4, label='HM_TP')
plt.plot(eps, V5, label='HM_NO_PM')
plt.plot(eps, V6, label='TO_NO_PM')


plt.legend()
plt.show()