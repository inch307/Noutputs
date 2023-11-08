import numpy as np
import matplotlib.pyplot as plt
import n_output

e = 2.7
no = n_output.NOUTPUT(1, e)

def NO_PM_var(eps, beta = None, x_star = None):
    a_n = (np.exp(eps) + 3) / (np.exp(eps) - 1)
    a_n_1 = (np.exp(eps) - 5) * (np.exp(eps) + 3) / ((np.exp(eps) + 11) * (np.exp(eps) - 1) )
    P = 1 / (np.exp(eps) + 3)

    N = (a_n + a_n_1) / 2

    bias = 2 * P * (a_n**2 + a_n_1**2)

    t = np.exp(eps / 3)
    A =  (t+1) / (np.exp(eps) -1)
    B = (t + np.exp(eps)) * ((t + 1)**3 + np.exp(eps) - 1) / 3 / t**2 / (np.exp(eps)- 1)**2

    T_1 = ((a_n + a_n_1)**2 + 4*(1+A)*(bias - a_n_1 - B) ) / 4 / (1 + A)**2
    T_2 = A**2 * (a_n + a_n_1)**2 / 4 / (1 + A)**2
    
    if beta is None:
        beta = (A + np.sqrt(T_1 / T_2)) / (1 + A)
    print(f'beta is {(A + np.sqrt(T_1 / T_2)) / (1 + A)}')

    if x_star is None:
        x_star = beta * (a_n + a_n_1) / 2 / (beta + A * beta - A)
    print(f'x_star is {beta * (a_n + a_n_1) / 2 / (beta + A * beta - A)}')

    return (1-beta) * (A* x_star**2 + B) + beta * (bias + (a_n + a_n_1)*x_star - a_n_1 - x_star**2)

beta = np.arange(0, 1, 0.001)
x = np.arange(0, 1, 0.001)

fo




