import math
import numpy as np

def PM_max_var(eps, t):
    return (t + 1) / (np.exp(eps) - 1) + (t + np.exp(eps)) * ((t + 1)**3 + np.exp(eps) - 1) / (3*t**2*(np.exp(eps)-1)**2)

def Duchi_max_var(eps):
    return ((np.exp(eps) + 1) / (np.exp(eps)-1))**2

def HM_Duchi_PM(eps):
    if eps < 0.61:
        return ((np.exp(eps) + 1) / (np.exp(eps)-1))**2
    else:
        return (np.exp(eps/2) + 3) / (3 * np.exp(eps/2) * (np.exp(eps/2) - 1)) + (np.exp(eps) +  1) ** 2 / (np.exp(eps/2) * (np.exp(eps)-1)**2)
    
    
def N_max_var(eps):
    return