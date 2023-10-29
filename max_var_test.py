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
# plt.plot(eps, Duchi_max_var(eps), label='duchi')
# plt.plot(eps, PM_max_var(eps), label='PM')

plt.plot(eps, PM_sub_max_var(eps), label='PM_sub')
print(PM_sub_max_var(2))

# V1 = []
V2 = []
V3 = []
for e in eps:
    # V1.append(HM_Duchi_PM(e))
    to = topm.TOPM(1, e, np.exp(e/3))
    V2.append(to.TO_max_var())
    no = n_output.NOUTPUT(1, e)
    V3.append(no.get_max_var())
    

# plt.plot(eps, V1, label='Duchi-Pm')
plt.plot(eps, V2, label='TO')
plt.plot(eps, V3, label='N_outputs')

plt.legend()
plt.show()