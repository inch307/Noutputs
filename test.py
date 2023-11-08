import random
import math
import cmath
import numpy as np
import topm
import matplotlib.pyplot as plt


def PM_sub_max_var(eps):
    t = np.exp(eps/3)
    return (t + 1) / (np.exp(eps) - 1) + (t + np.exp(eps)) * ((t + 1)**3 + np.exp(eps) - 1) / (3*t**2*(np.exp(eps)-1)**2)


e = 3
to = topm.TOPM(1, e)
# to.HM_max_var()
# to.beta = 1
# to.beta = 0
print(to.HM_max_var())
# print(to.TO_max_var())
# print(to.HM_var(1))
# print(PM_sub_max_var(e))

xx = np.arange(-1, 1, 0.01)

V = []
for x in xx:
    V.append(to.TO_var(x))

plt.plot(xx, V)
plt.show()