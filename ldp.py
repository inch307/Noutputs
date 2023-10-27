import random
import math
import numpy as np
from matplotlib import pyplot as plt

eps = 8


x = np.linspace(-1, 1, 1000)
ais = [0.32]
n = len(ais) + 1
P =  1 / (math.exp(eps) + 2*n - 1)

def make_a(ais):
    print(1 / (P * (math.exp(eps) - 1)))
    ais.append( 1 / (P * (math.exp(eps) - 1)) )
    rev_ais = []
    for i in reversed(ais):
        rev_ais.append(-i)
    
    return np.array(rev_ais + ais)

ais = make_a(ais)
print(ais)

# print(ais[-2] / (ais[-1] - ais[-2]))
# print(1 / (2 * (math.exp(eps)-1) * P))

print(  (2*(math.exp(eps)-1)*P - 1) * ais[-3])
print((ais[-1] + ais[-2]) / 2)
print((math.exp(eps-1) * P * (0.5 * ais[-1] + 0.5 * ais[-2]) ))

def split_point(x, ais):
    transform_x = x / ((math.exp(eps) -1)*P)
    split_x = []
    idx_ai = 1
    b = transform_x
    num_idx = 0
    for idx, elem in enumerate(transform_x):
        if elem < ais[idx_ai]:
            num_idx +=1
        else:
            if idx == len(transform_x)-1:
                num_idx += 1
            a, b = np.split(b, [num_idx])
            # print(b)
            split_x.append(a)
            idx_ai += 1
            num_idx = 1
    return split_x

# print(split_point(x, ais))

split_x = split_point(x, ais)
# get Variance
var = []
bias = np.sum(ais**2 * P)
# print(var)

idx = 0
for a_idx, l in enumerate(split_x):    
    for i in l:
        v = bias - x[idx]**2
        # print(i)
        # print(x[idx])
        q = (x[idx] - (math.exp(eps) - 1) * P * ais[a_idx]) / ((math.exp(eps)-1) *P * (ais[a_idx+1] - ais[a_idx]))
        # print(q)
        v += (1-q)*(math.exp(eps)-1)*P * ais[a_idx]**2 + q*(math.exp(eps)-1)*P * ais[a_idx+1]**2
        var.append(v)
        idx+=1
        # exit()

var = np.array(var)

mv = bias + (ais[-2] + ais[-1])**2/4 - (math.exp(eps)-1) * P * ais[-1] * ais[-2]
print(f'max var: {mv}')
dv = (4*P + 0.5)*ais[-2] - ((math.exp(eps)-1) * P - 0.5)*ais[-1]
print(f'd var: {dv}')
mmv = bias + (ais[-3] + ais[-2])**2/4 - (math.exp(eps)-1) * P * ais[-2] * ais[-3]
print(f'second max var: {mmv}')
# zv = bias + (math.exp(eps)-1)*P*ais[-2]**2
# print(f'z var: {zv}')

plt.plot(x, var)
plt.show()


