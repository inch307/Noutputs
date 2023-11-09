import random
import math
import numpy as np
from matplotlib import pyplot as plt
import cmath

eps = 3.7


x = np.linspace(-1, 1, 1000)
ais = []
n = 2
P =  1 / (math.exp(eps) + 2*n)

print(f'e^eps = {math.exp(eps)}')
print(f'2n+2 = {2*n+2}')
print(f'a_i-1 condition:  {6*n+8 + 4 *math.sqrt(2*n**2 + 6*n -4)}')
print(f'(4*n**2+8)/(12*n+11) = {(4*n**2+8)/(12*n+11)}')
print(f'6*n+5 - 4*math.sqrt(2*n**2+4*n+1) = {6*n+5 - 4*math.sqrt(2*n**2+4*n+1)}')

# append a_n
a_n = 1 / (P * (math.exp(eps) - 1))
ais.append(a_n)
# print(a_n)
# append min a_n-1
a_n1 = ((2*(math.exp(eps) -1)*P - 1) * a_n) / (8*P + 1)
ais.append( ((2*(math.exp(eps) -1)*P - 1) * a_n) / (8*P + 1))
# print( ((2*(math.exp(eps) -1)*P - 1) * a_n) / (8*P + 1) )
# print( ((2*(math.exp(eps) -1)*P - 1) ) / (8*P + 1) )

# print(ais[-1])
# print(ais[-2])

 
# print(ais[-1] * (2*(math.exp(eps)-1)*P - 1 ) - math.sqrt( ais[-1]**2 * (2*(math.exp(eps)-1)*P - 1 )**2 + (ais[-1] + ais[-2])**2 - ais[-1]**2 - 4*(math.exp(eps)-1)*P*ais[-1]*ais[-2] )      ) 
# print(2*ais[-1]*( 2*(math.exp(eps)-1)*P - 1 ) - ais[-2])


# exit()

# ais.append(ais[-1] * (2*(math.exp(eps)-1)*P - 1 ) - math.sqrt( ais[-1]**2 * (2*(math.exp(eps)-1)*P - 1 )**2 + (ais[-1] + ais[-2])**2 - ais[-1]**2 - 4*(math.exp(eps)-1)*P*ais[-1]*ais[-2] ))

for i in range(n-2):
    print('a_'+str(i))
    print(2*ais[-1]*( 2*(math.exp(eps)-1)*P - 1 ) - ais[-2])
    ais.append(2*ais[-1]*( 2*(math.exp(eps)-1)*P - 1 ) - ais[-2])

T = 2*(math.exp(eps)-1)*P - 1
print(f'T**2 - 1 is {T**2 - 1}')
for i in range(1, n+1):
    print('a_'+str(i))
    ak = ((a_n * (T+cmath.sqrt(T**2 - 1)) - a_n1) * (T-cmath.sqrt(T**2-1))**(n-i) + (a_n1 - a_n*(T-cmath.sqrt(T**2-1))) * (T+cmath.sqrt(T**2-1))**(n-i)) / (2*cmath.sqrt(T**2-1))
    print(ak)

def make_a(ais):
    # for i in range(n-2):
        
    rev_ais = list(reversed(ais))
    minus_ais = []
    for i in ais:
        minus_ais.append(-i)
    
    return np.array(minus_ais + [0] + rev_ais)

ais = make_a(ais)
# ais[-7] = ais[-7]-0.01
print(ais)

# print(ais[-2] / (ais[-1] - ais[-2]))
# print(1 / (2 * (math.exp(eps)-1) * P))

# print(  (2*(math.exp(eps)-1)*P - 1) * ais[-3])
# print((ais[-1] + ais[-2]) / 2)
# print((math.exp(eps-1) * P * (0.5 * ais[-1] + 0.5 * ais[-2]) ))

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
# mmv = bias + (ais[-7] + ais[-6])**2/4 - (math.exp(eps)-1) * P * ais[-6] * ais[-7]
# print(f'second max var: {mmv}')
# ddv = (4*P + 0.5)*ais[-7] - ((math.exp(eps)-1) * P - 0.5)*ais[-6]
# print(f'second d var: {ddv}')
# zv = bias + (math.exp(eps)-1)*P*ais[-2]**2
# print(f'z var: {zv}')

plt.plot(x, var)
plt.show()


