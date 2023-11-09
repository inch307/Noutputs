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
print(f'2n+1 = {2*n+1}')
print(f'a_i-1 condition:  {6*n+5 + 4 *math.sqrt(2*n**2 + 4*n + 1)}')
print(f'(4*n**2+8)/(12*n+11) = {(4*n**2+8)/(12*n+11)}')
print(f'6*n+5 - 4*math.sqrt(2*n**2+4*n+1) = {6*n+5 - 4*math.sqrt(2*n**2+4*n+1)}')

# append a_n
a_n = 1 / (P * (math.exp(eps) - 1))
ais.append(a_n)
# print(a_n)
# append min a_n-1
# A = a_n
# B = math.exp(eps)
# --- 4
# ais.append(((16*A*B**2-32*A*B+16*A)*P**2+(12*A-12*A*B)*P+A)/((64*B**3-192*B**2+192*B-64)*P**3+((-80*B**2)+160*B-80)*P**2+(24*B-24)*P-1))
# ais.append(((4*A*B-4*A)*P-A)/((64*B**3-192*B**2+192*B-64)*P**3+((-80*B**2)+160*B-80)*P**2+(24*B-24)*P-1))
# ais.append(A/((64*B**3-192*B**2+192*B-64)*P**3+((-80*B**2)+160*B-80)*P**2+(24*B-24)*P-1))
# --- 4

# --- 5
# ais.append(((64*A*B**3-192*A*B**2+192*A*B-64*A)*P**3+((-80*A*B**2)+160*A*B-80*A)*P**2+(24*A*B-24*A)*P-A)/((256*B**4-1024*B**3+1536*B**2-1024*B+256)*P**4+((-448*B**3)+1344*B**2-1344*B+448)*P**3+(240*B**2-480*B+240)*P**2+(40-40*B)*P+1))
# ais.append(((16*A*B**2-32*A*B+16*A)*P**2+(12*A-12*A*B)*P+A)/((256*B**4-1024*B**3+1536*B**2-1024*B+256)*P**4+((-448*B**3)+1344*B**2-1344*B+448)*P**3+(240*B**2-480*B+240)*P**2+(40-40*B)*P+1))
# ais.append(A/((64*B**3-192*B**2+192*B-64)*P**3+((-96*B**2)+192*B-96)*P**2+(36*B-36)*P-1))
# ais.append(A/((256*B**4-1024*B**3+1536*B**2-1024*B+256)*P**4+((-448*B**3)+1344*B**2-1344*B+448)*P**3+(240*B**2-480*B+240)*P**2+(40-40*B)*P+1))
# --- 5

# n
t = (math.exp(eps)-1)*P
coef_lst = [1/(4*t-1)]

for i in range(n-2):
    C = coef_lst[i]
    print(f'C at {i}: {C**2 + 2*C - 4*t*C +4*t**2 - 4*t + 1}')
    print(f'root plus at {i}:  {(1-2*t + math.sqrt(C**2 + 2*C - 4*t*C +4*t**2 - 4*t + 1)) / (C**2 + 2*C - 4*t*C)}')
    print(f'root minus at {i}: {(1-2*t - math.sqrt(C**2 + 2*C - 4*t*C +4*t**2 - 4*t + 1)) / (C**2 + 2*C - 4*t*C)}')
    coef_lst.append((1-2*t + math.sqrt(C**2 + 2*C - 4*t*C +4*t**2 - 4*t + 1)) / (C**2 + 2*C - 4*t*C))

# xx = np.arange(0, 1, 0.0001)
# x = []
# y = []
# for i in xx:
#     x.append(i)
#     y.append((1-2*t + math.sqrt(i**2 + 2*i - 4*t*i +4*t**2 - 4*t + 1)) / (i**2 + 2*i - 4*t*i))
# # y = (1-2*t + math.sqrt(xx**2 + 2*xx - 4*t*xx +4*t**2 - 4*t + 1)) / (xx**2 + 2*xx - 4*t*xx)
# plt.plot(x, y)
# plt.show()

# xx = np.arange(0.1, 1, 0.0001)
# x = []
# y = []
# for i in xx:
#     x.append(i)
#     y.append((1-2*t - math.sqrt(i**2 + 2*i - 4*t*i +4*t**2 - 4*t + 1)) / (i**2 + 2*i - 4*t*i))
# # y = (1-2*t + math.sqrt(xx**2 + 2*xx - 4*t*xx +4*t**2 - 4*t + 1)) / (xx**2 + 2*xx - 4*t*xx)
# plt.plot(x, y)
# plt.show()
for C in reversed(coef_lst):
    ais.append(ais[-1] * C)
ais[1] += 0.28


# print(ais[-1])
# print(ais[-2])

 
# print(ais[-1] * (2*(math.exp(eps)-1)*P - 1 ) - math.sqrt( ais[-1]**2 * (2*(math.exp(eps)-1)*P - 1 )**2 + (ais[-1] + ais[-2])**2 - ais[-1]**2 - 4*(math.exp(eps)-1)*P*ais[-1]*ais[-2] )      ) 
# print(2*ais[-1]*( 2*(math.exp(eps)-1)*P - 1 ) - ais[-2])


# exit()

# ais.append(ais[-1] * (2*(math.exp(eps)-1)*P - 1 ) - math.sqrt( ais[-1]**2 * (2*(math.exp(eps)-1)*P - 1 )**2 + (ais[-1] + ais[-2])**2 - ais[-1]**2 - 4*(math.exp(eps)-1)*P*ais[-1]*ais[-2] ))

# for i in range(n-2):
#     print('a_'+str(i))
#     print(2*ais[-1]*( 2*(math.exp(eps)-1)*P - 1 ) - ais[-2])
#     ais.append(2*ais[-1]*( 2*(math.exp(eps)-1)*P - 1 ) - ais[-2])

# T = 2*(math.exp(eps)-1)*P - 1
# print(f'T**2 - 1 is {T**2 - 1}')
# for i in range(1, n+1):
#     print('a_'+str(i))
#     ak = ((a_n * (T+cmath.sqrt(T**2 - 1)) - a_n1) * (T-cmath.sqrt(T**2-1))**(n-i) + (a_n1 - a_n*(T-cmath.sqrt(T**2-1))) * (T+cmath.sqrt(T**2-1))**(n-i)) / (2*cmath.sqrt(T**2-1))
#     print(ak)

def make_a(ais):
    # for i in range(n-2):
        
    rev_ais = list(reversed(ais))
    minus_ais = []
    for i in ais:
        minus_ais.append(-i)
    
    return np.array(minus_ais + [0] + rev_ais)

ais = make_a(ais)
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
mmv = bias + (ais[-3] + ais[-2])**2/4 - (math.exp(eps)-1) * P * ais[-2] * ais[-3]
print(f'second max var: {mmv}')
# zv = bias + (math.exp(eps)-1)*P*ais[-2]**2
# print(f'z var: {zv}')

plt.plot(x, var)
plt.show()


