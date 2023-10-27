import matplotlib.pyplot as plt
import numpy as np
import math
import cmath

s = np.arange(0.5, 20, 0.1)

N = 4
if N % 2 == 0:
    n = N // 2
else:
    n = (N-1) // 2
     
# dks = np.arange(1, 20, 1)
dks = [10]

def make_a(ais):
        # for i in range(n-2):
            
        rev_ais = list(reversed(ais))
        minus_ais = []
        for i in ais:
            minus_ais.append(-i)
        
        return np.array(minus_ais + rev_ais)

for dk in dks:
    V1 = []
    for eps in s:
        print(eps)
        ais = []
        if N % 2 == 0:
            P =  1 / (math.exp(eps) + 2*n - 1)
        else:
            P =  1 / (math.exp(eps) + 2*n)

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
            ais.append(2*ais[-1]*( 2*(math.exp(eps)-1)*P - 1 ) - ais[-2])

        T = 2*(math.exp(eps)-1)*P - 1
        # print(f'T**2 - 1 is {T**2 - 1}')
        for i in range(1, n+1):
            ak = ((a_n * (T+cmath.sqrt(T**2 - 1)) - a_n1) * (T-cmath.sqrt(T**2-1))**(n-i) + (a_n1 - a_n*(T-cmath.sqrt(T**2-1))) * (T+cmath.sqrt(T**2-1))**(n-i)) / (2*cmath.sqrt(T**2-1))
        

        ais = make_a(ais)

        bias = np.sum(ais**2 * P)

        V1.append(dk * (bias * 2  + (a_n + a_n1)**2 / 2 - a_n1) - (a_n + a_n1)**2 / 4)
        # print(f'front: {dk * (bias * 2  + (a_n + a_n1)**2 / 2 - a_n1)}')
        # print(f'front: {2.5 * 4 * (bias * 2  + (a_n + a_n1)**2 / 2 - a_n1)}')
        # print((a_n + a_n1)**2 / 4)

        label = f'{dk}'
    # print(V1)

    plt.plot(s, V1, label=label)


plt.legend()
plt.show()