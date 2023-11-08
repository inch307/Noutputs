import topm
import numpy as np


e = 2.5
to = topm.TOPM(1, e)

sum = 0
sum_2 = 0
N = 300000
# for i in range(N):
#     noisy_x = to.HM_single(to.x_star)
#     sum += noisy_x
#     sum_2 += noisy_x**2

# print(f'HM Var is: {sum_2 / N - (sum/N)**2}')

# sum = 0
# sum_2 = 0
# N = 300000
# for i in range(N):
#     noisy_x = to.TO_single(to.C/2)
#     sum += noisy_x
#     sum_2 += noisy_x**2

# print(f'TO Var is: {sum_2 / N - (sum/N)**2}')

sum = 0
sum_2 = 0
N = 300000
for i in range(N):
    noisy_x = to.PM_single(1)
    sum += noisy_x
    sum_2 += noisy_x**2

print(f'PM Var is: {sum_2 / N - (sum/N)**2}')