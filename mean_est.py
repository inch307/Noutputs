import numpy as np
import utils

import duchi
import pm
import topm
import n_output

def create_normal_data(data_shape, mean, std):
    data = np.random.normal(mean, std, data_shape)
    data = np.clip(data,-1, 1)

    return data

std = 1/4
mean = 0
d = 16
num_sam = 10000
data_shape = (num_sam, d)

eps = 3

# for m in mean:
data = create_normal_data(data_shape, mean, std)

duchi_mech = duchi.Duchi(d, eps)
pm_mech = pm.PM(d, eps)
to_pm_mech = topm.TOPM(d, eps)
no_mech = n_output.NOUTPUT(d, eps)

duchi_mean_single = np.array([0. for i in range(d)])
TO_mean = np.array([0. for i in range(d)])
NO_mean = np.array([0. for i in range(d)])
for sample in data:
    noisy_sample = duchi_mech.Duchi_single_unif(sample)
    duchi_mean_single += noisy_sample
    noisy_sample = to_pm_mech.TO_multi(sample)
    TO_mean += noisy_sample
    noisy_sample = no_mech.NO_multi(sample)
    NO_mean += noisy_sample


duchi_mean_single = duchi_mean_single / num_sam
TO_mean /= num_sam
NO_mean /= num_sam

print(duchi_mean_single)
print(TO_mean)
print(NO_mean)

print(((duchi_mean_single - mean)**2).mean())
print(((TO_mean-mean)**2).mean())
print(((NO_mean-mean)**2).mean())

