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

eps = 4

# for m in mean:
data = create_normal_data(data_shape, mean, std)

duchi_mech = duchi.Duchi(d, eps)
pm_mech = pm.PM(d, eps)
to_pm_mech = topm.TOPM(d, eps)
no_mech = n_output.NOUTPUT(d, eps)

duchi_mean = np.array([0. for i in range(d)])
duchi_mean_single = np.array([0. for i in range(d)])
for sample in data:
    noisy_sample = duchi_mech.Duchi_multi(sample)
    duchi_mean += noisy_sample
    noisy_sample = duchi_mech.Duchi_single_unif(sample)
    duchi_mean_single += noisy_sample

duchi_mean = duchi_mean / num_sam
duchi_mean_single = duchi_mean_single / num_sam

print(duchi_mean)
print(duchi_mean_single)

