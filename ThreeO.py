import math
import numpy as np
import PM

def TO_single(x, eps):
    eps_ = np.log((3 + np.sqrt(65)) / 2)
    if eps < np.log(2):
        P0_0 = 0
    elif eps <= eps_:
        delta_0 = np.exp(4*eps) + 14*np.exp(3*eps) + 50*np.exp(2*eps) - 2 * np.exp(eps) + 25
        delta_1 = -2*np.exp(6*eps) -42*np.exp(5*eps) -270*np.exp(4*eps) -404*np.exp(3*eps) -918*np.exp(2*eps) +30*np.exp(eps) -250
        
        P0_0 = (-1/6) * ( -2*np.exp(2*eps) - 4*np.exp(eps) - 5 + 2 * np.sqrt(delta_0) * np.cos(np.pi/3 + (1/3) * np.arccos(-delta_1 / (2*np.sqrt(delta_0**3))) ))
    else:
        P0_0 = np.exp(eps) / (np.exp(eps) + 2)

    C = (np.exp(eps) + 1) / ((np.exp(eps)-1) * (1 - P0_0 / np.exp(eps)))

    if x >= 0 and x <= 1:
        P_mC_x = (1 - P0_0) / 2 + ( (1 - P0_0) / 2 - (np.exp(eps) - P0_0) / (np.exp(eps) * (np.exp(eps) + 1)) ) * x
    else:
        P_mC_x = (1 - P0_0) / 2 + ( (np.exp(eps) - P0_0) / (np.exp(eps) + 1) - (1 - P0_0) / 2 ) * x

    if x >= 0 and x <= 1:
        P_C_x = (1 - P0_0) / 2 + ( (np.exp(eps) - P0_0) / (np.exp(eps) + 1) - (1 - P0_0) / 2 ) * x
    else:
        P_C_x = (1 - P0_0) / 2 + ( (1 - P0_0) / 2 - (np.exp(eps) - P0_0) / (np.exp(eps) * (np.exp(eps) + 1)) ) * x
    
    P_0_x = P0_0 + (P0_0 / np.exp(eps) - P0_0) * x

    u = np.random.choice([-1, 0, 1], 1, p = [P_mC_x, P_0_x, P_C_x])

    return u * C

def TO_multi(x, eps):
    d = x.size
    original_shape = x.shape
    x = x.reshape(d)
    y = np.zeros_like(x)
    k = max(1, min(d, math.floor(eps / 2.5)))

    js = np.random.choice([i for i in range(d)], k, False, p=[1/d for i in range(d)])
    
    for j in js:
        y[j] = d * TO_single(x[j], eps / k) / k

    return y.reshape(original_shape)

# for here PM_SUB
def TO_PM_HM_single(x, eps):
    # :TODO
    P0_0 = 0
    a = P0_0
    t=0

    # beta is prob of PM
    if eps > 0 and eps < 0.610986:
        beta = 0
    elif eps < np.log(2):
        beta = (2*(np.exp(eps) - a)**2 * (np.exp(eps) -1) - a * np.exp(eps) * (np.exp(eps) + 1)**2) / (2 * (np.exp(eps)-a)**2 * (np.exp(eps) + t) - a * np.exp(eps) * (np.exp(eps) + 1)**2)
    else:
        c = np.exp(eps)
        A = a**2 * c**2 * (c+1)**4 / (4 * (c + t)**2 * (c - a)**4 * (c-1)) - a**2 * c**2 * (c+1)**4 / (2 * (c+t)**2 * (c-1) * (c-a)**4) + ( (t+1)**3 + c - 1) / (3 * t**2 * (c - 1)**2) - (1 - a) * c**2 * (c+1)**2 / ((c+t)*(c-1)**2 * (c-a)**2)
        B = (1 + t)**2 * a**2 * c**2 * (c+1)**4 / (4* (c+t)**2 * (c-a)**4 * (c-1))

        beta = (-np.sqrt(B/A) + np.exp(eps) -1) / (np.exp(eps) + np.exp(eps/3))

    if np.random.rand() > beta:
        y = TO_single(x,eps)
    else:
        y = PM.PM_sub(x, eps)

    return y

def TO_PM_HM_multi(x, eps):
    d = x.size
    original_shape = x.shape
    x = x.reshape(d)
    y = np.zeros_like(x)
    k = max(1, min(d, math.floor(eps / 2.5)))

    js = np.random.choice([i for i in range(d)], k, False, p=[1/d for i in range(d)])
    
    for j in js:
        y[j] = d * TO_PM_HM_single(x[j], eps / k) / k

    return y.reshape(original_shape)