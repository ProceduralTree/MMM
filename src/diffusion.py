

# #+RESULTS:
# : None


import numpy as np

# Code

def oscillation(x, eps = 0.1):
    return 1 / (2+1.9 * np.cos(2 * np.pi* x / eps))

# Code

def osc2D_point(x,y , eps = 0.25):
    return oscillation(x, eps=eps) * oscillation(y, eps=eps)
def osc2D_line(x,y , eps = 0.25):
    return np.maximum(oscillation(x, eps=eps) , oscillation(y, eps=eps))

# 2D Box Condition
# To test numerical stability of our methods we introduce a box constrain condition, that traps some concentration in the center.


alpha = 0.99
gamma = 0.002
depth = 1e-3
a = 4
b = 200

exp_kernel_smooth = lambda r: 1. - 0.99 * np.exp(-(1.1**b) * a*r**a)
exp_kernel = lambda r: alpha * np.exp( - r / gamma)

def R(x,y , p=2):
    center = np.array([0.5,0.5])
    r = 0.2
    thicc = 0.005
    return np.maximum(0. , np.abs((np.abs(x -center[0])**p + np.abs(y - center[1])**p)**(1/p) - r) - thicc)

def radius(x,y , p=2):
    center = np.array([0.5,0.5])
    return np.abs((np.abs(x -center[0])**p + np.abs(y - center[1])**p)**(1/p))

def smooth_box(x,y):
    r = 0.2
    return exp_kernel_smooth(np.abs(radius(x,y, p=100) - r))



def box(x,y , p=2):
    return np.maximum(depth , 1. -  exp_kernel(R(x,y , p=100)))
def circle(x,y , p=2):
    return np.maximum(depth , 1. -  exp_kernel(R(x,y , p=2)))
def rhombus(x,y , p=2):
    return np.maximum(depth , 1. -  exp_kernel(R(x,y , p=1)))

# Code

def noise1D(x,scale=10.  , frequencies=5):
    s = lambda x ,f , a , o: a* np.sin(f*2*np.pi*(x + o))
    rng = np.random.default_rng(69)
    coeffs = rng.random((frequencies,3))
    res = np.zeros(len(x))
    for i in range(frequencies):
        res += s(x, scale *coeffs[i,0] ,coeffs[i,1] , coeffs[i,2] )
    res = res / (2*np.sum(coeffs[:,1])) + 0.5
    return res

# Code

def noise2D(x,y , scale=8. , frequencies=20):
    s = lambda x ,f , a , o: a* np.sin(f*2*np.pi*(x + o))
    rng = np.random.default_rng(6)
    coeffs = rng.random((frequencies,6))
    res = np.zeros_like(x)
    for i in range(frequencies):
        gamma = 1.1**(i+scale)
        theta = np.pi * coeffs[i,5]
        x_prime = x * np.cos(theta) - y * np.sin(theta)
        y_prime = x * np.cos(theta) - y * np.sin(theta)
        res += 1/gamma * (s(x_prime, gamma ,coeffs[i,1] ,coeffs[i,2] ) + s(y_prime, gamma ,coeffs[i,2] , coeffs[i,4] ))

    res = res*10 + 20
    return res
    return
