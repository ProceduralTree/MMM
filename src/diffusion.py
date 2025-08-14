import numpy as np

# 1D
# Since the Aim of multiscale Finite Volume, is to improve the results for highly fluctuating diffusivities, we test with the following oscillating function
# \begin{align*}
# D(x) &= \frac{1}{2+ 1.9 \cos \left( \frac{2 \pi x}{\epsilon} \right)}
# \end{align*}

def oscillation(x, eps = 0.1):
    return 1 / (2+1.9 * np.cos(2 * np.pi* x / eps))

# 2D Box Condition
# To test numerical stability of our methods we introduce a box constrain condition, that traps some concentration in the center.

alpha = 1.
gamma = 0.002

exp_kernel = lambda r: alpha * np.exp( - r / gamma)

def R(x,y , p=2):
    center = np.array([0.5,0.5])
    r = 0.2
    thicc = 0.03
    return np.maximum(0. , np.abs((np.abs(x -center[0])**p + np.abs(y - center[1])**p)**(1/p) - r) - thicc)

def box(x,y , p=2):
    return np.maximum(0.0005 , 1. -  exp_kernel(R(x,y , p=100)))
def circle(x,y , p=2):
    return np.maximum(0.0005 , 1. -  exp_kernel(R(x,y , p=2)))
def rhombus(x,y , p=2):
    return np.maximum(0.0005 , 1. -  exp_kernel(R(x,y , p=1)))

# 2D Oscillation

def osc2D_point(x,y , eps = 0.25):
    return oscillation(x, eps=eps) * oscillation(y, eps=eps)
def osc2D_line(x,y , eps = 0.25):
    return oscillation(x, eps=eps) + oscillation(y, eps=eps)

# 1D Noise

def noise1D(x,scale=10.  , frequencies=5):
    s = lambda x ,f , a , o: a* np.sin(f*2*np.pi*(x + o))
    coeffs = np.random.rand(frequencies,3)
    res = np.zeros(len(x))
    for i in range(frequencies):
        res += s(x, scale *coeffs[i,0] ,coeffs[i,1] , coeffs[i,2] )
    res = res / (2*np.sum(coeffs[:,1])) + 0.5
    return res
