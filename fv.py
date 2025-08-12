# Finite Volume
# Integral Form
# \begin{align*}
# \int_{Q} \nabla \cdot (D(x) \nabla c )  &= \int_{Q} f(x) \, \mathrm{d}x \\
# \int_{\partial Q} D(x) \nabla c \cdot \vec{n} \mathrm{d}S \, &=   \int_{Q} f(x) \, \mathrm{d} x
# \end{align*}
# We assume constant c on \(Q\)
# \(q =D(x) \nabla c\) is not uniquely defined on \(\partial\Omega\) since we assuse c constant and therefore discontinous on \(\partial Q\). We therefore introduce a Numerical Flux \(q = g(c^+ , c^{-} )\)
# for example upwind

# \begin{align*}
# g(c^+ , c^-) = - D(x^{\frac{1}{2} +}) \frac{c^+ - c^-}{h}
# \end{align*}

# \begin{align*}
# g(c^+ , c^-) &= T_{\pm } * \left( c^+ - c^- \right) \\
# T_{\pm } &= - D(x^{\frac{1}{2}+}) \frac{1}{h}
# \end{align*}


# With \(D(x) = \frac{1}{2+ 1.9 \cos \left( \frac{2 \pi x}{\epsilon} \right)}\)
# Linear System
# \begin{align*}
# \int_{\partial Q_{i}} D(x_{i}) \nabla c \cdot \vec{n}  \, \mathrm{d}S &= |Q| \overline{f}(x_{i}) \\
# \sum_{j \in \left\{ -1,1 \right\} } j *  g(c_{i+j+1} , c_{i+j})  &=   h \overline{f}(x_{i})
# \end{align*}


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
epsilon = 0.1 #* np.pi
N = 10
n = 100
h = 1/(N-1)
D = lambda x: 1 / (2+1.9 * np.cos(2 * np.pi* x / epsilon))
x = np.linspace(0,1 ,N)
x_highres = np.linspace(0,1 , 100000)
c = np.zeros(N)
source = np.ones(N) * h
# Boundary
source[0] = 0
source[-1] = 0
plt.plot(x , D(x))
plt.plot(x_highres , D(x_highres))
plt.legend([r"$D$ Sampled on the course grid" , r"$D$"])
plt.title("Diffusion Coefficient")



# #+RESULTS:
# [[file:images/D.svg]]


# Matrix Assembly

from scipy.sparse import spdiags
import seaborn as sns
def assemble_matrix(N ,x, h, D ):
    diagp1 = np.zeros(N)
    diagp1[2:] = np.ones(N-2) * -1/h * D(x[1:-1] + 0.5*h)
    diagm1 = np.zeros(N)
    diagm1[:-2] = np.ones(N-2) * -1/h * D(x[1:-1] - 0.5*h)
    diag0 = np.ones(N)
    diag0[1:-1] *= 1/h * (D(x[1:-1]-0.5*h) + D(x[1:-1] + 0.5*h))
    A = spdiags([diagm1 , diag0 , diagp1] , np.array( [-1, 0, 1] ))
    return A
A = assemble_matrix(10 , np.linspace(0,1.,10), h , D_1D)
#plt.spy(A)
sns.heatmap(A.todense())
plt.title("Sparsity Patter of A")



# #+RESULTS:
# [[file:images/A-sparsity.svg]]



c = sp.sparse.linalg.spsolve(A.tocsr(),source)
plt.plot(c)
plt.title("Course Grid Solution")

# Multiscale
# In 1D

# \begin{align*}
# T_{\pm } &= -\int_{Q} D(x) (\phi'_{\pm} (x))^2\, \mathrm{d}x
# \end{align*}

# calculate c integral

np.sum((c[1:] - c[:-1])/h * -D(x[1:]))



# #+RESULTS:


from scipy.sparse.linalg import spsolve
micro_basis = np.zeros(N*n)
T = np.zeros(N)
for i,x_m in enumerate(x):
    xm = np.linspace(x_m , x_m + h , n)
    hm = h/(n-1)
    A_m = assemble_matrix(n, xm , hm )
    fm = np.ones(n) * hm
    fm[0] = 0
    fm[-1] = 1
    phi = spsolve(A_m.tocsr(),fm)
    micro_basis[n * i:n*(i+1)] = phi
    T[i] =   hm * np.sum(((phi[1:] - phi[:-1])/hm)**2 * D(xm[:-1]))
plt.plot(x,T)
plt.xlabel(r"$x$")
plt.ylabel(r"$T(x)$")
plt.title(r"Multiscale Transmission Coeficcients $T$")

diagp1 = np.zeros(N)
diagp1[2:] = np.ones(N-2) * -1* T[1:-1]
diagm1 = np.zeros(N)
diagm1[:-2] = np.ones(N-2) * -1*  T[:-2]
diag0 = np.ones(N)
diag0[1:-1] *=  (T[1:-1] + T[:-2])
A_macro = spdiags([diagm1 , diag0 , diagp1] , np.array( [-1, 0, 1] ))
sns.heatmap(A_macro.todense())



# #+RESULTS:
# [[file:A.png]]


N_fine = n*N
x_fine = np.linspace(0,1 , n*N)
h_fine = 1/(N*n -1)
A_fine = assemble_matrix(N_fine , x_fine , h_fine)
f_fine = np.ones(N*n) * h_fine
f_fine[0] = 0
f_fine[-1] = 0
c_fine = sp.sparse.linalg.spsolve(A_fine.tocsr(),f_fine)



# #+RESULTS:
# [[file:images/fine.svg]]


c_macro = sp.sparse.linalg.spsolve(A_macro.tocsr(),source)
c_multi = np.zeros((N-1)* n)
x = np.linspace(0,1,N)
x_multi = np.linspace(0,1 , n*(N-1))
for i in range(len(c_macro)-1):
    t = micro_basis[n*i:n*(i+1)]
    c_multi[n*i:n*(i+1)] = (1-t) * c_macro[i] + t * c_macro[i+1]
plt.plot(x,c)
plt.plot(x,c_macro)
plt.plot(x_multi,c_multi)
plt.plot(x_fine , c_fine)
plt.title("Comparison Of Different Solvers")
plt.xlabel(r"$x$")
plt.ylabel(r"$c(x)$")
plt.legend(["macro" , "multiscale", "multi_fine" , "reference"])



# #+name: Assemble 2D Matrix

   def assemble_matrix(self)->None:
       main_diag = np.ones((  self.N,self.M))
       diag_north = np.zeros((self.N,self.M))
       diag_south = np.zeros((self.N,self.M))
       diag_east = np.zeros(( self.N,self.M))
       diag_west = np.zeros(( self.N,self.M))
       main_diag[1:-1,1:-1] =  -1* (self._T_x[:-1,1:-1] + self._T_x[1:,1:-1] + self._T_y[1:-1,:-1] + self._T_y[1:-1,1:])
       main_diag = np.ravel(main_diag)

       diag_north[1:-1,1:-1] =  self._T_y[1:-1,:-1]
       diag_south[1:-1,1:-1] =  self._T_y[1:-1,1:]
       diag_east[1:-1,1:-1] =   self._T_x[1:,1:-1]
       diag_west[1:-1,1:-1] =   self._T_x[:-1,1:-1]
       diag_north = diag_north.ravel()
       diag_south = diag_south.ravel()
       diag_west = diag_west.ravel()
       diag_east = diag_east.ravel()

       A = sp.sparse.spdiags([main_diag , diag_north , diag_south ,  diag_west , diag_east] , [0 , -self.N  , self.N , 1 , -1] , self.N*self.M , self.M*self.N)
       self._A = A.T

# Numerical Flux in 2D
# \begin{align*}
# g_{x}(c_{i+1,j} , c_{ij}) &= - \Delta_y D(x_{i+ \frac{1}{2},j }) \frac{c_{i+1,j} - c_{ij}}{\Delta_x}\\
# g_y(c_{i,j+1} , c_{ij}) &= - \Delta_x D(x_{i,j+ \frac{1}{2}}) \frac{c_{i,j+1} - c_{ij}}{\Delta_y} \\
# g_x(c_{i+1j} , c_{ij}) &=   T^x_{i+1j} \left( c_{i+1j} - c_{ij}  \right)\\
# g_y(c_{ij+1} , c_{ij}) &=   T^y_{ij+1} \left( c_{i+1j} - c_{ij}  \right)
# \end{align*}
# The boundary term can then be approximated by
# \begin{align*}
#  - g_{x}(c_{i,j} , c_{i-1,j}) + g_{x}(c_{i+1,j} , c_{ij})  -  g_y(c_{i,j} , c_{i,j-1}) + g_y(c_{i,j+1} , c_{ij}) &= \Delta_x \Delta_y f(x_{ij})
# \end{align*}
# One Dimensionalize the index
# \begin{align*}
#  - g_{x}(c_{i + Nj} , c_{i-1 + Nj}) + g_{x}(c_{i+1 + Nj} , c_{i + Nj})  -  g_y(c_{i + Nj} , c_{i + N(j-1)}) + g_y(c_{i + N(j+1)} , c_{i + Nj}) &= \Delta_x \Delta_y f(x_{i + Nj})
# \end{align*}
# plug in Flux Approach with \(\Delta_x = \Delta_y = h\)
# \begin{align*}
# & \left(D(x-\frac{h}{2},y)c_{i+Nj}-D(x-\frac{h}{2},y)c_{i-1+Nj}\right)\\
# &-\left(D(x+\frac{h}{2},y)c_{i+1+Nj}-D(x+\frac{h}{2},y)c_{i+Nj}\right)\\
# &+\left(D(x,y-\frac{h}{2})c_{i+Nj}-D(x,y-\frac{h}{2})c_{i+N(j-1)}\right)\\
# &-\left(D(x,y+\frac{h}{2})c_{i+N(j+1)}-D(x,y+\frac{h}{2})c_{i+Nj}\right)
# \end{align*}

# \begin{align*}
# & D(x-\frac{h}{2},y)c_{i+Nj}-D(x-\frac{h}{2},y)c_{i-1+Nj}  \\
# &-D(x+\frac{h}{2},y)c_{i+1+Nj}+D(x+\frac{h}{2},y)c_{i+Nj}  \\
# & D(x,y-\frac{h}{2})c_{i+Nj}-D(x,y-\frac{h}{2})c_{i+N(j-1)}\\
# &-D(x,y+\frac{h}{2})c_{i+N(j+1)}+D(x,y+\frac{h}{2})c_{i+Nj}
# \end{align*}

# \begin{align*}
# & -D(x-\frac{h}{2},y)c_{i-1+Nj}  \\
# &-D(x+\frac{h}{2},y)c_{i+1+Nj}  \\
# & -D(x,y-\frac{h}{2})c_{i+N(j-1)}\\
# &-D(x,y+\frac{h}{2})c_{i+N(j+1)}\\
# \left(D(x-\frac{h}{2},y) + D(x+\frac{h}{2},y) + D(x,y-\frac{h}{2}) + D(x,y+\frac{h}{2}) \right) c_{i+Nj}
# \end{align*}


import os

# Set this before importing NumPy/SciPy
os.environ["OMP_NUM_THREADS"] = "16"       # For MKL/OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # For OpenBLAS
os.environ["MKL_NUM_THREADS"] = "16"       # For Intel MKL
os.environ["NUMEXPR_NUM_THREADS"] = "16"   # Just in case

import numpy as np
import scipy

epsilon =0.25
D_1D = lambda x: 1 / (2+1.9 * np.cos(2 * np.pi* x / epsilon))
D = lambda x,y: D_1D(x) * D_1D(y)

alpha = 1.
gamma = 0.001

center = np.array([0.5,0.5])
exp_kernel = lambda r: alpha * np.exp( - r / gamma)
r = 0.2
p = 100.0
thicc = 0.02
R = lambda x,y: np.maximum(0. , np.abs((np.abs(x -center[0])**p + np.abs(y - center[1])**p)**(1/p) - r) - thicc)
D = lambda x,y:   np.maximum(0.0005 , 1. -  exp_kernel(R(x,y)))
D_lin = lambda x,y: x



# #+RESULTS:
# : None


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
N = 1000
M = 1000
x = np.linspace(0.,1., N)
y= np.linspace(0.,1., M)
grid = np.meshgrid(x,y)
diffusion = D(grid[0] , grid[1])
diffusion = np.reshape(diffusion , (N,M))
sns.heatmap(diffusion)



# #+RESULTS:
# [[file:images/2D_Diffusion.png]]



reload(src.fvsolver)
from src.fvsolver import FVSolver2D
smol_fv = FVSolver2D(10,10,D)
smol_fv.assemble_matrix()
sns.heatmap(smol_fv._A.todense())
#plt.spy(A.T, markersize=1)



# #+RESULTS:
# [[file:images/spy.svg]]


fv2D = FVSolver2D(N,M,D)
fv2D.assemble_matrix()
fv2D.set_boundary()
c = fv2D.solve()
sns.heatmap(c, cmap="magma")



# #+RESULTS:
# [[file:images/2d-result.png]]


error =np.linalg.norm(A@c_vec - f)
print(error)



# #+RESULTS:
# : 1.025105313314805e-12


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid[0] ,grid[1],c , cmap="magma")

# 2D Multiscale


from scipy.sparse.linalg import cg , spsolve
import numpy as np
def solve_microscale(p0 , p1,resolution , D):
    step = np.linspace(0.,1.,resolution)
    hm = 1/resolution
    range_x = lambda x: p0[0] + x * p1[0]
    range_y = lambda x: p0[0] + x * p1[0]
    D_micro = lambda x: D(range_x(x) , range_y(x))
    A = assemble_matrix(resolution , step, hm , D_micro)
    fm = np.ones_like(step) * hm
    fm[0] = 0
    phi = spsolve(A.tocsr(),fm)
    return phi



# #+RESULTS:
# : None


def microscale_basis(N , M , resolution , h , D):
    micro_basis = np.zeros((N,M ,2, resolution))
    for i in range(N):
        for j in range(M):
            p0 = np.array([x[i] + 0.5 * h, y[j] + 0.5 * h])
            p_north = np.array([x[i+1]+ 0.5 * h, y[j]+ 0.5 * h])
            p_east = np.array([x[i+1]+ 0.5 * h, y[j]+ 0.5 * h])
            phi_north = solve_microscale(p0 , p_north ,resolution , D )
            phi_east = solve_microscale(p0 , p_east ,resolution , D )
            micro_basis[i,j,0,:] = phi_north
            micro_basis[i,j,1,:] = phi_east
    return micro_basis



# #+RESULTS:
# : None


m = microscale_basis(10 ,10 , 10 , 1/100 , D_lin)

# Cleanup
# #+name: Assemble Matrix

   def assemble_matrix(self)-> None:
      diagp1 = np.zeros(self.N)
      diagp1[2:] =  self._T[1:]
      diagm1 = np.zeros(self.N)
      diagm1[:-2] =  self._T[:-1]
      diag0 = np.ones(self.N)
      diag0[1:-1] = -1 * (self._T[1:] + self._T[:-1])
      self._A = spdiags([diagm1 , diag0 , diagp1] , np.array( [-1, 0, 1] ))



# #+name: Microscale Transmissions

   def set_multiscale_transmissions(self, resolution)->NDArray[np.float64]:
      micro_basis = np.zeros((self.N -1)*resolution)
      for i in range(self.N -1):
         micro_fv = FVSolver(resolution , self.D , domain=(self.x[i] , self.x[i+1]))
         micro_fv.set_boundary(bc=(0.,1.))
         micro_fv.assemble_matrix()
         phi = micro_fv.solve()

         micro_basis[resolution * i:resolution*(i+1)] = phi
         hm = micro_fv.h
         self._T[i] = -hm * np.sum(((phi[1:] - phi[:-1])/hm)**2 * self.D(micro_fv.x[:-1]))
      return micro_basis



# #+RESULTS:
# : None


from importlib import reload
import src.fvsolver
from src.fvsolver import FVSolver
reload(src.fvsolver)
epsilon = 0.1
D = lambda x: 1 / (2+1.9 * np.cos(2 * np.pi* x / epsilon))
fv = FVSolver(10 ,  D)
fv.assemble_matrix()
fv.set_boundary()
c_course = fv.solve()
plt.plot(c_course)



# #+RESULTS:
# [[file:images/course1D.png]]


mb = fv.set_multiscale_transmissions(100)
plt.plot(mb)



# #+RESULTS:
# [[file:images/msbasis.png]]


fv.assemble_matrix()
c_multi = fv.solve()
plt.plot(c_multi)
