import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# #+name: 1D Diffusion

import numpy as np
import matplotlib.pyplot as plt
import src.diffusion as D
reload(D)
x = np.linspace(0,1 ,10)
plt.plot(x , D.oscillation(x))
x_highres = np.linspace(0,1 , 100000)
plt.plot(x_highres , D.oscillation(x_highres))
plt.legend([r"$D$ Sampled on a course grid" , r"$D$"] , loc="upper right")
plt.title("1D Diffusion Coefficient")



# #+name: 2D Box Constraints

import src.diffusion as D
reload(D)

N = 1000
M = 1000
x = np.linspace(0.,1., N)
y= np.linspace(0.,1., M)
grid = np.meshgrid(x,y)
diffusion_b = D.box(grid[0] , grid[1])
diffusion_b = diffusion_b.reshape((N,M))
diffusion_c = D.circle(grid[0] , grid[1])
diffusion_c = diffusion_c.reshape((N,M))
diffusion_r = D.rhombus(grid[0] , grid[1])
diffusion_r = diffusion_r.reshape((N,M))

fig,axis= plt.subplots(1,3)
im1 = axis[0].imshow(diffusion_b , cmap="magma" , extent=[0,1,0,1])
axis[0].set_title(r"Square with $L^{100}$ norm")
im2 = axis[1].imshow(diffusion_c , cmap="magma" , extent=[0,1,0,1])
axis[1].set_title(r"Circle with $L^{2}$ norm")
im2 = axis[2].imshow(diffusion_r , cmap="magma" , extent=[0,1,0,1])
axis[2].set_title(r"Rhombus with $L^{1}$ norm")

#fig.colorbar()
fig.suptitle(r"2D Box Constraints")
fig.colorbar(im1 ,ax=axis , fraction=0.025)


# #+name: 2D Ocillation

import src.diffusion as D
reload(D)


N = 1000
M = 1000
x = np.linspace(0.,1., N)
y= np.linspace(0.,1., M)
grid = np.meshgrid(x,y)
diffusion_b = D.osc2D_point(grid[0] , grid[1])
diffusion_b = diffusion_b.reshape((N,M))
diffusion_c = D.osc2D_line(grid[0] , grid[1])
diffusion_c = diffusion_c.reshape((N,M))

fig,axis= plt.subplots(1,2)
im1 = axis[0].imshow(diffusion_b , cmap="magma" , extent=[0,1,0,1])
axis[0].set_title(r"0D Obstacles")
im2 = axis[1].imshow(diffusion_c , cmap="magma" , extent=[0,1,0,1])
axis[1].set_title(r"1D Obstacles")

#fig.colorbar()
fig.suptitle(r"Osscillating Diffusion")
fig.colorbar(im1 ,ax=axis , fraction=0.025)

import src.diffusion as D
reload(D)
x = np.linspace(0,1)
plt.plot(D.noise1D(x))



# #+RESULTS:
# : None


import src.diffusion as D
reload(D)
N = 1000
M = 1000
x = np.linspace(0.,1., N)
y= np.linspace(0.,1., M)
grid = np.meshgrid(x,y)
noise = D.noise2D(grid[0].ravel() , grid[1].ravel(), scale=5 , frequencies=50)
sns.heatmap(noise.reshape(N,M))




# #+name: Init

   def __init__(self , N :int , D :Callable  , domain=(0.,1.))->None:
       self.h = (domain[1] - domain[0]) / N
       self.N = N
       self.D = D
       self.x = np.linspace(domain[0] , domain[1] , N)
       self._T =  -1/self.h * D((self.x[:-1] + self.x[1:]) * 0.5)
       self.f = self.h* np.ones(N)



# #+name: Solve

   def solve(self):
      self.c = spsolve(self._A.tocsr() , self.f)
      return self.c



# #+name: Boundary

   def set_boundary(self , bc=(0.,0.)):
      self.f[0] = bc[0]
      self.f[-1] = bc[1]







# Matrix Assembly
# #+name: Assemble Matrix

   def assemble_matrix(self)-> None:
      diagp1 = np.zeros(self.N)
      diagp1[2:] =  self._T[1:]
      diagm1 = np.zeros(self.N)
      diagm1[:-2] =  self._T[:-1]
      diag0 = np.ones(self.N)
      diag0[1:-1] = -1 * (self._T[1:] + self._T[:-1])
      self._A = spdiags([diagm1 , diag0 , diagp1] , np.array( [-1, 0, 1] ))

sns.heatmap(A.todense())
plt.title("Sparsity Patter of A")

# Multiscale
# In 1D
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
      self.micro_basis = micro_basis
      return micro_basis



# \begin{align*}
# T_{\pm } &= -\int_{Q} D(x) (\phi'_{\pm} (x))^2\, \mathrm{d}x
# \end{align*}


# #+name: Reconstruct Microscale Solution

   def reconstruct_multiscale(self)->NDArray[np.float64]:
        self.reconstruction = np.zeros_like(self.micro_basis)
        for i in range(len(self.c)-1):
            n = len(self.micro_basis) // self.N + self.N
            t = self.micro_basis[n*i:n*(i+1)]
            self.reconstruction[n*i:n*(i+1)] = (1-t) * self.c[i] + t * self.c[i+1]

from importlib import reload
import src.fvsolver
from src.fvsolver import FVSolver
import src.diffusion as D
reload(src.fvsolver)
reload(D)
fv = FVSolver(10 ,  D.oscillation)
fv.set_boundary()
mb = fv.set_multiscale_transmissions(100)
fv.assemble_matrix()
c_course = fv.solve()
fv.reconstruct_multiscale()
plt.plot(fv.reconstruction)



# #+RESULTS:
# [[file:images/reconstruction.png]]


plt.plot(mb)



# #+RESULTS:
# [[file:images/msbasis.png]]


fv.assemble_matrix()
c_multi = fv.solve()
plt.plot(c_multi)



# #+end_src


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

# Cleanup

# #+RESULTS:
# : None


from importlib import reload
import src.fvsolver
from src.fvsolver import FVSolver
reload(src.fvsolver)
epsilon = 0.1
D = lambda x: 1 / (2+1.9 * np.cos(2 * np.pi* x / epsilon))
fv = FVSolver(100 ,  D)
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



# #+name: Init 2D

   def __init__(self ,
                N:int,
                M:int ,
                D :Callable  ,
                domain=np.array([[0.,0.] , [1.,1.]]),
                )->None:
      self.h_x = (domain[1,0] - domain[0,0]) / N
      self.h_y = (domain[1,1] - domain[0,1]) / M
      self.x = np.linspace(domain[0,0] , domain[1,0] , N)
      self.y = np.linspace(domain[0,1] , domain[1,1] , M)
      x_h = self.x[:-1] + 0.5 * self.h_x
      y_h = self.y[:-1] + 0.5 * self.h_y
      halfgrid_x = np.meshgrid(x_h,self.y,indexing="ij")
      halfgrid_y = np.meshgrid(self.x,y_h , indexing="ij")
      self._T_x = -self.h_y/self.h_x * D(halfgrid_x[0] , halfgrid_x[1])
      self._T_y = -self.h_x/self.h_y * D(halfgrid_y[0] , halfgrid_y[1])
      self.N = N
      self.M = M
      self.D = D
      self.f = self.h_x * self.h_y* np.ones((N, M))



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



# #+RESULTS:
# : None


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



# #+RESULTS:
# [[file:images/2D_Diffusion.png]]



reload(src.fvsolver)
from src.fvsolver import FVSolver2D
smol_fv = FVSolver2D(10,10,D)
smol_fv.assemble_matrix()
plt.imshow(smol_fv._A.todense())
#plt.spy(A.T, markersize=1)



# #+RESULTS:
# [[file:images/spy.svg]]


fv2D = FVSolver2D(N,M,D)
sns.heatmap(fv2D._T_y, cmap="magma")



# #+RESULTS:
# [[file:images/_T_x.png]]


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
# #+name:2D Microscale Transmissions

   def set_multiscale_transmissions(self, resolution):
      microscale_basis_x = np.zeros((self._T_x.shape[0] , self._T_x.shape[1] , resolution))
      microscale_basis_y = np.zeros((self._T_y.shape[0] , self._T_y.shape[1] , resolution))
      for i in range(self._T_x.shape[0]):
         for j in range(self._T_x.shape[1]):
            #Do mircroscale x
            D_micro = lambda x: self.D(x, self.y[j])
            fv_micro = FVSolver(resolution , D_micro, domain=(self.x[i] , self.x[i+1]))
            fv_micro.assemble_matrix()
            fv_micro.set_boundary(bc=(0.,1.))
            phi =fv_micro.solve()
            microscale_basis_x[i,j,:] = phi
            self._T_x[i,j] =   -fv_micro.h * self.h_y* np.sum(((phi[1:] - phi[:-1])/fv_micro.h)**2 * D_micro(fv_micro.x[:-1]))

      for i in range(self._T_y.shape[0]):
         for j in range(self._T_y.shape[1]):
            # Do microscale y
            D_micro = lambda y: self.D(self.x[i], y)
            fv_micro = FVSolver(resolution , D_micro, domain=(self.y[j] , self.y[j+1]))
            fv_micro.assemble_matrix()
            fv_micro.set_boundary(bc=(0.,1.))
            phi =fv_micro.solve()
            microscale_basis_y[i,j,:] = phi
            self._T_y[i,j] =   -fv_micro.h * self.h_x  * np.sum(((phi[1:] - phi[:-1])/fv_micro.h)**2 * D_micro(fv_micro.x[:-1]))

      return microscale_basis_x , microscale_basis_y

reload(src.fvsolver)
from src.fvsolver import FVSolver2D
fv2D = FVSolver2D(100,100,D)
mx,my = fv2D.set_multiscale_transmissions(100)
fv2D.assemble_matrix()
fv2D.set_boundary()
c = fv2D.solve()
sns.heatmap(c, cmap="magma")
