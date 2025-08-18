

# #+RESULTS:
# : None



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Diffusivity
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

# Diffusion
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

fig,axis= plt.subplots(1,2 , figsize=(10,4))
im1 = axis[0].imshow(diffusion_b , cmap="magma" , extent=[0,1,0,1])
axis[0].set_title(r"0D Obstacles")
im2 = axis[1].imshow(diffusion_c , cmap="magma" , extent=[0,1,0,1])
axis[1].set_title(r"1D Obstacles")

#fig.colorbar()
fig.suptitle(r"Oscillating Diffusion")
fig.colorbar(im1 ,ax=axis , fraction=0.025)

# Diffusivity
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

fig,axis= plt.subplots(1,3 , figsize=(14,4))
im1 = axis[0].imshow(diffusion_b , cmap="magma" , extent=[0,1,0,1])
axis[0].set_title(r"Square with $L^{100}$ norm")
im2 = axis[1].imshow(diffusion_c , cmap="magma" , extent=[0,1,0,1])
axis[1].set_title(r"Circle with $L^{2}$ norm")
im2 = axis[2].imshow(diffusion_r , cmap="magma" , extent=[0,1,0,1])
axis[2].set_title(r"Rhombus with $L^{1}$ norm")

#fig.colorbar()
fig.suptitle(r"2D Box Constraints")
fig.colorbar(im1 ,ax=axis , fraction=0.025)

# Initialization
# #+name: Init

   def __init__(self , N :int , D :Callable  , domain=(0.,1.))->None:
       self.h = (domain[1] - domain[0]) / (N-1)
       self.N = N
       self.D = D
       self.x = np.linspace(domain[0] , domain[1] , N)
       self._T =  -1/self.h * D((self.x[:-1] + self.x[1:])*0.5)
       self.f = self.h* np.ones(N)

# Solving
# #+name: Solve

   def solve(self):
      self.c = spsolve(self._A.tocsr() , self.f)
      return self.c

# Boundary
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

# Sparsity Pattern of the linear system
# #+name: A Sparsity

from importlib import reload
import src.fvsolver
from src.fvsolver import FVSolver
reload(src.fvsolver)
f10 = FVSolver(20,  D.oscillation)
f10.assemble_matrix()
A = f10._A
sparsity = np.full(A.shape , np.nan)
Idx = A.nonzero()
sparsity[Idx] = A.todense()[Idx]
plt.imshow(sparsity , cmap="viridis")
plt.title("Sparsity Patter of A")

# 1D Results
# #+name: fig:comparison-1d

from importlib import reload
import src.fvsolver
from src.fvsolver import FVSolver
import src.diffusion as D
reload(src.fvsolver)
reload(D)
fv = FVSolver(10 ,  D.oscillation)
fv.assemble_matrix()
fv.set_boundary()
c_course = fv.solve()

fv_ref = FVSolver(10000,  D.oscillation)
fv_ref.set_boundary()
fv_ref.assemble_matrix()
c_fine = fv_ref.solve()

fvmulti = FVSolver(10 ,  D.oscillation)
mb = fvmulti.set_multiscale_transmissions(100)
fvmulti.set_boundary()
fvmulti.assemble_matrix()
c_multi = fvmulti.solve()
fvmulti.reconstruct_multiscale()

fig , ax = plt.subplots(figsize=(10,4))
plt.plot(fv.x , c_course)
plt.plot(fvmulti.x , c_multi)
x_fine = np.linspace(0,1, len(fvmulti.micro_basis))
plt.plot(x_fine,fvmulti.reconstruction)
plt.plot(fv_ref.x,c_fine)
plt.title("Comparison Of Different Solvers\n with oscilating Diffusion" , fontsize=22)
plt.xlabel(r"$x$")
plt.ylabel(r"$c(x)$")
plt.legend(["macro" , "multiscale", "multi_fine" , "reference"] , fontsize=14)
plt.tight_layout()



# #+name: fig:microscale-basis

import seaborn
import seaborn as sns
plt.style.use('default')
fig , ax = plt.subplots(figsize=(10,4))
for i , line in zip([3,5,9] , [":" , "-." , "--"]) :
    fv = FVSolver(i ,  D.oscillation)
    mb = fv.set_multiscale_transmissions(100)
    fineX = np.linspace(0.,1. , mb.shape[0] )
    plt.plot(fineX,mb , linestyle=line)
plt.title("Microscale Basis" , fontsize=22)
plt.xlabel(r"$x$")
plt.ylabel(r"$\phi(x)$")
plt.legend(["3 Cells" , "5 Cells", "9 Cells"], fontsize=14)
plt.tight_layout()

# Oscillations

plot_comparison(D.osc2D_line , 50 , "Line Diffusion with 4 Spikes")



# #+RESULTS:
# [[file:images/2d-multi-result-line.png]]


plot_comparison(D.osc2D_point , 50 ,"Point Diffusion with 4 Spikes" )

# Box Conditions

plot_comparison(D.box , 50 , "Box Obstacle")



# #+RESULTS:
# [[file:images/2d-multi-result-box.png]]


plot_comparison(D.smooth_box , 50 , "Circle Obstacle")



# #+RESULTS:
# [[file:images/2d-multi-result-circle.png]]


plot_comparison(D.rhombus , 50 , "Diamond Obstacle")

# 1D Error

# #+name: fig:error-1d

import src.diffusion as diffusionModule
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
reload(diffusionModule)
import src.fvsolver as fvModule
reload(fvModule)
diffusionFunction = lambda x: diffusionModule.oscillation(x,eps=1/20)


fineX = np.linspace(0, 1, 10000)
solver = fvModule.FVSolver(10000, diffusionFunction, (0,1))
solver.set_boundary()
solver.assemble_matrix()
referenceSolution = solver.solve()

# plt.plot(fineX , referenceSolution)

gridCoarseLevels = np.arange(1, 500, 1)
gridCoarseLevelsMulti = np.arange(2, 110, 1)

singleScaleErrorLevels = []
multiScaleErrorLevels = []
multiScaleReconstructErrorLevels = []


for coarseLevel in gridCoarseLevels:
    # solve single scale
    coarseX = np.linspace(0,1 ,coarseLevel)
    solver = fvModule.FVSolver(coarseLevel, diffusionFunction, (0,1))
    solver.set_boundary()
    solver.assemble_matrix()
    coarseSolution = solver.solve()
    interpolatedCoarseSolution = np.interp(fineX, coarseX, coarseSolution)
    error = np.sqrt(np.mean(np.square(referenceSolution - interpolatedCoarseSolution)))
    singleScaleErrorLevels.append(error)

for coarseLevel in gridCoarseLevelsMulti:
    #solve multi scale
    coarseX = np.linspace(0,1 ,coarseLevel)
    solver = fvModule.FVSolver(coarseLevel, diffusionFunction, (0,1))
    solver.set_boundary()
    mb = solver.set_multiscale_transmissions(1000)
    solver.assemble_matrix()
    coarseSolution = solver.solve()
    interpolatedCoarseSolution = np.interp(fineX, coarseX, coarseSolution)
    error = np.sqrt(np.mean(np.square(referenceSolution - interpolatedCoarseSolution)))
    multiScaleErrorLevels.append(error)


    reconstructedSolution = solver.reconstruct_multiscale()
    reconstructedX = np.linspace(0,1,len(solver.micro_basis))
    interpolatedCoarseSolution = np.interp(fineX, reconstructedX, solver.reconstruction)
    error = np.sqrt(np.mean(np.square(referenceSolution - interpolatedCoarseSolution)))
    multiScaleReconstructErrorLevels.append(error)

plt.figure(figsize=(6,4))
plt.scatter(gridCoarseLevels, singleScaleErrorLevels, marker=".", label="single-scale")
plt.scatter(gridCoarseLevelsMulti, multiScaleErrorLevels, marker="x", alpha=0.5, label="multi-scale")
plt.scatter(gridCoarseLevelsMulti, multiScaleReconstructErrorLevels, marker="+", alpha=0.5, label="multi-scale reconstructed")

plt.title("1D MSE Single vs Multiscale\n1000 subgrid cells\n1D Diffusion 20 Spikes")

plt.xlabel("1D coarse grid resolution")
plt.ylabel("Mean Square Error")
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.legend()
plt.tight_layout()

# 2D Error
# #+name: plot-2d-error

import src.diffusion as diffusionModule
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
reload(diffusionModule)
import src.fvsolver as fvModule
reload(fvModule)
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import src.fvsolver as fvModule
from scipy.interpolate import RegularGridInterpolator
reload(fvModule)

def plot_error_2d(diffusionFunction  , gridCoarseLevels , gridCoarseLevelsMulti , subtitle):
    singleScaleErrorLevels = []
    multiScaleErrorLevels = []
    multiScaleReconstructErrorLevels = []

    fineN = 500
    fineX = np.linspace(0, 1, fineN)
    fineY = np.linspace(0, 1, fineN)
    fineXX, fineYY = np.meshgrid(fineX, fineY)
    finePoints = np.column_stack([fineXX.ravel(), fineYY.ravel()])
    solver = fvModule.FVSolver2D(fineN, fineN, diffusionFunction)
    solver.set_boundary()
    solver.assemble_matrix()
    referenceSolution = solver.solve()
    for coarseLevel in gridCoarseLevels:
        # solve single scale
        coarseX = np.linspace(0, 1, coarseLevel)
        coarseY = np.linspace(0, 1, coarseLevel)
        coarseXX, coarseYY = np.meshgrid(coarseX, coarseY)
        coarsePoints = np.column_stack([coarseXX.ravel(), coarseYY.ravel()])

        solver = fvModule.FVSolver2D(coarseLevel,coarseLevel, diffusionFunction)
        solver.set_boundary()
        solver.assemble_matrix()
        coarseSolution = solver.solve()

        interpolator = RegularGridInterpolator((coarseX , coarseY), coarseSolution)
        interpolatedCoarseSolution = interpolator(finePoints).reshape(fineXX.shape)

        error = np.sqrt(np.mean(np.square(referenceSolution - interpolatedCoarseSolution)))
        singleScaleErrorLevels.append(error)

    for coarseLevel in gridCoarseLevelsMulti:
        #solve multi scale
        coarseX = np.linspace(0, 1, coarseLevel)
        coarseY = np.linspace(0, 1, coarseLevel)
        coarseXX, coarseYY = np.meshgrid(coarseX, coarseY)
        coarsePoints = np.column_stack([coarseXX.ravel(), coarseYY.ravel()])

        solver = fvModule.FVSolver2D(coarseLevel,coarseLevel, diffusionFunction)
        solver.set_boundary()
        mb = solver.set_multiscale_transmissions(100)
        solver.assemble_matrix()
        coarseSolution = solver.solve()

        interpolator = RegularGridInterpolator((coarseX , coarseY), coarseSolution)
        interpolatedCoarseSolution = interpolator(finePoints).reshape(fineXX.shape)

        error = np.sqrt(np.mean(np.square(referenceSolution - interpolatedCoarseSolution)))
        multiScaleErrorLevels.append(error)


        reconstructedSolution = solver.reconstruct_multiscale()
        reconstructedX = np.linspace(0, 1, (solver.N-1) * solver.resolution)
        reconstructedY = np.linspace(0, 1, (solver.M-1) * solver.resolution)
        rcXX, rcYY = np.meshgrid(reconstructedX, reconstructedY)
        reconstructedPoints = np.column_stack([rcXX.ravel(), rcYY.ravel()])

        interpolator = RegularGridInterpolator((reconstructedX , reconstructedY), reconstructedSolution)
        interpolatedCoarseSolution = interpolator(finePoints).reshape(fineXX.shape)

        error = np.sqrt(np.mean(np.square(referenceSolution - interpolatedCoarseSolution)))
        multiScaleReconstructErrorLevels.append(error)

    # print(singleScaleErrorLevels.shape)
    fig , ax = plt.subplots(figsize=(6,4))
    ax.scatter(gridCoarseLevels, singleScaleErrorLevels, marker=".", label="single-scale")
    ax.scatter(gridCoarseLevelsMulti, multiScaleErrorLevels, marker="x", alpha=0.5, label="multi-scale")
    ax.scatter(gridCoarseLevelsMulti, multiScaleReconstructErrorLevels, marker="+", alpha=0.5, label="multiscale reconstructed")

    fig.suptitle(f"2D MSE Single vs Multiscale\n{fineN} subgrid cells\n{subtitle}")

    ax.set_xlabel("2D coarse grid resolution")
    ax.set_ylabel("Mean Square Error")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    return fig






# #+name: fig:error-2d-circle

gridCoarseLevels = np.arange(5, 100, 2)
gridCoarseLevelsMulti = np.arange(5, 50, 2)
fig = plot_error_2d(diffusionModule.circle  , gridCoarseLevels , gridCoarseLevelsMulti , "Circle Diffusion")



# #+RESULTS: fig:error-2d-circle
# [[file:error-2d-circle.png]]


# #+name: fig:error-2d-box

gridCoarseLevels = np.arange(5, 50, 2)
gridCoarseLevelsMulti = np.arange(5, 50, 2)
fig = plot_error_2d(diffusionModule.box  , gridCoarseLevels , gridCoarseLevelsMulti , "Box Diffusion")



# #+RESULTS: fig:error-2d-box
# [[file:error-2d-box.png]]

# #+name: fig:error-2d-diamond

gridCoarseLevels = np.arange(5, 50, 2)
gridCoarseLevelsMulti = np.arange(5, 50, 2)
fig = plot_error_2d(diffusionModule.rhombus  , gridCoarseLevels , gridCoarseLevelsMulti , "Diamond Diffusion")



# #+RESULTS: fig:error-2d-diamond
# [[file:error-2d-diamond.png]]

# #+name: fig:error-2d-line

gridCoarseLevels = np.arange(5, 50, 2)
gridCoarseLevelsMulti = np.arange(5, 50, 2)
fig = plot_error_2d(lambda x,y: diffusionModule.osc2D_line(x,y , eps = 1/5)  , gridCoarseLevels , gridCoarseLevelsMulti , "Line Diffusion 5 Spikes")



# #+RESULTS: fig:error-2d-line
# [[file:error-2d-line.png]]
# #+name: fig:error-2d-point

gridCoarseLevels = np.arange(5, 50, 2)
gridCoarseLevelsMulti = np.arange(5, 50, 2)
fig = plot_error_2d(lambda x,y:diffusionModule.osc2D_point(x,y , eps=1/5)  , gridCoarseLevels , gridCoarseLevelsMulti , "Point Diffusion 5 Spikes")

# Multiscale :noexport:
# In 1D
# #+name: Microscale Transmissions

   def set_multiscale_transmissions(self, resolution)->NDArray[np.float64]:
      self.resolution = resolution
      micro_basis = np.zeros((self.N-1)*resolution)
      for i in range(1,self.N):
         micro_fv = FVSolver(resolution , self.D , domain=(self.x[i-1], self.x[i]))
         micro_fv.set_boundary(bc=(0.,1.))
         micro_fv.assemble_matrix()
         phi = micro_fv.solve()

         micro_basis[resolution * (i-1):resolution*i] = phi
         hm = micro_fv.h
         self._T[i-1] = -hm * np.sum(((phi[1:] - phi[:-1])/hm)**2 * self.D(micro_fv.x[:-1]))
      self.micro_basis = micro_basis
      return micro_basis



# \begin{align*}
# T_{\pm } &= -\int_{Q} D(x) (\phi'_{\pm} (x))^2\, \mathrm{d}x
# \end{align*}


# #+name: Reconstruct Microscale Solution

   def reconstruct_multiscale(self)->NDArray[np.float64]:
        self.reconstruction = np.zeros_like(self.micro_basis)
        for i in range(len(self.c)-1):
            n = self.resolution
            t = self.micro_basis[n*i:n*(i+1)]
            self.reconstruction[n*i:n*(i+1)] = (1-t) * self.c[i] + t * self.c[i+1]

from importlib import reload
import src.fvsolver
from src.fvsolver import FVSolver
import src.diffusion as D
reload(src.fvsolver)
reload(D)
fv = FVSolver(20 ,  D.oscillation)
fv.assemble_matrix()
fv.set_boundary()
c_course = fv.solve()

fv_ref = FVSolver(10000,  D.oscillation)
fv_ref.set_boundary()
fv_ref.assemble_matrix()
c_fine = fv_ref.solve()

fvmulti = FVSolver(10 ,  D.oscillation)
mb = fvmulti.set_multiscale_transmissions(100)
fvmulti.set_boundary()
fvmulti.assemble_matrix()
c_multi = fvmulti.solve()
fvmulti.reconstruct_multiscale()

plt.plot(fv.x , c_course)
plt.plot(fvmulti.x , c_multi)
x_fine = np.linspace(0,1, len(fvmulti.micro_basis))
plt.plot(x_fine,fvmulti.reconstruction)
plt.plot(fv_ref.x,c_fine)
plt.title("Comparison Of Different Solvers")
plt.xlabel(r"$x$")
plt.ylabel(r"$c(x)$")
plt.legend(["macro" , "multiscale", "multi_fine" , "reference"])



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

# Cleanup :noexport:

# #+RESULTS:
# : None


from importlib import reload
import src.fvsolver
from src.fvsolver import FVSolver
import src.diffusion as D
reload(src.fvsolver)
reload(D)
epsilon = 0.1
diff = lambda x: D.circle(x,0.5)
fv = FVSolver(100 , diff)
fv.assemble_matrix()
fv.set_boundary()
c_course = fv.solve()
wall = fv.D(fv.x)
print(np.min(wall))
#plt.plot(fv.x,wall)
plt.plot(fv.x,c_course)



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
      self.h_x = (domain[1,0] - domain[0,0]) / (N-1)
      self.h_y = (domain[1,1] - domain[0,1]) / (M-1)
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

       A = sp.sparse.spdiags([main_diag , diag_east , diag_west ,  diag_north , diag_south] , [0 , -self.N  , self.N , 1 , -1] , self.N*self.M , self.M*self.N)
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


import seaborn as sns
import src.fvsolver
import src.diffusion as D
reload(src.fvsolver)
reload(D)
from src.fvsolver import FVSolver2D
N = 30
M = 30
fv2D = FVSolver2D(N,M,D.rhombus)
fv2D.set_boundary()
fv2D.set_multiscale_transmissions(50)
fv2D.assemble_matrix()
c = fv2D.solve()
fv2D.reconstruct_multiscale()
plt.subplots(figsize=(6,4))
sns.heatmap(fv2D.reconstruction, cmap="magma")
#sns.heatmap(c, cmap="magma")



# #+RESULTS:
# [[file:images/2d-result.png]]


error =np.linalg.norm(A@c_vec - f)
print(error)



# #+RESULTS:
# : 1.025105313314805e-12


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(grid[0] ,grid[1],c , cmap="magma")

# 2D Multiscale :noexport:
# \begin{align*}
# T_{\pm } &= -\int_{Q} D(x) \phi_x'(x)^2\, \mathrm{d}x
# \end{align*}
# #+name:2D Microscale Transmissions

   def set_multiscale_transmissions(self, resolution):
      self.resolution = resolution
      self.microscale_basis_x = np.zeros((self._T_x.shape[0] , self._T_x.shape[1] , resolution))
      self.microscale_basis_y = np.zeros((self._T_y.shape[0] , self._T_y.shape[1] , resolution))
      for i in range(self._T_x.shape[0]):
         for j in range(self._T_x.shape[1]):
            #Do mircroscale x
            D_micro = lambda x: self.D(x, self.y[j])
            fv_micro = FVSolver(resolution , D_micro, domain=(self.x[i] , self.x[i+1]))
            fv_micro.assemble_matrix()
            fv_micro.set_boundary(bc=(0.,1.))
            phi =fv_micro.solve()
            self.microscale_basis_x[i,j,:] = phi
            self._T_x[i,j] =   -fv_micro.h * self.h_y* np.sum(((phi[1:] - phi[:-1])/(fv_micro.h))**2 * D_micro(fv_micro.x[1:] - fv_micro.h/2))

      for i in range(self._T_y.shape[0]):
         for j in range(self._T_y.shape[1]):
            # Do microscale y
            D_micro = lambda y: self.D(self.x[i], y)
            fv_micro = FVSolver(resolution , D_micro, domain=(self.y[j] , self.y[j+1]))
            fv_micro.assemble_matrix()
            fv_micro.set_boundary(bc=(0.,1.))
            phi =fv_micro.solve()
            self.microscale_basis_y[i,j,:] = phi
            self._T_y[i,j] =   -fv_micro.h * self.h_x  * np.sum(((phi[1:] - phi[:-1])/(fv_micro.h))**2 * D_micro(fv_micro.x[1:] - fv_micro.h/2))

      return self.microscale_basis_x , self.microscale_basis_y

import src.fvsolver
import src.diffusion as D
reload(src.fvsolver)
reload(D)
from src.fvsolver import FVSolver2D
def plot_comparison(function , resolution , typestr):
    fvref = FVSolver2D(1000, 1000,function)
    fvref.set_boundary()
    fvref.assemble_matrix()
    c_ref = fvref.solve()
    fv2D = FVSolver2D(resolution, resolution,function)
    fv2D.assemble_matrix()
    fv2D.set_boundary()
    c_course = fv2D.solve()
    mx,my = fv2D.set_multiscale_transmissions(2000)
    fv2D.assemble_matrix()
    fv2D.set_boundary()
    c = fv2D.solve()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)  # 1 row, 2 columns
    fig.suptitle(f"{typestr} with {resolution}" + r"$\times$" + f"{resolution} Grid")
    im1 = axes[0].imshow(c_course , cmap="magma" , extent=[0,1,0,1])
    axes[0].set_title("Course")
    im2 = axes[1].imshow(c , cmap="magma" , extent=[0,1,0,1])
    axes[1].set_title("Multiscale")
    im2 = axes[2].imshow(c_ref , cmap="magma" , extent=[0,1,0,1])
    axes[2].set_title("Reference")
    plt.colorbar(im1, ax=axes)
    return fig



# #+RESULTS:
# : None

# #+name: 2D Reconstruction

   def reconstruct_multiscale(self):
       self.reconstruction = np.zeros(((self.N-1) * self.resolution  , (self.M-1) * self.resolution))
       for i in range(self.N-1):
           for j in range(self.M-1):
                 x_lower = self.microscale_basis_x[i, j, :]
                 x_upper = self.microscale_basis_x[i, j+1, :]
                 y_lower = self.microscale_basis_y[i, j, :]
                 y_upper = self.microscale_basis_y[i+1, j, :]
                 interp = np.linspace(0,1 , self.resolution)
                 X = np.outer(x_lower,(1-interp)) + np.outer(x_upper,interp)
                 Y = np.outer((1-interp) , y_lower) + np.outer(interp,y_upper)

                 w11 = (1 - X) * (1-Y)
                 w12 = (1-X) * Y
                 w21 = X * (1-Y)
                 w22 = X * Y
                 self.reconstruction[
                     i * self.resolution : (i + 1) * self.resolution,
                     j * self.resolution : (j + 1) * self.resolution,
                 ] = (
                     w11 * self.c[i, j]
                     + w12 * self.c[i, j + 1]
                     + w21 * self.c[i + 1, j]
                     + w22 * self.c[i + 1, j + 1]
                 )
       return self.reconstruction

# Diffusion

import src.diffusion as D
reload(D)
x = np.linspace(0,1)
plt.plot(D.noise1D(x))

# Difusion
# #+name: 2D Noise

import src.diffusion as D
reload(D)
N = 100
M = 100
x = np.linspace(0.,1., N)
y= np.linspace(0.,1., M)
grid = np.meshgrid(x,y)
noise = D.noise2D(grid[0].ravel() , grid[1].ravel(), scale=10, frequencies=20)
sns.heatmap(noise.reshape(N,M))
