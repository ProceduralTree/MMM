# Preamble :noexport:

from typing import Callable
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from numpy.typing import NDArray

# Class Structure

class FVSolver:
   N : int
   resolution : int
   h : np.float64
   x : NDArray[np.float64]
   D : Callable
   f : NDArray[np.float64]
   c : NDArray[np.float64]
   micro_basis : NDArray[np.float64]
   _T : NDArray[np.float64]

   def __init__(self , N :int , D :Callable  , domain=(0.,1.))->None:
       self.h = (domain[1] - domain[0]) / (N-1)
       self.N = N
       self.D = D
       self.x = np.linspace(domain[0] , domain[1] , N)
       self._T =  -1/self.h * D((self.x[:-1] + self.x[1:])*0.5)
       self.f = self.h* np.ones(N)

   def assemble_matrix(self)-> None:
      diagp1 = np.zeros(self.N)
      diagp1[2:] =  self._T[1:]
      diagm1 = np.zeros(self.N)
      diagm1[:-2] =  self._T[:-1]
      diag0 = np.ones(self.N)
      diag0[1:-1] = -1 * (self._T[1:] + self._T[:-1])
      self._A = spdiags([diagm1 , diag0 , diagp1] , np.array( [-1, 0, 1] ))
   def set_boundary(self , bc=(0.,0.)):
      self.f[0] = bc[0]
      self.f[-1] = bc[1]

   def solve(self):
      self.c = spsolve(self._A.tocsr() , self.f)
      return self.c

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


   def reconstruct_multiscale(self)->NDArray[np.float64]:
        self.reconstruction = np.zeros_like(self.micro_basis)
        for i in range(len(self.c)-1):
            n = self.resolution
            t = self.micro_basis[n*i:n*(i+1)]
            self.reconstruction[n*i:n*(i+1)] = (1-t) * self.c[i] + t * self.c[i+1]

# 2D :noexport:

import scipy as sp
import numpy as np
class FVSolver2D:
   N : int
   M : int
   h_x : np.float64
   h_y : np.float64
   x : NDArray[np.float64]
   y : NDArray[np.float64]
   D : Callable
   f : NDArray[np.float64]
   c : NDArray[np.float64]

   _T_x : NDArray[np.float64]
   _T_y : NDArray[np.float64]



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


   def set_boundary(self , bc=(0.,0. , 0. , 0.)):
      self.f[ 0,1:-1]= bc[0]
      self.f[-1,1:-1]= bc[1]
      self.f[1:-1, 0]= bc[2]
      self.f[1:-1,-1]= bc[3]


   def solve(self):
      self.c = spsolve(self._A.tocsr() , self.f.ravel()).reshape((self.N,self.M))
      return self.c

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

   def reconstruct_multiscale(self):
       self.reconstruction = np.zeros(((self.N-1) * self.resolution  , (self.M-1) * self.resolution))
       for i in range(self.N-1):
           for j in range(self.M-1):
                 x_lower = self.microscale_basis_x[i, j, :]
                 x_upper = self.microscale_basis_x[i, j+1, :]
                 y_lower = self.microscale_basis_y[i, j, :]
                 y_upper = self.microscale_basis_y[i+1, j, :]
                 interp_x = 0.5*( y_upper + y_lower)
                 interp_y = 0.5*( x_upper + x_lower)
                 #interp_x = np.linspace(0,1,self.resolution)
                 #interp_y = np.linspace(0,1,self.resolution)
                 X = np.outer(x_lower,(1-interp_x)) + np.outer(x_upper,interp_x)
                 Y = np.outer((1-interp_y) , y_lower) + np.outer(interp_y,y_upper)
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
