# Cleanup

from typing import Callable
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from numpy.typing import NDArray

class FVSolver:
   N : int
   h : np.float64
   x : NDArray[np.float64]
   D : Callable
   f : NDArray[np.float64]
   c : NDArray[np.float64]

   _T : NDArray[np.float64]

   def __init__(self , N :int , h : np.float64 , D :Callable  , domain=(0.,1.))->None:
       self.N = N
       self.D = D
       self.x = np.linspace(domain[0] , domain[1] , N)
       self._T = - 1/h * D((self.x[:-1] + self.x[1:]) * 0.5)
       self.f = h* np.ones(N)



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
      self.f[1] = bc[1]


   def solve(self):
      self.c = spsolve(self._A.tocsr() , self.f)
      return self.c
   def set_multiscale_transmissions(self):
      pass
