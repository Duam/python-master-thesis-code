#!/usr/bin/python3
                
import numpy as np
from numpy import dot
from numpy.linalg import matrix_power
from numpy.linalg import matrix_rank

#####################################################################################

def isControllable(A: np.ndarray, 
                   B: np.ndarray):
  # Fetch data and do checks
  nu = B.shape[1]
  nx = A.shape[1]
  assert A.shape[0] == nx
  assert B.shape[0] == nx
  I = np.eye(nx)

  # Get eigenvalues of A
  lams = np.linalg.eigvals(A)

  # Loop through all eigenvalues
  for lam in lams:
    # Create (sub)controllability matrix
    M = np.hstack((dot(lam,I)-A,B))
    # Check rankness and return if needed
    if matrix_rank(M) != nx:
      return False

  # If all checks were successful, the pair is controllable
  return True

#####################################################################################  

def isStabilizable(A: np.ndarray,
                   B: np.ndarray):
  # Fetch data and do checks
  nu = B.shape[1]
  nx = A.shape[1]
  assert A.shape[0] == nx
  assert B.shape[0] == nx
  I = np.eye(nx)

  # Get eigenvalues of A
  lams = np.linalg.eigvals(A)

  # Loop through all eigenvalues with positiv real part
  for lam in lams:
    if np.real(lam) >= 0:
      # Create (sub)stability matrix
      M = np.hstack((dot(lam,I)-A,B))
      # Check rankness and return if needed
      if matrix_rank(M) != nx:
        return False

  # If all checks were successful, the pair is stabilizable
  return True

#####################################################################################  

def isObservable(A: np.ndarray,
                 C: np.ndarray):
  # Fetch data and do checks
  nx = A.shape[1]
  ny = C.shape[0]
  assert A.shape[0] == nx
  assert C.shape[1] == nx
  I = np.eye(nx)

  # Get eigenvalues
  lams = np.linalg.eigvals(A)

  # Loop through all eigenvalues
  for lam in lams:
    # Create (sub)observability matrix
    M = np.vstack((dot(lam,I)-A,C))
    # Check rankness and return if needed
    if matrix_rank(M) != nx:
      return False

  # If all checks were successful, the pair is observable
  return True

#####################################################################################

def isDetectable(A: np.ndarray,
                 C: np.ndarray):
  # Fetch data and do checks
  nx = A.shape[1]
  ny = C.shape[0]
  assert A.shape[0] == nx
  assert C.shape[1] == nx
  I = np.eye(nx)

  # Get eigenvalues
  lams = np.linalg.eigvals(A)

  # Loop through all eigenvalues with positive real value
  for lam in lams:
    if np.real(lam) >= 0:
      # Create (sub)detectability matrix
      M = np.vstack((dot(lam,I)-A,C))
      # Check rankness and return if needed
      if matrix_rank(M) != nx:
        return False

  # If all checks were successful, the pair is detectable
  return True

#####################################################################################

# Unit-test
if __name__ == '__main__':
  A1 = np.array([[1,0], [0,1]])
  A2 = np.array([[0,1], [1,0]])
  A3 = np.array([[1,1], [0,1]])
  B1 = np.array([[1],[1]])
  B2 = np.array([[1],[0]])
  B3 = np.array([[0],[0]])
  C1 = np.array([[1]])
  C2 = np.array([[0]])
  
  sys1  = (A1,B1,C1)
  sys2  = (A1,B1,C2)
  sys3  = (A2,B1,C1)
  sys4  = (A2,B1,C2)
  sys5  = (A3,B1,C1)
  sys6  = (A3,B1,C2)

  sys7  = (A1,B2,C1)
  sys8  = (A1,B2,C2)
  sys9  = (A2,B2,C1)
  sys10 = (A2,B2,C2)
  sys11 = (A3,B2,C1)
  sys12 = (A3,B2,C2)

  sys13 = (A1,B3,C1)
  sys14 = (A1,B3,C2)
  sys15 = (A2,B3,C1)
  sys16 = (A2,B3,C2)
  sys17 = (A3,B3,C1)
  sys18 = (A3,B3,C2)

  print("No unit test here")

