#!/usr/bin/python3

from casadi import SX, Function, jacobian
import numpy as np
from numpy import dot, transpose
from numpy.linalg import matrix_rank, inv
from typing import Callable


class ILC:

  """ An Iterative Learning Controller (ILC) for MIMO systems
  as described in 'A discrete-time iterative learning algorithm
  for linear time-varying systems' by K.K. Tan, S.N. Huang, 
  T.H. Lee and S.Y. Lim.

  This module takes a discrete-time nonlinear system and linearizes
  it along a given trajectory, thus producing an LTV system. 

  TODO: Explain internal workings more

  Ideas for later:
  - Update A,B,C matrices after each step (Relinearize around last
    state) instead of linearizing only once around the reference?
  """

  def __init__(self, F: Callable[[SX,SX],SX], 
                     g: Callable[[SX,SX],SX],
                     N: int,
                     nx: int,
                     nu: int,
                     ny: int):
    """

    """
    print("=====================================")
    print("===    Creating ILC (MIMO,LTV)    ===")
    print("=====================================")

    # Check sizes
    assert F.n_in() == 2, "F: incorrect number of inputs (must be 2)!"
    assert F.n_out() == 1, "F: incorrect number of outputs (must be 1)!"
    F.assert_size_in(0, nx, 1)
    F.assert_size_in(1, nu, 1)
    F.assert_size_out(0, nx, 1)
    assert g.n_in() == 2, "h: incorrect number of inputs (must be 2)!"
    assert g.n_out() == 1, "h: incorrect number of outputs (must be 1)!"
    g.assert_size_in(0, nx, 1)
    g.assert_size_in(1, nu, 1)
    g.assert_size_out(0, ny, 1)

    # Fetch parameters
    self.N = N
    self.nx = nx
    self.nu = nu
    self.ny = ny

    # Create casadi expressions
    x = SX.sym('x', nx, 1)
    u = SX.sym('u', nu, 1)
    xnext = F(x,u)
    y = g(x,u)

    # Create linearization matrices as functions
    self.A_fun = Function('A_fun', [x,u], [jacobian(xnext,x)], ['x','u'], ['A'])
    self.B_fun = Function('B_fun', [x,u], [jacobian(xnext,u)], ['x','u'], ['B'])
    self.C_fun = Function('C_fun', [x,u], [jacobian(y,x)],     ['x','u'], ['C'])

    # Create (empty) A, B, C and learning gain matrices
    self.A = np.zeros((nx,nx,N))
    self.B = np.zeros((nx,nu,N))
    self.C = np.zeros((ny,nx,N))
    self.K = np.zeros((nu,ny,N))

    # Create (empty) deviation matrix
    self.yerr = np.zeros((ny,N))

    # Initialize sample-counter (resets after each iteration)
    self.k = 0

    # Ready to be initialized
    self.initialized = False
    print("ILC (MIMO,LTV) created.")


  def init(self, yref: np.array,
                 xref: np.array, 
                 uref: np.array):
    """ Initializes the ILC
    Parameters:
    yref: The output reference
    xref: The target state trajectory (precomputed by Target Selector)
    uref: The target control trajectory (precomputed by Target Selector)
    """
    print("=====================================")
    print("===  Initializing ILC (MIMO,LTV)  ===")
    print("=====================================")

    # Do not initialize multiple times
    assert self.initialized == False, "Already initialized!"

    # Fetch parameters for readability
    N  = self.N
    nx = self.nx
    nu = self.nu
    ny = self.ny

    # Check sizes
    assert xref.shape == (nx,N), "xref: Incorrect shape!"
    assert uref.shape == (nu,N), "uref: Incorrect shape!"
    assert yref.shape == (ny,N), "yref: Incorrect shape!"

    # Store references
    self.yref = np.copy(yref)
    self.xref = np.copy(xref)
    self.uref = np.copy(uref)

    # Check stability of phi TODO
    # (Fixed for now. Tuning parameter?)
    phi = 0.5 * np.eye(ny)

    # Compute linearization matrices
    print("Computing linearization matrices ..")
    for k in range(N):
      self.A[:,:,k] = self.A_fun(xref[:,k], uref[:,k]).full()
      self.B[:,:,k] = self.B_fun(xref[:,k], uref[:,k]).full()
      self.C[:,:,k] = self.C_fun(xref[:,k], uref[:,k]).full()

    # Check row-rankness of C(k+1)*B(k) for all k (Theorem 2)
    print("Checking row-rankness ..")
    for k in range(N):
      C_kp1 = self.C[:,:,k+1] if k+1 < N else self.C[:,:,0]
      B_k   = self.B[:,:,k]

      full_row_rank = matrix_rank(dot(C_kp1,B_k)) == ny
      assert full_row_rank, "C(k+1)*B(k) is row-deficient for k=" + str(k) + "!"

    # Compute learning gain matrices (Remark 2)
    print("Computing learning gains ..")
    for k in range(N):
      B_k   = self.B[:,:,k]
      B_kp1 = self.B[:,:,k+1] if k+1 < N else self.B[:,:,0]
      C_kp1 = self.C[:,:,k+1] if k+1 < N else self.C[:,:,0]

      eye = np.eye(ny)
      prod1 = transpose(dot(C_kp1, B_k))
      prod2 = dot(C_kp1, B_kp1)
      prod3 = inv(dot(prod2, prod1))
      prod4 = dot(prod1, prod3)
      sub1 = eye - phi

      self.K[:,:,k] = np.dot(prod4, sub1)

    ##################
    ## Double check stability (follows from Theorem 2, remove later)
    print("Double checking stability ..")
    for k in range(N):
      K_k   = self.K[:,:,k]
      B_k   = self.B[:,:,k]
      C_kp1 = self.C[:,:,k+1] if k+1 < N else self.C[:,:,0]
      eye = np.eye(ny)
      M = eye - dot(C_kp1, dot(B_k, K_k))

      eigvals, eigvecs = np.linalg.eig (M)
      eigvals_norm = [np.linalg.norm(lam) for lam in eigvals]
      for m in eigvals_norm:
        assert m < 1.0, "UNSTABLE FOR k=" + k + ", CHECK IMPLEMENTATION"

    # End initialization
    self.initialized = True
    print("ILC (MIMO,LTV) initialized.")


  def call(self, y, xest):
    """ Executes the ILC step
    """
    # Fetch paramters for readability
    k = self.k
    N = self.N

    # Fetch deviation (last iteration)
    e = self.yerr[:,k]

    # Fetch gain matrix
    K = self.K[:,:,k]
    
    # Compute control delta
    du = np.dot(K,e)

    # Update internal data
    self.yerr[:,k] = self.yref[:,k] - y
    self.xref[:,k] = np.copy(xest)
    self.uref[:,k] += du.flatten()
    
    # Check for NaN and inf
    if np.any(np.isnan(du)) or np.any(np.isinf(du)):
      raise ValueError("NaN/inf encountered in control output at k="+str(k))

    # Increase counter
    if k+1 < N:
      self.k += 1
    else:
      self.k = 0
      # Trigger relinearization (TODO)
    
    # Return control delta
    return du
