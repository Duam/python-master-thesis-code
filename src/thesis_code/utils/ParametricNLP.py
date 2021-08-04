#!/usr/bin/python3

#########################################
### @author Paul Daum
### @date 04.12.2018
### @brief This file implements a generic
### optimimization problem
#########################################

# CasADi is used for algorithmic differentiation
# and creating the optimization problem
import casadi as cas
from casadi import DM, MX, mtimes, vertcat, horzcat, nlpsol, inf, hessian, Function, blockcat
from casadi import jacobian, hessian
from casadi import qpsol, is_linear, is_quadratic
from casadi.tools import struct_symSX, struct_symMX, struct_MX, entry
from casadi.tools.structure3 import DMStruct

from utils.IterationCallback import IterationCallback

import os
import numpy
import scipy
import copy
import time
import itertools
import multiprocessing

from typing import Callable

###########################################################################
###                                                                     ###
###                          START OF OP CLASS                          ###
###                                                                     ###
###########################################################################

class ParametricNLP:

    """
    A representation for the parametric optimization problem

    S(p) = minimize:  J(w,p)
              w
          subject to: g(w,p) = 0
                      h(w,p) >= 0

       Usage is as follows:
    1. Create the ParametricNLP object
    2. Add decision variables and parameters
    3. Get the internal casadi variables
    4. Define a cost function
    5. Define constraints and add them
    (!) Keep step 2 and 3 separated
    """


    def __init__(self, name = 'OP', verbose = False):
        """ Creates the empty problem.
        Prepares casadi expressions and creates the optimization variables.

        TODO: More docu
        """
        self.name = name
        self.verbose = verbose

        self.J = 0
        self.w = {}
        self.lbx = {}
        self.ubx = {}
        self.p = {}
        self.g = {}
        self.h = {}

        self.nw = 0
        self.np = 0
        self.ng = 0
        self.nh = 0

        self.current_params = struct_symMX([])
        self.iterCb = None # The iteration-callback function
        self.variables_baked = False


    def add_decision_var(self, name:str, shape:tuple):
        """ Adds a decision variable to the problem. """
        # Check if baking allowed and if entry already exists
        assert not self.variables_baked, 'Adding variables prohibited after baking'
        assert name not in self.w.keys(), 'Key \'' + name + '\' already exists'
        # Add entry and increase variable counter
        self.w[name] = shape
        self.nw += numpy.prod(shape)
        # Print info string
        if self.verbose: print("Added decision variable: " + name + ' ' + str(shape))

    def add_parameter(self, name: str, shape: tuple):
        """ Adds a parameter to the problem """
        # Check if baking allowed and if entry already exists
        assert not self.variables_baked, 'Adding variables prohibited after baking'
        assert name not in self.p.keys(), 'Key ' + name + ' already exists'
        # Add entry and increase variable counter
        self.p[name] = shape
        self.np += numpy.prod(shape)
        # Print info string
        if self.verbose: print("Added parameter: " + name + ' ' + str(shape))

    def bake_variables(self):
        # Create structs for variables, parameters
        self.struct_w = struct_symMX([entry(key, shape=self.w[key]) for key in self.w])
        self.struct_p = struct_symMX([entry(key, shape=self.p[key]) for key in self.p])
        # Variables were baked successfully
        self.variables_baked = True

    def get_decision_var(self, name:str):
        """ Returns the desired decision variable. """
        # Return the desired variable
        return self.struct_w[name]

    def get_parameter(self, name: str):
        """ Returns the desired parameter. """
        # Return the desired parameter
        return self.struct_p[name]


    def set_cost(self, expr: MX):
        """ Sets the cost function of the problem.
        Parameters:
          expr (casadi.MX): A casadi MX expression defining the
            problem's cost function. Must only use decision
            variables and parameters defined in the optimizaion problem.
        """
        self.J = expr

    def add_equality(self, name:str, expr:MX):
        """ Adds an equality constraint to the problem
        Parameters:
          name (str): The name of the constraint
          expr (casadi.MX) A casadi MX/MX expression defining the equality
            constraint's LHS. Must only use decision variables and parameters
            defined in the optimization problem.
        """
        # Check if entry already exists
        assert name not in self.g.keys(), 'Key ' + name + ' already exists'
        # Add entry and increase constraint counter
        self.g[name] = expr
        self.ng += numpy.prod(expr.shape)
        # Print an info string
        if self.verbose: print("Added equality constraint: " + name + ' ' + str(expr.shape))

    def add_inequality(self, name:str, expr:MX):
        """ Adds an inequality constraint to the problem
        Parameters:
          name (str): The name of the constraint
          expr (casadi.MX) A casadi MX/MX expression defining the equality
            constraint's LHS. Must only use decision variables and parameters
            defined in the optimization problem.
        """
        # Check if entry already exists
        assert name not in self.h.keys(), 'Key \'' + name + '\' already exists'
        # Add entry and increase constraint counter
        self.h[name] = expr
        self.nh += numpy.prod(expr.shape)
        # Print an info string
        if self.verbose: print("Added inequality constraint: " + name + ' ' + str(expr.shape))

    def init(self, is_qp = False,
             nlpsolver: str ='ipopt',
             opts: dict = {},
             cbfun: Callable[[int,dict],None] = None,
             compile_solver: bool = False,
             create_analysis_functors:bool = True):
        """
        Initializes the problem by creating the NLP and the solver
        objects.
        Parameters:
        - nlpsolver (str): The name of the solver.
        - opts (dict): Options passed to the solver.
        - cbfun (Callable[[int,dict],None]): Function called after each iteration.
        """
        assert self.variables_baked, "Variables need to be baked before init()!"

        # Create structs for constraints
        self.struct_g = struct_symMX([entry(key, shape=self.g[key].shape) for key in self.g])
        self.struct_h = struct_symMX([entry(key, shape=self.h[key].shape) for key in self.h])

        # Create structs for multipliers of variables, parameters, constraints
        self.struct_lam_w = struct_symMX([entry(key, shape=self.w[key]) for key in self.w])
        self.struct_lam_p = struct_symMX([entry(key, shape=self.p[key]) for key in self.p])
        self.struct_lam_g = struct_symMX([entry(key, shape=self.g[key].shape) for key in self.g])
        self.struct_lam_h = struct_symMX([entry(key, shape=self.h[key].shape) for key in self.h])


        # Vectorize x and p
        #self.w_int = vertcat(*[w.reshape((w.shape[0]*w.shape[1],1)) for w in self.w.values()])
        #self.p_int = vertcat(*[p.reshape((p.shape[0]*p.shape[1],1)) for p in self.p.values()])

        self.J_int = struct_MX([entry('J', expr=self.J)])

        # Vectorize g and h, concatenate them
        self.g_int = struct_MX([entry(key, expr=self.g[key]) for key in self.g.keys()])
        self.h_int = struct_MX([entry(key, expr=self.h[key]) for key in self.h.keys()])
        self.c_int = vertcat(self.g_int.cat, self.h_int.cat)
        #self.g_int = vertcat(*[g.reshape((g.shape[0]*g.shape[1],1)) for g in self.g.values()])
        #self.h_int = vertcat(*[h.reshape((h.shape[0]*h.shape[1],1)) for h in self.h.values()])
        #self.c_int = vertcat(self.g_int, self.h_int)

        # Create the nlp object
        self.nlp = {
            'x': self.struct_w, #vertcat(*cas.symvar(self.w_int)),
            'p': self.struct_p, #vertcat(*cas.symvar(self.p_int)),
            'f': self.J,
            'g': self.c_int,
        }

        """
        # Check if all decision variables were used in J,g or h
        w = self.struct_w
        used_check = dict([ 
          (key+" "+str(idx),False) for key in w.keys() for idx in itertools.product(*[range(s) for s in w[key].shape])
        ])
    
        for key in w.keys():
          # Fetch decision variable vector
          elem = vertcat(*cas.symvar(w[key]))
          # Check all vector components
          for idx in itertools.product(*[range(s) for s in elem.shape]):
            if cas.depends_on(self.nlp['f'], *cas.symvar(elem[idx])):
              used_check[key+" "+str(idx)] = True
            print("Checking g..", key, idx)
            if cas.depends_on(self.nlp['g'], *cas.symvar(elem[idx])):
              used_check[key+" "+str(idx)] = True
        
        for key in used_check.keys():
          if used_check[key] == False:
            print("WARNING: In ParametricNLP \"", self.name, "\", element \"", key, "\" is unused.")
        """

        # Create the iteration callback (if given) and add it to the solver
        if cbfun != None:
            opts['iteration_callback'] = IterationCallback('IterationCallback', self.nw, self.ng + self.nh, self.np, cbfun)

        if 'hess_lag' not in opts.keys():
            # Create lagrange hessian
            w = self.struct_w
            p = self.struct_p
            lam_g = self.struct_lam_g.cat
            z = vertcat(w,lam_g) # Todo: needs lam_h also
            sigma = cas.MX.sym('sigma')
            # Create Lagrangian
            lag = self.J
            if self.g_int.shape != (0,0):
                lag += mtimes(lam_g.T, self.g_int)

            """
            f = self.nlp['f']
            #g = self.nlp['g'][:self.ng]
            g = self.nlp['g']
            h = self.nlp['g'][self.ng:]
            grad_f = jacobian(f,w).T
            #grad_f = cas.gradient(f,w)
            jac_g = jacobian(g,w)
            hes_f = hessian(f,w)[0]
            hes_g = [hessian(g[i],w)[0] for i in range(self.ng)]
            
            self.grad_f_fun = Function('grad_f', [w,p], [0,grad_f])
            self.jac_g_fun = Function('grad_g', [w,p], [0,jac_g])
            
            self.hess_lag_fun = Function('hess_lag', [w,p,sigma,lam_g], [
              sigma * hes_f + sum([ lam_g[i] * hes_g[i] for i in range(self.ng) ])
            ]).expand()
            
            #print(self.grad_f_fun.n_out())
        
            ## Initialize solver
            opts['grad_f'] = self.grad_f_fun
            opts['jac_g'] = self.jac_g_fun
            opts['hess_lag'] = self.hess_lag_fun
            """

        # Create solver
        if is_qp:
            assert is_quadratic(self.nlp['f'], self.nlp['x'])
            assert is_linear(self.nlp['g'], self.nlp['x'])
            self.solver = qpsol(self.name, nlpsolver, self.nlp, opts)
            print("ABORT. Quadratic stuff is deactivated atm.")
            quit(0)
        else:
            print("Creating NLP \"" + self.name + "\". This may take a while.")
            self.solver = nlpsol(self.name, nlpsolver, self.nlp, opts)

        # Compile solver if desired
        if compile_solver:
            opts['expand'] = False # Needed because we eval_sx bug when loading compiled .so
            path = "" + self.name # Apparently no subdirectory allowed?
            if not os.path.isfile(path + ".so"):
                print("Compiling NLP \"" + self.name + "\". This may take a while.")
                start_time = time.time()
                self.solver.generate_dependencies(path + ".c")
                flags  = "gcc "
                flags += " -pipe"
                flags += " -O3" # use -O3 for maximum evaluation speed # Try out O2, too
                flags += " -fPIC -shared -v"
                flags += " " + path + ".c"
                flags += " -o " + path + ".so"
                compile_flag = os.system(flags)
                # Use -03 for MAXIMUM SPEED
                end_time = time.time()
                print("Compilation took: " + str(time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))))
                print("Written solver \"" + self.name + "\" in file " + path + ".")
            else:
                print("Using already compiled solver \"" + path + "\".")

            self.solver = nlpsol(self.name, nlpsolver, os.path.abspath(path+'.so'), opts)
            #self.solver = cas.external(self.name, os.path.abspath(path+'.so'))

        # Define lower & upper bounds
        lbg = numpy.zeros((self.ng,1))
        ubg = numpy.zeros((self.ng,1))
        lbh = numpy.zeros((self.nh,1))
        ubh = inf * numpy.ones((self.nh,1))
        self.lb = vertcat(lbg,lbh)
        self.ub = vertcat(ubg,ubh)

        # Create some convenience functions
        if create_analysis_functors:
            w = self.struct_w
            p = self.struct_p
            lam_g = self.struct_lam_g.cat
            z = vertcat(w,lam_g) # Todo: needs lam_h also

            # Create Lagrangian
            lag = self.J
            if self.g_int.shape != (0,0):
                lag += mtimes(lam_g.T, self.g_int)

            # Create functors
            self.f_fun = Function('f', [w,p], [self.nlp['f']])
            self.g_fun = Function('g', [w,p], [self.nlp['g'][:self.ng]])
            self.h_fun = Function('h', [w,p], [self.nlp['g'][self.ng:]])
            self.jac_g_fun = Function('jac_g', [w,p], [jacobian(self.g_fun(w,p),w)])
            self.lag_fun = Function('lag', [w,lam_g,p], [lag])
            self.jac_lag_fun = Function('jac_lag', [w,lam_g,p], [jacobian(lag,w)])
            self.hess_lag_fun = Function('hess_lag', [w,lam_g,p], [hessian(lag,w)[0]])
            # NLP functors
            """
            self.kkt_fun = Function('kkt', [w,lam_g,p], [blockcat([
              [self.hess_lag_fun(w,lam_g,p), self.jac_g_fun(w,p).T],
              [self.jac_g_fun(w,p), DM.zeros((self.ng,self.ng))]
            ])])
            self.R_fun = Function('R', [w,lam_g,p], [blockcat([
              [ self.f_fun(w,p) + mtimes(self.jac_g_fun(w,p).T, lam_g) ],
              [ self.g_fun(w,p) ]
            ])])
            self.jac_R_fun = Function('jac_R', [w,lam_g,p], [jacobian(self.R_fun(w,lam_g,p),z)])
            """

    def eval_g(self, w:DMStruct, p:DMStruct):
        assert isinstance(w,DMStruct), "w is type " + str(type(w))
        assert isinstance(p,DMStruct), "p is type " + str(type(p))
        return self.struct_g(self.g_fun(w,p))

    def eval_h(self, w:DMStruct, p:DMStruct):
        assert isinstance(w,DMStruct), "w is type " + str(type(w))
        assert isinstance(p,DMStruct), "p is type " + str(type(p))
        return self.struct_h(self.h_fun(w,p))

    def eval_kernel(self, w:DMStruct, p:DMStruct):
        assert isinstance(w,DMStruct), "w is type " + str(type(w))
        assert isinstance(p,DMStruct), "p is type " + str(type(p))
        return DM(scipy.linalg.null_space(self.jac_g_fun(w,p)))

    def eval_reduced_hessian(self, w:DMStruct, lam_g:DMStruct, p:DMStruct):
        assert isinstance(w,DMStruct), "w is type " + str(type(w))
        assert isinstance(p,DMStruct), "p is type " + str(type(p))
        Z = self.eval_kernel(w,p)
        return mtimes([Z.T, self.hess_lag_fun(w,lam_g,p), Z])

    def eval_expanded_eigvecs(self, w:DMStruct, lam_g:DMStruct, p:DMStruct):
        assert isinstance(w,DMStruct), "w is type " + str(type(w))
        assert isinstance(lam_g,DMStruct), "lam_g is type " + str(type(lam_g))
        assert isinstance(p,DMStruct), "p is type " + str(type(p))
        Z = self.eval_kernel(w,p)
        reduced_hess = self.eval_reduced_hessian(w,lam_g,p)
        eigvals, eigvecs = scipy.linalg.eig(reduced_hess)
        eigvecs_expanded = Z.full().dot(eigvecs)
        return [ (eigvals[k], self.struct_w(eigvecs_expanded[:,k])) for k in range(eigvecs_expanded.shape[1]) ]

    def eval_kappa_exact(self, w:DMStruct, lam_g:DMStruct, p:DMStruct):
        assert isinstance(w,DMStruct), "w is type " + str(type(w))
        assert isinstance(lam_g,DMStruct), "lam_g is type " + str(type(lam_g))
        assert isinstance(p,DMStruct), "p is type " + str(type(p))
        M = self.kkt_fun(w,lam_g,p)
        J = self.jac_R_fun(w,lam_g,p)
        mat = DM.eye((M.rows())) - mtimes(cas.inv(M), J)
        eigvals, eigvecs = scipy.linalg.eig(mat)
        return max([abs(val) for val in eigvals])



    def solve(self, winit: struct_symMX, param: struct_symMX = None):
        """ Solves the Parametric NLP """

        # Fetch parameters
        self.current_params = param # needed?

        # Optimize!
        result = self.solver(
            p = param,
            x0 = winit,
            lbg = self.lb,
            ubg = self.ub
        )

        # Extract solution and solver statistics
        stats = self.solver.stats()
        f     = result['f']
        w     = self.struct_w     (result['x'])
        #lam_w = self.struct_lam_w (result['lam_x'])
        p     = self.struct_p     (param)
        #lam_p = self.struct_lam_p (result['lam_p'])
        #g     = self.struct_g     (result['g'][:self.ng])
        #lam_g = self.struct_lam_g (result['lam_g'][:self.ng])
        #h     = self.struct_h     (result['g'][self.ng:]) if self.nh > 0 else self.struct_h(0)
        #lam_h = self.struct_lam_h (result['lam_g'][self.ng:]) if self.nh > 0 else self.struct_h(0)

        # Bundle solution and return
        sol = {
            'f': f,
            'w': w, #'lam_w': lam_w,
            'p': p, #'lam_p': lam_p,
            #'g': g, 'lam_g': lam_g,
            #'h': h, 'lam_h': lam_h
        }

        return sol, stats, result, self.lb, self.ub


    def __str__(self):
        """ toString-method.
        Returns an info-string about the object
        """

        max_str_len = 120
        infostr =  "======================================\n"
        infostr += "== OPTIMIZATION PROBLEM INFO STRING ==\n"
        infostr += "\"" + self.name + "\" \n\n"
        infostr += "Shooting variables: " + str(self.nw) + "\n"
        for elem in self.w.items():
            infostr += "  " + elem[0] + ": " + str(elem[1]) + "\n"
        infostr += "Parameters: " + str(self.np) + "\n"
        for elem in self.p.items():
            infostr += "  " + elem[0] + ": " + str(elem[1]) + "\n"
        infostr += "Objective: \n"
        if len(str(self.J)) > max_str_len:
            infostr += str(self.J)[:max_str_len] + " ..."
        else:
            infostr += str(self.J)

        infostr += "\n\nEquality constraints: \n"
        for elem in self.g.items():
            infostr += elem[0] + ": " + str(elem[1].shape) + "\n"

        infostr += "\n\nInequality constraints: \n"
        for elem in self.h.items():
            infostr += elem[0] + ": " + str(elem[1].shape) + "\n"

        infostr += "======================================\n"
        return infostr

    def getf(self):
        """ Returns the cost function """
        # Functionize constraints and return
        return Function('f', [self.struct_w,self.struct_p], [self.J], ['w','p'], ['J'])

    def getG(self):
        """ Returns the equality constraint function """
        # Create symbolics
        w = self.struct_w #cas.vertcat(*cas.symvar(self.w_int))
        p = self.struct_p #cas.vertcat(*cas.symvar(self.p_int))

        # Get equality constraints
        g = self.g_int

        # Functionize constraints and return
        return Function('G', [w,p], [g], ['w','p'], ['G'])


    def getKKT(self):
        """ Returns the KKT matrix as a function object """
        from casadi import jacobian, hessian, blockcat

        # Create symbolics
        w = self.struct_w #cas.vertcat(*cas.symvar(self.w_int))
        p = self.struct_p #cas.vertcat(*cas.symvar(self.p_int))
        g = self.getG()(w,p)
        lam_g = MX.sym('lam_g', g.rows())

        # Create Lagrangian
        L = self.J
        if g.shape != (0,0):
            L += mtimes(lam_g.T, g)

        # Derive g w.r.t. w
        jg = MX.zeros(g.shape)
        if g.shape != (0,0):
            jg = jacobian(g,w)

        # Derive L w.r.t. w
        jg = self
        hL = self.hess_lag_fun(w,p)

        # Create KKT matrix
        KKT = blockcat([
            [hL, jg.T],
            [jg, MX.zeros((jg.rows(),jg.rows()))]
        ])

        # TODO: Do a KKT.is_singular() check, print out warnings if so

        # Functionize KKT matrix and return
        return Function('KKT', [w,lam_g,p], [KKT], ['w','lam_g','p'], ['KKT'])


    def evalG(self, w: numpy.array):
        """ Evaluates the constraint function at a specific point.
        Parameter: w (numpy.array): The decision variables
        """
        return self.getG()(w, self.current_params)


    def evalKKT(self, w, lam_g):
        """ Evaluates the KKT matrix at a specific point.
        Parameters:
        w (numpy.array): The decision variable
        lam_g (numpy.array): The constraint multipliers
        """
        return self.getKKT()(w, lam_g, self.current_params)


    def spy(self, fignum, gridspacing = 1):
        """ Returns the sparsity pattern plot of the hessian

        Parameters:
        fignum (int): The figure number
        gridspacing (int): Repetition interval of the thick lines

        Returns the figure handle
        """
        from utils.SparsityPlotter import SparsityPlotter
        from casadi import sparsify

        # Create symbolics
        w = self.struct_w #cas.vertcat(*cas.symvar(self.w_int))
        p = self.struct_p #cas.vertcat(*cas.symvar(self.p_int))
        g = self.getG()(w,p)
        lam_g = MX.sym('lam_g', g.rows())

        # Evaluate KKT matrix symbolically
        KKT = self.getKKT().expand()(w,lam_g,p)

        # Sparsify the KKT matrix
        KKT_sp = sparsify(KKT)
        rows = KKT.rows()
        cols = KKT.columns()

        # Return the figure handle of the sparsity plot
        plotter = SparsityPlotter(fignum, rows, cols, gridSpacing=gridspacing)
        return plotter.printSparsity(KKT_sp)


###########################################################################
###                                                                     ###
###                            END OF OP CLASS                          ###
###                                                                     ###
###########################################################################
