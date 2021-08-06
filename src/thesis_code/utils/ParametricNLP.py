import casadi as cas
from casadi import DM, MX, mtimes, vertcat, horzcat, nlpsol, inf, hessian, Function, blockcat
from casadi import jacobian, hessian
from casadi import qpsol, is_linear, is_quadratic
from casadi.tools import struct_symSX, struct_symMX, struct_MX, entry
from casadi.tools.structure3 import DMStruct
import os
import numpy
import scipy
import time
from typing import Callable


class ParametricNLP:
    def __init__(self, name = 'OP'):
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

        :param name: Identifier of the NLP
        """
        self.name = name
        self.cost = 0
        self.decision_vars = {}
        self.decision_vars_lb = {}  # Lower bound
        self.decision_vars_ub = {}  # Upper bound
        self.params = {}
        self.eq_constraints = {}
        self.ineq_constraints = {}
        self.num_decision_vars = 0
        self.num_params = 0
        self.num_eq_constraints = 0
        self.num_ineq_constraints = 0
        self.current_params = struct_symMX([])
        self.variables_baked = False

    def add_decision_var(self, name: str, shape: tuple):
        """ Adds a decision variable to the problem. """
        # Check if baking allowed and if entry already exists
        assert not self.variables_baked, 'Adding variables prohibited after baking'
        assert name not in self.decision_vars.keys(), 'Key \'' + name + '\' already exists'
        # Add entry and increase variable counter
        self.decision_vars[name] = shape
        self.num_decision_vars += numpy.prod(shape)
        # Print info string
        print("Added decision variable: " + name + ' ' + str(shape))

    def add_parameter(self, name: str, shape: tuple):
        """ Adds a parameter to the problem """
        # Check if baking allowed and if entry already exists
        assert not self.variables_baked, 'Adding variables prohibited after baking'
        assert name not in self.params.keys(), 'Key ' + name + ' already exists'
        # Add entry and increase variable counter
        self.params[name] = shape
        self.num_params += numpy.prod(shape)
        # Print info string
        print("Added parameter: " + name + ' ' + str(shape))

    def bake_variables(self):
        # Create structs for variables, parameters
        self.struct_w = struct_symMX([entry(key, shape=self.decision_vars[key]) for key in self.decision_vars])
        self.struct_p = struct_symMX([entry(key, shape=self.params[key]) for key in self.params])
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
        self.cost = expr

    def add_equality(self, name:str, expr:MX):
        """ Adds an equality constraint to the problem
        Parameters:
          name (str): The name of the constraint
          expr (casadi.MX) A casadi MX/MX expression defining the equality
            constraint's LHS. Must only use decision variables and parameters
            defined in the optimization problem.
        """
        assert name not in self.eq_constraints.keys(), 'Key ' + name + ' already exists'
        self.eq_constraints[name] = expr
        self.num_eq_constraints += numpy.prod(expr.shape)
        # Print an info string
        print("Added equality constraint: " + name + ' ' + str(expr.shape))

    def add_inequality(self, name:str, expr:MX):
        """ Adds an inequality constraint to the problem
        Parameters:
          name (str): The name of the constraint
          expr (casadi.MX) A casadi MX/MX expression defining the equality
            constraint's LHS. Must only use decision variables and parameters
            defined in the optimization problem.
        """
        assert name not in self.ineq_constraints.keys(), 'Key \'' + name + '\' already exists'
        # Add entry and increase constraint counter
        self.ineq_constraints[name] = expr
        self.num_ineq_constraints += numpy.prod(expr.shape)
        # Print an info string
        print("Added inequality constraint: " + name + ' ' + str(expr.shape))

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
        self.struct_g = struct_symMX([entry(key, shape=self.eq_constraints[key].shape) for key in self.eq_constraints])
        self.struct_h = struct_symMX([entry(key, shape=self.ineq_constraints[key].shape) for key in self.ineq_constraints])

        # Create structs for multipliers of variables, parameters, constraints
        self.struct_lam_w = struct_symMX([entry(key, shape=self.decision_vars[key]) for key in self.decision_vars])
        self.struct_lam_p = struct_symMX([entry(key, shape=self.params[key]) for key in self.params])
        self.struct_lam_g = struct_symMX([entry(key, shape=self.eq_constraints[key].shape) for key in self.eq_constraints])
        self.struct_lam_h = struct_symMX([entry(key, shape=self.ineq_constraints[key].shape) for key in self.ineq_constraints])

        self.J_int = struct_MX([entry('J', expr=self.cost)])

        # Vectorize g and h, concatenate them
        self.g_int = struct_MX([entry(key, expr=self.eq_constraints[key]) for key in self.eq_constraints.keys()])
        self.h_int = struct_MX([entry(key, expr=self.ineq_constraints[key]) for key in self.ineq_constraints.keys()])
        self.c_int = vertcat(self.g_int.cat, self.h_int.cat)

        # Create the nlp object
        self.nlp = {
            'x': self.struct_w,
            'p': self.struct_p,
            'f': self.cost,
            'g': self.c_int,
        }

        if 'hess_lag' not in opts.keys():
            # Create lagrange hessian
            w = self.struct_w
            p = self.struct_p
            lam_g = self.struct_lam_g.cat
            z = vertcat(w,lam_g) # Todo: needs lam_h also
            sigma = cas.MX.sym('sigma')
            # Create Lagrangian
            lag = self.cost
            if self.g_int.shape != (0,0):
                lag += mtimes(lam_g.T, self.g_int)

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
            opts['expand'] = False  # Needed because we eval_sx bug when loading compiled .so
            path = "" + self.name  # Apparently no subdirectory allowed?
            if not os.path.isfile(path + ".so"):
                print("Compiling NLP \"" + self.name + "\". This may take a while.")
                start_time = time.time()
                self.solver.generate_dependencies(path + ".c")
                flags = "gcc "
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

        # Define lower & upper bounds
        lbg = numpy.zeros((self.num_eq_constraints, 1))
        ubg = numpy.zeros((self.num_eq_constraints, 1))
        lbh = numpy.zeros((self.num_ineq_constraints, 1))
        ubh = inf * numpy.ones((self.num_ineq_constraints, 1))
        self.lb = vertcat(lbg,lbh)
        self.ub = vertcat(ubg,ubh)

        # Create some convenience functions
        if create_analysis_functors:
            w = self.struct_w
            p = self.struct_p
            lam_g = self.struct_lam_g.cat
            z = vertcat(w,lam_g)  # Todo: needs lam_h also

            # Create Lagrangian
            lag = self.cost
            if self.g_int.shape != (0,0):
                lag += mtimes(lam_g.T, self.g_int)

            # Create functors
            self.f_fun = Function('f', [w,p], [self.nlp['f']])
            self.g_fun = Function('g', [w,p], [self.nlp['g'][:self.num_eq_constraints]])
            self.h_fun = Function('h', [w,p], [self.nlp['g'][self.num_eq_constraints:]])
            self.jac_g_fun = Function('jac_g', [w,p], [jacobian(self.g_fun(w,p),w)])
            self.lag_fun = Function('lag', [w,lam_g,p], [lag])
            self.jac_lag_fun = Function('jac_lag', [w,lam_g,p], [jacobian(lag,w)])
            self.hess_lag_fun = Function('hess_lag', [w,lam_g,p], [hessian(lag,w)[0]])

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



    def solve(self, initial_guess: struct_symMX, params: struct_symMX = None):
        """ Solves the Parametric NLP.
        :param initial_guess:
        :param params:
        :returns:
        """
        self.current_params = param
        result = self.solver(p=param, x0=initial_guess, lbg=self.lb, ubg=self.ub)

        # Extract solution and solver statistics
        stats = self.solver.stats()
        optimal_cost = result['f']
        optimal_solution = self.struct_w(result['x'])
        params = self.struct_p(param)
        sol = {'f': optimal_cost, 'w': optimal_solution, 'p': params}
        return sol, stats, result, self.lb, self.ub


    def __str__(self):
        """toString-method.
        :returns: An info-string about the object.
        """

        max_str_len = 120
        infostr =  "======================================\n"
        infostr += "== OPTIMIZATION PROBLEM INFO STRING ==\n"
        infostr += "\"" + self.name + "\" \n\n"
        infostr += "Shooting variables: " + str(self.num_decision_vars) + "\n"
        for elem in self.decision_vars.items():
            infostr += "  " + elem[0] + ": " + str(elem[1]) + "\n"
        infostr += "Parameters: " + str(self.num_params) + "\n"
        for elem in self.params.items():
            infostr += "  " + elem[0] + ": " + str(elem[1]) + "\n"
        infostr += "Objective: \n"
        if len(str(self.cost)) > max_str_len:
            infostr += str(self.cost)[:max_str_len] + " ..."
        else:
            infostr += str(self.cost)

        infostr += "\n\nEquality constraints: \n"
        for elem in self.eq_constraints.items():
            infostr += elem[0] + ": " + str(elem[1].shape) + "\n"

        infostr += "\n\nInequality constraints: \n"
        for elem in self.ineq_constraints.items():
            infostr += elem[0] + ": " + str(elem[1].shape) + "\n"

        infostr += "======================================\n"
        return infostr

    def getf(self):
        """Returns the cost function
        :returns: A functional to evaluate the cost function.
        """
        return Function('f', [self.struct_w, self.struct_p], [self.cost], ['w', 'p'], ['J'])

    def getG(self):
        """Returns the equality constraint function.
        :returns: A functional to evaluate the constraint function.
        """
        return Function('G', [self.struct_w, self.struct_p], [self.g_int], ['w','p'], ['G'])
