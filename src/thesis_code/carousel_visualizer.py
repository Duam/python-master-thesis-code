import casadi as cas
from casadi import DM, horzcat, blockcat
import numpy as np
from thesis_code.model import CarouselModel
import matplotlib.pyplot as plt

def plotStates(
        model: CarouselModel,
        dt: float,
        Xs_ref: DM=None,
        Us_ref: DM=None,
        Xs_sim: DM=None,
        Us_sim: DM=None,
        Xs_est: DM=None) -> None:
    """ Plots the states and input as a timeseries

    :param model: The carousel model instance
    :param dt: Timestep in seconds
    :param Xs_ref: State trajectory reference (default None)
    :param Us_ref: Control trajectory reference (default None)
    :param Xs_sim: State trajectory actual (default None)
    :param Us_sim: Control trajectory actual (default None)
    :param Xs_est: State trajectory estimate (default None)
    """
    NX = model.NX()
    if Xs_ref is not None: assert Xs_ref.shape[0] == NX, "Xs_ref shape =" + str(Xs_ref.shape)
    if Xs_sim is not None: assert Xs_sim.shape[0] == NX, "Xs_sim shape =" + str(Xs_sim.shape)
    if Xs_est is not None: assert Xs_est.shape[0] == NX, "Xs_est shape =" + str(Xs_est.shape)
    NU = model.NU()
    if Us_ref is not None: assert Us_ref.shape[0] == NU, "Us_ref shape =" + str(Us_ref.shape)
    if Us_sim is not None: assert Us_sim.shape[0] == NU, "Us_sim shape =" + str(Us_sim.shape)

    # Define plots
    uplots = [ [ (0, 'Elevator setpoint') ] ]
    xplots = [
        [ (6, 'Deflector angle') ],
        [ (5, 'Carousel yaw rate') ],
        [ (0, 'Plane roll'), (3, 'Plane roll rate')],
        [ (1, 'Plane pitch'), (4, 'Plane pitch rate')]
    ]

    # Fetch plot sizes
    n_uplots = len(uplots)
    n_xplots = len(xplots)
    n_plots = n_uplots + n_xplots

    # Compute root-mean-square error for REFERENCE VS SIMULATION for the STATES
    rms_x_ref_sim = None
    if Xs_ref is not None and Xs_sim is not None:
        assert Xs_ref.shape[1] == Xs_sim.shape[1], "Xs_ref, Xs_sim shape = " + str(Xs_ref.shape) + ", " + str(Xs_sim.shape)
        N = Xs_ref.shape[1]
        err = horzcat(*[ Xs_ref[:,k] - Xs_sim[:,k] for k in range(N) ])
        rms_x_ref_sim = blockcat([[ cas.sqrt(err[i,k]**2) for k in range(N)] for i in range(NX)])

    # Compute root-mean-square error for REFERENCE VS SIMULATION for the CONTROLS
    rms_u_ref_sim = None
    if Us_ref is not None and Us_sim is not None:
        assert Us_ref.shape[1] == Us_sim.shape[1], "Us_ref, Us_sim shape = " + str(Us_ref.shape) + ", " + str(Us_sim.shape)
        N = Us_ref.shape[1]
        err = horzcat(*[ Us_ref[:,k] - Us_sim[:,k] for k in range(N) ])
        rms_u_ref_sim = blockcat([[ cas.sqrt(err[i,k]**2) for k in range(N)] for i in range(NU)])

    # Compute root-mean-square error for SIMULATION VS ESTIMATION for the STATES
    rms_x_sim_est = None
    if Xs_sim is not None and Xs_est is not None:
        assert Xs_sim.shape[1] == Xs_est.shape[1], "Xs_sim, Xs_est shape = " + str(Xs_sim.shape) + ", " + str(Xs_ref.shape)
        N = Xs_sim.shape[1]
        err = horzcat(*[ Xs_sim[:,k] - Xs_est[:,k] for k in range(N) ])
        rms_x_sim_est = blockcat([[ cas.sqrt(err[i,k]**2) for k in range(N)] for i in range(NX)])

    # Create a figure
    fig, ax = plt.subplots(n_plots,3, sharex='all')
    plt.sca(ax[0,0])
    plt.title('Controls')
    plt.sca(ax[0,1])
    plt.title('Tracking error')
    plt.sca(ax[0,2])
    plt.title('Estimation error')

    # Plot controls and their rms
    for k in range(n_uplots):
        # Left column: Control plots
        plt.sca(ax[k,0])
        for p in uplots[k]:
            # Fetch control-index and control-name
            i = p[0]
            s = p[1]
            # Plot reference and simulated control
            if Us_ref is not None: plt.step(dt*np.arange(Us_ref.shape[1]), Us_ref[i,:].T, '--', label=s+' reference', where='post')
            if Us_sim is not None: plt.step(dt*np.arange(Us_sim.shape[1]), Us_sim[i,:].T, label=s, where='post')
            #plt.legend(loc='best').draggable()
        plt.grid()
        # Middle column: Reference-Simulation RMS plots
        plt.sca(ax[k,1])
        for p in uplots[k]:
            i = p[0]
            s = p[1]
            if rms_u_ref_sim is not None: plt.plot(dt*np.arange(rms_u_ref_sim.shape[1]), rms_u_ref_sim[i,:].T, '--', label=s+' reference')
            #plt.legend(loc='best').draggable()
            plt.yscale('log')
        plt.grid()


    # Plot states and their rms
    plt.sca(ax[n_uplots,0])
    plt.title("States")
    for k in range(n_xplots):
        # Left column: Control plots
        plt.sca(ax[n_uplots+k,0])
        for p in xplots[k]:
            # Fetch control-index and control-name
            i = p[0]
            s = p[1]
            # Plot reference and simulated control
            if Xs_ref is not None: plt.plot(dt*np.arange(Xs_ref.shape[1]), Xs_ref[i,:].T, '--', label=s+' reference')
            if Xs_sim is not None: plt.plot(dt*np.arange(Xs_sim.shape[1]), Xs_sim[i,:].T, label=s)
            if Xs_est is not None: plt.plot(dt*np.arange(Xs_est.shape[1]), Xs_est[i,:].T, '-.', label=s+' estimate')
            #plt.legend(loc='best').draggable()
        plt.grid()
        # Middle column: Reference-Simulation RMS plots
        plt.sca(ax[n_uplots+k,1])
        for p in xplots[k]:
            i = p[0]
            s = p[1]
            if rms_x_ref_sim is not None: plt.plot(dt*np.arange(rms_x_ref_sim.shape[1]), rms_x_ref_sim[i,:].T, '--', label=s+' reference')
            #plt.legend(loc='best').draggable()
            plt.yscale('log')
        plt.grid()
        # Right column: Estimate-Simulation RMS plots
        plt.sca(ax[n_uplots+k,2])
        for p in xplots[k]:
            i = p[0]
            s = p[1]
            if rms_x_sim_est is not None: plt.plot(dt*np.arange(rms_x_sim_est.shape[1]), rms_x_sim_est[i,:].T, '-.', label=s+' estimate')
            #plt.legend(loc='best').draggable()
            plt.yscale('log')
        plt.grid()


def plotStates_withRef(Xs:np.ndarray, Xs_ref:np.ndarray,
                       Us:np.ndarray, Us_ref:np.ndarray,
                       dt:float, model:CarouselModel):
    """ Plots the states and input as a timeseries
    Args:
      Xs[np.ndarray] -- State trajectory
      Xs_ref[np.ndarray] -- State trajectory reference
      Us[np.ndarray] -- Control trajectory
      Us_ref[np.ndarray] -- Control trajectory reference
      dt[float] -- Timestep
      model[CarouselWhiteBoxModel] -- The used carousel model instance
    """
    assert Xs.shape[0] == model.NX()
    assert Us.shape[0] == model.NU()
    assert Xs.shape[1] == Us.shape[1] + 1
    assert Xs.shape[0] == Xs_ref.shape[0]
    assert Us.shape[0] == Us_ref.shape[0]
    assert Xs.shape[1] == Xs_ref.shape[1]
    assert Us.shape[1] == Us_ref.shape[1]

    N = Us.shape[1]
    tAxis = dt * np.arange(N+1)

    # Compute root-mean-square error for each state
    err_x = cas.horzcat(*[ Xs[:,k] - Xs_ref[:,k] for k in range(Xs.shape[1]) ])
    err_u = cas.horzcat(*[ Us[:,k] - Us_ref[:,k] for k in range(Us.shape[1]) ])
    rms_x = cas.blockcat([
        [ cas.sqrt(err_x[i,k]*err_x[i,k]) for k in range(err_x.shape[1])] for i in range(err_x.shape[0])
    ])
    rms_u = cas.blockcat([
        [ cas.sqrt(err_u[i,k]*err_u[i,k]) for k in range(err_u.shape[1])] for i in range(err_u.shape[0])
    ])

    plots = [
        [ (4, 'Deflector angle') ],
        [ (0, 'Plane roll'), (2, 'Plane roll rate') ],
        [ (1, 'Plane pitch'), (3, 'Plane pitch rate') ]
    ]
    n_plots = len(plots)

    fig, ax = plt.subplots(n_plots,2, sharex='all')
    plt.sca(ax[0,0])
    plt.title('States & Controls')
    plt.sca(ax[0,1])
    plt.title('RMS error')
    for k in range(n_plots):
        # Left column: State & control plots
        plt.sca(ax[k,0])
        # Draw plots
        for p in plots[k]:
            # Fetch the state-index and the state-name
            i = p[0]
            s = p[1]
            # The input and elevator deflection are plotted in one graph
            if p[0] == 6:
                plt.step(tAxis[:-1], Us[0,:].T, 'o--', color="orange", label="Input", where='post')
                plt.plot(tAxis[:-1], Us_ref[0,:].T, 'x', color="red", label='Input reference')
            # Plot the state
            print(i)
            print(s)
            plt.plot(tAxis, Xs[i,:].T, 'o-', label=s)
            plt.plot(tAxis, Xs_ref[i,:].T, 'x', label=s+' reference')
            plt.legend(loc='best')
        plt.grid()

        # Right column: RMS error
        plt.sca(ax[k,1])
        # Draw plots
        for p in plots[k]:
            print("Plotting plot", p)
            # Fetch the state-index and the state-name
            i = p[0]
            s = p[1]
            # The input and elevator deflection are plotted in one graph
            if p[0] == 6:
                plt.step(tAxis[:-1], rms_u[0,:].T, 'o--', color="orange", label="Input", where='post')
            # Plot the carousel yaw together with its roll angle RMS
            if i == 5:
                plt.plot(tAxis, rms_x[2,:].T, 'o-', label="Carousel yaw")
                plt.plot()
            # Plot the state
            plt.plot(tAxis, rms_x[i,:].T, 'o-', label=s)
            plt.legend(loc='best')
            plt.yscale('log')
        plt.grid()


def plotAerodynamics(Xs:np.ndarray, Us:np.ndarray, dt:float, model:CarouselModel):
    """ Plots the aerodynamics data (AoA, Fl, Fd) as a timeseries
    Args:
      Xs[np.ndarray] -- State trajectory
      Us[np.ndarray] -- Control trajectory
      dt[float] -- Timestep
      model[CarouselWhiteBoxModel] -- The used carousel model instance
    """
    assert Xs.shape[0] == model.NX()
    assert Us.shape[0] == model.NU()
    assert Xs.shape[1] == Us.shape[1] + 1
    N = Us.shape[1]
    tAxis = dt * np.arange(N+1)

    p = model.p0()
    param = model.params
    print(param)

    # Compute aerodynamics
    alpha_A = [model.alpha_A(Xs[:,k],p) for k in range(N+1)]
    FL_A =   [model.FL_A(Xs[:,k],p) for k in range(N+1)]
    FD_A =   [model.FD_A(Xs[:,k],p) for k in range(N+1)]

    alpha_E = [model.alpha_E(Xs[:,k],p) for k in range(N+1)]
    FL_E =   [model.FL_E(Xs[:,k],p) for k in range(N+1)]
    FD_E =   [model.FD_E(Xs[:,k],p) for k in range(N+1)]
    #AileronData = [model.computeAileronAerodynamicsData(Xs[:,k],param) for k in range(N+1)]
    #ElevatorData =  [model.computeElevatorAerodynamicsData(Xs[:,k],param) for k in range(N+1)]


    #print([data['AoA'] for data in AileronData])

    fig,ax = plt.subplots(3,1)
    plt.sca(ax[0])
    plt.plot(tAxis, alpha_A, label="Aileron AoA")
    plt.plot(tAxis, alpha_E, label="Elevator AoA")
    plt.legend(loc='best')
    plt.grid()

    plt.sca(ax[1])
    plt.ylabel('Lift forces')
    plt.plot(tAxis, [np.sign(alpha_A[k]) * np.linalg.norm(FL_A[k]) for k in range(N+1)], label="Aileron Lift")
    plt.plot(tAxis, [np.sign(alpha_E[k]) * np.linalg.norm(FL_E[k]) for k in range(N+1)], label="Elevator Lift")
    plt.legend(loc='best')
    plt.grid()

    plt.sca(ax[2])
    plt.ylabel('Drag forces')
    plt.plot(tAxis, [np.sign(alpha_A[k]) * np.linalg.norm(FD_A[k]) for k in range(N+1)], label="Aileron Drag")
    plt.plot(tAxis, [np.sign(alpha_E[k]) * np.linalg.norm(FD_E[k]) for k in range(N+1)], label="Elevator Drag")
    plt.legend(loc='best')
    plt.grid()


def plotMoments(Xs:np.ndarray, Zs:np.ndarray, Us:np.ndarray, dt:float, model:CarouselModel):
    """ Plots the mechanical moments as a timeseries
    Args:
      Xs[np.ndarray] -- State trajectory
      Us[np.ndarray] -- Control trajectory
      dt[float] -- Timestep
      model[CarouselWhiteBoxModel] -- The used carousel model instance
    """
    assert Xs.shape[0] == model.NX()
    assert Us.shape[0] == model.NU()
    assert Xs.shape[1] == Us.shape[1] + 1
    N = Us.shape[1]
    tAxis = dt * np.arange(N+1)

    p = model.p0()


    # Compute all moments in the joint-frame
    Ms_gravity = np.zeros((3,N+1))
    Ms_elevator = np.zeros((3,N+1))
    Ms_aileron  = np.zeros((3,N+1))
    Ms_pitchdamping = np.zeros((3,N+1))
    Ms_elevationdamping = np.zeros((3,N+1))
    Ms_jointconstraint = np.zeros((3,N+1))
    Ms_total = np.zeros((3,N+1))

    for k in range(N+1):
        x = Xs[:,k]
        z = Zs[:,k]
        Ms_gravity[:,k]          = model.rotateView(x,1,3,model.M_gravity(x,p)).full().flatten()
        Ms_elevator[:,k]         = model.rotateView(x,1,3,model.M_Elevator(x,p)).full().flatten()
        Ms_aileron[:,k]          = model.rotateView(x,1,3,model.M_Aileron(x,p)).full().flatten()
        Ms_pitchdamping[:,k]     = model.rotateView(x,1,3,model.M_PitchDamping(x,p)).full().flatten()
        Ms_elevationdamping[:,k] = model.rotateView(x,1,3,model.M_ElevationDamping(x,p)).full().flatten()
        Ms_jointconstraint[:,k]  = model.rotateView(x,1,3,model.M_JointConstraint(x,z,p)).full().flatten()
        Ms_total[:,k]  = Ms_gravity[:,k]
        Ms_total[:,k] += Ms_elevator[:,k] + Ms_aileron[:,k]
        Ms_total[:,k] += Ms_pitchdamping[:,k] + Ms_elevationdamping[:,k]
        Ms_total[:,k] += Ms_jointconstraint[:,k]

        # Plot all moments component-wise
    fig,ax = plt.subplots(3,1)
    plt.suptitle("Mechanical moments in joint frame")

    plot_string = ['x','y','z']

    for k in range(3):
        plt.sca(ax[k])
        plt.title(r"$"+plot_string[k]+"$-optimal_control_problems")
        plt.ylabel(r"$"+plot_string[k]+"$-Moment [Nm]")
        plt.plot(tAxis, Ms_gravity[k,:], label="Gravity")
        plt.plot(tAxis, Ms_elevator[k,:], label="Elevator")
        plt.plot(tAxis, Ms_aileron[k,:], label="Wing")
        plt.plot(tAxis, Ms_pitchdamping[k,:], label="Pitch damping")
        plt.plot(tAxis, Ms_elevationdamping[k,:], label="Elevation damping")
        plt.plot(tAxis, Ms_jointconstraint[k,:], label="Joint constraint")
        plt.plot(tAxis, Ms_total[k,:], label="Total sum")
        plt.grid()
    plt.xlabel(r"Time $t$")
    leg = plt.legend(loc='best')
    leg.set_draggable(True)



""" =========================================       FANCY PLOTS       ============================================= """


def plotFlightTrajectory(Xs:np.ndarray, model:CarouselModel, roll_ref:np.ndarray = None):
    """ Plots the positions and an abstract carousel in 3d
    Args:
      Xs[np.ndarray] -- State trajectory
      model[CarouselWhiteBoxModel] -- The used carousel model instance
    """
    assert Xs.shape[0] == model.NX()
    N = Xs.shape[1] - 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    p = model.p0()


    # Make the grid TODO
    pos_COM = np.array([ model.pos_COM(Xs[:,k],p).full() for k in range(N) ])
    pos_cs4 = np.array([ model.pos_cs4(Xs[:,k],p).full() for k in range(N) ])
    pos_cs3 = np.array([ model.pos_cs3(Xs[:,k],p).full() for k in range(N) ])

    ax.plot(pos_COM[:,0,0], pos_COM[:,1,0], -pos_COM[:,2,0], linestyle='solid')
    ax.plot(pos_cs4[:,0,0], pos_cs4[:,1,0], -pos_cs4[:,2,0], linestyle='dashed')
    ax.plot(pos_cs3[:,0,0], pos_cs3[:,1,0], -pos_cs3[:,2,0], linestyle='dotted')

    # Get roll reference (fake) states and plot roll reference
    if roll_ref is not None:
        Xs_fake = Xs
        Xs_fake[0,:] = roll_ref.full().flatten()
        pos_cs4_ref = np.array([ model.pos_cs4(Xs_fake[:,k],p).full() for k in range(N) ])
        ax.plot(pos_cs4_ref[:,0,0], pos_cs4_ref[:,1,0], -pos_cs4_ref[:,2,0], linestyle='dashed')

    # Plot connecting lines
    for k in range(N):
        ax.plot(
            [pos_cs4[k,0,0], pos_cs3[k,0,0]],
            [pos_cs4[k,1,0], pos_cs3[k,1,0]],
            [-pos_cs4[k,2,0], -pos_cs3[k,2,0]],
            alpha=0.5,
            color='grey'
        )
        ax.plot(
            [pos_COM[k,0,0], pos_cs4[k,0,0]],
            [pos_COM[k,1,0], pos_cs4[k,1,0]],
            [-pos_COM[k,2,0], -pos_cs4[k,2,0]],
            alpha=0.5,
            color='grey'
        )
    ax.auto_scale_xyz([-3,3],[-3,3],[-3,3])


