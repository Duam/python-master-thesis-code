import casadi as cas
from casadi import DM, horzcat, blockcat
import numpy as np
from thesis_code.model import CarouselModel
import matplotlib.pyplot as plt


def plotStates(
        model: CarouselModel,
        dt: float,
        Xs_ref: DM = None,
        Us_ref: DM = None,
        Xs_sim: DM = None,
        Us_sim: DM = None,
        Xs_est: DM = None
) -> None:
    """ Plots the states and input as a timeseries
    :param model: The carousel model instance
    :param dt: Timestep in seconds
    :param Xs_ref: State trajectory reference
    :param Us_ref: Control trajectory reference
    :param Xs_sim: State trajectory actual
    :param Us_sim: Control trajectory actual
    :param Xs_est: State trajectory estimate
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


def plotStates_withRef(
        Xs: np.ndarray,
        Xs_ref: np.ndarray,
        Us: np.ndarray,
        Us_ref: np.ndarray,
        dt: float,
        model: CarouselModel
) -> None:
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


def plotFlightTrajectory(
        Xs: np.ndarray,
        model: CarouselModel,
        roll_ref: np.ndarray = None
) -> None:
    """Plots the positions and an abstract carousel in 3d
    :param Xs: The state trajectory.
    :param model: The used carousel model instance.
    :param roll_ref: The roll reference trajectory.
    :returns: Nothing
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


