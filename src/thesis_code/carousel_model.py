import numpy as np
import casadi as cas
from casadi import cos, sin, inv, SX, DM, mtimes, Function, vertcat, horzcat, blockcat, jacobian, hessian, sqrt, acos, norm_2
from thesis_code.utils.ParametricNLP import ParametricNLP

va = SX.sym('va',3)
vb = SX.sym('vb',3)
cross_product = Function('cross_product', [va,vb], [
    vertcat(
        va[1]*vb[2] - va[2]*vb[1],
        va[2]*vb[0] - va[0]*vb[2],
        va[0]*vb[1] - va[1]*vb[0]
    )
])

angle = SX.sym('angle',1)
RotMatX = Function('RotMatX', [angle], [
    blockcat([[ 1,          0,           0 ],
              [ 0, cos(angle), -sin(angle) ],
              [ 0, sin(angle),  cos(angle) ]])
])
RotMatY = Function('RotMatY', [angle], [
    blockcat([[  cos(angle), 0, sin(angle) ],
              [  0,          1,          0 ],
              [ -sin(angle), 0, cos(angle) ]])
])
RotMatZ = Function('RotMatZ', [angle],[
    blockcat([[ cos(angle), -sin(angle), 0 ],
              [ sin(angle),  cos(angle), 0 ],
              [        0,         0, 1 ]])
])


class CarouselWhiteBoxModel:

    @staticmethod
    def get_vector_defined_in_x_viewed_from_y(
            frame_x: int,
            frame_y: int,
            vector: SX,
            psi: SX,
            phi: SX,
            theta: SX):
        """ Rotates a vector from one reference frame into another. This algorithm looks at the source
        and destination reference systems and applies the corresponding transformations one after another.
        Args:
          X[int] -- The source reference frame (1-4)
          Y[int] -- The destination reference frame (1-4)
          vector[SX] -- The vector to be viewed
          psi[float] -- The carousel arm angle
          phi[float] -- The airplanes elevation angle
          theta[float] -- The airplane's pitch angle
        Returns:
          The vector (defined in coordinate system X) as viewed from coordinate system Y.
        """
        assert 1 <= frame_x <= 4 and 1 <= frame_y <= 4, "Source/Dest must be one of the four frames."
        source = frame_x
        dest = frame_y

        # R^1_2 = View a vector (def. in carousel frame) in arm frame
        RotSys_1to2 = lambda psi: blockcat([
            [  cos(psi), sin(psi), 0 ],
            [ -sin(psi), cos(psi), 0 ],
            [        0,         0, 1 ]])
        # R^2_3 = View a vector (def. in arm frame) from joint frame
        RotSys_2to3 = lambda phi: blockcat([
            [ 1,         0,        0 ],
            [ 0,  cos(phi), sin(phi) ],
            [ 0, -sin(phi), cos(phi) ]])
        # R^3_4 = View a vector (def. in joint frame) from plane frame
        RotSys_3to4 = lambda theta: blockcat([
            [ cos(theta), 0, -sin(theta) ],
            [ 0,          1,           0 ],
            [ sin(theta), 0,  cos(theta) ]])

        if dest == 1:
            if source > 3: vector = mtimes([RotSys_3to4(theta).T,  vector])
            if source > 2: vector = mtimes([RotSys_2to3(phi).T,    vector])
            if source > 1: vector = mtimes([RotSys_1to2(psi).T,    vector])
        elif dest == 2:
            if source > 3: vector = mtimes([RotSys_3to4(theta).T,  vector])
            if source > 2: vector = mtimes([RotSys_2to3(phi).T,    vector])
            if source < 2: vector = mtimes([RotSys_1to2(psi),      vector])
        elif dest == 3:
            if source > 3: vector = mtimes([RotSys_3to4(theta).T,  vector])
            if source < 2: vector = mtimes([RotSys_1to2(psi),      vector])
            if source < 3: vector = mtimes([RotSys_2to3(phi),      vector])
        elif dest == 4:
            if source < 2: vector = mtimes([RotSys_1to2(psi),      vector])
            if source < 3: vector = mtimes([RotSys_2to3(phi),      vector])
            if source < 4: vector = mtimes([RotSys_3to4(theta),    vector])
        return vector

    def get_position_defined_in_x_viewed_from_y(
            self,
            frame_x: int,
            frame_y: int,
            pos: SX,
            psi: SX,
            phi: SX,
            theta: SX):
        """get_PositionDefinedIn_X_ViewedFrom_Y Returns the given vector as viewed from a different frame.
        Args:
          X[int] -- The source reference frames (1-4)
          Y[int] -- The destination reference frame (1-4)
          pos[SX] -- The position vector to be viewed
          psi[SX] -- The carousel arm angle
          phi[SX] -- The airplane's elevation angle
          theta[float] -- The airplane's pitch angle
        Results:
          The vector 'pos' in a different reference frame.
        """
        assert 1 <= frame_x <= 4 and 1 <= frame_y <= 4, "Source/Dest must be one of the four frames."
        source = frame_x
        dest = frame_y

        # Shortcut to view vectors/directions from a different view
        transform = lambda start,end,v: self.get_vector_defined_in_x_viewed_from_y(start, end, v, psi, phi, theta, )

        if dest == 1:
            if source > 3: pos = self.pos_4_wrt_3 + transform(4,3,pos)
            if source > 2: pos = self.pos_3_wrt_2 + transform(3,2,pos)
            if source > 1: pos = self.pos_2_wrt_1 + transform(2,1,pos)
        elif dest == 2:
            if source > 3: pos = self.pos_4_wrt_3 + transform(4,3,pos)
            if source > 2: pos = self.pos_3_wrt_2 + transform(3,2,pos)
            if source < 2: pos = transform(1,2,pos-self.pos_2_wrt_1)
        elif dest == 3:
            if source > 3: pos = self.pos_4_wrt_3 + transform(4,3,pos)
            if source < 2: pos = transform(1,2,pos-self.pos_2_wrt_1)
            if source < 3: pos = transform(2,3,pos-self.pos_3_wrt_2)
        elif dest == 4:
            if source < 2: pos = transform(1,2,pos-self.pos_2_wrt_1)
            if source < 3: pos = transform(2,3,pos-self.pos_3_wrt_2)
            if source < 4: pos = transform(3,4,pos-self.pos_4_wrt_3)
        return pos


    @staticmethod
    def getConstants():
        return {
            'g': 9.81, # Gravity (m/s^2)
            'rho': 1.22, # Air density (kg/m^3)
            'wind_speed': [0., 0., 0.],
            'AoA_stall': 20.0 * (2*np.pi/360),
            'safety_factor_stall': 0.93,
            'carousel_speed': -2.0, # Carousel angular velocity in rad/s
            'carousel_tau': 1.0, # Carousel speed PT1 constant
            'roll_sensor_offset': np.pi,
            'pitch_sensor_offset': 0.9265,
            'max_roll':  135 * (2*np.pi/360),
            'min_roll': -135 * (2*np.pi/360),
            'max_pitch':  53 * (2*np.pi/360),
            'min_pitch': -53 * (2*np.pi/360),
        }

    @staticmethod
    def getDefaultParams():
        return {
            'm': 1.4, # Airplane mass (kg)
            'Ixx_COM': 2.5, # Inertia tensor wrt. COM around x-axis
            'Iyy_COM': 0.3, # Inertia tensor wrt. COM around y-axis
            'Izz_COM': 1.0, # Inertia tensor wrt. COM around z-axis
            'pos_center_of_mass_wrt_4': [0.03, 0., 0.],
            'pos_aileron_aerodynamic_center_wrt_4': [0., 0.5, 0.],
            'pos_elevator_aerodynamic_center_wrt_4': [-0.56, 0., -0.1],
            'pos_imu_wrt_4': [0.15, 0., -0.05],
            'rot_imu_wrt_4': [0., np.pi, 0.],
            'pos_4_wrt_3': [0., 0.5, 0.],
            'pos_3_wrt_2': [0., 2.05, 0.],
            'S_A': 0.2, # Main wing area (m^2)
            'S_E': 0.08, # Elevator wing area (m^2)
            'C_LA_0': 2.5,    # Wing lift coeff. for AoA = 0
            'C_LA_max': 3.75,  # Wing lift coeff. for AoA = max
            'C_DA_0': 0.25,    # Wing drag coeff. for AoA = 0
            'C_DA_max': 0.375,  # Wing drag coeff. for AoA = max
            'C_LE_0': 0.3,    # Elevator lift coeff. for AoA = 0
            'C_LE_max': 0.75,  # Elevator lift coeff. for AoA = max
            'C_DE_0': 0.03,  # Elevator drag coeff. for AoA = 0
            'C_DE_max': 0.075, # Elevator drag coeff. for AoA = max
            'elevator_deflector_gain': -(15 * (2*np.pi/360)), # Gain of elevator trim tab (max tab angle)
            'elevator_deflector_tau': 0.2, # Time constant of elevator trim tab
            'mu_phi': 0.01, # Elevation friction coeff. (kg * m^2 / s)
            'mu_theta': 0.15, # Pitch friction coeff. (kg * m^2 / s)
        }

    def writeParamsToJson(self, filename:str):
        import json
        with open(filename, 'w') as outfile:
            json.dump(self.params, outfile, indent=4)

    def NX(self):
        return self.x_aug_sym.rows()

    def NZ(self):
        return self.z_sym.rows()

    def NU(self):
        return self.u_aug_sym.rows()

    def NP(self):
        return self.p_sym.rows()

    def NY(self):
        return self.out_sym.rows()


    def __init__(self, params=None, with_angle_output=False, with_imu_output=True):
        # Model parameters
        self.params = params if params != None else self.getDefaultParams()
        self.constants = self.getConstants()

        # --- Preparation ---

        # Create differential algebraic equation
        self.dae = cas.DaeBuilder()
        # .. differential states:
        phi      = self.dae.add_x('plane_roll')
        theta    = self.dae.add_x('plane_pitch')
        psi      = self.dae.add_x('carousel_yaw')
        phidot   = self.dae.add_x('plane_roll_vel')
        thetadot = self.dae.add_x('plane_pitch_vel')
        psidot   = self.dae.add_x('carousel_yaw_vel')
        delta_e  = self.dae.add_x('elevator_deflection_angle')
        # .. inputs
        delta_e_setp = self.dae.add_u('elevator_deflection_angle_setpoint')
        psidot_setp  = self.dae.add_u('carousel_speed_setpoint')
        # .. algebraic variables and dummies
        phiddot   = self.dae.add_z('plane_roll_acc')
        thetaddot = self.dae.add_z('plane_pitch_acc')
        z_constr  = self.dae.add_z('joint_constraint_moment')
        # .. parameters (constant in time)
        param = {}
        for key in self.getDefaultParams().keys():
            param[key] = self.dae.add_p(key, cas.vertcat(np.array(self.getDefaultParams()[key]).flatten()).rows())

        for key in self.getConstants().keys():
            if key in param.keys():
                print("CarouselWhiteBoxModel: Key " + key + " found in constants and parameters. ABORT.")
                quit(0)
            else:
                param[key] = self.getConstants()[key]

        # Some wrappers
        x = cas.vertcat(*self.dae.x)
        z = cas.vertcat(*self.dae.z)
        u = cas.vertcat(*self.dae.u)
        p = cas.vertcat(*self.dae.p)

        # Create shorthand functions
        self.rotateView    = lambda x,src,dest,vec: self.get_vector_defined_in_x_viewed_from_y(src, dest, vec, x[2],
                                                                                               x[0], x[1], )
        self.transformView = lambda x,src,dest,pos: self.get_position_defined_in_x_viewed_from_y(src, dest, pos, x[2],
                                                                                                 x[0], x[1])

        # Create generalized coordinate vectors
        q     = vertcat(phi,    theta,    psi)
        qdot  = vertcat(phidot, thetadot, psidot)
        #qddot = vertcat(phiddot, thetaddot, psiddot)

        qhat = vertcat(phi, theta)
        qhatdot = vertcat(phidot, thetadot)
        qhatddot = vertcat(phiddot, thetaddot)

        # Some important data
        self.pos_COM_wrt_4 = param['pos_center_of_mass_wrt_4']
        self.pos_ACA_wrt_4 = param['pos_aileron_aerodynamic_center_wrt_4']
        self.pos_ACE_wrt_4 = param['pos_elevator_aerodynamic_center_wrt_4']
        self.pos_IMU_wrt_4 = param['pos_imu_wrt_4']
        self.pos_4_wrt_3   = param['pos_4_wrt_3']
        self.pos_3_wrt_2   = param['pos_3_wrt_2']
        self.pos_2_wrt_1   = cas.DM([0,0,0])
        #self.I_COM         = param['I_COM'].reshape((3,3))
        self.I_COM         = cas.diag(vertcat(param['Ixx_COM'],param['Iyy_COM'],param['Izz_COM']))
        self.F_gravity     = cas.vertcat(cas.MX.zeros(2),param['m']*param['g'])

        # Position and velocity getters
        self.pos_COM = cas.Function('pos_COM', [x,p], [self.transformView(x,4,1,self.pos_COM_wrt_4)])
        self.vel_COM = cas.Function('vel_COM', [x,p], [mtimes([jacobian(self.pos_COM(x,p),q),qdot])])
        self.pos_ACA = cas.Function('pos_ACA', [x,p], [self.transformView(x,4,1,self.pos_ACA_wrt_4)])
        self.vel_ACA = cas.Function('vel_ACA', [x,p], [mtimes([jacobian(self.pos_ACA(x,p),q),qdot])])
        self.pos_ACE = cas.Function('pos_ACE', [x,p], [self.transformView(x,4,1,self.pos_ACE_wrt_4)])
        self.vel_ACE = cas.Function('vel_ACE', [x,p], [mtimes([jacobian(self.pos_ACE(x,p),q),qdot])])
        self.pos_cs4 = cas.Function('pos_cs4', [x,p], [self.transformView(x,3,1,self.pos_4_wrt_3)])
        self.vel_cs4 = cas.Function('vel_cs4', [x,p], [mtimes([jacobian(self.pos_cs4(x,p),q),qdot])])
        self.pos_cs3 = cas.Function('pos_cs3', [x,p], [self.transformView(x,2,1,self.pos_3_wrt_2)])
        self.vel_cs3 = cas.Function('vel_cs3', [x,p], [mtimes([jacobian(self.pos_cs3(x,p),q),qdot])])

        # Direction getters
        self.dir_wing = cas.Function('dir_wing', [x,p], [self.rotateView(x,4,1,cas.vertcat(0,1,0))])
        self.dir_nose = cas.Function('dir_nose', [x,p], [self.rotateView(x,4,1,cas.vertcat(1,0,0))])


        # Angular velocity getters (body frame)
        self.omega = cas.Function('omega_wrt_body', [x,p], [
            psidot   * self.rotateView(x,1,4,cas.vertcat(0,0,1))
            + phidot   * self.rotateView(x,2,4,cas.vertcat(1,0,0))
            + thetadot * self.rotateView(x,4,4,cas.vertcat(0,1,0))
        ])

        # Compute current position, velocity and acceleration in world frame
        pos_COM = self.pos_COM(x,p)
        vel_COM = self.vel_COM(x,p)
        pos_ACA = self.pos_ACA(x,p)
        vel_ACA = self.vel_ACA(x,p)
        pos_ACE = self.pos_ACE(x,p)
        vel_ACE = self.vel_ACE(x,p)
        pos_cs4 = self.pos_cs4(x,p)
        vel_cs4 = self.vel_cs4(x,p)
        pos_cs3 = self.pos_cs3(x,p)
        vel_cs3 = self.vel_cs3(x,p)
        omega   = self.omega(x,p)

        # --- Compute internal moments in joint frame ---
        # Compute absolute angular momentum in world frame
        L = cas.MX.zeros(3)
        L += self.rotateView(x,4,1, mtimes([self.I_COM, omega]))
        L += param['m'] * cross_product(pos_COM-pos_cs3,vel_COM)
        self.L = cas.Function('L', [x], [L])

        # Compute change in absolute angular momentum wrt. time in world frame
        #dL = mtimes([ jacobian(L,q), qdot ]) + mtimes([ jacobian(L,qdot), qddot ])
        dL = mtimes([ jacobian(L,q), qdot ]) + mtimes([ jacobian(L,qhatdot), qhatddot ])
        self.dL = cas.Function('dL', [x,z], [dL])

        # Compute M
        M_int = cas.MX.zeros(3)
        M_int += dL
        M_int += param['m'] * cross_product(vel_cs3, vel_COM)
        M_int_wrt_3 = self.rotateView(x,1,3,M_int)
        self.M_int = cas.Function('M_int', [x,z], [M_int])

        # --- Compute external moments in joint frame ---
        # Gravity
        M_gravity       = cross_product(pos_COM-pos_cs3, self.F_gravity)
        M_gravity_wrt_3 = self.rotateView(x,1,3,M_gravity)

        # Aileron Lift Moment
        """weff_ACA = -vel_ACA
        alpha_ACA = cas.atan2(-weff_ACA[2],-weff_ACA[0])
        CL_ACA = params['C_LA_0'] + params['C_LA_max'] * cas.sin(2*alpha_ACA)
        CD_ACA = param['C_LA_0'] / param['LiftToDragRatio_A'] + 0.5 * param['C_DA_max'] * (1 - cas.cos(2*alpha_ACA))
        FL_ACA = -0.5 * param['S_A'] * C_L * w_eff_norm2 * cross_product(dir_wing, w_eff)
        FD =  0.5 * param['S_A'] * C_D * w_eff_norm2 * w_eff
        """

        AileronData     = self.computeAileronAerodynamicsData(x,param)
        F_AileronLift   = AileronData['F_Lift']
        F_AileronDrag   = AileronData['F_Drag']
        M_Aileron       = cross_product(pos_ACA-pos_cs3, F_AileronLift+F_AileronDrag)
        M_Aileron_wrt_3 = self.rotateView(x,1,3,M_Aileron)


        # Elevator Lift Moment
        ElevatorData     = self.computeElevatorAerodynamicsData(x,param)
        F_ElevatorLift   = ElevatorData['F_Lift']
        F_ElevatorDrag   = ElevatorData['F_Drag']
        M_Elevator       = cross_product(pos_ACE-pos_cs3, F_ElevatorLift+F_ElevatorDrag)
        M_Elevator_wrt_3 = self.rotateView(x,1,3,M_Elevator)

        # Joint damping (elevation, pitch) and joint constraint moment
        M_ElevationDamping_wrt_3 = vertcat(1,0,0) * phidot * (-param['mu_phi'])
        M_PitchDamping_wrt_3     = vertcat(0,1,0) * thetadot * (-param['mu_theta'])
        M_JointConstraint_wrt_3  = vertcat(0,0,1) * z_constr

        # Moment functions
        self.M_gravity = cas.Function('M_gravity', [x,p], [M_gravity])
        self.M_Aileron = cas.Function('M_Aileron', [x,p], [M_Aileron])
        self.M_Elevator  = cas.Function('M_Elevator', [x,p], [M_Elevator])
        self.M_ElevationDamping = cas.Function('M_ElevationDamping', [x,p], [self.rotateView(x,3,1,M_ElevationDamping_wrt_3)])
        self.M_PitchDamping = cas.Function('M_PitchDamping', [x,p], [self.rotateView(x,3,1,M_PitchDamping_wrt_3)])
        self.M_JointConstraint = cas.Function('M_JointConstraint', [x,z,p], [self.rotateView(x,3,1,M_JointConstraint_wrt_3)])

        # --- Compute sum of moments ---
        M_ext_wrt_3  = cas.MX.zeros(3)
        M_ext_wrt_3 += M_gravity_wrt_3
        M_ext_wrt_3 += M_Aileron_wrt_3
        M_ext_wrt_3 += M_Elevator_wrt_3
        M_ext_wrt_3 += M_PitchDamping_wrt_3
        M_ext_wrt_3 += M_ElevationDamping_wrt_3
        M_ext_wrt_3 += M_JointConstraint_wrt_3

        # -- Compute elevator deflection dynamics --
        elevator_deflection_velocity  = cas.MX(0)
        elevator_deflection_velocity -= 1/param['elevator_deflector_tau'] * delta_e
        elevator_deflection_velocity += 1/param['elevator_deflector_tau'] * delta_e_setp

        # --- Formulate fully implicit DAE functions ---
        self.dae.add_ode('plane_roll_vel', phidot)
        self.dae.add_ode('plane_pitch_vel', thetadot)
        self.dae.add_ode('carousel_yaw_vel',  psidot)
        self.dae.add_ode('plane_roll_acc', phiddot)
        self.dae.add_ode('plane_pitch_acc', thetaddot)
        self.dae.add_ode('carousel_yaw_acc', 1/param['carousel_tau'] * (psidot_setp - psidot))
        self.dae.add_ode('elevator_deflection_vel', elevator_deflection_velocity)

        self.dae.add_alg('moments_x', M_ext_wrt_3[0] - M_int_wrt_3[0])
        self.dae.add_alg('moments_y', M_ext_wrt_3[1] - M_int_wrt_3[1])
        self.dae.add_alg('moments_z', M_ext_wrt_3[2] - M_int_wrt_3[2])

        # --- Formulate sensor outputs ---
        self.omega_IMU = cas.Function('omega_IMU', [x,p], [self.omega(x,p)])
        self.pos_IMU = cas.Function('pos_IMU', [x,p], [self.transformView(x,4,1,self.pos_IMU_wrt_4)])
        self.vel_IMU = cas.Function('vel_IMU', [x,p], [mtimes([jacobian(self.pos_IMU(x,p),q),qdot])])
        self.acc_IMU = cas.Function('acc_IMU', [x,z,p], [
            mtimes([jacobian(self.vel_IMU(x,p),q),qdot]) + mtimes([jacobian(self.vel_IMU(x,p),qhatdot),qhatddot])
            #mtimes([jacobian(self.vel_IMU(x,p),q),qdot]) + mtimes([jacobian(self.pos_IMU(x,p),qhat),qhatddot])
            - vertcat(0,0,param['g'])
        ])
        self.acc_IMU_wrt_4 = cas.Function('acc_IMU_wrt_4', [x,z,p], [self.rotateView(x,1,4,self.acc_IMU(x,z,p))])

        # Add outputs
        #offset_gyro = cas.vertcat(-0.01735971, 0.01450223, 0.01142167)
        Rx = RotMatX(param['rot_imu_wrt_4'][0])
        Ry = RotMatY(param['rot_imu_wrt_4'][1])
        #Rz = RotMatZ(param['rot_imu_wrt_4'][2])
        Rz = RotMatX(param['rot_imu_wrt_4'][2])
        #Ry = RotMatY(np.pi)

        #R_IMU = mtimes([Rz, Ry, Rx])
        #R_IMU = mtimes([Rx, Ry, Rz])
        #R_IMU = mtimes([Rz, Ry, Rx])
        #R_IMU = mtimes([Ry, Rx])
        #R_IMU = mtimes([Rz, Ry])
        #R_IMU = mtimes([Ry, Rz])
        R_IMU = mtimes([Rx, Ry]) # This one is the only one that really works well

        self.meas_gyro = cas.Function('meas_gyro', [x,p], [mtimes(R_IMU, self.omega_IMU(x,p))])
        self.meas_acc = cas.Function('meas_acc', [x,z,p], [mtimes(R_IMU, self.acc_IMU_wrt_4(x,z,p))])

        if with_angle_output:
            self.dae.add_y('carousel_roll_out', param['roll_sensor_offset'] - phi)
            self.dae.add_y('carousel_pitch_out', param['pitch_sensor_offset'] - theta)

        if with_imu_output:
            self.dae.add_y('plane_linear_acceleration', self.meas_acc(x,z,p))
            self.dae.add_y('plane_angular_velocity', self.meas_gyro(x,p))

        self.dae.scale_variables()
        self.dae.make_semi_explicit()

        # Convenience wrappers
        self.x_sym = x = cas.vertcat(*self.dae.x)
        self.z_sym = z = cas.vertcat(*self.dae.z)
        self.u_sym = u = cas.vertcat(*self.dae.u)
        self.p_sym = p = cas.vertcat(*self.dae.p)
        self.ode_sym = ode = cas.vertcat(*self.dae.ode)
        self.alg_sym = alg = cas.vertcat(*self.dae.alg)
        self.out_sym = out = cas.vertcat(*self.dae.ydef)

        # Create functionals
        self.ode_fun = cas.Function('ode', [x, z, u, p], [ode], ['x', 'z', 'u', 'p'], ['f']).expand()
        self.alg_fun = cas.Function('alg', [x, z, u, p], [alg], ['x', 'z', 'u', 'p'], ['g']).expand()
        self.out_fun = cas.Function('out', [x, z, u, p], [out], ['x', 'z', 'u', 'p'], ['h']).expand()

        # Augment carousel symbolics
        NX = 5
        NU = 1
        self.x_aug_sym = x = cas.vertcat(x[0], x[1], x[3], x[4], x[6]) # x = [roll,pitch,roll_rate,pitch_rate,deflection]
        self.u_aug_sym = u = cas.vertcat(u[0]) # u = [deflection_setpoint]
        args = [x,z,u,p]
        args_aug = [
            vertcat(x[0], x[1], 0.0, x[2], x[3], param['carousel_speed'], x[4]),
            z,
            vertcat(u[0], param['carousel_speed']),
            p
        ]

        # Re-evaluate dae
        ode_aug = self.ode_fun(*args_aug)
        ode_aug = cas.vertcat(ode[0],ode[1],ode[3],ode[4],ode[6])
        alg_aug = self.alg_fun(*args_aug)
        out_aug = self.out_fun(*args_aug)

        # Re-create functionals
        self.ode_aug_fun = cas.Function('ode_aug', args, [ode_aug], ['x','z','u','p'], ['f_aug'])
        self.alg_aug_fun = cas.Function('alg_aug', args, [alg_aug], ['x','z','u','p'], ['g'])
        self.out_aug_fun = cas.Function('out_aug', args, [out_aug], ['x','z','u','p'], ['h'])


    def u0(self):
        return cas.vertcat(
            0.5
        )

    def x0(self):
        return cas.vertcat(
            0.0, 0.0,# Angles
            0.0, 0.0,   # Angular velocities
            0.5, # Elevator deflection angle
        )

    def p0(self):
        return cas.vertcat(np.concatenate([np.array(val).flatten() for val in self.params.values()]))


    def computeAileronLiftAndDragCoefficients(self, alpha,param):
        CL_0    = param['C_LA_0']
        CL_max  = param['C_LA_max']
        CD_0    = param['C_DA_0']
        CD_max  = param['C_DA_max']
        CL = CL_0 + CL_max * cas.sin(2*alpha)
        CD = CD_0 + 0.5 * CD_max * (1 - cas.cos(alpha))
        return CL, CD


    def computeAileronAerodynamicsData(self, x, param):
        # Compute the effective wind wrt. aileron aerodynamic center
        vel = self.vel_ACA(x,cas.vertcat(*self.dae.p)) #wrt 1
        #vel_norm2 = norm_2(vel)
        dir_nose = self.rotateView(x,4,1,vertcat(1,0,0))
        dir_wing = self.rotateView(x,4,1,vertcat(0,1,0))
        w_eff = - vel # assumes zero wind
        #w_eff_norm2 = norm_2(w_eff)
        w_eff_norm2 = cas.sqrt(cas.mtimes([w_eff.T, w_eff]))

        # Project the effective wind into the body frame's x-z plane
        # Note: This also leads to the simplification in the lift direction's computation (norm_2(dir_lift) = w_eff_norm2)
        w_eff_proj = cas.mtimes([cas.diag(cas.DM([1,0,1])),self.rotateView(x,1,4,w_eff)])
        #w_eff_proj_norm2 = norm_2(w_eff_proj)
        w_eff_proj_norm2 = cas.sqrt(cas.mtimes([w_eff_proj.T, w_eff_proj]))
        alpha = cas.atan2(-w_eff_proj[2],-w_eff_proj[0])

        # Compute aerodynamic coefficients
        C_L, C_D = self.computeAileronLiftAndDragCoefficients(alpha,param)

        # Compute lift and drag force
        F_L = -0.5 * param['S_A'] * C_L * w_eff_norm2 * cross_product(dir_wing, w_eff)
        F_D =  0.5 * param['S_A'] * C_D * w_eff_norm2 * w_eff
        #F_L = -0.5 * self.params['S_A'] * C_L * w_eff_proj_norm2 * cross_product(dir_wing, w_eff_proj)
        #F_D = 0.5 * self.params['S_A'] * C_D * w_eff_proj_norm2 * w_eff_proj

        p = vertcat(*self.dae.p)
        self.alpha_A = cas.Function('alpha_E', [x,p], [alpha])
        self.CL_A = cas.Function('CL_A', [x,p], [C_L])
        self.CD_A = cas.Function('CD_A', [x,p], [C_D])
        self.FL_A = cas.Function('FL_A', [x,p], [F_L])
        self.FD_A = cas.Function('FD_A', [x,p], [F_D])

        return {
            'F_Lift': F_L,
            'F_Drag': F_D,
            'C_Lift': C_L,
            'C_Drag': C_D,
            'AoA': alpha,
            'wind_eff': w_eff,
            'dir_nose': dir_nose,
            'dir_wing': dir_wing
        }

    def computeElevatorLiftAndDragCoefficients(self, alpha, param):
        CL_0    = param['C_LE_0']
        CL_max  = param['C_LE_max']
        CD_0    = param['C_DE_0']
        CD_max  = param['C_DE_max']
        CL = CL_0 + CL_max * cas.sin(2*alpha)
        CD = CD_0 + 0.5 * CD_max * (1 - cas.cos(alpha))
        return CL, CD


    def computeElevatorAerodynamicsData(self, x, param):
        # Compute the effective wind wrt. elevator aerodynamic center
        vel = self.vel_ACE(x,cas.vertcat(*self.dae.p)) #wrt 1
        #vel_norm2 = norm_2(vel)

        # In our model, the elevator tab changes the orientation of the elevator
        delta_e = x[-1]
        elevator_angle = param['elevator_deflector_gain'] * (delta_e - 0.5)
        dir_elv_wrt_4 = vertcat(cas.cos(elevator_angle),0,-cas.sin(elevator_angle))
        dir_elv = self.rotateView(x,4,1,dir_elv_wrt_4)

        dir_nose = self.rotateView(x,4,1,vertcat(1,0,0))
        dir_wing = self.rotateView(x,4,1,vertcat(0,1,0))
        w_eff = - vel # assumes zero wind
        #w_eff_norm2 = norm_2(w_eff)
        w_eff_norm2 = cas.sqrt(cas.mtimes([w_eff.T, w_eff]))


        # Project the effective wind into the body frame's x-z plane
        # Note: This also leads to the simplification in the lift direction's computation (norm_2(dir_lift) = w_eff_norm2)
        w_eff_proj = cas.mtimes([cas.diag(cas.DM([1,0,1])),self.rotateView(x,1,4,w_eff)])
        #w_eff_proj_norm2 = norm_2(w_eff_proj)
        w_eff_proj_norm2 = cas.sqrt(cas.mtimes([w_eff_proj.T, w_eff_proj]))
        alpha = cas.atan2(-w_eff_proj[2],-w_eff_proj[0])
        #alpha += param['elevator_deflector_gain'] * (delta_e - 0.5)
        alpha += elevator_angle

        # Compute aerodynamic coefficients
        C_L, C_D = self.computeElevatorLiftAndDragCoefficients(alpha,param)

        # Compute lift and drag force
        F_L = 0.5 * param['S_E'] * C_L * w_eff_norm2 * cross_product(dir_wing, w_eff)
        F_D = 0.5 * param['S_E'] * C_D * w_eff_norm2 * w_eff
        #F_L = 0.5 * self.params['S_E'] * C_L * w_eff_proj_norm2 * cross_product(dir_wing, w_eff_proj)
        #F_D = 0.5 * self.params['S_E'] * C_D * w_eff_proj_norm2 * w_eff_proj

        p = vertcat(*self.dae.p)
        self.alpha_E = cas.Function('alpha_E', [x,p], [alpha])
        self.CL_E = cas.Function('CL_E', [x,p], [C_L])
        self.CD_E = cas.Function('CD_E', [x,p], [C_D])
        self.w_eff_E = cas.Function('w_eff_E', [x,p], [w_eff])
        self.FL_E = cas.Function('FL_E', [x,p], [F_L])
        self.FD_E = cas.Function('FD_E', [x,p], [F_D])

        return {
            'F_Lift': F_L,
            'F_Drag': F_D,
            'C_Lift': C_L,
            'C_Drag': C_D,
            'AoA': alpha,
            'wind_eff': w_eff,
            'dir_nose': dir_nose,
            'dir_wing': dir_wing
        }


    def get_steady_state (self):

        # Get carousel symbolics
        x,z,u,p = (self.x_aug_sym, self.z_sym, self.u_aug_sym, self.p_sym)
        ode = self.ode_aug_fun(x,z,u,p)
        alg = self.alg_aug_fun(x,z,u,p)
        out = self.out_aug_fun(x,z,u,p)

        # Create ode derivative
        dfdx_fun = Function('dfdx', [x,z,u,p], [jacobian(ode, x)])

        # Create determinant function
        s = SX.sym('s', 1)
        A = SX.sym('A', self.NX(), self.NX())
        det_fun = Function('det', [A], [cas.det(A)])

        # Create the rootfinder
        finder = ParametricNLP('steady_state_finder')
        finder.add_decision_var('x', (self.NX(),1))
        finder.add_decision_var('z', (self.NZ(),1))
        finder.add_parameter('xdot', (self.NX(),1))
        finder.add_parameter('u', (self.NU(),1))
        finder.add_parameter('p', (self.NP(),1))
        finder.bake_variables()
        x = finder.get_decision_var('x')
        z = finder.get_decision_var('z')
        xdot = finder.get_parameter('xdot')
        u = finder.get_parameter('u')
        p = finder.get_parameter('p')
        eps = 0

        # Set NLP cost
        residual_ode = xdot - self.ode_aug_fun(x,z,u,p)
        finder.set_cost(mtimes(residual_ode.T, residual_ode))

        # Set constraints
        # 1. Algebraic constraints satisfied
        finder.add_equality('alg', self.alg_aug_fun(x,z,u,p))
        # 2. Some variables in x, xdot, and z have the same meaning
        finder.add_equality('z0=xdot3', z[0] - xdot[3])
        finder.add_equality('z1=xdot4', z[1] - xdot[4])
        # 3. Make sure our steady state is feasible
        max_pitch = 53 * (2*np.pi/360)
        min_pitch = -53 * (2*np.pi/360)
        max_roll = 135 * (2*np.pi/360)
        min_roll = -135 * (2*np.pi/360)
        finder.add_inequality('max_roll', max_roll - x[0])
        finder.add_inequality('min_roll', x[0] - min_roll)
        finder.add_inequality('max_pitch', max_pitch - x[1])
        finder.add_inequality('min_pitch', x[1] - min_pitch)
        # 4. Make sure the steady state is stable
        finder.add_inequality('stable', - eps - det_fun(dfdx_fun(x, z, u, p)))
        finder.init(opts={'ipopt.print_level':0,'print_time': 0,'ipopt.sb': 'yes'})

        # Create guess and parameters
        finder_guess = finder.struct_w(0)
        finder_guess['x'] = DM( self.x0() )
        finder_guess['z'] = DM([0., 0., 3.])
        finder_param = finder.struct_p(0)
        finder_param['xdot'] = vertcat(0,0,0,0,0)
        finder_param['u'] = DM( self.u0() )
        finder_param['p'] = DM( self.p0() )

        # Solve the rootfinding problem
        sol, stats, sol_orig, lb, ub = finder.solve(finder_guess, finder_param)
        residual_sol = sol['f']
        x_sol = sol['w']['x']
        z_sol = sol['w']['z']

        print("Model steady state computed.")
        print("Model steady state: " + str(x_sol))

        return x_sol, z_sol, self.u0()
