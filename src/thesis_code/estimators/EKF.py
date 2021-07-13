import numpy as np

## 
# @brief Implementation of a time-discrete Extended Kalman Filter
class EKF:
    
    ## 
    # @brief Initialization of the time discrete EKF
    # @param f State transition equation
    # @param F Linearization of f
    # @param h Output equation
    # @param H Linearization of h
    # @param x0 Initial state estimate
    # @param P0 Initial state covariance matrix
    # @param Q Process noise covariance matrix
    # @param R Measurement noise covariance matrix
    def __init__ (self, f, F, h, H, x0, P0, Q, R):
        # Get dimensions
        self.nx = x0.shape[0]
        self.ny = h(x0, np.zeros([h.size1_in(1),1])).shape[0]

        # Assign members
        self.Q = Q  # Process noise
        self.R = R  # Measurement noise
        self.P = P0 # Initial covariance
        self.f = f  # State function
        self.F = F  # Jacobian of f
        self.h = h  # Output function
        self.H = H  # Jacobian of h
        self.xhat = x0 # Initial state
        self.I = np.eye(self.nx,self.nx) # Unit matrix
        self.t = 0 # Initial timestamp
        self.xhat_predicted = x0 # Intermediate variable
        self.P_predicted = P0 # Intermediate variable
        
    ##
    # @brief Estimates an entire trajectory
    # @param ys A vector of measurement vectors
    # @param us A vector of control vectors
    # @param ts A vector of timestamps
    '''
    def estimateTrajectory(self, ys, us, ts):
        # Set prediction horizon
        N = us.shape[1]
        
        # Initialize states and covariances
        xs = np.zeros([self.nx, N])
        Ps = np.zeros([self.nx, self.nx, N])
        
        # Estimate trajectory
        for i in np.arange(0,us.shape[1]):
            # Estimate the i-th step
            step = self(ys[:,i], us[:,i], ts[i])
            xs[:,i]   = step[0].squeeze()
            Ps[:,:,i] = step[1]

            if np.any(np.isnan(xs)):
                print('A value in x is NaN')
            
        # Return estimates and covariances
        return xs, Ps
    '''

    ##
    # @brief Prediction step of the EKF
    # @param u The applied control vector
    # @param t The timestamp of the control
    def predict(self, u, t):
        # Get timestep
        dt = t - self.t
        self.t = t

        # Predict next state
        self.xhat_predicted = self.f(self.xhat,u)

        # Predict uncertainty
        Fk = self.F(self.xhat,u)
        self.P_predicted = np.dot(Fk, np.dot(self.P, Fk.T)) + self.Q
        
        # Return prediction
        return self.xhat_predicted, self.P_predicted

    ##
    # @brief Correction step of the EKF
    # @param y The measurement resulting from the last control
    ''' Added u_cur '''
    def correct(self, u, y):
        # Get predicted measurement and measurement matrix
        hk = self.h(self.xhat_predicted, u)
        Hk = self.H(self.xhat_predicted, u)

        # Compute Kalman gain 
        nom = np.dot(self.P_predicted, Hk.T)
        den = np.dot(Hk, np.dot(self.P_predicted, Hk.T)) + self.R
        K = np.dot(nom, np.linalg.inv(den))

        # Compute corrected state and covariance
        self.xhat = self.xhat_predicted + np.dot(K, (y - hk))
        self.P = np.dot((self.I - np.dot(K, Hk)), self.P_predicted)
        
        # Return corrected state and covariance
        return self.xhat, self.P

    ##
    # @brief Convenience Function. Predicts and Corrects in one step
    # @param u_prev The previous control vector
    # @param y_cur The current measurement vector
    # @param timestamp The timestamp of the control
    def __call__(self, u_prev, u_cur, y_cur, timestamp):
        self.predict(u_prev, timestamp)
        self.correct(u_cur, y_cur)
        return self.xhat, self.P
