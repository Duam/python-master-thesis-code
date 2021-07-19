# Measurement- and Process-noise

For "Idle": The carousel is not moving, the half-wing is hanging down. It's also pitched down
due to some unknown/unmodelled downward torque.

For "Rotating". The carousel is rotating at 2 rad per second. The half-wing is being forced 
outwards by centrifugal forces and flies almost horizontally.

The data is measured and logged in the files
* Idle: "./data/zcmlog-2019-09-18-idle-data-for-variance-analysis.0001"
* Rotating: "./data/zcmlog-2019-09-18-mhe-test-with-measured-angles.0000"

Preprocessing: Reordering for ascending timestamps.

#### Accelerometer (VE1, carousel arm)

* Mean:
  * Idle: [-0.31489269  0.27179736  9.3450156 ]
  * Rotating: [-0.34268266 -6.22943987  9.310575  ] 

* Variance:
  * Idle: [3.34286974e-05 3.66999911e-05 7.39442451e-05]
  * Rotating: [0.00991952 0.00788354 0.00081035]
   
#### Acceleromter (VE2, half-wing)

* Mean:
  * Idle: [ 6.49089603e-03 -9.64159142e+00  4.60983727e-01]
  * Rotating: [ 0.67506635 -15.47344897   6.60201611]

* Variance:
  * Idle: [3.01953954e-05 3.25337061e-05 6.58931721e-05]
  * Rotating: [0.09296443 0.04286172 0.15367722]

#### Gyroscope (VE1, carousel arm)
   
* Mean:
  * Idle: [-0.01401519  0.01072913  0.00047893]
  * Rotating: [-0.01489896  0.0070821   2.00380106]
  
* Variance:
  * Idle: [3.10483711e-06 3.19853932e-06 3.72359904e-06]
  * Rotating: [4.08848172e-06 6.53497466e-06 1.56970396e-05] 

#### Gyroscope (VE2, half-wing)
   
* Mean:
  * Idle: [-0.01735971  0.01450223  0.01142167]
  * Rotating: [-0.18439707 -0.42622639  1.95446436]
  
* Variance:
  * Idle: [3.23053703e-06 3.11135387e-06 2.93546719e-06]
  * Rotating:  [0.00592503 0.00236551 0.00118049]

#### Elevation sensor (roll) (VE1, carousel arm)

* Mean:
  * Idle: [1.49814677],
  * Rotating: [2.78920058]
  
* Variance:
  * Idle: [0.]
  * Rotating: [0.00135375]
   
#### Rotation sensor (pitch) (VE1, carousel arm)
   
* Mean:
  * Idle: [1.69694662]
  * Rotating: [0.72029013]

* Variance: 
  * Idle: [2.42926759e-09]
  * Rotating: [0.00060121]
   
## Conclusions

* IMU 2 orientation: from accelerometers mean data: Idle orientation shows g on negative y-axis. Indicates that y_IMU = -y_O4. 
Rotating mode: g shows on the positive z-axis. Indicates that z_IMU = -z_4. I conclude that x_IMU = -x_O4.
* Corresponding rotation matrix: R = -1 * eye(3)
* Idle variances:
  * Accelerometers, Gyroscopes have neglibily small measurement noise  (1e-5, 1e-6)
  * Elevation, Rotation sensors have even smaller measurement noise (1e-9 to 0.)
  * Conclusion: Pure measurement noise is virtually nonexistent and all disturbances are probably process noise
* Rotating variances on half-wing:
  * Accelerometer: Pretty big noise: 0.05 to 0.15
  * Gyroscope: Medium noise: 1e-3 to 6e-3
  * Elevation: about 1e-3
  * Rotation: about 6e-4
  * Conclusion (angle sensors): Sensors are extremely accurate, so the variances that we see here are probably pure process
    noise
  * Conclusion (acc, gyro): The noise that we see here is probably a mix of:
    * Torques produced by the angle disturbances itself, measurable with the angle sensors.
    * Vibrations in the half-wing that can not be measured. This is our actual in-flight measurement noise.
    * It probably takes a lot of effort separating these two noise-sources.
    * Because the gyroscope measurements are still relatively accurate, we can pretend that all the variances we see
      are actually measurement noises (for simplicity).
    * If we're going to pre-filter acceleration and angular velocity, we need to re-compute the variances.
