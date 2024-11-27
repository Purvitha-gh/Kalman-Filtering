Kalman-Filtering using python
STEP 1: Import Libraries
First, we need to import the necessary libraries. We'll use numpy for numerical operations and matplotlib for plotting.
import numpy as np
import matplotlib.pyplot as plt
STEP 2: Define the System
We need to define the system we're modeling. This includes the state transition matrix (A), control matrix (B), measurement matrix (H), process noise covariance (Q), and measurement noise covariance (R).
dt = 1.0  # time step
A = np.array([[1, dt], [0, 1]])  # state transition matrix
B = np.array([[0.5 * dt**2], [dt]])  # control input matrix
H = np.array([[1, 0]])  # measurement matrix
Q = np.array([[1, 0], [0, 3]])  # process noise covariance
R = np.array([[10]])  # measurement noise covariance
u = np.array([[0]])  # control vector (acceleration)
STEP 3: Initialize the State and Covariance
We initialize the state vector (x) and the error covariance matrix (P).
x = np.array([[0], [1]])  # initial state (position, velocity)
P = np.array([[1000, 0], [0, 1000]])  # initial error covariance
STEP 4: Simulate Measurements
To simulate the process, we generate noisy measurements for a number of time steps.
np.random.seed(0)
num_steps = 50
true_positions = []
measurements = []

for _ in range(num_steps):
    # Simulate the true position
    true_position = x[0, 0] + x[1, 0] * dt
    true_positions.append(true_position)
    
    # Generate a noisy measurement
    z = np.dot(H, x) + np.random.normal(0, R[0, 0] ** 0.5)
    measurements.append(z[0, 0])
    
    # Update the state (true movement)
    x = np.dot(A, x) + np.dot(B, u)
    STEP 5: Implement the Kalman Filter
We implement the prediction and update steps of the Kalman filter.
estimated_positions = []
x = np.array([[0], [1]])  # re-initialize the state
P = np.array([[1000, 0], [0, 1000]])  # re-initialize the error covariance

for z in measurements:
    # Prediction step
    x = np.dot(A, x) + np.dot(B, u)
    P = np.dot(A, np.dot(P, A.T)) + Q
    
    # Update step
    S = np.dot(H, np.dot(P, H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
    y = z - np.dot(H, x)
    x = x + np.dot(K, y)
    P = P - np.dot(K, np.dot(H, P))
    
    estimated_positions.append(x[0, 0])
STEP 6 : Plot the Results
Finally, we plot the true positions, measurements, and estimated positions.
plt.plot(true_positions, label='True Position')
plt.plot(estimated_positions, label='Estimated Position')
plt.scatter(range(num_steps), measurements, color='red', label='Measurements', marker='+')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter Position Estimation')
plt.show()

