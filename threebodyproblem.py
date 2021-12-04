# avoids annoying deprecation warnings with scipy
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# scipy and numpy used for vector functions and ODE solver
import scipy as sc
import numpy as num
# matplotlib used for animations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# Define some constants
# Universal gravitation constant
G = 6.67408e-11  # units of N-m2/kg2
# Define some reference values to simplify calculations
m_ref = 1.989e+30  # units of kg, mass of the sun used as reference mass
r_ref = 9.461e+12  # units of m, 1 light year used as reference distance
v_ref = 30000  # units of m/s, relative velocity of earth around the sun used as reference velocity
t_ref = 365 * 24 * 3600 * 0.51  # units of s, 100 orbital period of Earth (1 year)
# Non-dimensionalized constants
C1 = G * t_ref * m_ref / (r_ref**2 * v_ref)
C2 = v_ref * t_ref / r_ref

# Choose masses relative to the reference mass
m1 = 1.0  # dimensionless
m2 = 2.0  # dimensionless
m3 = 3.0  # dimensionless

# Choose initial position vectors relative to reference
r1 = [0.1, 0, 0]  # dimensionless
r2 = [-0.1, 0, 0]  # dimensionless
r3 = [-1, 0, 0]  # dimensionless

# Convert position vectors to arrays
r1 = num.array(r1, dtype="float64")
r2 = num.array(r2, dtype="float64")
r3 = num.array(r3, dtype="float64")

# Calculate Center of Mass position (from a different strategy I tried but didn't want to remove)
r_cm = (m1*r1 + m2*r2 + m3*r3) / (m1 + m2 + m3)

# Choose initial velocities relative to reference
v1 = [-0.1, -0.01, 0]  # dimensionless
v2 = [0.1, 0.01, 0]  # dimensionless
v3 = [0.1, 0, 0]  # dimensionless

# Convert velocity vectors to arrays
v1 = num.array(v1, dtype="float64")
v2 = num.array(v2, dtype="float64")
v3 = num.array(v3, dtype="float64")

# Calculate Center of Mass velocity
v_cm = (m1*v1 + m2*v2 + m3*v3) / (m1 + m2 + m3)

# The function defining the equations of motion
def MotionEquations(w, t, G, m1, m2, m3):
    r1 = w[:3]
    r2 = w[3:6]
    r3 = w[6:9]
    v1 = w[9:12]
    v2 = w[12:15]
    v3 = w[15:18]

    # normal vectors
    r12 = sc.linalg.norm(r2 - r1)
    r13 = sc.linalg.norm(r3 - r1)
    r23 = sc.linalg.norm(r3 - r2)

    # the ODEs
    dv1bydt = C1 * m2 * (r2 - r1) / r12**3 + C1 * m3 * (r3 - r1) / r13 ** 3
    dv2bydt = C1*m1*(r1-r2)/r12**3+C1*m3*(r3-r2)/r23**3
    dv3bydt = C1 * m1 * (r1 - r3) / r13**3 + C1 * m2 * (r2 - r3) / r23 ** 3
    dr1bydt = C2 * v1
    dr2bydt = C2 * v2
    dr3bydt = C2 * v3

    # position derivatives
    r12_derivs = sc.concatenate((dr1bydt, dr2bydt))
    r_derivs = sc.concatenate((r12_derivs, dr3bydt))

    # velocity derivatives
    v12_derivs = sc.concatenate((dv1bydt, dv2bydt))
    v_derivs = sc.concatenate((v12_derivs, dv3bydt))

    # all derivatives
    derivs = sc.concatenate((r_derivs, v_derivs))
    return derivs


# Organize initial parameters
init_params = sc.array([r1,r2,r3,v1,v2,v3])  # the initial parameters
init_params = init_params.flatten()  # convert to 1D array
time_span = sc.linspace(0,100,1000)  # 20 orbital periods and 500 data points

# solve the ODEs!
import scipy.integrate
three_body_sol = sc.integrate.odeint(MotionEquations,init_params,time_span,args=(G,m1,m2,m3))

# solved position vectors
r1_sol = three_body_sol[:,:3]
r2_sol = three_body_sol[:,3:6]
r3_sol = three_body_sol[:,6:9]

# Create a figure
fig = plt.figure(figsize=(15, 15))
# Create the 3D axes
ax = fig.add_subplot(111, projection="3d")
# Plot the orbit paths
ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="darkblue")
ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="tab:red")
ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2], color="tab:green")
# Plot the final positions of the objects
ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="darkblue", marker="o", s=100, label="Mass 1.0")
ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="tab:red", marker="o", s=100, label="Mass 2.0")
ax.scatter(r3_sol[-1, 0], r3_sol[-1, 1], r3_sol[-1, 2], color="tab:green", marker="o", s=100, label="Mass 3.0")
# Label the axes, add a title
ax.set_xlabel("x-coordinate", fontsize=14)
ax.set_ylabel("y-coordinate", fontsize=14)
ax.set_zlabel("z-coordinate", fontsize=14)
ax.set_title("Visualization of orbits of masses in a three-body system\n", fontsize=14)
ax.legend(loc="upper left", fontsize=14)

plt.show()


