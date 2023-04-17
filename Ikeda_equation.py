import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def reservoir_system(x, s, beta, mu, phi_0):
    eps = 240e-9 / 20.87e-6 # calculate the value of epsilon from the given values of T_R and tau_D
    dxds = (-x + beta*np.sin(mu*x + phi_0)**2) / eps
    return dxds

def u(t):
    # define the input signal here
    return 0.0

x0 = 0.0

# set parameter values
beta = 0.3 # nonlinearity gain
mu = 2.5 # feedback scaling
rho = 0.0 # relative weight of input information compared to feedback signal
phi_0 = np.pi * 0.89 # offset phase of the MZM

s = np.linspace(0, 0.1, 1000)
ds = s[1] - s[0]

# calculate time values based on delay time
tau_d = 20.87E-6 # delay time
t = [s1 * tau_d for s1 in s]

# integrate the system over time
x = odeint(reservoir_system, x0, s, args=(beta, mu, phi_0))

# plot the results
plt.plot(t, x)
plt.xlabel('Time [s]')
plt.ylabel('x(s)')
plt.title(f'Reservoir Computing without External Signal \n(beta={beta}, mu={mu}, phi_0={phi_0})')
plt.show()
