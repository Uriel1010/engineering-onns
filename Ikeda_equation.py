import numpy as np
from scipy import integrate as spode
from scipy.io import wavfile
from scipy import interpolate as spint
import matplotlib.pyplot as plt


def reservoir_system(s, x, eps, beta, mu, phi_0, rho, u, s_history, x_history):
    # get previous state at s-1 using interpolation
    s_i = len(s_history)
    if s_i > 1:
        prev_x = np.interp(s - 1, s_history, x_history, left=0.0, right=0.0)
    else:
        prev_x = x_history[0]

    dxds = (-x + beta * np.sin(mu * prev_x + rho * float(u(s - 1)) + phi_0) ** 2) / eps

    # append current state to history
    x_history.append(x[0])
    s_history.append(s)

    return dxds


def u(s):
    # define the input signal here
    return 0.0

def u_delta(s, t_0=5):
    return 1.0 if np.abs(s - t_0) < ds else 0.0

def u_sin(s, omega=2*np.pi):
    return np.sin(omega * s)

def u_step(s, t_0=5):
    return 1.0 if s > t_0 else 0.0

def interpolate_audio(fname):
    # TODO: load audio and create interp object
    samplerate, data = wavfile.read(fname)
    s = np.linspace(0, 1, data[::100].shape[0])
    interp_obj = spint.interp1d(s, data[::100], bounds_error=False, fill_value=0)
    # func = func(s)
    return interp_obj


u_signal = interpolate_audio('./free-spoken-digit-dataset/test.wav')

# calculate time values based on delay time
tau_d = 20.87E-6  # delay time
Tr = 240e-9
eps = Tr / tau_d

# initialize history arrays
s_history = [0.0]
x0 = 0
x_history = [x0]

# set parameter values
beta = 0.3  # nonlinearity gain
mu = 2.5  # feedback scaling
rho = 0.0  # relative weight of input information compared to feedback signal
phi_0 = np.pi * 0.89  # offset phase of the MZM

s_eval = np.linspace(0, 10, 1000)
ds = s_eval[1] - s_eval[0]
t_eval = s_eval * tau_d


def initial_value_problem(tau_d, Tr, beta, mu, phi_0, rho, u):
    # initialize history arrays
    s_history = [0.0]
    x0 = 0
    x_history = [x0]
    eps = Tr / tau_d

    return spode.solve_ivp(
        reservoir_system, t_span=s_eval[[0, -1]], t_eval=s_eval, y0=[x0],
        args=(eps, beta, mu, phi_0, rho, u, s_history, x_history),
        method='RK23'
    )


# integrate the system over time
sol = spode.solve_ivp(
    reservoir_system, t_span=s_eval[[0, -1]], t_eval=s_eval, y0=[x0],
    args=(eps, beta, mu, phi_0, rho, u, s_history, x_history),
    method='RK23'
)

# plot the results
fig, ax = plt.subplots(figsize=(5, 3), layout='tight')
plt.plot(sol.t, sol.y[0])
plt.xlabel('s')
plt.ylabel('x')
plt.title(
    'Reservoir Dynamics without External Signal\n'
    fr'($\beta$={beta}, $\mu$={mu}, $\Phi_0$={phi_0:.2f})'
)
plt.show()

if __name__ == "__main__":
    pass
