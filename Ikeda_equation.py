import numpy as np
from scipy import integrate as spode
from scipy.io import wavfile
from scipy import interpolate as spint
import matplotlib.pyplot as plt


def reservoir(s, x, eps, beta, mu, phi_0, rho, u):
    global x_history
    global s_history

    # get previous state at s-1 using interpolation
    s_i = len(s_history)
    if s_i > 1:
        prev_x = np.interp(s - 1, s_history, x_history, left=0.0, right=0.0)
    else:
        prev_x = x_history[0]

    dxds = \
        (-x + beta * np.sin(mu * prev_x + rho * u(s - 1) + phi_0) ** 2) / eps

    # append current state to history
    x_history = np.r_[x_history, x[0]]
    s_history = np.r_[s_history, s]

    return dxds


def u_delta(s, t_0=5):
    return 1.0 if np.abs(s - t_0) < ds else 0.0


def u_sin(s, omega=2*np.pi):
    return np.sin(omega * s)


def u_step(s, t_0=5):
    return 1.0 if s > t_0 else 0.0


def interpolate_audio(fname, s_start=-1, s_end=0, decim=1):
    # TODO: load audio and create interp object
    samplerate, data = wavfile.read(fname)
    data = data / np.max(np.abs(data))
    s = np.linspace(s_start, s_end, data[::decim].shape[0])
    interp_obj = spint.interp1d(
        s, data[::decim], kind='linear',
        bounds_error=False, fill_value=0
    )
    return interp_obj


def initial_value_problem(tau_d, Tr, beta, mu, phi_0, rho, u, smax):
    global s_history
    global x_history
    # initialize history arrays
    s_history = [0.0]
    x_history = [0.0]
    x0 = 0
    eps = Tr / tau_d
    s_eval = np.linspace(0, smax, 5000)

    return spode.solve_ivp(
        reservoir, t_span=s_eval[[0, -1]], t_eval=s_eval, y0=[x0],
        args=(eps, beta, mu, phi_0, rho, u),
        method='RK45'
    )


if __name__ == "__main__":

    # calculate time values based on delay time
    tau_d = 20.87E-6  # delay time
    Tr = 240e-9
    eps = Tr / tau_d

    # initialize history arrays
    s_history = [0.0]
    x0 = 0
    x_history = [x0]

    # set parameter values
    beta = 1.5  # nonlinearity gain
    mu = 1  # feedback scaling
    rho = 0.5  # relative weight of input information compared to feedback signal
    phi_0 = np.pi * 0.89  # offset phase of the MZM

    s_eval = np.linspace(0, 10, 5000)
    ds = s_eval[1] - s_eval[0]
    t_eval = s_eval * tau_d

    u_audio = interpolate_audio(
        './free-spoken-digit-dataset/test.wav',
    )

    u = u_audio
    # integrate the system over time
    sol = spode.solve_ivp(
        reservoir, t_span=s_eval[[0, -1]], t_eval=s_eval, y0=[x0],
        args=(eps, beta, mu, phi_0, rho, u),
        method='RK45'
    )

    # plot the results
    fig, ax = plt.subplots(figsize=(5, 3), layout='tight')
    plt.plot(sol.t, sol.y[0], label='$x(s)$')
    plt.plot(sol.t, u(s_eval - 1), label='$u(s - 1)$')
    plt.xlabel('s')
    plt.ylabel('x')
    plt.title(
        'Reservoir Dynamics without External Signal\n'
        fr'($\beta$={beta}, $\mu$={mu}, $\rho$={rho}, $\Phi_0$={phi_0:.2f})'
    )
    plt.legend()
    plt.show()

