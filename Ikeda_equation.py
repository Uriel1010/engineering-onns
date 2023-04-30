import functools

import numpy as np
from scipy import integrate as spode
from scipy.io import wavfile
from scipy import interpolate as spint
import matplotlib.pyplot as plt


class IkedaEquation:
    def __init__(self, eps, beta, mu, phi_0, rho):
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.phi_0 = phi_0
        self.rho = rho
        self.x_history = None
        self.s_history = None
        self.u = None
        self.sol = None

    def derivative(self, s, x):
        if len(self.s_history > 1):
            prev_x = np.interp(s - 1, self.s_history, self.x_history, left=self.x_history[0], right=0.0)
        else:
            prev_x = self.x_history[0]

        if self.u:
            u = self.u
        else:
            u = lambda s: 0

        dxds = (
            -x + self.beta * np.sin(self.mu * prev_x + self.rho * u(s - 1) + self.phi_0) ** 2
        ) / self.eps

        # append current state to history
        self.x_history = np.r_[self.x_history, x[0]]
        self.s_history = np.r_[self.s_history, s]

        return dxds

    def solve_ivp(self, x0, smax, nsamples=5000, method='RK45'):
        self.s_history = np.array([0.0])
        self.x_history = np.array([x0])
        s_eval = np.linspace(0, smax, nsamples)
        self.sol = spode.solve_ivp(
            self.derivative, t_span=s_eval[[0, -1]], t_eval=s_eval, y0=[x0],
            method=method
        )
        return self.sol

    def plot_solution(self, ax=None):
        # plot the results
        if not ax:
            fig, ax = plt.subplots(figsize=(5, 3), layout='tight')
        else:
            fig = ax.gcf()
        if self.u:
            ax.plot(self.sol.t, self.u(self.sol.t), label='$u(s)$', color='r')
        ax.plot(self.sol.t, self.sol.y[0], label='$x(s)$', color='k')
        ax.set_xlabel('s')
        ax.set_title(
            'Reservoir Dynamics\n'
            fr'($\epsilon$={self.eps:.2f}, $\beta$={self.beta}, $\mu$={self.mu},'
            fr' $\rho$={self.rho}, $\Phi_0$={self.phi_0:.2f})'
        )
        ax.legend()
        return fig, ax


def u_delta(s, t_0=5):
    return 1.0 if np.abs(s - t_0) < ds else 0.0


def u_sin(s, omega=2*np.pi):
    return np.sin(omega * s)


def u_step(s, t_0=5):
    return 1.0 if s > t_0 else 0.0


def interpolate_audio(fname):
    samplerate, data = wavfile.read(fname)
    data = data / np.max(np.abs(data))
    ndata = data.shape[0]
    s = np.arange(0, ndata)
    interp_obj = spint.interp1d(
        s, data, kind='linear',
        bounds_error=False, fill_value=0
    )
    return interp_obj


def sample_and_hold(fun):
    @functools.wraps(fun)
    def wrapped(s):
        i = np.array(s, dtype=int)
        return fun(i)

    return wrapped


if __name__ == "__main__":

    # calculate time values based on delay time
    tau_d = 20.87E-6  # delay time
    Tr = 240e-9
    eps = Tr / tau_d

    # set parameter values
    beta = 1.5  # nonlinearity gain
    mu = 1  # feedback scaling
    rho = 0.5  # relative weight of input information compared to feedback signal
    phi_0 = np.pi * 0.89  # offset phase of the MZM

    eq = IkedaEquation(eps, beta, mu, phi_0, rho)

    # Solve without external signal to reach equilibrium state
    x0 = 0
    eq.u = None
    sol = eq.solve_ivp(x0, 10, nsamples=2, method='RK23')
    x0 = sol.y[0, -1]

    eq.u = sample_and_hold(interpolate_audio(
        './free-spoken-digit-dataset/test.wav',
    ))
    eq.solve_ivp(x0, 50, nsamples=1000, method='RK23')
    eq.plot_solution()
    plt.show()

