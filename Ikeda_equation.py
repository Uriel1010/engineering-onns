import numpy as np
from scipy import integrate as spode
from scipy.io import wavfile
import matplotlib.pyplot as plt


class IkedaEquation:
    def __init__(self, eps, beta, mu, phi_0, rho, N):
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.phi_0 = phi_0
        self.rho = rho
        self.x_history = None
        self.s_history = None
        self.u = np.array([0])
        self.N = N
        np.random.seed(42)
        self.mask = 1 - 2 * np.random.randint(2, size=(N,))
        self.sol = None

    def derivative(self, s, x):
        prev_x = np.interp(s - 1, self.s_history, self.x_history)

        dxds = (
            -x
            + self.beta * np.sin(
                self.mu * prev_x
                + self.rho * self.J(s - 1)
                + self.phi_0
            ) ** 2
        ) / self.eps

        # Update history
        self.x_history[:-1] = self.x_history[1:]
        self.x_history[-1] = x[0]
        self.s_history[:-1] = self.s_history[1:]
        self.s_history[-1] = s
        return dxds

    def I(self, s):
        i = np.asarray(s, dtype=int)
        res = np.zeros(i.shape, dtype=np.double)
        idx = np.logical_and(i >= 0, i < self.Q)
        res[idx] = self.u[i[idx]]
        return res

    def J(self, s):
        i = np.array((s % 1) * self.N, dtype=int)
        return self.mask[i] * self.I(s)

    @property
    def Q(self):
        return self.u.shape[0]

    def solve_ivp(self, x0, s_eval, method='RK45'):
        self.s_history = np.linspace(-1, 0, self.N)
        self.x_history = np.full(self.N, x0)
        self.sol = spode.solve_ivp(
            self.derivative, t_span=[0, s_eval[-1]], t_eval=s_eval, y0=[x0],
            method=method, vectorized=True
        )
        return self.sol

    def plot_solution(self, ax=None, J=True, I=False):
        # plot the results
        if not ax:
            fig, ax = plt.subplots(figsize=(5, 3), layout='tight')
        else:
            fig = ax.gcf()
        if I:
            ax.plot(self.sol.t, self.I(self.sol.t), label='$I(s)$', color='r')
        if J:
            ax.plot(self.sol.t, self.rho * self.J(self.sol.t), label=r'$\rho J(s)$', color='g')
        ax.plot(self.sol.t, self.sol.y[0], label='$x(s)$', color='k')
        ax.set_xlabel('s')
        ax.set_title(
            'Reservoir Dynamics\n'
            fr'($\epsilon$={self.eps:.2f}, $\beta$={self.beta}, $\mu$={self.mu},'
            fr' $\rho$={self.rho}, $\Phi_0$={self.phi_0:.2f})'
        )
        ax.legend(loc='right')
        return fig, ax


if __name__ == "__main__":
    # set parameter values
    beta = 0.3  # nonlinearity gain
    mu = 1  # feedback scaling
    rho = np.pi  # relative weight of input information compared to feedback signal
    phi_0 = np.pi * 0.89  # offset phase of the MZM
    N = 400  # number of virtual nodes
    eps = 5 / N  # response time
    eq = IkedaEquation(eps, beta, mu, phi_0, rho, N=N)

    x0 = 0
    sol = eq.solve_ivp(x0, np.linspace(0, 100, 2), method='RK23')
    x0 = sol.y[0, -1]
    sr, data = wavfile.read('tests/test.wav')
    eq.u = data / np.max(np.abs(data))
    s_eval = np.linspace(0, len(data) / 100, N * len(data))
    eq.solve_ivp(x0, s_eval, method='RK23')
    fig, ax = eq.plot_solution(I=True)
    plt.show()

