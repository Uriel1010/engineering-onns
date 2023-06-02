import numpy as np
from scipy.constants import speed_of_light
from matplotlib import pyplot as plt


class CROW:
    def __init__(self, R, kappa, neff, N=None, kappa_wg=None, c=1.) -> None:
        if N is None:
            R, neff = np.atleast_1d(R, neff)
            R, neff = np.broadcast_arrays(R, neff)
            self.N = len(R)
            if np.isscalar(kappa):
                kappa = np.full(self.N - 1, kappa, dtype=np.complex128)
            self.R = R
            self.neff = neff
            if kappa_wg is None:
                kappa_wg = kappa[[0, -1]]
            elif np.isscalar(kappa_wg):
                kappa_wg = np.r_[kappa_wg, kappa_wg]
            else:
                assert len(kappa_wg) == 2
            self.kappa = np.r_[kappa_wg[0], kappa, kappa_wg[1]]
        else:
            assert np.isscalar(R)
            assert np.isscalar(kappa)
            assert np.isscalar(neff)
            self.N = N
            self.R = np.full(self.N, R, dtype=np.double)
            self.kappa = np.full(self.N + 1, kappa, dtype=np.complex128)
            self.neff = np.full(self.N, neff, dtype=np.complex128)
            if kappa_wg is not None:
                self.kappa[[0, -1]] = kappa_wg
        self.c = c

    def P(self, kappa):
        t = np.sqrt(1 - np.abs(kappa) ** 2)
        return 1 / kappa * np.array(
            [[-t, 1], [-1, t.conj()]]
        )

    def Q(self, phi):
        return np.array([[0, np.exp(-1j * phi)], [np.exp(1j * phi), 0]])

    def compute_matrices(self, f):
        k0 = 2 * np.pi * f / self.c
        T = self.P(self.kappa[0])
        for n in range(self.N):
            # TODO: compute t of MRR more wisely
            phi = k0 * self.neff[n] * np.pi * self.R[n]
            T = self.P(self.kappa[n + 1]).dot(self.Q(phi)).dot(T)
        self.T = T

    def _tf_drop(self, f):
        self.compute_matrices(f)
        A, B, C, D = self.T.ravel()
        return C - A * D / B

    def tf_drop(self, f):
        f = np.atleast_1d(f)
        res = np.empty_like(f, dtype=np.complex)
        for i in range(f.shape[0]):
            res[i] = self._tf_drop(f[i])
        return res

    def _tf_through(self, f):
        self.compute_matrices(f)
        A, B, C, D = self.T.ravel()
        return -A / B

    def tf_through(self, f):
        f = np.atleast_1d(f)
        res = np.empty_like(f, dtype=np.complex)
        for i in range(f.shape[0]):
            res[i] = self._tf_through(f[i])
        return res


if __name__ == "__main__":
    kappa = 0.1j
    neff = 1.45
    wl0 = 1.55e-6
    R = 60e-6
    c = speed_of_light
    f0 = c / wl0
    df = 300e9
    freq = np.linspace(f0 - df, f0 + df, 1000)
    crow = CROW(R, kappa, neff, N=1, c=c)
    tau_D = 100e-12

    fig, ax = plt.subplots(
        1, 2,
        dpi=200, figsize=(6, 3),
        layout='tight'
    )
    fig.suptitle = 'Transfer Function'
    delta_f_GHz = (freq - f0) / 1e9
    tf_crow = crow.tf_drop(freq)
    tf_ideal = np.exp(2j * np.pi * (freq - f0) * tau_D)
    for tf, label in zip([tf_ideal, tf_crow], ['Ideal', 'CROW']):
        ax[0].plot(delta_f_GHz, np.abs(tf))
        phase = np.angle(tf)
        ax[1].plot(delta_f_GHz, np.unwrap(phase), label=label)
    ax[0].set_title('Magnitude')
    ax[1].set_title('Phase')
    for _ in ax:
        _.set_xlabel(r"$f-f_0$ [GHz]")
        _.grid(True)
    ax[-1].legend()
    plt.show()
