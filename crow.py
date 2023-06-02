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
    df = 10e9
    freqs = np.linspace(f0 - df, f0 + df, 1000)
    crow = CROW(R, kappa, neff, N=1, c=c)
    TF_drop = np.asarray([crow.tf_drop(f) for f in freqs])
    tau_D = 100e-12
    TF_ideal = np.exp(2j * np.pi * (freqs - f0) * tau_D)
    fig, ax = plt.subplots(
        dpi=200, figsize=(4, 3),
        layout='tight'
    )
    delta_f_GHz = (freqs - f0) / 1e9
    ax.plot(delta_f_GHz, np.abs(TF_drop), label='Transmittivity')
    # ax.plot(delta_f_GHz, np.unwrap(np.angle(TF_ideal)), label='Ideal delay')
    phase = np.angle(TF_drop)
    ax.plot(delta_f_GHz, np.unwrap(phase), label='Phase')
    ax.legend()
    ax.set_xlabel(r"$f-f_0$ [GHz]")
    ax.grid(True)
    ax.set_title('Phase')
    plt.show()
