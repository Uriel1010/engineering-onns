import numpy as np
import scipy.constants
import scipy.optimize as spopt
from matplotlib import pyplot as plt
from crow import CROW


def mse(params, tau, freq, neff):
    kappa_wg = 1j * params[-1]
    params = params[:-1]
    R, kappa = np.array_split(params, 2)
    tf_ideal = np.exp(2j * np.pi * freq * tau)
    crow = CROW(R, 1j * kappa, neff, kappa_wg=kappa_wg, c=scipy.constants.c)
    tf_crow = crow.tf_drop(freq)
    return np.mean(np.abs(tf_ideal - tf_crow) ** 2)


if __name__ == '__main__':
    kappa = 0.1j
    neff = 1.45
    wl0 = 1.55e-6
    R = 60e-6
    f0 = scipy.constants.c / wl0
    df = 10e9
    freq = np.linspace(f0 - df, f0 + df, 100)
    tau = 1e-12
    R_init = [60e-6, 60e-6]
    imkappa_init = 0.1
    x0 = np.r_[R_init, imkappa_init, imkappa_init]
    res = spopt.minimize(
        mse, x0, args=(tau, freq, neff),
        method='nelder-mead',
        options=dict(xatol=1e-8, maxfev=5000)
    )
    print(res)
    R, kappa = np.array_split(res.x[:-1], 2)
    kappa_wg = 1j * res.x[-1]
    crow = CROW(R, 1j * kappa, neff, kappa_wg=kappa_wg)
    tf_crow = crow.tf_drop(freq)

    fig, ax = plt.subplots(
        1, 2,
        dpi=200, figsize=(6, 3),
        layout='tight'
    )
    fig.suptitle = 'Transfer Function'
    delta_f_GHz = (freq - f0) / 1e9
    tf_ideal = np.exp(2j * np.pi * freq * tau)
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
