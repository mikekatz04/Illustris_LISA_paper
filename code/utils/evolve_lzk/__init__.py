"""Binary hardening calculations.
"""
# flake8: noqa  --- ignore imported but unused flake8 warnings

import numpy as np

import zcode.math as zmath

PC = 3.085677e+18
MSOL = 1.9884754e+33
NWTG = 6.674079e-08
SPLC = 2.9979e10
MPRT = 1.67262e-24
SIGMA_T = 6.65246e-25
YR = 3.15576e+07

# Derived Constants
GYR = YR * 1e9
_CONST_SCHWARZSCHILD = 2*NWTG/SPLC**2


class Hardening_Mechanism:

    def __init__(self, evolver):
        self._evolver = evolver
        verbose = evolver._verbose
        self._verbose = verbose

        # if verbose:
        #     print("Initializing `{}`".format(self.__class__.__name__))

        return

    def harden(self, *args, **kwargs):
        raise NotImplementedError("`harden()` has not been overridden!")

    @staticmethod
    def dvdt_to_dadt(rads, vels, dvdt):
        """Calculate the hardening rate (da/dt) and timescale given a deceleration.

        Protects against divide-by-zero (no warnings or errors).

        Arguments
        ---------
        rads : (N,) array_like of float
            Binary separation (radius from center).
        vels : (N,) array_like of float
            Velocity of hardening object at given radius.
        dvdt : (N,) array_like of float
            Deceleration at given radius.

        Returns
        -------
        dadt : (N,) array_like of float
            Hardening rate, i.e. velocity of hardening.
        tau  : (N,):
            Hardening timescale, i.e. ``a/(da/dt)``

        """
        dadt = (2 * rads / vels) * dvdt
        tau = np.zeros_like(dadt)
        use_rads = np.ones_like(dadt) * rads[np.newaxis, :]
        inds = (dadt != 0.0)
        tau[inds] = - use_rads[inds] / dadt[inds]
        if np.any(tau < 0.0):
            nbad = np.count_nonzero(tau < 0.0)
            nall = np.size(tau)
            frac = nbad/nall
            print("Bads = {}/{} = {}".format(nbad, nall, frac))
            bads = np.where(tau < 0.0)
            print(dadt[bads][0], dvdt[bads][0], tau[bads][0])
            raise ValueError("Negative inspiral timescales!")

        return dadt, tau

    def check_timescale(self, name, tau, rad=PC, extr=None, dadt=None):
        EXTR = [1e6, 1e16]
        if extr is None:
            extr = EXTR
        else:
            for ii in range(2):
                extr[ii] = EXTR[ii] if extr[ii] is None else extr[ii]

        if tau is None:
            tau = - self._evolver.rads / dadt

        # rad_ind = self._evolver._rad_ind
        rad_ind = zmath.argnearest(self._evolver.rads, rad)
        rad_rad = self._evolver.rads[rad_ind]/PC
        stats = zmath.stats_str(tau[:, rad_ind]/YR, log=False)
        print(name + " T[{:.0e} pc]/YR = ".format(rad_rad) + stats)
        tau_med = np.median(tau[:, rad_ind]/YR)
        if (tau_med < extr[0]) or (tau_med > extr[1]):
            err = name + " timescale looks wrong!  (vs. {:.1e}, {:.1e} [yr])".format(*extr)
            raise ValueError(err)

        return


def radius_schwarzschild(mass):
    return _CONST_SCHWARZSCHILD * mass


def vel_circ(mt, mr, sep):
    vels = np.sqrt(NWTG*mt/sep)
    vels = vels / (1.0 + mr)
    return vels


from .evolveLZK import EvolveLZK
