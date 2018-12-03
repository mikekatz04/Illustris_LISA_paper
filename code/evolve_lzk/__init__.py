"""Binary hardening calculations.
"""
# flake8: noqa  --- ignore imported but unused flake8 warnings

import numpy as np

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
        tau[inds] = use_rads[inds] / dadt[inds]
        return dadt, tau


def radius_schwarzschild(mass):
    return _CONST_SCHWARZSCHILD * mass


def vel_circ(mt, mr, sep):
    vels = np.sqrt(NWTG*mt/sep)
    vels = vels / (1.0 + mr)
    return vels


from .evolveLZK import EvolveLZK
