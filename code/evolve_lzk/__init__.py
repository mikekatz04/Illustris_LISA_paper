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
_CONST_SCHWARZSCHILD = 2*NWTG/SPLC**2


class Hardening_Mechanism:

    def __init__(self):
        return

    def harden(self, *args, **kwargs):
        raise NotImplementedError("`harden()` has not been overridden!")


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
    dadt = (2*rads/vels)*dvdt
    tau = np.zeros_like(dadt)
    inds = dadt.nonzero()[0]
    tau[inds] = rads[inds]/dadt[inds]
    return dadt, tau


def radius_schwarzschild(mass):
    return _CONST_SCHWARZSCHILD * mass


from .evolveLZK import EvolveLZK

# from . import DistFunc
# # from . import dynamical_friction
# from .dynamical_friction import Dynamical_Friction
# # from . import circumbinary_disk
# from .circumbinary_disk import Disk_Torque
# # from . import grav_waves
# from .grav_waves import Grav_Waves
# from . import stellar_scattering
