"""
Calculate Gravitational Wave Driven Inspiral.

Functions
---------
    dadt_gw      : Derivative of the semi-major axis in time.
    dedt_gw      : Derivative of the eccentricity in time.
    tau_gw       : Calculate the GW Inspiral timescale.
    sep_gw       : Calculate the separation to inspiral within the given time.
    orbitalFreq  : Convert from separation to orbital frequency.
    orbitalSep   : Convert from orbital frequency to separation.

Notes
-----


"""

import numpy as np
from zcode.constants import NWTG, SPLC, YR

from mbhmergers.hardening import Hardening_Mechanism

_CONST_DIMENS = np.power(NWTG, 3.0)/np.power(SPLC, 5.0)


class Grav_Waves(Hardening_Mechanism):

    def harden(self, m1, m2, seps, eccs):
        """GW-Driven Hardening and Eccentricity evolution.

        Returns
        -------
        dadt : (N,) scalar
            Derivative of semi-major axis vs. time (negative).
        decdt : scalar or (N,) scalar
            Derivative of eccentricity vs. time (negative).
        taus : (N,) scalar
            Hardening timescale (a / da/dt) due to GW emission.

        """

        dadt = -_CONST_DIMENS * (64.0/5.0) * m1 * m2 * (m1+m2) * np.power(seps, -3.0)
        decdt = np.zeros_like(dadt)
        # Only bother with the eccentricity terms if non-zero
        if np.any(eccs > 0.0):
            e2 = np.square(eccs)
            ecc_fact_a = (1 + (73.0/24)*e2 + (37.0/96)*e2*e2)*np.power((1.0-e2), -3.5)
            ecc_fact_e = (eccs + (121.0/304)*eccs*e2)*np.power((1.0-e2), -2.5)
            decdt = -_CONST_DIMENS * (304.0/15.0) * m1 * m2 * (m1+m2) * np.power(seps, -4.0)

            dadt *= ecc_fact_a
            decdt *= ecc_fact_e

        taus = -seps/dadt

        return dadt, decdt, taus


# def dadt_gw(m1, m2, rads, ee=0.0):
#     """Derivative of the semi-major axis in time.
#     """
#     eccFact = 1.0
#     # Only bother with the eccentricity terms if non-zero
#     if np.any(ee > 0.0):
#         e2 = np.square(ee)
#         eccFact = (1 + (73.0/24)*e2 + (37.0/96)*e2*e2)*np.power((1.0-e2), -3.5)
#
#     dadt = -_CONST_DIMENS*(64.0/5.0)*m1*m2*(m1+m2)*np.power(rads, -3.0)*eccFact
#     taus = -rads/dadt
#
#     return dadt, taus
#
#
# def dedt_gw(m1, m2, aa, ee=0.0):
#     """ Derivative of the eccentricity in time """
#
#     if(ee == 0.0): return 0.0
#     e2      = np.square(ee)
#     eccFact = (ee + (121.0/304)*ee*e2)*np.power((1.0-e2), -2.5)
#     dedt    = _CONST_DIMENS*(304.0/15.0)*m1*m2*(m1+m2)*np.power(aa, -4.0)*eccFact
#
#     return dedt


def tau_gw(m1, m2, aa):
    """ Calculate the GW Inspiral timescale """
    const = (5.0/256.0)/_CONST_DIMENS
    mterm = m1*m2*(m1+m2)
    return const*np.power(aa, 4.0)/mterm


def sep_gw(m1, m2, tau=1.0e10*YR):
    """ Calculate the separation to inspiral within time ``tau`` """
    mterm = m1*m2*(m1+m2)
    const = (256.0/5.0)*_CONST_DIMENS
    return np.power(const*mterm*tau, 0.25)


def orbitalFreq(m1, aa, m2=0.0):
    """ Convert from separation to orbital frequency. """
    return np.sqrt(NWTG*(m1+m2)/(np.square(np.pi)*np.power(aa, 3.0)))


def orbitalSep(m1, ff, m2=0.0):
    """ Convert from orbital frequency to separation. """
    return np.power(NWTG*(m1+m2) / (np.square(np.pi)*np.square(ff)), 1.0/3.0)
