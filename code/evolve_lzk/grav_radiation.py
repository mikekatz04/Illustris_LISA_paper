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

import zcode.math as zmath

from . import Hardening_Mechanism, SPLC, NWTG, PC, MSOL

_CONST_DIMENS = np.power(NWTG, 3.0)/np.power(SPLC, 5.0)


class Grav_Radiation(Hardening_Mechanism):

    # def harden(self, m1, m2, seps, eccs):
    def dadt(self):
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
        m1 = self._evolver.m1[:, np.newaxis]
        m2 = self._evolver.m2[:, np.newaxis]
        rads = self._evolver.rads[np.newaxis, :]
        # eccs = 0.0
        print(m1.shape, m2.shape, rads.shape)
        print(zmath.stats_str(m1/MSOL))
        print(zmath.stats_str(m2/MSOL))
        dadt = -_CONST_DIMENS * (64.0/5.0) * m1 * m2 * (m1+m2) * np.power(rads, -3.0)

        '''
        decdt = np.zeros_like(dadt)
        # Only bother with the eccentricity terms if non-zero
        if np.any(eccs > 0.0):
            e2 = np.square(eccs)
            ecc_fact_a = (1 + (73.0/24)*e2 + (37.0/96)*e2*e2)*np.power((1.0-e2), -3.5)
            ecc_fact_e = (eccs + (121.0/304)*eccs*e2)*np.power((1.0-e2), -2.5)
            decdt = -_CONST_DIMENS * (304.0/15.0) * m1 * m2 * (m1+m2) * np.power(seps, -4.0)

            dadt *= ecc_fact_a
            decdt *= ecc_fact_e
        '''

        self.check_timescale("GW", None, 1e-2*PC, extr=[1e4, 1e12], dadt=dadt)

        # taus = -rads/dadt
        # return dadt, decdt, taus

        return dadt
