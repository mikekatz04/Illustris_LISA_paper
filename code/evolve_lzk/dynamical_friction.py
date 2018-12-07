import numpy as np
import scipy as sp

# from zcode.constants import NWTG, MSOL

# from mbhmergers import constants
# from mbhmergers.hardening import Hardening_Mechanism, dvdt_to_dadt

# WHICH_BH = constants.WHICH_BH
# ATTENUATION = constants.ATTENUATION

from . import Hardening_Mechanism, NWTG, MSOL


class Dynamical_Friction(Hardening_Mechanism):

    DF_WHICH_BH = 2
    COULOMB_LOGARITHM = 15.0
    DF_ATTENUATED = True

    GAS_VEL_SOFT = 1.0
    SELF_GRAV_RAD_MULT = 1.0

    def dadt(self):
        """Calculate the dynamical friction hardening rate.

        Arguments
        ---------
        m1 : scalar or (N,) array_like scalars,
            Mass of the primary BH.
        m2 : scalar or (N,) array_like scalars,
            Mass of the secondary BH.
        rads : scalar or (N,) array_like scalars,
            Effective distance of object from center or potential/environment.
        dens : scalar or (N,) array_like scalars,
            Density of the background (e.g. stars).
        vel_obj : scalar or (N,) array_like scalars,
            Velocity of the primary object (e.g. binary velocity).
        vdisp : scalar or (N,) array_like scalars,
            Velocity dispersion of the background (e.g. stars).
        bmin : scalar or (N,) array_like scalars,
            Minimum effective impact parameter.
        bmax : scalar or (N,) array_like scalars,
            Maximum effective impact parameter.

        rads_hard : scalar or (N,) array_like scalars or `None`,
            Radius (separation) at which the binary becomes 'hard'.
            Required if ``attenuated == True``.
        mstar : scalar or (N,) array_like scalars or `None`,
            Mass of a single characteristic star.
            Required if ``attenuated == True``.
        mass_stars : scalar or (N,) array_like scalars or `None`,
            Mass of all stars interacting with central object.
            Required if ``attenuated == True``.

        """

        evolver = self._evolver
        mass_star = evolver.mass_star
        mstar = evolver.MSTAR * MSOL
        # Use a fixed Coulomb-Logarithm
        coulomb = self.COULOMB_LOGARITHM

        rads = evolver.rads
        vcirc = evolver.vcirc

        vdisp = evolver.vdisp[:, np.newaxis]

        dens_other = evolver.dens_star + evolver.dens_dm

        which_bh = self.DF_WHICH_BH
        if which_bh == 1:
            mass_obj = evolver.m1[:, np.newaxis]
        elif which_bh == 2:
            mass_obj = evolver.m2[:, np.newaxis]
        elif which_bh == 0:
            mass_obj = evolver.mtot[:, np.newaxis]
        else:
            errStr = "Unrecognized value for `self.DF_WHICH_BH` = '{}'!".format(which_bh)
            raise ValueError(errStr)

        # Calculate Gas-Drag separately
        gas_vel = vdisp * self.GAS_VEL_SOFT
        dvdt_g = _dvdt_full(mass_obj, rads, evolver.dens_gas, vcirc, gas_vel, coulomb)
        # Stop the gas-drag dynamical friction when the viscous-circumbinary-disk takes over
        if evolver.SELF_GRAV_UNSTABLE:
            rad_sg = evolver.rad_sg
            # Find where binary separations are larger than circumbinary disk
            #    Make sure self-gravity cutoff is valid (i.e. positive)
            inds = (rads < rad_sg) & (rad_sg > 0.0)
            dvdt_g[inds] = 0.0

        # Use smoothly-attenuated DF
        if self.DF_ATTENUATED:
            rad_hard = evolver.rad_hard[:, np.newaxis]
            dvdt_o = _dvdt_attenuated(mass_obj, rads, dens_other, vcirc, vdisp, coulomb,
                                      rad_hard, mstar, mass_star)
        # Use 'full' unattenuated DF everywhere
        else:
            dvdt_o = _dvdt_full(mass_obj, rads, dens_other, vcirc, vdisp, coulomb)

        dvdt = dvdt_o + dvdt_g
        dadt, tau = self.dvdt_to_dadt(rads, vcirc, dvdt)

        # dadt, tau = dvdt_to_dadt(rads, vcirc, dvdt)
        # dadt_g, tau_g = dvdt_to_dadt(rads, vcirc, dvdt_g)
        # return dadt, dadt_g
        return dadt


def _dvdt_full(mass_obj, rads, dens, vel_obj, vdisp, coul):
    """Calculate the dynamical friction deceleration assuming full loss-cone interaction.

    See `dyn_fric` for parameter documentation.
    Note that if the velocity dispersion (`vdisp`) is small, compared to `vel_obj`, then the
    velocity factor (`velf`) approaches a constant (near unity).  `velf` only matters if
    ``vel_obj << vdisp``, in which case there is an exponential cutoff.
    """
    velf = vel_obj / (vdisp * np.sqrt(2.0))
    const = -4 * np.pi * np.square(NWTG)
    pref = mass_obj * dens / np.square(vel_obj)
    # Calculate error-function term, based on assuming maxwellian velocity distribution
    erfs = np.fabs(sp.special.erf(velf) - (2.0*velf/np.sqrt(np.pi))*np.exp(-np.square(velf)))
    # erfs = 1.0
    dvdt = const * pref * erfs * coul
    return dvdt


def _dvdt_attenuated(mass_obj, rads, dens, vel_obj, vdisp, coul, rads_hard, mstar, mass_stars):
    """Attentuated dvdt accounting for depletion of the loss-cone.

    See [Begelman, Blandford & Rees 1980].  Once the binary becomes hard, i.e. r < r_hard, the
    loss-cone must be taken into account.  The hardening radius is defined as the point at which
    the binary velocity becomes larger than the velocity dispersion of the system.  This is a very
    simple procedure for taking loss-cone effects into account.  More accurately the dynamical
    friction should be stopped at this point, and a full scattering calculation should be used.

    See `dyn_fric` for parameter documentation.
    """
    # Get un-attenuated `dvdt`
    dvdt_atten = _dvdt_full(mass_obj, rads, dens, vel_obj, vdisp, coul)

    # Calculate attentuation factor
    numStars = mass_stars / mstar
    atten1 = np.log(numStars) * rads_hard / rads
    atten2 = np.square(mass_obj / mass_stars) * numStars
    lcAtten = np.maximum(atten1, atten2)

    # Apply attentuation to hard-separations
    inds = (rads < rads_hard)
    dvdt_atten[inds] /= lcAtten[inds]

    return dvdt_atten
