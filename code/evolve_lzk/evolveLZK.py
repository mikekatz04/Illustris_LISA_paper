"""

ALSO NEED
- mdot
- binned, resolved profiles (not just fits)
- vel-disp fit profiles

"""
import os
import sys
import logging
import warnings
from datetime import datetime

import scipy as sp  # noqa
import scipy.interpolate   # noqa
# from scipy.interpolate import interp1d, interp2d
# from scipy.special import gamma as Gamma_Function
# from scipy.integrate import quad
# from scipy.special import hyp2f1
import numpy as np
# import pdb

import zcode.math as zmath
import zcode.astro as zastro

import cosmopy

from utils.mbhbinaries import MassiveBlackHoleBinaries
from . import PC, MSOL, NWTG, GYR, radius_schwarzschild, vel_circ
from . dynamical_friction import Dynamical_Friction
from . circumbinary_disk import Disk_Torque
from . grav_radiation import Grav_Radiation
from . stellar_scattering import Stellar_Scattering_Scalings


class EvolveLZK(MassiveBlackHoleBinaries):

    NUM_STEPS = 200
    SELF_GRAV_UNSTABLE = True

    MASS_STELLAR_MIN = 1.0e5           # [Msol] 0.0 for none
    RAD_STELLAR_MIN = 0.1              # [pc]  0.0 for none
    RAD_BOUND_MIN = 1.0e-2
    RAD_BOUND_MAX = 1.0e2
    RAD_LOSS_CONE_MIN = 1.0e-3
    RAD_LOSS_CONE_MAX = 1.0e3
    RAD_HARD_MIN = 1.0e-2
    RAD_HARD_MAX = 1.0e2
    PROFILE_BMAX_BMIN_RATIO = 10.0

    MSTAR = 0.6

    # def __init__(self, m1, m2, vel_disp_1, vel_disp_2, star_gamma, separation, redz, e_0=0.0):
    def __init__(self, fname, e_0=0.0, verbose=False):
        names = ['dm', 'gas', 'star']
        self._names = names
        num_steps = self.NUM_STEPS

        input_data = np.genfromtxt(fname, names=True, dtype=None)
        self._input_data = input_data
        self._verbose = verbose
        self._cosmo = cosmopy.get_cosmology()

        if verbose:
            print("input_data.shape = {}".format(input_data.shape))
            print("names = {}".format(", ".join(input_data.dtype.names)))

        keys_masses = ['mass_new_prev_in', 'mass_new_prev_out']
        masses = np.array([input_data[kk] for kk in keys_masses]).T * MSOL
        self.masses = masses

        keys_snaps = ['snap_prev_in', 'snapshot_prev_out', 'snapshot_fin_out']
        self.snaps = np.array([input_data[kk] for kk in keys_snaps]).T

        # keys_gammas = ['dm_gamma', 'gas_gamma', 'star_gamma']
        self.gammas = np.array([-input_data[kk + "_gamma"] for kk in names]).T

        # m2, m1 = masses.T
        # mt = m1 + m2
        # mr = m2/m1
        #
        self.m1 = masses.max(axis=-1)
        self.m2 = masses.min(axis=-1)
        mt = self.m1 + self.m2
        self.mtot = mt
        mr = self.m2 / self.m1
        self.mrat = mr

        self.redz_form = input_data['redshift']
        self.tage_form = self._cosmo.age(self.redz_form).cgs.value

        self.sep0 = input_data['separation']*1000*PC
        self.vdisp = input_data['vel_disp_fin_out'] * 1e5
        self.gal_mstar = input_data['stellar_mass_fin_out'] * MSOL
        self.gal_mtot = input_data['total_mass_fin_out'] * MSOL
        self.eccen = e_0

        self.num_binaries = self.m1.size
        self.num_steps = num_steps
        self._shape = (self.num_binaries, self.num_steps)
        self.rad_isco = 3 * radius_schwarzschild(self.m2)
        sep_extr = [self.rad_isco.min(), self.sep0.max()]
        self.rads = np.logspace(*np.log10(sep_extr), self.NUM_STEPS)

        # Binary circular velocity
        self.vcirc = vel_circ(mt[:, np.newaxis], mr[:, np.newaxis], self.rads[np.newaxis, :])

        # TODO: FIX THIS accretion rates needed from Illustris
        warnings.warn("Fix accretion rates!")
        self.mdot = zastro.eddington_accretion(mt)

        if verbose:
            print("Mtot = " + zmath.stats_str(mt, log=True))
            print("Mrat = " + zmath.stats_str(mr, log=False))
            print("redz_form = " + zmath.stats_str(self.redz_form, log=False))
            print("sepa = " + zmath.stats_str(self.sep0, log=False))
            print("gal_mstar = " + zmath.stats_str(self.gal_mstar, log=True))
            print("gal_mtot = " + zmath.stats_str(self.gal_mtot, log=True))
            for gg, nn in zip(self.gammas.T, names):
                print("{} = ".format(nn) + zmath.stats_str(gg))

        self.init_mass_profile_arrays()
        self.calc_critical_radii()

        return

    def init_mass_profile_arrays(self):
        SF_EFF_FRAC = 0.2
        RMAX_FACT = 5.0
        PROFILE_POWLAW_MAX = -0.1
        # PROFILE_POWLAW_MAX = -2.9
        PROFILE_POWLAW_MIN = -2.9

        verbose = self._verbose
        names = self._names

        if verbose:
            print("Running `EvolveLZK.init_mass_profile_arrays()")

        # TODO: FIX ALL THIS
        warnings.warn("FIX TEMPORARY MASS PROFILES!")

        rads = self.rads
        mstar = self.gal_mstar
        mtot = self.gal_mtot
        mdm = 5 * mstar

        mgas = mstar / SF_EFF_FRAC
        mdm = np.maximum(mdm, mtot - (mstar + mgas))
        mdm = np.minimum(mdm, mtot - mstar)

        mgas = np.maximum(mtot - (mstar + mdm), 0.0)
        if verbose:
            print("dm = " + zmath.stats_str(mdm/mtot))
            print("gas = " + zmath.stats_str(mgas/mtot))
            print("star = " + zmath.stats_str(mstar/mtot))

        rmax = self.sep0 * RMAX_FACT
        vols = 4*np.pi * np.power(rads, 3) / 3
        dv = np.concatenate(([vols[0]], vols[1:] - vols[:-1]))
        # dv = 4*np.pi*np.power(rads, 2) * np.concatenate(([rads[0]], np.diff(rads)))

        gammas = np.clip(self.gammas, PROFILE_POWLAW_MIN, PROFILE_POWLAW_MAX)
        # NOTE: this ordering must match `_names` (and thus `gammas`)
        mass_tots = [mdm, mgas, mstar]
        # dens = []
        # mass = []
        for mm, gg, nn in zip(mass_tots, gammas.T, names):
            dd = ((gg+3) / (4*np.pi*(PC**3))) * mm / np.power(rmax/PC, gg+3)
            _dens = dd[:, np.newaxis] * np.power(rads[np.newaxis, :]/PC, gg[:, np.newaxis])
            # dens.append(_dens)
            _mass = np.cumsum(_dens * dv[np.newaxis, :], axis=-1)
            # mass.append(_mass)
            if verbose:
                print(nn + " dens = " + zmath.stats_str(dd*PC**3/MSOL))

            dvar = "dens_" + nn
            mvar = "mass_" + nn
            if verbose:
                print("Setting `{}`, `{}`".format(dvar, mvar))
            setattr(self, dvar, _dens)
            setattr(self, mvar, _mass)

        return

    def calc_critical_radii(self):
        verbose = self._verbose
        if verbose:
            print("Running `EvolveLZK.calc_critical_radii()")

        m1 = self.m1
        rads = self.rads

        # Sphere of Influence of BH
        #     Use Velocity Dispersion: R_infl = GM_bh/sigma^2
        rad_infl = NWTG * m1 / np.square(self.vdisp)

        # Stellar Effective Radius and Mass
        #    Scale stellar radius to that of MW (0.5pc) (mass 2e12 Msol),
        #    See: http://adsabs.harvard.edu/abs/2009A%26A...499..483B
        rad_stellar = 0.5 * PC * self.gal_mtot / (2.0e12 * MSOL)
        rad_stellar = np.maximum(rad_stellar, self.RAD_STELLAR_MIN*PC)

        # Find mass out to stellar radius by interpolation
        # beg = datetime.now()
        # interp_func = zmath.interp_func(rads, self.mass_star,
        #                                 axis=-1, bounds_error=True, assume_sorted=True)
        # mass_stellar = interp_func(rad_stellar)
        # This is much faster:
        mass_stellar = [zmath.interp(rs, rads, ms, valid=False)
                        for rs, ms in zip(rad_stellar, self.mass_star)]
        mass_stellar = np.maximum(mass_stellar, self.MASS_STELLAR_MIN*MSOL)

        # 'Binding' radius of BH
        rad_bound = np.power(m1 / mass_stellar, 1/3) * rad_stellar
        rad_bound = np.clip(rad_bound, self.RAD_BOUND_MIN*PC, self.RAD_BOUND_MAX*PC)

        # 'Loss Cone' Radius
        rad_lc  = np.power(self.MSTAR / m1, 1/4)
        rad_lc *= rad_stellar * np.power(rad_bound/rad_stellar, 9/4)

        rad_lc = np.clip(rad_lc, self.RAD_LOSS_CONE_MIN*PC, self.RAD_LOSS_CONE_MAX*PC)

        # 'Hardened' radius for binary
        rad_hard = rad_stellar * np.power(rad_bound / rad_stellar, 3)
        rad_hard = np.clip(rad_hard, self.RAD_HARD_MIN*PC, self.RAD_HARD_MAX*PC)

        # Impact Parameters
        #  -----------------

        # Minimum Impact Parameter
        use_vel = np.maximum(self.vcirc, self.vdisp[:, np.newaxis])
        bmin = NWTG * self.m2[:, np.newaxis] / np.square(use_vel)

        # Maximum impact parameter (See [Begelman, Blandford, Rees 1980])
        #     When Binary is Bound --- Decrease as b~(r/r_B)^{3/2}*r_c
        bound = np.power(rads[np.newaxis, :] / rad_bound[:, np.newaxis], 3/2)
        bound_mask = rads[np.newaxis, :] < rad_bound[:, np.newaxis]
        #     Start with stellar effective radius
        bmax = np.ones_like(bound) * rad_stellar[:, np.newaxis]
        bmax[bound_mask] *= bound[bound_mask]
        #     When Binary is Hard  --- Decrease as b~(r r_h)^{1/2}
        hard = np.sqrt(rads[np.newaxis, :] * rad_hard[:, np.newaxis])
        hard_mask = rads[np.newaxis, :] < rad_hard[:, np.newaxis]
        bmax[hard_mask] = hard[hard_mask]

        # Make sure ``bmax/bmin`` is at least ``bratio``
        bmax = np.maximum(bmax, bmin*self.PROFILE_BMAX_BMIN_RATIO)

        self.rad_infl = rad_infl
        self.rad_stellar = rad_stellar
        self.mass_stellar = mass_stellar
        self.rad_bound = rad_bound
        self.rad_lc = rad_lc
        self.rad_hard = rad_hard
        self.bmin = bmin
        self.bmax = bmax
        return

    def init_integral_arrays(self):
        shape = self._shape

        self.dadt = np.zeros(shape)
        # self.durs = np.zeros(shape)
        # self.times = np.zeros(shape)
        # self.redz = np.zeros(shape)

        self.dadt_df = np.zeros(shape)
        self.dadt_sc = np.zeros(shape)
        self.dadt_cd = np.zeros(shape)
        self.dadt_gw = np.zeros(shape)

        return

    def integrate(self):
        verbose = self._verbose
        memory = False
        if verbose:
            print("Running `EvolveLZK.integrate()")

        def printer(init=None, run=None, beg=None):
            if verbose:
                if beg is not None:
                    print("    Done after '{}'".format(datetime.now()-beg))
                if memory:
                    log_memory("  ")
                if init is not None:
                    print("")
                    print("Initializing '{}'".format(init))
                if run is not None:
                    print("Running '{}'".format(run))

            beg = datetime.now()
            return beg

        # Disk Torque
        # -------------------------------------
        beg_all = printer(init="Disk_Torque")
        disk_torq = Disk_Torque(self)

        beg = printer(run="Disk_Torque.dadt()", beg=beg_all)
        self.dadt_cd[:], _, disk_regs, _, disk_vtime = disk_torq.dadt()
        printer(beg=beg)
        self.rad_sg = disk_regs.rads[2]

        # Dynamical Friction
        # -------------------------------------
        beg = printer(init="Dynamical_Friction")
        dyn_fric = Dynamical_Friction(self)

        beg = printer(run="Dynamical_Friction.dadt()", beg=beg)
        self.dadt_df[:] = dyn_fric.dadt()
        printer(beg=beg)

        # Stellar Scattering
        # -------------------------------------
        beg = printer(init="Stellar_Scattering_Scalings")
        stel_scat = Stellar_Scattering_Scalings(self)

        beg = printer(run="Stellar_Scattering_Scalings.dadt()", beg=beg)
        self.dadt_sc[:] = stel_scat.dadt()
        printer(beg=beg)

        # Gravitational Waves
        # -------------------------------------
        beg = printer(init="Grav_Radiation")
        grav_rad = Grav_Radiation(self)

        beg = printer(run="Grav_Radiation.dadt()", beg=beg)
        self.dadt_gw[:], *_ = grav_rad.dadt()
        printer(beg=beg)

        self.dadt[:] = self.dadt_df + self.dadt_sc + self.dadt_cd + self.dadt_gw

        print("\nAll done after {}".format(datetime.now()-beg_all))

        return

    def calculate_timescale(self):
        # time_coal = 0.0
        eccen_final = 0.0
        verbose = self._verbose
        if verbose:
            print("EvolveLZK.calculate_timescale()")

        self.init_integral_arrays()
        self.integrate()

        rads = self.rads
        rads_2d = rads[np.newaxis, :]
        dadt = self.dadt
        num_binaries, num_rads = dadt.shape

        dr = np.concatenate((np.diff(rads), [0.0])) * np.ones_like(dadt)
        alive = (self.rad_isco[:, np.newaxis] <= rads_2d) & (rads_2d <= self.sep0[:, np.newaxis])

        # Calculate the durations for normal, mid-evolution steps
        times = np.zeros_like(dadt)
        inds = (dadt != 0.0)
        inds[:, -1] = False
        times[inds] = - dr[inds] / dadt[inds]

        # Calculate first step duration
        # -----------------------------------------------

        # Find the last valid index (largest separation) for each binary
        # `argmax` returns the first index, so reverse it to get the last index
        beg = np.argmax(np.flip(alive, axis=-1), axis=-1)
        # (un)reverse the index numbers
        beg = (num_rads - 1) - beg

        dr_beg = self.sep0 - rads[beg]
        times[:, beg] = - dr_beg / dadt[:, beg]

        # Calculate last step duration
        # -----------------------------------------------

        ''' # NOTE: this will always be a tiny duration... so just ignore it
        # Find the first valid index (smallest separation) for each binary
        end = np.argmax(alive)
        # Then go one index lower, minimum 0 (binary with r[0] == ISCO)
        end = np.maximum(end - 1, 0)
        # Set 'alive' values to True for this extra piece of a step
        dr_end = rads[end] - self.rad_isco
        times[end] = - dr_end / dadt[end]
        '''

        # Find cumulative lifetimes
        durs = np.sum(times * alive, axis=-1)

        # Find coalescence redshift
        tage_coal = self.tage_form + durs
        redz_coal = self._cosmo.tage_to_z(tage_coal)
        inds_coal = np.isfinite(redz_coal)
        num_coal = np.count_nonzero(inds_coal)
        frac = num_coal/num_binaries

        if verbose:
            print("EvolveLZK.calculate_timescale()")
            print("Lifetimes: " + zmath.stats_str(durs/GYR) + " [Gyr]")
            print("Coalescence redshifts: " + zmath.stats_str(redz_coal))
            print("\t(valid) Coalescence redshifts: " + zmath.stats_str(redz_coal[redz_coal >= 0.0]))
            print("{}/{} = {:.4f} coalesce before z=0".format(num_coal, num_binaries, frac))

        self.times = times
        self.durs = durs
        self.tage_coal = tage_coal
        self.redz_coal = redz_coal
        self.inds_coal = inds_coal

        return durs, eccen_final


def log_memory(pref=None, log=None, lvl=logging.DEBUG):
    """Log the current memory usage.
    """
    cyc_str = ""
    KB = 1024.0

    if log is None:
        print_main = print
        print_error = print
    else:
        print_main = lambda xx: log.log(lvl, xx)
        print_error = lambda xx: log.debug(xx)

    if pref is not None:
        cyc_str += "{}: ".format(pref)

    if sys.platform.startswith('linux'):
        RUSAGE_UNIT = 1024.0
    elif sys.platform.startswith('darwin'):
        RUSAGE_UNIT = 1024.0*1024.0

    try:
        import resource
        max_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        _str = "RSS Max Self: {:.2f} [MB], Child: {:.2f} [MB]".format(
            max_self/RUSAGE_UNIT, max_child/RUSAGE_UNIT)
        print_main(cyc_str + _str)
    except Exception as err:
        msg = "resource.getrusage failed.  '{}'".format(str(err))
        print_error(msg)

    try:
        import psutil
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        cpu_perc = process.cpu_percent()
        mem_perc = process.memory_percent()
        num_thr = process.num_threads()
        _str = "RSS: {:.2f} [MB], {:.2f}%; Threads: {:3d}, CPU: {:.2f}%".format(
            rss/KB/KB, mem_perc, num_thr, cpu_perc)
        print_main(cyc_str + _str)
    except Exception as err:
        msg = "psutil.Process failed.  '{}'".format(str(err))
        print_error(msg)

    return
