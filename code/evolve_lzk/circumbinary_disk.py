"""
"""
import numpy as np
from collections import namedtuple

# from zcode.constants import NWTG, SPLC, MSOL, MPRT, YR, SIGMA_T, PC
import zcode.astro as zastro

# from mbhmergers import physics
from . import Hardening_Mechanism, NWTG, SPLC, MSOL, PC, SIGMA_T, MPRT, YR

# Tuple describing the 3 critical (transition) radii, and which array elements belong in each
#    `rads` : 3x[(N,) array], list of three arrays -- each critical radius for each array element
#    `inds` : 4x[[]], list of four lists --- for each region, list of array elements in that region
#    `map`  : (N,) array of int --- for each array element, the region it belongs in.
Regions = namedtuple('regions', ('rads', 'inds', 'map'))


class Disk_Torque(Hardening_Mechanism):

    #    Units of each are given in square-brackets '[]'
    # Cutoff the circumbinary disk with self-gravity (`None` or `0.0` for no cutoff)
    SELF_GRAV_RAD = 1.0e0              # (1.0) units of sg-critical radius
    # Maximum self-gravity cutoff radius (only if SG is on, i.e. if ``SELF_GRAV_RAD``)
    #    0.1
    SELF_GRAV_RAD_MAX = 1.0e0            # (1.0) [pc]
    #    Apply maximum SG cutoff to region 1 (normally never SG unstable)
    SG_MAX_REG1 = True
    # Model parameters
    DISK_GAP_RAD = 1.0                 # Edge of disk-gap in binary radii [1.0]
    BETA_VISC = False                  # Visc-pres relation (`False`: alpha-disk)
    MEAN_MOL_WEIGHT_TIL_MU_0 = 1.0     # Mean molecular weight [0.615]
    MEAN_MASS_PER_ELEC_TIL_MU_E = 1.0  # Mean mass per electron [0.875]
    TEMP_OPAC_CONST_TIL_F_T = 1.0      # Temperature-opacity constant [0.75]
    KAPPA_ES = 1.0                 # Electron scattering opacity [6.65e-25 cm^2]
    KAPPA_FF = 1.0                 # Free-free scattering opacity [8e22 cm^2/g]
    ALPHA_VISC_03 = 1.0                # Alpha viscocity parameter [0.3]
    RAD_EFF_EPS_01 = 1.0

    def dadt(self):
        """Binary Hardening Rate from a Circumbinary Gaseous Disk.

        Arguments
        ---------
        rads : (N,) array_like of scalar
            Binary separation radius in [cm].
        m1 : array_like of scalar
            Mass of one of the two BHs in [g].
        m2 : array_like of scalar
            Mass of another of the two BHs in [g].
        mdot : array_like of scalar
            Mass accretion rate in units of [g/s].

        See the [Common Arguments and Parameters] section of the `circumbinary_disk.py` docs
        for more argument information.

        Returns
        -------
        dadt_visc : (N,) array_like of scalar
            Binary hardening rate in [cm/s] for the given binary separations.
        tgas : (N,) array_like of scalar
            Timescale for hardening due to circumbinary gas-drag at the given binary separations.
        regs : namedtuple('regions', ('rads', 'inds', 'map'))
            Which disk-regions each array element belongs in.  See `circumbinary_disk._rad_regions`.
        qb : (N,) array_like of scalar
            Disk-dominance parameter.  Ignores self-gravity.
        tv : (N,) array_like of scalar
            Viscous timescale.  Ignores self-gravity.

        """
        evolver = self._evolver
        rads = evolver.rads[np.newaxis, :]
        m1 = evolver.m1[:, np.newaxis]
        m2 = evolver.m2[:, np.newaxis]
        mdot = evolver.mdot[:, np.newaxis]

        # Find which array elements are in which disk regions
        regs = self._rad_regions(rads, m1+m2, mdot)

        # Disk dominance parameter (modulates viscous-time)
        #    This is [q1, q2, q3, qb]
        qset = self.qdom(rads, m1, m2, mdot, regs=regs)

        # Viscous-Timescale evaluated at binary position (relevant for disk-dominated regions)
        #    This is [v1, v2, v3, vt]
        vset = self.tvisc(rads, m1+m2, mdot, regs=regs)

        # Inspiral Timescale, overall
        t1, t2, t3, ti = self.time_inspiral(rads, m1, m2, mdot, regs=regs, qs=qset, vs=vset)

        # Turn off drag (set time to inf) in self-gravity instability region (4)
        ti = self._self_grav(regs, ti)

        # Convert to hardening rate (negative) da/dt (i.e. dR/dt)
        dadt_visc = -rads/ti

        return dadt_visc, ti, regs, qset[-1], vset[-1]

    def time_inspiral(self, rads, m1, m2, mdot, regs=None, qs=None, vs=None):
        lam = self.DISK_GAP_RAD

        if regs is None:
            regs = self._rad_regions(rads, m1+m2, mdot)

        # Disk dominance parameter (modulates viscous-time)
        if qs is None:
            qs = self.qdom(rads, m1, m2, mdot, regs=regs)
        q1, q2, q3, qb = qs
        # Viscous-Timescale evaluated at binary position (relevant for disk-dominated regions)
        if vs is None:
            vs = self.tvisc(rads, m1+m2, mdot, regs=regs)
        v1, v2, v3, vt = vs

        # Viscous-Timscale evaluated at edge of gap (relevant for secondary-dominated regions)
        if np.isclose(lam, 1.0):
            v1_l, v2_l, v3_l = np.array([v1, v2, v3])
        else:
            v1_l, v2_l, v3_l = self.tvisc(rads*lam, m1+m2, mdot, regs=regs)
            # regs_l = _rad_regions(rads*lam, m1+m2, mdot, sets)
            # tv_lambda = _merge_regions(regs_l, np.inf*np.ones_like(rads), t1_l, t2_l, t3_l, None)

        # Calculate Timescales
        # --------------------
        # Convert from viscous-times to actual hardening times
        t1, t2, t3 = self._inspiral_time([v1, v2, v3], [v1_l, v2_l, v3_l], [q1, q2, q3])
        ti = self._merge_regions(regs, np.inf*np.ones_like(t1), t1, t2, t3, None)

        return t1, t2, t3, ti

    def tvisc(self, rads, mtot, mdot, regs=None):
        """Viscous Timescale in the Appropriate Regions.

        Region-4 (Self-gravity unstable) is *ignored* in this calculation!

        Arguments
        ---------
        rads : array_like of scalar
            Binary separation radius in [cm].
        lam : array_like of scalar
            Gap size relative to the binary separation, $R_\lambda = \lambda R$.
        mtot : array_like of scalar
            Combined Mass of the two BHs in [g].
        mdot : array_like of scalar
            Mass accretion rate in units of [g/s].

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------

        """
        lam = self.DISK_GAP_RAD
        tmu_0 = self.MEAN_MOL_WEIGHT_TIL_MU_0
        tmu_e = self.MEAN_MASS_PER_ELEC_TIL_MU_E
        tf_t = self.TEMP_OPAC_CONST_TIL_F_T
        tkap_es = self.KAPPA_ES
        tkap_ff = self.KAPPA_FF
        alpha_03 = self.ALPHA_VISC_03
        eps_01 = self.RAD_EFF_EPS_01

        rads = np.atleast_1d(rads)
        if regs is None:
            regs = self._rad_regions(rads, mtot, mdot)

        v1 = self._tvisc_1(rads, lam, mtot, mdot,
                           tmu_e=tmu_e, tf_t=tf_t, tkap_es=tkap_es,
                           alpha_03=alpha_03, eps_01=eps_01)
        v2 = self._tvisc_2(rads, lam, mtot, mdot,
                           tmu_0=tmu_0, tmu_e=tmu_e, tf_t=tf_t, tkap_es=tkap_es,
                           alpha_03=alpha_03, eps_01=eps_01)
        v3 = self._tvisc_3(rads, lam, mtot, mdot,
                           tmu_0=tmu_0, tmu_e=tmu_e, tf_t=tf_t, tkap_ff=tkap_ff,
                           alpha_03=alpha_03, eps_01=eps_01)

        vs = self._merge_regions(regs, np.inf*np.ones_like(v1), v1, v2, v3, None)
        return v1, v2, v3, vs

    def qdom(self, rads, m1, m2, mdot, regs=None):
        """Disk Dominance Parameter q_B in the appropriate region.

        Region-4 (Self-gravity unstable) is *ignored* in this calculation!

        Arguments
        ---------
        rads : array_like of scalar
            Binary separation radius in [cm].
        lam : array_like of scalar
            Gap size relative to the binary separation, $R_\lambda = \lambda R$.
        m1 : array_like of scalar
            Mass of one of the two BHs in [g].
        m2 : array_like of scalar
            Mass of another of the two BHs in [g].
        mdot : array_like of scalar
            Mass accretion rate in units of [g/s].

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------
        qb : array_like of scalar
            Disk-dominance parameter q_B at the given radii.

        """
        lam = self.DISK_GAP_RAD
        tmu_0 = self.MEAN_MOL_WEIGHT_TIL_MU_0
        tmu_e = self.MEAN_MASS_PER_ELEC_TIL_MU_E
        tf_t = self.TEMP_OPAC_CONST_TIL_F_T
        tkap_es = self.KAPPA_ES
        tkap_ff = self.KAPPA_FF
        alpha_03 = self.ALPHA_VISC_03
        eps_01 = self.RAD_EFF_EPS_01

        rads = np.atleast_1d(rads)
        if regs is None:
            regs = self._rad_regions(rads, m1+m2, mdot)

        q1 = self._qdom_1(rads, lam, m1, m2, mdot,
                          tmu_e=tmu_e, tf_t=tf_t, tkap_es=tkap_es,
                          alpha_03=alpha_03, eps_01=eps_01)
        q2 = self._qdom_2(rads, lam, m1, m2, mdot,
                          tmu_0=tmu_0, tmu_e=tmu_e, tf_t=tf_t, tkap_es=tkap_es,
                          alpha_03=alpha_03, eps_01=eps_01)
        q3 = self._qdom_3(rads, lam, m1, m2, mdot,
                          tmu_0=tmu_0, tmu_e=tmu_e, tf_t=tf_t, tkap_ff=tkap_ff,
                          alpha_03=alpha_03, eps_01=eps_01)

        qb = self._merge_regions(regs, np.ones_like(q1), q1, q2, q3, None)
        return q1, q2, q3, qb

    def _self_grav(self, regs, tinsp):
        """Set inspiral timescale in Self-Gravity unstable region to infinite.
        """
        sg_rad, sg_rad_max = self._parse_sg_rads()
        if sg_rad > 0.0:
            tinsp[regs.inds[3]] = np.inf
        return tinsp

    def _inspiral_time(self, tvs, tvs_lam, qbs):
        """Use the viscous-time and disk-dominance-parameter (q) to determine the actual inspiral time.
        """
        t1, t2, t3 = tvs
        t1_l, t2_l, t3_l = tvs_lam
        q1, q2, q3 = qbs
        # Inspiral times start as viscous times
        s1, s2, s3 = np.array([t1, t2, t3])

        # Exponent `k` is 0 when ``q_b > 1.0``, and otherwise:
        #    3/8  in the electron-scattering region (2),
        #    7/17 in the free-free region (3).
        #    Evaluate at the edge of the gap (``R*lam``, i.e. `tv_lambda`)
        #    See Haiman+09 Eq.20.
        # Region 2 - ES-dominated
        sec_dom = (q2 < 1)
        s2[sec_dom] = np.power(q2[sec_dom], -3/8) * t2_l[sec_dom]
        # Region 3 - FF-dominated
        sec_dom = (q3 < 1)
        s3[sec_dom] = np.power(q3[sec_dom], -7/17) * t3_l[sec_dom]

        return s1, s2, s3

    def _mass_ratio_sym(self, m1, m2):
        """Calculate the normal and symmetric mass-ratios

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------
        qq : scalar or array_like of scalar
            Normal mass-ratio, defined as ``qq <= 1.0``.
        qs : scalar or array_like of scalar
            Normalized, symmetric mass-ratio.
        """
        # Normal mass ratio (Make sure this is ``qq <= 1.0``).
        qq = np.min([m1, m2], axis=0) / np.max([m1, m2], axis=0)
        # Normalized, Symmetric Mass-Ratio
        qs = 4*qq / np.square(1+qq)
        return qs

    def _mdot_edd(self, tmass, eps_01=1.0, tmu_e=1.0):
        """Eddington Accretion Rate.

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------
        mde : scalar or array_like of scalar
            Eddington accretion rate in grams/sec.

        """
        mde = (4*np.pi/0.0875)*NWTG*MPRT/(eps_01*tmu_e*SIGMA_T*SPLC)
        mde *= tmass
        return mde

    def _rad_regions(self, rads, tmass, mdot):
        """Find the indices for radii in each disk region.

        Arguments
        ---------
        rads : (N,) array_like of scalar
            Binary separation radius in [cm].
        tmass : array_like of scalar
            Total (combined) mass of both blackholes in [grams].
        mdot : array_like of scalar
            Mass accretion rate in units of [g/s].
        sg_rad : float
            Factor in radius (``sg_rad * r^{sg}``) at which to call the disk unstable.

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------
        regs: namedtuple('regions', ('rads', 'inds', 'map'))
            `rads`: [cr_12, cr_23, cr_sg]
                Which are the three critical radii for each element of the input arrays.
            `inds`: [inds1, inds2, inds3, inds4]
                For each disk-region, the indices of array elements which belong in it.
            `map`: rmap
                For each array-element, the (int) disk-region it belongs in {1,2,3,4}.

        """
        beta_visc = self.BETA_VISC
        sg_rad, sg_rad_max = self._parse_sg_rads()
        tmu_0 = self.MEAN_MOL_WEIGHT_TIL_MU_0
        tmu_e = self.MEAN_MASS_PER_ELEC_TIL_MU_E
        tf_t = self.TEMP_OPAC_CONST_TIL_F_T
        tkap_es = self.KAPPA_ES
        tkap_ff = self.KAPPA_FF
        alpha_03 = self.ALPHA_VISC_03
        eps_01 = self.RAD_EFF_EPS_01

        cr_12 = self._rad_crit_12(tmass, mdot,
                                  tmu_0=tmu_0, tmu_e=tmu_e, tkap_es=tkap_es,
                                  alpha_03=alpha_03, eps_01=eps_01)
        cr_23 = self._rad_crit_23(tmass, mdot,
                                  tmu_0=tmu_0, tmu_e=tmu_e, tf_t=tf_t,
                                  tkap_es=tkap_es, tkap_ff=tkap_ff, eps_01=eps_01)

        # Determine Radial Regions
        # ------------------------
        # Region 1: $b=0 and r_3 < r_3^{gas/rad}$
        _inds1 = ((not beta_visc) & (rads <= cr_12))
        inds1 = np.where(_inds1)[0]
        # Region 2: $(b=1 and r_3 < r_3^{gas/rad})$ or $r_3^{Gas/rad} < r_3 < r_3^{es/ff}$
        _inds2 = ((rads > cr_12) & (rads <= cr_23)) | ((beta_visc) & (rads <= cr_12))
        inds2 = np.where(_inds2)[0]
        # Region 3: $r_3 > r_3^{es/ff}$ and $r_3 < r_3^{sg}$
        _inds3 = (rads >= cr_23)
        inds3 = np.where(_inds3)[0]

        # bads = np.where(_inds1 & _inds2)[0]
        # if bads.size:
        if np.any(_inds1 & _inds2):
            bads = np.where(_inds1 & _inds2)[0]
            err_str = "Overlap between region 1 and region 2\n{}".format(bads)
            raise ValueError(err_str)

        # bads = np.where(_inds1 & _inds3)[0]
        # if bads.size:
        if np.any(_inds1 & _inds3):
            bads = np.where(_inds1 & _inds3)[0]
            err_str = "Overlap between region 1 and region 3\n{}".format(bads)
            raise ValueError(err_str)

        # For each radius, indicate which region (for Regions 1,2,3)
        #    Region 4 (self-gravity unstable, is added later)
        rmap = np.zeros_like(_inds1, dtype=int)
        for ii, rr in enumerate([inds1, inds2, inds3], 1):
            rmap[rr] = ii

        # Make sure all array elements are in *some* region
        # bads = np.where(rmap == 0)[0]
        # if bads.size:
        if np.any(rmap == 0):
            bads = np.where(rmap == 0)[0]
            err_str = ("Some radii not mapped to regions {1,2,3}."
                       "\n\tbads = {}\n\trads = {}\n\ttmass = {}\n\tmdot = {}")
            err_str = err_str.format(bads, rads[bads], tmass[bads], mdot[bads])
            raise ValueError(err_str)

        # Self-Gravity
        # ------------
        if sg_rad > 0.0:
            # Check for self-gravity instability separately
            #    Region 2
            cr_sg_es = self._rad_crit_sg_es(
                tmass, mdot, tmu_0=tmu_0, tmu_e=tmu_e, tf_t=tf_t,
                tkap_es=tkap_es, tkap_ff=tkap_ff, alpha_03=alpha_03, eps_01=eps_01)
            #    Region 3
            cr_sg_ff = self._rad_crit_sg_ff(
                tmass, mdot, tmu_0=tmu_0, tmu_e=tmu_e, tf_t=tf_t,
                tkap_es=tkap_es, tkap_ff=tkap_ff, alpha_03=alpha_03, eps_01=eps_01)
            # Combine into single array of radii appropriate for each element
            #    Values in region 1 (i.e. *not* in 2 or 3, will have ``cr_sg == np.inf``).
            #    Apply radial scaling
            cr_sg = sg_rad * self._rad_crit_sg(cr_sg_es, cr_sg_ff, _inds2, _inds3)
            # Apply maximum self-gravity radius
            if sg_rad_max > 0.0:
                cr_sg[inds2] = np.minimum(sg_rad_max, cr_sg[inds2])
                cr_sg[inds3] = np.minimum(sg_rad_max, cr_sg[inds3])
                # Also apply maximum cutoff to region 1 (not normally SG unstable)
                if self.SG_MAX_REG1:
                    cr_sg[inds1] = np.minimum(sg_rad_max, cr_sg[inds1])

            # Check for radii larger than self-gravity critical
            num_reg1 = np.count_nonzero(rmap == 1)
            inds4 = np.where(rads > cr_sg)[0]
            # Add self-gravity region to mapping
            rmap[inds4] = 4
            # Make sure only regions 2 and regions 3 are SG unstable, is settings says so
            if not self.SG_MAX_REG1 and num_reg1 != np.count_nonzero(rmap == 1):
                i_reg1 = np.where(rmap == 1)[0]
                j_reg1 = np.where(rmap == 1)[0]
                for ii in i_reg1:
                    if ii not in j_reg1:
                        print(ii, "overwritten")
                        print("\trads = ", rads[ii]/PC)
                        print("\tcr_sg = ", cr_sg[ii]/PC)
                        print("\t_inds2 = ", _inds2[ii])
                        print("\t_inds3 = ", _inds3[ii])
                        break

                diff = num_reg1 - np.count_nonzero(rmap == 1)
                err_str = "Region 1 ({}) has been overwritten by self-gravity.".format(diff)
                raise ValueError(err_str)

        else:
            cr_sg = np.zeros_like(cr_12)
            inds4 = []

        regs = Regions([cr_12, cr_23, cr_sg], [inds1, inds2, inds3, inds4], rmap)
        return regs

    def _rad_crit_12(self, tmass, mdot, tmu_0=1.0, tmu_e=1.0, tkap_es=1.0, alpha_03=1.0, eps_01=1.0):
        """Critical Radius between regions 1 and 2, i.e. $r^{gas/rad}$.

        Arguments
        ---------
        tmass : scalar or array_like of scalar
            Total (combined) mass of both blackholes in [grams].
        mdot : scalar or array_like of scalar
            Accretion rate in [grams/sec].

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------
        cr_12 : scalar or array_like of scalar
            Critical radius between regions 1 and 2.

        """
        _r3 = 1e3 * zastro.schwarzschild_radius(tmass)
        _mdot = mdot / (0.1*self._mdot_edd(tmass, eps_01, tmu_e))
        _tmass = tmass/(1e7*MSOL)
        cr_12 = 0.482*np.power(tmu_0, 8./21)*np.power(tmu_e, 2./21)*np.power(tkap_es, 6./7)
        cr_12 *= np.power(alpha_03, 2./21)*np.power(_mdot, 16./21)*np.power(_tmass, 2./21)
        # Convert from r_3 (1e3*R_s) to CGS
        cr_12 *= _r3
        return cr_12

    def _rad_crit_23(self, tmass, mdot, tmu_0=1.0, tmu_e=1.0, tf_t=1.0,
                     tkap_es=1.0, tkap_ff=1.0, eps_01=1.0):
        """Critical Radius between regions 2 and 3, i.e. $r^{es/ff}$.

        Arguments
        ---------
        tmass : scalar or array_like of scalar
            Total (combined) mass of both blackholes in [grams].
        mdot : scalar or array_like of scalar
            Accretion rate in [grams/sec].

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------
        cr_23 : scalar or array_like of scalar
            Critical radius between regions 1 and 2.

        """
        _r3 = 1e3 * zastro.schwarzschild_radius(tmass)
        _mdot = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))
        cr_23 = 4.1*np.power(tmu_0, -1./3)*np.power(tf_t, 17./12)*np.power(tkap_ff/tkap_es, -2./3)
        cr_23 *= np.power(_mdot, 2./3)
        # Convert from r_3 (1e3*R_s) to CGS
        cr_23 *= _r3
        return cr_23

    def _rad_crit_sg_es(self, tmass, mdot, tmu_0=1.0, tmu_e=1.0, tf_t=1.0,
                        tkap_es=1.0, tkap_ff=1.0, alpha_03=1.0, eps_01=1.0):
        """Critical Radius when the disk becomes unstable to self-gravity in Region-2, i.e. $r^{sg}$.

        Arguments
        ---------
        tmass : scalar or array_like of scalar
            Total (combined) mass of both blackholes in [grams].
        mdot : scalar or array_like of scalar
            Accretion rate in [grams/sec].

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------
        cr_sg_es : scalar or array_like of scale-factor
            Critical radius at which the circumbinary disk becomes unstable to self-gravity in the
            ES-dominated regime (Region-2).

        """
        _r3 = 1e3*zastro.schwarzschild_radius(tmass)
        _mdot = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))
        _tmass = tmass/(1e7*MSOL)

        # Electron-Scattering Regime ($\tilde{\kappa}_{es} \rightarrow 1$)
        cr_sg_es = 12.6*np.power(tmu_0, -8.0/9.0)*np.power(tmu_e, 14.0/27.0)*np.power(tf_t, 20.0/9.0)
        cr_sg_es *= np.power(tkap_es, 2.0/9.0)*np.power(alpha_03, 8.0/9.0)
        cr_sg_es *= np.power(_mdot, -8.0/27.0)*np.power(_tmass, -26.0/27.0)
        cr_sg_es *= _r3
        return cr_sg_es

    def _rad_crit_sg_ff(self, tmass, mdot, tmu_0=1.0, tmu_e=1.0, tf_t=1.0,
                        tkap_es=1.0, tkap_ff=1.0, alpha_03=1.0, eps_01=1.0):
        """Critical Radius when the disk becomes unstable to self-gravity in Region-3, i.e. $r^{sg}$.

        Arguments
        ---------
        tmass : scalar or array_like of scalar
            Total (combined) mass of both blackholes in [grams].
        mdot : scalar or array_like of scalar
            Accretion rate in [grams/sec].

        See the [Common Arguments and Parameters] section for more argument information.

        Returns
        -------
        cr_sg_ff : scalar or array_like of scale-factor
            Critical radius at which the circumbinary disk becomes unstable to self-gravity in the
            FF-dominated regime (Region-3).

        """
        _r3 = 1e3*zastro.schwarzschild_radius(tmass)
        _mdot = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))
        _tmass = tmass/(1e7*MSOL)

        # Free-Free Regime ($\tilde{\kappa}_{ff} \rightarrow 1$)
        cr_sg_ff = 30.99*np.power(tmu_0, -1.0)*np.power(tmu_e, 28.0/45.0)*np.power(tf_t, 143.0/60.0)
        cr_sg_ff *= np.power(tkap_ff, 2.0/15.0)*np.power(alpha_03, 28.0/45.0)
        cr_sg_ff *= np.power(_mdot, -22.0/45.0)*np.power(_tmass, 52.0/45.0)
        cr_sg_ff *= _r3
        return cr_sg_ff

    def _rad_crit_sg(self, cr_sg_es, cr_sg_ff, inds2, inds3):
        """Combine the Self-Gravity critical radius for both regimes.

        Arguments
        ---------
        cr_sg_es : (N,) array of scalar
            Radii in the ES-regime ('1')
        cr_sg_ff : (N,) array of scalar
            Radii in the FF-regime ('2')
        inds2 : (N,) array of bool
            True for array elements in region '1'
        inds3 : (N,) array of bool
            True for array elements in region '2'

        Returns
        -------
        cr_sg : (N,) array of scalar
            Critical-Radius for self-gravity in the appropriate regime.
            Set to infinite outside of regions 2 and 3 (i.e. for region 1).

        """
        ones = np.ones(np.shape(inds2))
        # Set all values to infinite
        cr_sg = np.inf * ones
        # Set values in regions 2 and 3 appropriately
        cr_sg[inds2] = (ones * cr_sg_es)[inds2]
        cr_sg[inds3] = (ones * cr_sg_ff)[inds3]
        return cr_sg

    def _qdom_1(self, rad, lam, m1, m2, mdot, tmu_e=1.0, tf_t=1.0, tkap_es=1.0, alpha_03=1.0, eps_01=1.0):
        """Disk Dominance Parameter q_B in region 1.
        """
        tmass = m1+m2  # Total mass
        _mde = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))  # MDot in units of 0.1 Eddington
        _tmass = tmass/(1.0e7*MSOL)  # Total Mass in units of $1e7*Msol$
        _r3 = rad/(1e3*zastro.schwarzschild_radius(tmass))  # Radius in units of $1e3*R_s$
        qs = self._mass_ratio_sym(m1, m2)  # Normal and symmetric mass-ratios
        qb1 = 1.2e-3*np.power(tmu_e*alpha_03, -1.)*np.power(tkap_es*tf_t, -2.)
        qb1 *= (_tmass/_mde)*np.power(lam*_r3, 7./2)/qs
        return qb1

    def _qdom_2(self, rad, lam, m1, m2, mdot, tmu_0=1.0, tmu_e=1.0, tf_t=1.0,
                tkap_es=1.0, alpha_03=1.0, eps_01=1.0):
        """Disk Dominance Parameter q_B in region 2.
        """
        tmass = m1+m2  # Total mass
        _mde = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))  # MDot in units of 0.1 Eddington
        _tmass = tmass/(1.0e7*MSOL)  # Total Mass in units of $1e7*Msol$
        _r3 = rad/(1e3*zastro.schwarzschild_radius(tmass))  # Radius in units of $1e3*R_s$
        qs = self._mass_ratio_sym(m1, m2)  # Normal and symmetric mass-ratios
        qb2 = 1.1e-2*np.power(tmu_0/(tmu_e*alpha_03), 4./5)*np.power(tkap_es, -1./5)*np.power(tf_t, -2.)
        qb2 *= np.power(_tmass, 6./5)*np.power(_mde, 3./5)*np.power(lam*_r3, 7./5)/qs
        return qb2

    def _qdom_3(self, rad, lam, m1, m2, mdot, tmu_0=1.0, tmu_e=1.0, tf_t=1.0,
                tkap_ff=1.0, alpha_03=1.0, eps_01=1.0):
        """Disk Dominance Parameter q_B in region 3.
        """
        tmass = m1+m2  # Total mass
        _mde = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))  # MDot in units of 0.1 Eddington
        _tmass = tmass/(1.0e7*MSOL)  # Total Mass in units of $1e7*Msol$
        _r3 = rad/(1e3*zastro.schwarzschild_radius(tmass))  # Radius in units of $1e3*R_s$
        qs = self._mass_ratio_sym(m1, m2)  # Normal and symmetric mass-ratios
        qb3 = 1.49e-3*np.power(tmu_e*alpha_03, -4./5)*np.power(tmu_0, 3./4)*np.power(tkap_ff, -1./10)
        qb3 *= np.power(tf_t, -143./80)
        qb3 *= np.power(_tmass, 6./5)*np.power(_mde, 7./10)*np.power(lam*_r3, 5./4)/qs
        return qb3

    def _tvisc_1(self, rad, lam, tmass, mdot, tmu_e=1.0, tf_t=1.0, tkap_es=1.0, alpha_03=1.0, eps_01=1.0):
        """Viscous-Timescale in region 1.

        HKM09 - Eq. 21a
        """
        _mde = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))  # MDot in units of 0.1 Eddington
        _tmass = tmass/(1.0e7*MSOL)  # Total Mass in units of $1e7*Msol$
        _r3 = rad/(1e3*zastro.schwarzschild_radius(tmass))  # Radius in units of $1e3*R_s$
        tv_1 = (2.82e7*YR)*np.power(tkap_es*tf_t, -2.)/alpha_03
        tv_1 *= _tmass*np.power(_mde, -2.)*np.power(lam*_r3, 7./2)
        return tv_1

    def _tvisc_2(self, rad, lam, tmass, mdot, tmu_0=1.0, tmu_e=1.0, tf_t=1.0,
                 tkap_es=1.0, alpha_03=1.0, eps_01=1.0):
        """Viscous-Timescale in region 2.
        """
        _mde = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))  # MDot in units of 0.1 Eddington
        _tmass = tmass/(1.0e7*MSOL)  # Total Mass in units of $1e7*Msol$
        _r3 = rad/(1e3*zastro.schwarzschild_radius(tmass))  # Radius in units of $1e3*R_s$
        tv_2 = (5.96e4*YR)*np.power(tmu_e, 1./5)*np.power(tmu_0/alpha_03, 4./5)*np.power(tkap_es, -1./5)
        tv_2 *= np.power(tf_t, -2.)
        tv_2 *= np.power(_tmass, 6./5)*np.power(_mde, -2./5)*np.power(lam*_r3, 7./5)
        return tv_2

    def _tvisc_3(self, rad, lam, tmass, mdot, tmu_0=1.0, tmu_e=1.0, tf_t=1.0,
                 tkap_ff=1.0, alpha_03=1.0, eps_01=1.0):
        """Viscous-Timescale in region 3.
        """
        _mde = mdot/(0.1*self._mdot_edd(tmass, eps_01, tmu_e))  # MDot in units of 0.1 Eddington
        _tmass = tmass/(1.0e7*MSOL)  # Total Mass in units of $1e7*Msol$
        _r3 = rad/(1e3*zastro.schwarzschild_radius(tmass))  # Radius in units of $1e3*R_s$
        tv_3 = (7.37e4*YR)*np.power(tmu_e, 1./5)*np.power(tmu_0, 3./4)*np.power(tkap_ff, -1./10)
        tv_3 *= np.power(tf_t, -143./80)*np.power(alpha_03, -4./5)
        tv_3 *= np.power(_tmass, 6./5)*np.power(_mde, -3./10)*np.power(lam*_r3, 5./4)
        return tv_3

    def _parse_sg_rads(self):
        """Convert the Self-Gravity radii parameters in `settings` into consistent usable form.

        Input settings can be 'None'.
        And `SELF_GRAV_RAD_MAX` is in [pc] -- convert to [cm] always.
        """
        sg_rad = self.SELF_GRAV_RAD
        sg_rad_max = self.SELF_GRAV_RAD_MAX

        # If `SELF_GRAV_RAD` is 'None' or '0.0', then turn SG off --- i.e. no limits to drag
        if sg_rad is None or sg_rad <= 0.0:
            sg_rad_max = 0.0
            sg_rad = 0.0
        # If `SELF_GRAV_RAD_MAX` is 'None' or '0.0', then *no maximum*
        elif sg_rad_max is None or sg_rad_max <= 0.0:
            sg_rad_max = 0.0

        # Otherwise, if both SG is enabled, and max-rad is given, convert to parsec
        sg_rad_max *= PC

        return sg_rad, sg_rad_max

    def _merge_regions(self, regs, vals, v1, v2, v3, v4):
        """Merge the values from regions [1,2,3,4] into the output array `vals`.

        If the given value array (e.g. `v1`) is not 'None', then the elements with the corresponding
        indices (e.g. `regs.inds[0]`) are copied into `vals`.
        """
        for vv, ii in zip([v1, v2, v3, v4], regs.inds):
            if vv is None or ii.size == 0: continue
            vals[ii] = vv[ii]

        return vals
