"""Methods for calculating flux of stars into the scattering loss-cone.

Methods
-------
-   scattering                 - Calculate loss-cone scattering props based on galaxy parameters.
-   plotScattering             - Illustrate the scattering calculation for a single, sample system.
-   diffusionCoef              - diffusion coefficient for stars in galaxy
-   numStars_all               - Differential number of all stars.
-   numStars_flc               - Differential number of stars in the full loss-cone (FLC).
-   fluxStars_flc              - flux of stars in full loss-cone (FLC)
-   fluxStars_sslc             - flux of stars in steady-state loss-cone (SSLC)
-   interactionCriteria_binary - Characteristic interaction criteria for MBH binary with stars.
-   _MT1999_lnR0inv            - parameter for calculation of steady-state flux
-   dadt_scattering            - Hardening rate for a given flux of stars and binary parameters.
-   netFlux                    - Effective, net flux.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import scipy as sp
import scipy.special  # noqa
from matplotlib import pyplot as plt
import tqdm

from zcode.constants import MSOL, NWTG, PC
from zcode import math as zmath
from zcode import inout as zio
from zcode import plot as zplot

from mbhmergers.hardening import Hardening_Mechanism


class Scalings:
    """Class to hold Scattering-Rate Scaling Constants.
    """

    def __init__(self, sets, log):
        self.log = log
        self.sets = sets
        self._bound_H = [0.0, np.inf]
        self._bound_K = [0.0, np.inf]

        # Get the data filename
        input_fname = os.path.join(sets.DIR_DATA, sets.LC_DATA_FILENAME)
        log.info("Scattering Data filename: '{}'.".format(input_fname))
        if not os.path.isfile(input_fname):
            err = "File not does exist '{}'.".format(input_fname)
            log.error(err)
            raise ValueError(err)

        # Load Data
        import json
        data = json.load(open(input_fname, 'r'))
        self._store_data(data)
        log.debug("Data loaded.  Storing interpolants...")
        # 'H' : Hardening Rate
        self._init_h()
        # 'K' : Eccentricity growth
        self._init_k()
        return

    def _store_data(self, data):
        self._data = None
        return

    def _init_k(self):
        return

    def _init_h(self):
        return

    def _calc_H(self, *args):
        return None

    def _calc_K(self, *args):
        return None

    def _bound(self, val, bounds):
        # No lower than minimum
        val = np.maximum(val, bounds[0])
        # No higher than maximum
        val = np.minimum(val, bounds[1])
        return val

    def H(self, *args):
        hh = self._calc_H(*args)
        hh = self._bound(hh, self._bound_H)
        return hh

    def K(self, *args):
        kk = self._calc_K(*args)
        kk = self._bound(kk, self._bound_K)
        return kk


class Scalings_SHM06(Scalings):
    """Class to hold Scattering-Rate Scaling Constants.

    Specific for [Sesana, Hardt & Madau 2006]
    """

    def __init__(self, sets, log):
        super().__init__(sets, log)
        # Set bounds to zero, and to twice the highest value seen in plots.
        self._bound_H = [0.0, 40.0]
        self._bound_K = [0.0, 0.4]
        return

    def _store_data(self, data):
        self._data = data['SHM06']
        return

    def _init_k(self):
        data = self._data['K']
        #    Get all of the mass ratios (ignore other keys)
        _kq_keys = list(data.keys())
        kq_keys = []
        for kq in _kq_keys:
            try:
                int(kq)
                kq_keys.append(kq)
            except (TypeError, ValueError):
                pass

        nq = len(kq_keys)
        if nq < 2:
            raise ValueError("Something is wrong... `kq_keys` = '{}'\ndata:\n{}".format(
                kq_keys, data))
        k_mass_ratios = 1.0/np.array(sorted([int(kq) for kq in kq_keys]))
        k_eccen = np.array(data[kq_keys[0]]['e'])
        ne = len(k_eccen)
        k_A = np.zeros((ne, nq))
        k_a0 = np.zeros((ne, nq))
        k_g = np.zeros((ne, nq))
        k_B = np.zeros((ne, nq))

        for ii, kq in enumerate(kq_keys):
            _dat = data[kq]
            k_A[:, ii] = _dat['A']
            k_a0[:, ii] = _dat['a0']
            k_g[:, ii] = _dat['g']
            k_B[:, ii] = _dat['B']

        self.K_A = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_A, kind='linear')
        self.K_a0 = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_a0, kind='linear')
        self.K_g = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_g, kind='linear')
        self.K_B = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_B, kind='linear')
        return

    def _init_h(self):
        _dat = self._data['H']
        k_mass_ratios = 1.0/np.array(_dat['q'])
        k_A = np.array(_dat['A'])
        k_a0 = np.array(_dat['a0'])
        k_g = np.array(_dat['g'])

        self.H_A = sp.interpolate.interp1d(
            k_mass_ratios, k_A, kind='linear', fill_value='extrapolate')
        self.H_a0 = sp.interpolate.interp1d(
            k_mass_ratios, k_a0, kind='linear', fill_value='extrapolate')
        self.H_g = sp.interpolate.interp1d(
            k_mass_ratios, k_g, kind='linear', fill_value='extrapolate')
        return

    def _calc_H(self, q, aa):
        """

        Note: `aa` must be in units of a_h (hardening radius).
        """
        use_a = aa/self.H_a0(q)
        hh = self.H_A(q) * np.power((1 + use_a), self.H_g(q))
        return hh

    def _calc_K(self, q, e, aa):
        """

        Note: `aa` must be in units of a_h (hardening radius).
        """
        # `interp2d` return a matrix of X x Y results... want diagonal of that
        use_a = (aa/self.K_a0(q, e))
        A = self.K_A(q, e)
        g = self.K_g(q, e)
        B = self.K_B(q, e)
        try:
            use_a = use_a.diagonal()
            A = A.diagonal()
            g = g.diagonal()
            B = B.diagonal()
        except ValueError:
            wstr = "WARNING: `diagonal()` call failed.\n\tshape(use_a) = {}\n\tshape(A) = {}"
            wstr = wstr.format(np.shape(use_a), np.shape(A))
            self.log.warning(wstr)
            pass

        kk = A * np.power((1 + use_a), g) + B
        return kk


class Loss_Cone_Scalings(Hardening_Mechanism):

    # def harden(BINS, rads_hard, sets):
    def harden(self, m1, m2, sep, eccen, rads_hard, dens_stars, vdisp_stars, lc_scalings):
        """Loss-Cone Hardening Rates based on simple scaling relations.

        Formerly `scaling_harden()`

        See [Sesana 2010](2010ApJ...719..851S), Eq.8 & 9
        """
        # sep = BINS.sep
        a_a0 = sep/rads_hard
        # ecc = BINS.eccen
        # mrats = BINS.m2 / BINS.m1
        mrats = m2 / m1
        HH = lc_scalings.H(mrats, a_a0)

        dadt = - HH * NWTG * dens_stars * np.square(sep) / vdisp_stars
        dedt = np.zeros_like(dadt)
        if np.any(eccen > 0.0):
            KK = lc_scalings.K(mrats, a_a0, eccen)
            dedt[:] = sep * NWTG * dens_stars * HH * KK / vdisp_stars

        return dadt, dedt


class Loss_Cone_Explicit(Hardening_Mechanism):

    def harden(self, m1, m2, sep, rads, eps, periods, j2Circs, diffCoef,
               dist_funcs, inds, rads_hard):
        """Calculate loss-cone scattering properties based on galaxy parameters.

        Arguments
        ---------
            m1
            m2
            sep
            rads
            eps
            periods
            j2Circs
            diffCoef
            dist_funcs
            leeway
            refillFrac
            mstar
            inds

        Returns
        -------
            radLC
            j2LC
            enrLC
            dnStarsFLC
            numStarsFLC
            dfStarsFLC
            dfStarsSSLC
            fluxStarsFLC
            fluxStarsSSLC
            flux
            dadt_lc

        """
        sets = self.sets
        leeway = sets.BINARY_INTERACTION_FACTOR
        refillFrac = sets.LC_REFILL
        mstar = sets.MSTAR * MSOL

        # Rearrange to sort by energies, instead of radii
        rads = rads[::-1]
        eps = eps[..., ::-1]
        periods = periods[..., ::-1]
        j2Circs = j2Circs[..., ::-1]
        diffCoef = diffCoef[..., ::-1]
        dist_funcs = dist_funcs[..., ::-1]

        # integFunc = zmath.cumtrapz_rev
        integFunc = sp.integrate.cumtrapz
        # Determine the Loss-Cone defining parameters
        radLC, j2LC, enrLC = interactionCriteria_binary(m1+m2, sep, leeway)
        # Calculate number of stars in the Full Loss-Cone (FLC)
        dnStarsFLC = numStars_flc(dist_funcs, periods, j2Circs, j2LC)
        #     integrate to get number of stars
        numStarsFLC = integFunc(dnStarsFLC, eps, initial=0.0)

        bads = _findInvalid(numStarsFLC)
        if bads:
            print("bads = ", bads)
            #    Choose sample bad element
            badInd = bads[0][0]
            print("badInd = ", badInd)
            print("inds[badInd]", inds[badInd])
            print("stellar_scattering = ", radLC, j2LC, enrLC)
            print("dist_funcs = ", dist_funcs[badInd])
            print("periods = ", periods[badInd])
            print("j2Circs = ", j2Circs[badInd])
            print("\n\n")
            print("eps = ", eps[badInd])
            print("dnStarsFLC = ", dnStarsFLC[badInd])
            print("numStarsFLC = ", numStarsFLC[badInd])
            raise ValueError("Invalid values detected in `numStarsFLC`!")

        # Calculate flux distribution of Full Loss-Cone (FLC)
        dfStarsFLC = fluxStars_flc(dnStarsFLC, periods)
        #     integrate to get net flux of FLC
        fluxStarsFLC = integFunc(dfStarsFLC, eps, initial=0.0)

        if _findInvalid(fluxStarsFLC):
            print("\nperiod = ", periods)
            print("eps = ", eps)
            print("sep = ", sep)
            print("dnStarsFLC = ", dnStarsFLC)
            print("dfStarsFLC = ", dfStarsFLC)
            print("\nfluxStarsFLC = ", fluxStarsFLC)
            raise ValueError("Invalid values detected in `fluxStarsFLC`!")

        # Calculate flux distribution of steady-state Loss-Cone (SSLC)
        dfStarsSSLC = fluxStars_sslc(rads, dnStarsFLC, diffCoef, periods, j2Circs, j2LC)
        #     integrate to get net flux of SSLC
        fluxStarsSSLC = integFunc(dfStarsSSLC, eps, initial=0.0)

        if _findInvalid(fluxStarsSSLC):
            print("rads = ", rads)
            print("eps = ", eps)
            print("sep = ", sep)
            print("leeway = ", leeway)
            print("diffCoef = ", diffCoef)
            print("j2Circs = ", j2Circs)
            print("radLC = ", radLC)
            print("enrLC = ", enrLC)
            print("j2LC = ", j2LC)
            print("dnStarsFLC = ", dnStarsFLC)
            print("dfStarsSSLC = ", dfStarsSSLC)
            print("fluxStarsSSLC = ", fluxStarsSSLC)
            raise ValueError("Invalid values detected in `fluxStarsSSLC`!")

        # Infer an 'effective' number of stars in the steady state
        numStarsSSLC = fluxStarsSSLC*periods
        if _findInvalid(numStarsSSLC):
            raise ValueError("Invalid values detected in `numStarsSSLC`!")

        # Net flux (interpolated between steady and full)
        flux = netFlux(fluxStarsFLC, fluxStarsSSLC, refillFrac)
        # Hardening Rate
        dadt_lc = dadt_scattering(m1+m2, sep, flux, mstar)
        #     `_findInvalid` checks for negative values, `dadt_lc` *should* be negative.  Reverse.
        bads = _findInvalid(-dadt_lc)
        if bads:
            print("Bads = {}\ndadt_lc[bads] = {}".format(bads, dadt_lc))
            raise ValueError("Invalid values detected in `dadt_lc`.  Bads = {}".format(bads))

        return radLC, j2LC, enrLC, dnStarsFLC, numStarsFLC, numStarsSSLC, \
            dfStarsFLC, dfStarsSSLC, fluxStarsFLC, fluxStarsSSLC, flux, dadt_lc


def plotScattering(sample, snap, mbhb, log, plotNames):
    """Illustrate the scattering calculation for a single, sample system.

    Performs calculation by calling the 'scattering()' method, just like in
    "MBHBinaryEvolution.py".

    Arguments
    ---------
    sample : int
        Target galaxy/merger number to examine (this is the number out of *all* binaries,
        not just the valid ones [included in ``mbhb.evolution``]).
    snap : int
        Illustris snapshot number {1, 135}.
    mbhb : `Binaries.MBHBinaries` object
    log : `logging.Logger` object
    plotNames : str
        Base name of plots.

    Returns
    -------
    plotNames : list of str
        Filenames of the plots created.

    """
    log.debug("plotScattering()")
    PLOT_A10 = True     # LC Occupancy
    PLOT_A11 = True     # Flux
    PLOT_A15 = True      # Flux vs. Separation
    PLOT_A02 = False     # Model Galaxy
    PLOT_A08 = False      # Dist Func

    from . import GM_Figures

    figNames = []

    # Initialization
    # --------------
    radialRange = np.array(mbhb.sets.RADIAL_RANGE_MODELS) * PC
    numSeps = mbhb.sets.PLOT_SCATTERING_SAMPLE_SEPS
    # Convert from `sample` number, of all binaries to index for valid (``mbhb.evolution``) ones
    val_inds = np.where(mbhb.valid)[0]
    valid_sample = np.where(val_inds == sample)[0]
    if valid_sample.size == 1:
        valid_sample = valid_sample[0]
    else:
        raise ValueError("`sample` '{}' returned '{}' from `val_inds`".format(sample, valid_sample))

    mstar = mbhb.galaxies.mstar

    log.debug(" - Sample subhalo %d, Snapshot %d" % (sample, snap))

    # Binary Properties (snapshot dependent)
    #    `evolution` class includes only valid binaries, so use `valid_sample`
    # m1 = np.max(mbhb.evolution.masses[valid_sample, snap])
    # m2 = np.min(mbhb.evolution.masses[valid_sample, snap])
    m1 = np.max(mbhb.initMasses[sample])
    m2 = np.min(mbhb.initMasses[sample])

    # Galaxy properties (snapshot independent)
    #    `galaxies` class includes *all* binaries, so use `sample` itself
    gals       = mbhb.galaxies
    eps        = gals.eps[sample]
    #     functions of energy
    rads       = gals.rads
    periods    = gals.perOrb[sample]
    dist_func   = gals.dist_func[sample]
    diffCoef   = gals.diffCoef[sample]
    j2Circs    = gals.j2Circ[sample]
    dnStarsAll = gals.dnStarsAll[sample]
    ndensStars = gals.densStars[sample]/mstar
    radHard    = gals.rads_hard[sample]

    numStarsAll = sp.integrate.cumtrapz(dnStarsAll[::-1], eps[::-1], initial=0.0)[::-1]
    loss_cone = Loss_Cone_Explicit(mbhb.sets, log)

    # Wrap the Scattering function (for convenience)
    def evalScattering(binSep):
        # retvals = scattering(m1, m2, binSep, rads, eps, periods, j2Circs, diffCoef, dist_func,
        #                      [sample], radHard, mbhb.sets)
        retvals = loss_cone.harden(
            m1, m2, binSep, rads, eps, periods, j2Circs, diffCoef, dist_func,
            [sample], radHard, mbhb.sets)

        radLC, j2LC, enrLC, dnStarsFLC, numStarsFLC, numStarsSSLC, \
            dfStarsFLC, dfStarsSSLC, fluxStarsFLC, fluxStarsSSLC, flux, dadt_lc = retvals

        return dnStarsFLC, numStarsFLC, dfStarsFLC, fluxStarsFLC, dfStarsSSLC, fluxStarsSSLC

    num_flc   = np.zeros(numSeps)
    flx_flc   = np.zeros(numSeps)
    flx_sslc  = np.zeros(numSeps)
    dadt_sslc = np.zeros(numSeps)
    dadt_flc  = np.zeros(numSeps)

    subPlotNames = zio.modify_filename(plotNames, prepend='stellar_scattering/')
    zio.check_path(subPlotNames)

    # Iterate over range of binary separations, plot profiles for each
    log.debug(" - Calculting scattering for %d binary separations" % (numSeps))
    flx_seps = zmath.spacing(radialRange, scale='log', num=numSeps)
    for ii, binSep in enumerate(tqdm.tqdm(flx_seps, desc="Calculating scattering")):
        dnStarsFLC, numStarsFLC, dfStarsFLC, fluxStarsFLC, dfStarsSSLC, fluxStarsSSLC = \
            evalScattering(binSep)

        hard_sslc = dadt_scattering(m1+m2, binSep, np.max(fluxStarsSSLC), mstar)
        hard_flc  = dadt_scattering(m1+m2, binSep, np.max(fluxStarsFLC),  mstar)

        eps_rad = zmath.spline(gals.rads, gals.eps[sample], log=True, pos=True, extrap=True)

        sepEner = eps_rad(binSep)

        # Plot Fig A10 - Loss-Cone Occupancy
        if PLOT_A10:
            fig = GM_Figures.figa10_lc_occupancy(
                gals.rads, eps, dnStarsAll, dnStarsFLC, numStarsAll, numStarsFLC,
                binr=binSep, bine=sepEner)
            fname1 = subPlotNames + "lcPipe_figa10-occ_%02d.png" % (ii)
            zplot.saveFigure(fig, fname1, verbose=False)   # , log=log)
            plt.close(fig)

        # Plot Fig A11 - loss-cone fluxes
        if PLOT_A11:
            fig = GM_Figures.figa11_lc_flux(
                eps, gals.rads, dfStarsFLC, dfStarsSSLC, fluxStarsFLC, fluxStarsSSLC,
                binr=binSep, bine=sepEner)
            fname2 = subPlotNames + "lcPipe_figa11-flx_%02d.png" % (ii)
            zplot.saveFigure(fig, fname2, verbose=False)    # , log=log)
            plt.close(fig)

        # Store values to arrays
        num_flc[ii]   = np.max(numStarsFLC)
        flx_flc[ii]   = np.max(fluxStarsFLC)
        flx_sslc[ii]  = np.max(fluxStarsSSLC)
        dadt_sslc[ii] = hard_sslc
        dadt_flc[ii]  = hard_flc

        # pbar.update(ii)

    # pbar.finish()

    if PLOT_A10:
        log.debug(" - - Saved FigA10 to (e.g.) '%s'" % (fname1))
    if PLOT_A11:
        log.debug(" - - Saved FigA11 to (e.g.) '%s'" % (fname2))

    # Plot Figs A15 - Scattering vs. Separations
    if PLOT_A15:
        log.debug(" - Plotting Fig15")
        fig = GM_Figures.figA15_flux_sep(flx_seps, flx_flc, flx_sslc, dadt_flc, dadt_sslc,
                                         sample, snap)
        fname = plotNames + "lcPipe_figa15-sep.png"
        zplot.saveFigure(fig, fname, verbose=False, log=log)
        plt.close(fig)
        figNames.append(fname)

    # Plot Figure A02 - Density/Mass Profiles
    if PLOT_A02:
        log.info(" - Plotting FigA02")
        dens  = [gals.densStars[sample], gals.densDM[sample]]
        mass  = [gals.massStars[sample], gals.massDM[sample], gals.massTot[sample]]
        names = ['Stars', 'DM', 'Total']
        vels  = [gals.vdisp_stars[sample]]

        fig = GM_Figures.figA02_ModelGalaxy(gals.rads, dens, mass, vels, names=names)
        fname = plotNames + "lcPipe_figa02-dens.png"
        zplot.saveFigure(fig, fname, verbose=False, log=log)
        plt.close(fig)
        figNames.append(fname)

    # Plot Figure A08 - Reconstructed Number Density
    if PLOT_A08:
        log.debug(" - Plotting FigA08")
        df_pot = zmath.spline(eps, dist_func, mono=True, pos=True, sort=True, log=True)
        dfErrs = np.zeros(np.size(eps))
        fig = GM_Figures.figA08_distFunc(eps, gals.rads, ndensStars, dist_func, dfErrs, df_pot, log,
                                         sample=sample)
        fname = plotNames + "lcPipe_figa08-distfunc.png"
        zplot.saveFigure(fig, fname, verbose=False, log=log)
        plt.close(fig)
        figNames.append(fname)

    return figNames


def diffusionCoef(ndens, vel, massBG, coulLog):
    """Calculate the diffusion coefficient based on Binney & Tremaine.

    Sec.8.3, pg.513, Eq.8-68.
    Assume that the velocity equals the velocity dispersion (i.e. v = sigma),
    thus xx = v/(sigma*sqrt(2)) = 1/sqrt(2).

    Arguments
    ---------
        ndens   <flt>[N] : number-density or number-density profile
        vel     <flt>[N] : velocity or velocity profile
        massBG  <flt>    : mass of both background and test particles
        coulLog <flt>    : coulomb logarithm

    Returns
    -------
        diffCoeff <flt>[N] : diffusion coefficient (or profile of)

    """

    xx      = 1.0/np.sqrt(2.0)
    erfx    = sp.special.erf(xx)
    gerf    = (erfx - (2.0*xx/np.sqrt(np.pi))*np.exp(-np.square(xx)))
    gerf   /= (2.0*np.square(xx))
    erfTerm = (erfx - gerf)/xx

    diffCoef = np.pi*np.sqrt(8.0)*np.square(NWTG)*ndens*np.square(massBG)/vel
    diffCoef *= coulLog*erfTerm

    return diffCoef


def numStars_all(dist_func, period, j2Circ):
    """Calculate the differential number of all stars.

    i.e.  `N(E)` such that the number of stars between energy E and dE is,
          `N(E) dE`

    Arguments
    ---------
        dist_func <flt>[N] : distribution function at the target energies
        period   <flt>[N] : keplerian period
        j2Circ   <flt>[N] : circular angular momentum squared

    Returns
    -------
        dN <flt>[N] : differential number of all stars in the galaxy

    """
    dN = 4.0*np.square(np.pi)*dist_func*period*j2Circ
    return dN


def numStars_flc(dist_func, period, j2Circ, j2Cone):
    """Calculate the differential number of stars in the full loss-cone (FLC).

    i.e. `N_lc(E)`  such that the number of stars in the full loss-cone between
         energy E and dE is   `N_lc(E) dE`

    Arguments
    ---------
        dist_func <flt>[N, M] : distribution function at all energies
        period   <flt>[N, M] : keplerian period
        j2Circ   <flt>[N, M] : circular angular momentum at all energies
        j2Cone   <flt>[N]   : loss-cone critical angular momentum squared

    Returns
    -------
        dNlc <flt>[N] : differential number of stars in the full loss-cone

    """
    # ``j2Circ`` might have shape (N, M), convert shape of j2Cone from (N, ) to (N, M)
    if np.ndim(j2Circ) == 2:
        j2LC = np.vstack([j2Cone]*np.shape(j2Circ)[-1]).T
    elif np.ndim(j2Circ) == 1:
        j2LC = j2Cone
    else:
        raise RuntimeError("Not configured for this shape of ``j2Circ``!")

    # Naive occupancy
    dNlc = 4.0*np.square(np.pi)*dist_func*period*j2LC
    # If circular angular momentum is *less* than loss-cone, occupancy is zero
    inds = (j2Circ < j2LC)
    dNlc[inds] = 0.0

    return dNlc


def fluxStars_flc(dn_flc, period):
    """Calculate flux of stars from the full loss-cone into the central object.

    Parameters
    ----------
        dn_flc <flt>[N] : differential number of stars in the full-loss-cone
        period <flt>[N] : orbital period

    Returns
    -------
    flux_full <flt>[N]
    """
    flux_full = dn_flc/period
    return flux_full


def fluxStars_sslc(rads, dn_flc, diffCoef, period, j2Circ, j2Cone):
    """Steady-state loss-cone (SSLC) flux of stars.

    From [Magorrian & Tremaine 1999] via the diffusion coefficient

    Parameters
    ----------
        rads     <flt>[N, M] :
        dn_flc   <flt>[N, M] : differential number of stars in the full loss-cone.
        diffCoef <flt>[N, M] : diffusion coefficient profile
        period   <flt>[N, M] : orbital periods
        j2Circ   <flt>[N, M] : circular angular momentum squared
        j2Cone   <flt>[N]   : angular momentum of the loss-cone squared

    Returns
    -------
        df_sslc <flt>[N] : steady state differential flux distribution.

    """
    muDiff   = 2.0*np.square(rads)*diffCoef/j2Circ
    # ``j2Circ`` may have shape (N, M), convert shape of j2Cone from (N, ) to (N, M)
    if np.ndim(j2Circ) == 2:
        j2LC = np.vstack([j2Cone]*np.shape(j2Circ)[-1]).T
    elif np.ndim(j2Circ) == 1:
        j2LC = j2Cone
    else:
        raise RuntimeError("Not configured for this shape of ``j2Circ``!")

    # Calculate R_lc  -- angular mometum ratio for loss-cone
    rlc   = j2LC/j2Circ
    qFact = period*muDiff/rlc
    # Calculate Flux
    lnr0inv = _MT1999_lnR0inv(rlc, qFact)
    df_sslc = dn_flc*(2.0*np.square(rads)*diffCoef)/(lnr0inv*j2LC)
    return df_sslc


def interactionCriteria_binary(centMass, binSep, leeway):
    """Characteristic interaction criteria for MBH binary with losscone stars

    Arguments
    ---------
        centMass : <scalar>, combined mass of the central MBH binary
        binSep   : <scalar>, separation of the binary system

    Returns
    -------
        actRad  : <scalar>, interaction radius ('R_bin*L')
        actJ2   : <scalar>, interaction angular momentum squared ('2*G*M*R_act')
        actEner : <scalar>, characteristic binding energy ('G*M/R_act')


    Notes
    -----
        Assumptions
            - Angular momentum and energy are given by circular-orbit equations with a central
              object of mass ``centMass``

    """
    actRad = binSep*leeway
    # Calculate Loss-cone angular momentum (approximately)
    actJ2 = 2.0*NWTG*centMass*actRad
    # Characteristic binding energy
    actEner = NWTG*centMass/actRad
    return actRad, actJ2, actEner


def _MT1999_lnR0inv(rlc, qfact):
    """Calculate simplified `ln(R_0(E)^-1)` term from [Magorrian & Tremaine 1999].

    R_0 = R_lc * q0
          where  q0 = exp(-q)                        for q >  1
          and    q0 = exp(-0.186*q - 0.824*sqrt(q))  for q <= 1

    Simplifying,
    ln(R_0^-1) = ln(R_lc) + y0
                   where    y0 = q                           for q >  1
                   and      y0 = 0.186q + 0.824*sqrt(q)      for q <= 1

    """
    # for an array of ``qfact``
    if np.iterable(qfact):
        y0 = np.zeros(np.shape(qfact))
        indsHi = (qfact > 1.0)
        indsLo = (qfact <= 1.0)
        y0[indsHi] = qfact[indsHi]
        y0[indsLo] = 0.186*qfact[indsLo] + 0.824*np.sqrt(qfact[indsLo])

    # Scalar ``qfact``
    else:
        if qfact > 1.0:
            y0 = qfact
        else:
            y0 = 0.186*qfact + 0.824*np.sqrt(qfact)

    yfact = -np.log(rlc) + y0
    return yfact


def dadt_scattering(centMass, sep, flux, mstar):
    """Calculate the hardening rate for a given flux of stars and binary parameters.

    Arguments
    ---------
        centMass <flt>[N] : total central mass (i.e. m1+m2)
        sep      <flt>[N] : binary separation
        flux     <flt>[N] : net flux of stars, i.e. rate of interactions
        mstar    <flt>[N] : mass of background stars

    Returns
    -------
        dadt     <flt>[N] : hardening rate

    """
    # Change in radius per encounter
    deltaRad = 3.0 * mstar * sep / centMass
    # Change in radius over time
    dadt = -deltaRad * flux
    return dadt


def netFlux(fluxFLC, fluxSSLC, refilling_log_frac):
    """Based on the Full and Steady-Steady loss-cone fluxes, calculate the effective, net flux.

    Arguments
    ---------
        fluxFLC            <flt>([N]) : full-loss-cone flux (or distribution of fluxes)
        fluxSSLC           <flt>([N]) : steady-state-loss-cone flux (or distribution of fluxes)
        refilling_log_frac <flt>      :

    """
    #     If multiple profiles are given, calculate a flux for each profile (axis=-1)
    flux = np.max(fluxSSLC, axis=-1)
    # With no refilling, return the steady-state flux
    if refilling_log_frac == 0.0: return flux

    # Interpolate between steady-state and full, in log-space
    ref_flux = np.max(fluxFLC, axis=-1)
    flux = zmath.midpoints([flux, ref_flux], axis=0, log=True, frac=refilling_log_frac)
    return flux


def _findInvalid(arr, val='both'):
    """Find elements of arrays with unreasonable values (negative and or infinite).

    Returns indices of invalid elements of `arr` if any exist.  Otherwise returns `None`.
    """
    if val.startswith('pos'):
        func = lambda xx: np.greater_equal(xx, 0.0)
    elif val.startswith('fin'):
        func = np.isfinite
    elif val.startswith('both'):
        # Check 'pos'itive
        inds = _findInvalid(arr, 'pos')
        if inds: return inds
        # Check 'fin'ite
        inds = _findInvalid(arr, 'fin')
        if inds: return inds
        return None
    else:
        raise ValueError("Unrecognized ``val`` '%s'" % (val))

    if np.ndim(arr) > 0:
        inds = np.where(func(arr) != True)
        if np.size(inds[0]) > 0:
            return inds
    else:
        if func(arr) != True:
            return arr

    return None
