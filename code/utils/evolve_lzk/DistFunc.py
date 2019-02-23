"""Calculate Distribution Functions, for all Galaxies in Parallel.

This submodule should first be executed as a parallel script, e.g.

    $ mpirun -np 64 python -m GalaxyModels.DistFunc

to calculate distribution function data arrays for all galaxies in the `MBHBinaries` class, loaded
from `Binaries.py`.  Once these arrays have been pre-calculated, spline functions can be rapidly
loaded by single processor runs.  As an executable, this submodule uses the `argparse` package,
usage information can thus be accessed via the command line as,

    $ python DistFunc.py -h

When used as a loaded python module, the `loadDistFuncs` method is used to load
distribution-function interpolation functions.

Functions
---------
-   loadDistFuncs            - Load precalculated Distribution Functions as spline functions.
-   main                     - Calculate distribution functions for all galaxies in parallel.
-   _runMaster               - Load data for all galaxies, distribute to 'slave' processes.
-   _runSlave                - Receive individual galaxy profiles and calculate the dist functions.
-   dist_func                 - Calculate the distribution function for a single galaxy profile.
-   densityFromDistFunc      - Reconstruct density profile from the given distribution function.
-   plotDistFunc             - Calc the distribution function for a sample galaxy, and plot results.

-   _plotPlotZoom            - Plot a full range in one axis, and a zoom-in in another.
-   _derivs                  - Calc the first and second derivatives.
-   _parseArguments          - Prepare argument parser and load command line arguments.
-   _mpiError                - Raise an error through MPI and exit all processes.
-   _loadMPILogger           - Init a logger for output, specifically in parallel configuration.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import xrange

import numpy as np
import scipy as sp
import scipy.interpolate  # noqa
import matplotlib.pyplot as plt
from datetime import datetime
import tqdm

import sys
import os
import logging
import argparse
# from mpi4py import MPI

from illpy_lib.constants import CONV_CGS_TO_SOL

from zcode import inout as zio
from zcode import math as zmath
from zcode import plot as zplot
from zcode.constants import MSOL

from mbhmergers import settings, binaries, constants
TAGS = constants.MPI_TAGS

__version__ = '0.3'


def loadDistFuncs(sets, log=None):
    """Load precalculated Distribution Functions from save file as spline functions.

    Arguments
    ---------
        run : int
            illustris run number {1, 3}
        validFrac : float or `None`
            Required fraction of galaxy distribution function values to be valid (> 0.0)
            for the galaxy as a whole to be considered valid.  `None` means no requirement.
        log : logging.Logger` object

    Returns
    -------
        valid : ndarray of bools, shape (N, )
            Flag describing whether each galaxy is valid or not.  `N` is the number of galaxies.

        dfs : ndarray of callable, scalar functions; shape (N, )
            log-splines of distribution functions for each galaxy.  Invalid galaxies have
            the `zeroFunc` given - which return ndarrays of 0.0

    """
    if log is None:
        log = constants.load_logger(True, False, __file__)
    log.debug("loadDistFuncs()")
    validFrac = sets.DIST_FUNC_VALID_FRAC

    # Load Data
    # ---------
    fname = sets.GET_DIST_FUNC_FILENAME(vers=__version__)
    if not os.path.exists(fname):
        errStr = "File '%s' does not exist!" % (fname)
        log.error(errStr)
        log.error("Run ``%s`` to generate the distribution functions!" % (__file__))
        raise RuntimeError(errStr)

    dfData = zio.npzToDict(fname)
    dfRun = dfData['run']
    log.debug("Loaded distribution function data from '%s', run %d" % (fname, dfRun))
    assert sets.RUN == dfRun, "Run numbers do not match!"

    eps = dfData['eps']
    dist_funcs = dfData['distfuncs']
    numMergers, numEners = np.shape(eps)
    log.debug(" - %d Mergers, with %d Energies" % (numMergers, numEners))

    # Determine `valid` galaxies
    # ----------------------------
    # Initially `valid` are those with any nonzero entries
    valid = np.any(dist_funcs != 0.0, axis=1)
    numVal = np.count_nonzero(valid)
    frac = 1.0 * numVal / numMergers
    log.info("Loaded %d/%d = %.4f nonzero distribution functions." % (numVal, numMergers, frac))

    # Find galaxies with non-finite values, set to invalid
    bad_map = ~np.isfinite(dist_funcs)
    num_bad = np.count_nonzero(bad_map, axis=1)
    num_gals_with_bad = np.count_nonzero(num_bad[valid])
    frac = num_gals_with_bad / numVal
    log.info("%d/%d = %.4f Galaxies have non-finite values" % (num_gals_with_bad, numVal, frac))
    ave_bad = num_bad[valid].mean()
    log.debug("Average number of non-finite values = %.2f" % (ave_bad))
    valid[num_bad > 0] = False

    # Find bad entries in valid galaxies
    bad_map = (dist_funcs <= 0.0)
    num_bad = np.count_nonzero(bad_map, axis=1)
    ave_bad = num_bad[valid].mean()
    log.debug("Average number of bad values in valid galaxies = %.2f" % (ave_bad))

    # Implement required fraction of good entries
    if validFrac:
        log.debug("Requiring fraction: %.4f good entries for valid galaxies" % (validFrac))
        #     Find the fraction of bad values for each galaxy
        fracBad = num_bad / numEners
        #     Only look at those which start as valid
        # valInds = np.where(valid)[0]
        # inds = valInds[np.where(fracBad > 1.0-validFrac)[0]]
        # numBad = np.size(inds)
        # valInds = np.where(valid)[0]
        bads = valid & (fracBad > 1.0 - validFrac)
        numBad = np.count_nonzero(bads)
        frac = 1.0 * numBad / numVal
        log.debug(" - %d/%d = %.4f Galaxies are below threshold" % (numBad, numVal, frac))
        valid[bads] = False

    # Construct Splines of Distribution Functions
    #  -------------------------------------------
    dfs = np.empty(numMergers, dtype=object)
    valInds = np.where(valid)[0]
    numVal = np.size(valInds)
    log.debug("Interating over %d valid galaxies" % (numVal))
    # Iterate over valid galaxies
    for ii, val in enumerate(tqdm.tqdm(valInds, desc="Constructing splines")):
        dfs[val] = _useSpline(eps[val], dist_funcs[val])

    # Set invalid galaxies to the 'zero function'
    dfs[~valid] = constants._zeroFunc
    return valid, dfs


def main():
    """
    Calculate distribution functions for all galaxies in parallel.

    Runs the `_runMaster` on process 0 which communicates with, and distributes jobs to, each of the
    `_runSlave` processes on all the other processors.
    """
    # Initialize MPI Parameters
    # -------------------------
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    if size <= 1: raise RuntimeError("Not setup for serial runs!")

    sets = settings.Settings()
    mstar = sets.MSTAR * MSOL

    if rank == 0:
        NAME = sys.argv[0]
        print("\n%s\n%s\n%s" % (NAME, '='*len(NAME), str(datetime.now())))
        zio.check_path(sets.GET_DIR_LOGS())

    # Make sure log-path is setup before continuing
    comm.Barrier()

    # Parse Arguments
    #  ---------------
    args = _parseArguments(sets)
    run = args.run
    verbose = args.verbose
    smooth = args.smooth
    relAcc = args.relAcc
    intSteps = args.intSteps

    # Load logger
    log = _loadMPILogger(rank, verbose, sets)
    if rank < 2: print("Log (Rank %d) filename '%s'" % (rank, log.filename))

    # log runtime parameters
    log.info("run           = %d  " % (run))
    log.info("version       = %s  " % (__version__))
    log.info("MPI comm size = %d  " % (size))
    log.info("Rank          = %d  " % (rank))
    log.info("")
    log.info("verbose       = %s  " % (str(verbose)))
    log.info("smooth        = %d  " % (smooth))
    log.info("relAcc        = %e  " % (relAcc))
    log.info("intSteps      = %d  " % (intSteps))
    log.info("")

    # Master Process
    # --------------
    if rank == 0:
        beg_all = datetime.now()

        try:
            log.info("Running Master")
            eps, ndens, ndD1, ndD2, dist_funcs, dfErrs, recDens = _runMaster(run, comm, log)
        except Exception as err:
            _mpiError(comm, log, err)

        end_all = datetime.now()
        log.debug("Done after '%s'" % (str(end_all-beg_all)))

        fname = sets.GET_DIST_FUNC_FILENAME(run=run, vers=__version__)
        zio.check_path(fname)

        data = {}
        data['run'] = run
        data['eps'] = eps
        data['ndens'] = ndens
        data['ndD1'] = ndD1
        data['ndD2'] = ndD2
        data['distfuncs'] = dist_funcs
        data['dferrs'] = dfErrs
        data['recdens'] = recDens
        data['version'] = __version__
        zio.dictToNPZ(data, fname, verbose=True)
        log.info("Saved data to '%s'" % (fname))

    # Slave Processes
    # ---------------
    else:
        try:
            log.info("Running slave")
            _runSlave(comm, smooth, relAcc, intSteps, mstar, log)
        except Exception as err:
            _mpiError(comm, log, err)

        log.info("Done.")

    return


def _runMaster(run, comm, log):
    """
    Load data for all mergers/galaxies, distribute to 'slave' processes for computation.

    Arguments
    ---------
    run : int
        illustris simulation number {1, 3}
    comm : MPI intracommunicator object
    log : `logging.Logger` object

    Returns
    -------
    eps, ndens, ndD1, ndD2, dist_funcs, dfErrs, recDens : ndarray of floats,
        each array has shape ``(N, M)``, for `N` mergers/galaxies and `M` radial bins
        See the description in ``DistFunc.dist_func`` for description of each parameter.

    """

    from mpi4py import MPI
    stat = MPI.Status()
    rank = comm.rank
    size = comm.size

    log.info("_runMaster()")
    log.debug("Rank %d/%d" % (rank, size))

    # Load Basic MBHBinaries
    log.info("Loading MBHBinaries")
    mbhb = binaries.binaries.MBHBinaries(run, scattering=False, log=log)
    gals = mbhb.galaxies
    numMergers = mbhb.numMergers
    valid_inds = np.where(mbhb.valid)[0]
    numVal = np.size(valid_inds)
    frac = 1.0*numVal/numMergers
    log.info(" - Loaded %d/%d = %.4f valid binaries" % (numVal, numMergers, frac))

    countDone = 0

    # Storage for results
    numRads = gals.numRads
    eps = np.zeros([numMergers, numRads])
    ndens = np.zeros([numMergers, numRads])
    ndD1 = np.zeros([numMergers, numRads])
    ndD2 = np.zeros([numMergers, numRads])
    dist_funcs = np.zeros([numMergers, numRads])
    dfErrs = np.zeros([numMergers, numRads])
    recDens = np.zeros([numMergers, numRads])

    # Duration of slave processes
    slaveDur = np.zeros(numMergers)
    # Duration of master interations
    cycleDur = np.zeros(numMergers)

    # Iterate Over Valid Binaries
    # ---------------------------
    log.info("Iterating over binaries")
    for ii, bin in enumerate(tqdm.tqdm(valid_inds)):
        beg = datetime.now()

        # Look for available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        src = stat.Get_source()
        tag = stat.Get_tag()

        # Store Results
        if tag == TAGS.DONE:
            # retBin, ener, dfunc, errs, dens, durat = data
            retBin, ener, den, dn, dn2, dfunc, errs, dens, durat = data
            eps[retBin] = ener
            ndens[retBin] = den
            ndD1[retBin] = dn
            ndD2[retBin] = dn2
            dist_funcs[retBin] = dfunc
            dfErrs[retBin] = errs
            recDens[retBin] = dens
            slaveDur[retBin] = durat
            countDone += 1

        # Distribute tasks
        comm.send([bin, gals.gravPot[bin], gals.densStars[bin]], dest=src, tag=TAGS.START)

        end = datetime.now()
        cycleDur[bin] = (end-beg).total_seconds()

    # Close out all Processes
    # -----------------------
    numActive = size-1
    log.info("Exiting %d active processes" % (numActive))
    while numActive > 0:

        # Find available slave process
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=stat)
        src = stat.Get_source()
        tag = stat.Get_tag()

        # If we're recieving exit confirmation, count it
        if tag == TAGS.EXIT:
            numActive -= 1
        else:
            # Store completed results
            if tag == TAGS.DONE:
                # retBin, ener, dfunc, errs, dens, durat = data
                retBin, ener, den, dn, dn2, dfunc, errs, dens, durat = data
                eps[retBin] = ener
                ndens[retBin] = den
                ndD1[retBin] = dn
                ndD2[retBin] = dn2
                dist_funcs[retBin] = dfunc
                dfErrs[retBin] = errs
                recDens[retBin] = dens
                slaveDur[retBin] = durat
                countDone += 1

            # Send exit command
            comm.send(None, dest=src, tag=TAGS.EXIT)

    fracDone = 1.0*countDone/numMergers
    log.info("%d/%d = %.4f Completed tasks!" % (countDone, numVal, fracDone))

    inds = (slaveDur > 0.0)
    slaveAve = np.average(slaveDur[inds])
    slaveStd = np.std(slaveDur[inds])

    inds = (cycleDur > 0.0)
    cycleAve = np.average(cycleDur[inds])
    cycleStd = np.std(cycleDur[inds])

    log.debug("Average Process time %.2e +- %.2e" % (slaveAve, slaveStd))
    log.debug("Average Cycle   time %.2e +- %.2e" % (cycleAve, cycleStd))
    log.debug("Total Process Time = %.2e" % (np.sum(slaveDur)))

    return eps, ndens, ndD1, ndD2, dist_funcs, dfErrs, recDens


def _runSlave(comm, smooth, relAcc, intSteps, mstar, log):
    """
    Receive individual galaxy profiles and calculate the distribution function.

    This is the process that should be run on ``rank != 0`` processes during parallel runs.

    Arguments
    ---------
    comm : MPI intracommunicator object (e.g. ``MPI.COMM_WORLD``)
    log : ``logging.Logger`` object

    Details
    -------
     - Waits for ``master`` process to send subhalo numbers
     - Returns status to ``master``

    """
    from mpi4py import MPI
    stat = MPI.Status()
    rank = comm.rank
    size = comm.size
    numReady = 0

    data = {}

    log.debug("_runSlave()")
    log.debug("Rank %d/%d" % (rank, size))

    # Keep looking for tasks until told to exit
    while True:
        # Tell Master this process is ready
        comm.send(None, dest=0, tag=TAGS.READY)
        # Receive ``task`` ([number, gravPot, ndensStars])
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=stat)
        tag = stat.Get_tag()

        if tag == TAGS.START:
            # Extract parameters
            bin, gravPot, densStars = task
            ndens = densStars/mstar
            beg = datetime.now()
            # Calculate distribution function
            data = dist_func(gravPot, ndens, smooth, relAcc, intSteps)
            #     unpack results
            eps, den, dn, dn2, df, dfErrs, reconDen = data
            end = datetime.now()
            durat = (end-beg).total_seconds()
            # Re-pack results
            data = [bin, eps, den, dn, dn2, df, dfErrs, reconDen, durat]
            comm.send(data, dest=0, tag=TAGS.DONE)
        elif tag == TAGS.EXIT:
            break

        numReady += 1

    # Finish, return done
    log.info("Done.  Sending Exit.")
    comm.send(None, dest=0, tag=TAGS.EXIT)
    return


def dist_func(gpot, ndens, smooth, relAcc, intSteps):
    """
    Calculate the distribution function for a single galaxy profile.

    Arguments
    ---------
    gpot : 1D ndarray of floats
        gravitational potential energy profile of this galaxy.  These must be *negative* values.
        The length of this array, `M` is the number of radial bins of the galaxy.
    ndens : 1D ndarray of floats
        stellar number density profile of the galaxy, and the same points as the given `gpot`.
    smooth : int
        the width of the top-hat smoothing function to use.  If `None` or `0`, no smoothing
    relAcc : float
        Target relative accuracy during integration
    intSteps : int
        Maximum number of sub-intervals used for integration (with `scipy`).

    Returns
    -------
    eps : ndarray of floats, all arrays have length `M`, for `M` radial bins.
        Array of sorted, *positive* energies (independent variable) for this galaxy.  Equal to
        ``-gpot``, and sorted.
    ndens : ndarray of floats,
        The number density of this galaxy corresponding to each `eps`.
    ndD1 : ndarray of floats,
        The first derivative of number-density as a function of energy, i.e. (dn/de)
    ndD2 : ndarray of floats,
        The second derivative of number-density as a function of energy, i.e. (d^2n/de^2)
    dist_funcs : ndarray of floats,
        The resulting distribution function values
    dfErrs : ndarray of floats,
        Errors (specifically from integration) on the distribution function.
    recDens : ndarray of floats,
        Number-density as reconstructed from the distribution function `dist_funcs`

    """
    # Load Parameters
    ppot = -gpot

    # Sort, and copy input arrays
    inds = np.argsort(ppot)
    eps = np.array(ppot[inds])
    den = np.array(ndens[inds])

    # Smooth the Density Array with a top-hat function of width ``smooth``
    if smooth:
        window = np.ones(int(smooth))/float(smooth)
        den = np.convolve(den, window, mode='same')

    # Get derivatives dn/de
    dn, dn2 = _derivs(eps, den)

    numEner = np.size(eps)
    grals = np.zeros(numEner)
    dfErrs  = np.zeros(numEner)
    ndensPotD1 = sp.interpolate.InterpolatedUnivariateSpline(eps, dn, k=2)
    ndensPotD2 = sp.interpolate.InterpolatedUnivariateSpline(eps, dn2, k=2)

    absAcc = (np.min(den)/np.max(eps))*relAcc

    # Calculate Distribution Function from Derivatives
    # ------------------------------------------------
    #     Iterate over each enery and integrate
    #     DF(x) is the integral from [0, x]
    for ii in range(1, numEner):
        func = lambda xx: ndensPotD2(xx)/np.sqrt(eps[ii] - xx)
        tg, te = sp.integrate.quad(func, eps[ii-1], eps[ii],
                                   limit=intSteps, epsrel=relAcc, epsabs=absAcc)
        grals[ii] = grals[ii-1] + tg
        dfErrs[ii]  = np.square(dfErrs[ii-1]) + np.square(te)

    const = 1.0/(np.square(np.pi)*np.sqrt(2.0))

    # Calculate boundary term
    boundary = ndensPotD1(np.min(eps))/np.sqrt(eps)

    # Construct Complete Distribution Function f(E)
    dfunc = const*(grals + boundary)
    dfErrs = const*np.sqrt(dfErrs)

    # Create DF interpolant based on positive values
    # dfpos = np.where(dfunc > 0.0)[0]
    #
    # if np.size(dfpos) > 10:
    dfpos = (dfunc > 0.0)

    if np.count_nonzero(dfpos) > 10:
        df_func = _useSpline(eps[dfpos], dfunc[dfpos])
        # Calculate Reconstructed Density
        reconDen, errs = densityFromDistFunc(eps, df_func)
    else:
        reconDen = np.zeros(numEner)
        # errs = np.zeros(numEner)

    return eps, den, dn, dn2, dfunc, dfErrs, reconDen


def densityFromDistFunc(eps, dfunc):
    """
    Reconstruct density profile from the given distribution function.

    See eq. 7 of Magorrian & Tremaine 1999
    Also discussed in Binney & Tremaine

    Arguments
    ---------
    eps : 1D ndarray of floats,
        positive energies, the independent variable, for galaxy distribution functions
    dfunc : 1D ndarray of floats
        the distributiion function for this galaxy

    Returns
    -------
    ndens : 1D ndarray of floats,
        the reconstructed number-density profile for this galaxy
    errs : 1D ndarray of floats
        The errors associated with the `scipy` integral to calculate densities.

    """
    NUM = len(eps)
    ndens = np.zeros(NUM)
    errs = np.zeros(NUM)

    for ii in range(1, NUM):
        func = lambda xx: 4.0*np.pi*dfunc(xx)*np.sqrt(2.0*(eps[ii] - xx))
        tempNdens, tempErr = sp.integrate.quad(func, eps[ii-1], eps[ii])
        ndens[ii] = ndens[ii-1] + tempNdens
        errs[ii] = np.square(errs[ii-1]) + np.square(tempErr)

    errs = np.sqrt(errs)
    return ndens, errs


def plotDistFunc(sample, gals, log, mstar=None, smooth=None, relAcc=None, intSteps=None):
    """
    Calculate the distribution function for a sample galaxy, and plot the results.

    Plots 8-panels on a single figure, showing different aspects of the calculation in each row,
    with a zoom-in in the second column of panels.

    Arguments
    ---------
    sample : int
        Index number of the target galaxy profile (used as index for the `gals` arrays).
    gals : `MBHBinaries.Galaxies` subclass object
        Used to retrieve the sample profiles
    log : `logging.Logger` class

    smooth : int or `None`, see `DistFunc.dist_func`,
        If `None`, the default value from `settings.py` is used.
    relAcc : float or `None`, see `DistFunc.dist_func`
        If `None`, the default value from `settings.py` is used.
    intSteps : int or `None`, see `DistFunc.dist_func`
        If `None`, the default value from `settings.py` is used.

    Returns
    -------
    fig : ``matplotlib.figure.Figure`` object
        Resulting figure

    """
    enerConv = CONV_CGS_TO_SOL.ENER.value

    # Initialization
    #  --------------
    sets = settings.Settings()
    # Load default parameters
    if mstar is None: mstar = sets.MSTAR * MSOL
    if smooth is None: smooth = sets.DIST_FUNC_DENS_SMOOTHING
    if relAcc is None: relAcc = sets.DIST_FUNC_REL_ACC
    if intSteps is None: intSteps = sets.DIST_FUNC_INT_STEPS

    gpot = gals.gravPot[sample]
    ppot = -gpot
    ndens = gals.densStars[sample]/mstar

    # Calculate derivatives, distribution function, and reconstructed density
    eps, den, dn, dn2, dfunc, dfErrs, reconDen = dist_func(gpot, ndens, smooth, relAcc, intSteps)

    # Find the location to zoom-in on, based on the second derivative
    zoomLoc = eps[np.argmax(np.fabs(dn2))]
    ALPHA = 0.5

    # Initialize figure
    fig, axes = plt.subplots(figsize=[16, 16], nrows=4, ncols=2)
    plt.subplots_adjust(top=0.98, bottom=0.05, left=0.05, right=0.95)
    for ii in xrange(np.shape(axes)[0]):
        for jj in xrange(np.shape(axes)[1]):
            ax = axes[ii, jj]
            ax.set(xscale='log', yscale='log', xlabel='Energy [(km/s)$^2$]')
            zplot.set_grid(ax, False)

    # Plot Density vs. Energy (Row-0)
    # -------------------------------
    _plotPlotZoom(axes[0, 0], axes[0, 1], zoomLoc, eps*enerConv, den)
    for ax in axes[0]: ax.set_ylabel('Number Density [cm$^{-3}$]')

    # Find threshold for symlog
    thresh = np.min(np.fabs(dn2))

    # Plot derivatives (Row-1)
    # ------------------------
    #     Negative values, use symlog (set scale before limits are set)
    for ax in axes[1]: ax.set_yscale('symlog', linthreshy=thresh)
    _plotPlotZoom(axes[1, 0], axes[1, 1], zoomLoc, eps*enerConv, dn2)
    for ax in axes[1]: ax.set_ylabel('Density Derivative')

    # Create DF interpolant based on positive values
    dfpos = (dfunc > 0.0)
    df_func = _useSpline(eps[dfpos], dfunc[dfpos])

    # Plot Distributiion Function (Row-2)
    # -----------------------------------
    _plotPlotZoom(axes[2, 0], axes[2, 1], zoomLoc, eps*enerConv, df_func(eps), zz=dfunc)
    for ax in axes[2]: ax.set_ylabel('Distribution Function')

    # Calculate Reconstructed Density
    reden, errs = densityFromDistFunc(eps, df_func)
    inds = np.argsort(ppot)

    # Plot Densities, Original and Reconstructed (Row-3)
    # --------------------------------------------------
    axes[3, 0].plot(ppot[inds]*enerConv, ndens[inds], 'k--', alpha=ALPHA)
    axes[3, 1].plot(ppot[inds]*enerConv, ndens[inds], 'k--', alpha=ALPHA)
    _plotPlotZoom(axes[3, 0], axes[3, 1], zoomLoc, eps*enerConv, reden)
    for ax in axes[3]: ax.set_ylabel('Number Density [cm$^{-3}$]')

    return fig


def _plotPlotZoom(ax1, ax2, zoomLoc, xx, yy, zz=None, zoomScale=20.0):
    """
    Plot a full range in one axis, and a zoom-in in another.
    """
    ALPHA = 0.5
    SIZE = 10
    COL = 'k'

    if(zz is None): zz = yy

    # Plot for Full Range
    ax1.plot(xx, yy, '-', color=COL, alpha=ALPHA)
    ax1.scatter(xx, zz, s=SIZE, color=COL, alpha=ALPHA)
    ax1.set_xlim(zmath.minmax(xx))

    # Plot Density vs. Energy for Zoom
    ax2.plot(xx, yy, '-', color=COL, alpha=ALPHA)
    ax2.scatter(xx, zz, s=SIZE, color=COL, alpha=ALPHA)
    xlim = zplot.zoom(ax2, zoomLoc, axis='x', scale=zoomScale)
    ax2.set_xscale('linear')
    ylim = zplot.limits(xx, yy, xlim)
    ax2.set_ylim(ylim)
    return


def _derivs(ppot, ndens, sort=False):
    """
    Calculate the first and second derivatives of the given function and parameter.
    """
    # Find derivatives of density as a function of energy (ppot)
    dp  = np.gradient(ppot,      edge_order=2)
    dn  = np.gradient(ndens, dp, edge_order=2)    # dn/dp
    dn2 = np.gradient(dn,    dp, edge_order=2)    # d^2n/dp^2
    return dn, dn2


def _parseArguments(sets):
    """
    Prepare argument parser and load command line arguments.
    """
    parser = argparse.ArgumentParser()
    versStr = '%s %s' % (sys.argv[0], __version__)
    parser.add_argument('--version', action='version', version=versStr)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output', default=sets.VERBOSE)
    parser.add_argument("run", type=int, nargs='?', choices=[1, 2, 3],
                        help="illustris simulation number", default=sets.RUN)
    parser.add_argument("smooth", type=int, nargs='?',
                        help="number of smoothing points for density array (<=1 is off)",
                        default=sets.DIST_FUNC_DENS_SMOOTHING)
    parser.add_argument("relAcc", type=float, nargs='?',
                        help="target relative accuracy for integrals",
                        default=sets.DIST_FUNC_REL_ACC)
    parser.add_argument("intSteps", type=int, nargs='?',
                        help="number of smoothing points for density array (<=1 is off)",
                        default=sets.DIST_FUNC_INT_STEPS)
    args = parser.parse_args()

    return args


def _mpiError(comm, log, err="ERROR"):
    """
    Raise an error through MPI and exit all processes.

    Arguments
    ---------
       comm <...> : mpi intracommunicator object (e.g. ``MPI.COMM_WORLD``)
       err  <str> : optional, extra error-string to print

    """
    rank = comm.rank
    errStr = "\nERROR: rank %d\n%s\n%s\n" % (rank, str(datetime.now()), err)
    log.exception(errStr)   # , exc_info=True)
    comm.Abort(rank)
    return


def _loadMPILogger(rank, verbose, sets):
    """
    Initialize a logger for output, specifically in parallel configuration.

    Directory should be checked by rank=0 processor before calling this method.
    """
    from zcode.inout.log import get_logger
    # Get logger and log-file names
    name = "logger_%03d" % (rank)
    logFile = _GET_LOG_NAME(rank, sets)
    # Determine verbosity level
    if verbose: lvl_str = logging.INFO
    else:       lvl_str = logging.WARNING
    lvl_fil = logging.DEBUG
    # Create logger
    if rank == 0:
        log = get_logger(name, tofile=logFile, level_file=lvl_fil, level_stream=lvl_str)
    else:
        log = get_logger(name, tofile=logFile, level_file=lvl_fil, tostr=False)
    # Store log filename for convenience
    log.filename = logFile
    return log


def _GET_LOG_NAME(rank, sets, version=__version__):
    logDir = sets.DIR_LOGS
    if rank == 0: logName = "DistFunc_v%s.log" % (version)
    else:         logName = "DistFunc_%03d_v%s.log" % (rank, version)
    logName = os.path.join(logDir, logName)
    return logName


def _useSpline(xx, yy):
    return zmath.spline(xx, yy, order=1, log=True, pos=True, mono=False, sort=True, extrap=True)


if __name__ == "__main__":
    main()
