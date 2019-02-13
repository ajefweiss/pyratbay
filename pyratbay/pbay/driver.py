# Copyright (c) 2016-2019 Patricio Cubillos and contributors.
# Pyrat Bay is currently proprietary software (see LICENSE).

import os
import sys
import time

from .. import lineread   as lr
from .. import tools      as pt
from .. import constants  as pc
from .. import plots      as pp
from .. import pyrat      as py
from .. import atmosphere as pa

from .  import argum     as ar
from .  import pyratfit  as pf

rootdir = os.path.realpath(os.path.dirname(__file__) + "/../../")

sys.path.append(rootdir + "/modules/MCcubed/")
import MCcubed as mc3

__all__ = ["run"]


def run(argv, main=False):
  """
  Pyrat Bay (Python Radiative Transfer in a Bayesian framework)
  initialization driver.

  Parameters
  ----------
  argv: List or string
     If called from the shell, the list of command line arguments; if
     called from the Python interpreter, the configuration-file name.
  main: Bool
     Flag to indicate if Pyrat was called from the shell (True) or from
     the Python interpreter.
  """

  # Put everything into a try--except to catch the sys.exit() Traceback.
  try:
    # Setup the command-line-arguments input:
    if main is False:
      sys.argv = ['pbay.py', '-c', argv]

    # Setup time tracker:
    timestamps = []
    timestamps.append(time.time())

    # Parse command line arguments:
    args, log = ar.parse()
    timestamps.append(time.time())

    # Check run mode:
    if args.runmode not in pc.rmodes:
      log.error("Invalid runmode ({:s}). Select from: {:s}.".
                 format(args.runmode, str(pc.rmodes)))

    # Call lineread package:
    if args.runmode == "tli":
      parser = lr.parser()
      lr.makeTLI(parser.dblist,  parser.pflist, parser.dbtype,
                 parser.outfile, parser.iwl, parser.fwl, parser.verb)
      return

    # Get gplanet from mplanet and rplanet if necessary:
    if (args.gplanet is None and args.rplanet is not None and
        args.mplanet is not None):
      args.gplanet = (pc.G
                    * pt.getparam(args.mplanet, "gram", log)
                    / pt.getparam(args.rplanet, args.radunits, log)**2)

    # Compute pressure-temperature profile:
    if args.runmode in ["pt", "atmosphere"] or pt.isfile(args.atmfile) != 1:
      # Check if PT file is provided:
      if args.ptfile is None:
        ar.checkpressure(args, log)  # Check pressure inputs
        pressure = pa.pressure(args.ptop, args.pbottom, args.nlayers,
                               args.punits, log)
        ar.checktemp(args, log)      # Check temperature inputs
        temperature = pa.temperature(args.tmodel, pressure,
             args.rstar, args.tstar, args.tint, args.gplanet, args.smaxis,
             args.radunits, args.nlayers, log, args.tparams)
      # If PT file is provided, read it:
      elif os.path.isfile(args.ptfile):
        log.msg("\nReading pressure-temperature file: '{:s}'.".
                format(args.ptfile))
        pressure, temperature = pa.read_ptfile(args.ptfile)

    # Return temperature-pressure if requested:
    if args.runmode == "pt":
      return pressure, temperature

    # Compute or read atmospheric abundances:
    if args.runmode == "atmosphere" or pt.isfile(args.atmfile) != 1:
        ar.checkatm(args, log)
        xsolar = pt.getparam(args.xsolar, "none", log)
        abundances = pa.abundances(args.atmfile, pressure, temperature,
            args.species, args.elements, args.uniform, args.punits, xsolar,
            args.solar, log)

    # Return atmospheric model if requested:
    if args.runmode == "atmosphere":
        return pressure, temperature, abundances

    # Check status of extinction-coefficient file if necessary:
    if args.runmode != "spectrum" and pt.isfile(args.extfile) == -1:
      log.error("Unspecified extinction-coefficient file (extfile).")

    # Force to re-calculate extinction-coefficient file if requested:
    if args.runmode == "opacity" and pt.isfile(args.extfile):
      os.remove(args.extfile)

    # Initialize pyrat object:
    dummy_log = mc3.utils.Log(None, width=80)
    if args.resume: # Bypass writting all of the initialization log:
      pyrat = py.init(args.cfile, log=dummy_log)
      pyrat.log = log
    else:
      pyrat = py.init(args.cfile, log=log)

    # Compute spectrum and return pyrat object if requested:
    if args.runmode == "spectrum":
      pyrat = py.run(pyrat)
      return pyrat

    # End if necessary:
    if args.runmode == "opacity":
      return pyrat

    # Parse retrieval info into the Pyrat object:
    pf.init(pyrat, args, log)
    pyrat.log = dummy_log
    dummy_log.verb = 0    # Mute logging in PB, but not in MC3
    pyrat.outspec = None  # Avoid writing spectrum file during MCMC

    # Basename of the output files:
    outfile = os.path.splitext(os.path.basename(log.logname))[0]
    # Run MCMC:
    freeze   = True  # Freeze abundances evoer iterations
    retmodel = False # Return only the band-integrated spectrum
    mc3_out = mc3.mcmc(data=args.data, uncert=args.uncert,
           func=pf.fit, indparams=[pyrat,freeze,retmodel], params=args.params,
           pmin=args.pmin, pmax=args.pmax, stepsize=args.stepsize,
           prior=args.prior, priorlow=args.priorlow, priorup=args.priorup,
           walk=args.walk, nsamples=args.nsamples, nchains=args.nchains,
           burnin=args.burnin, thinning=args.thinning,
           grtest=True, grbreak=args.grbreak, grnmin=args.grnmin,
           hsize=10, kickoff='normal', log=log, nproc=args.nproc,
           plots=True, pnames=pyrat.ret.pnames, texnames=pyrat.ret.texnames,
           showbp=False,
           resume=args.resume, savefile="{:s}.npz".format(outfile))

    if mc3_out is None:
      log.error("Error in MC3.")
    else:
      bestp, CRlo, CRhi, stdp, posterior, Zchain = mc3_out

    # Best-fitting model:
    pyrat.outspec = "{:s}_bestfit_spectrum.dat".format(outfile)
    bestbandflux = pf.fit(bestp, pyrat, retmodel=False)

    # Best-fit atmfile header:
    header = "# MCMC best-fitting atmospheric model.\n\n"
    # Write best-fit atmfile:
    bestatm = "{:s}_bestfit_atmosphere.atm".format(outfile)
    pa.writeatm(bestatm, pyrat.atm.press, pyrat.atm.temp,
                pyrat.mol.name, pyrat.atm.q, pyrat.atm.punits,
                header, radius=pyrat.atm.radius, runits='km')

    # Best-fitting spectrum:
    pp.spectrum(pyrat=pyrat, logxticks=args.logxticks, yran=args.yran,
                filename="{:s}_bestfit_spectrum.png".format(outfile))
    # Posterior PT profiles:
    if pyrat.ret.tmodelname in ["TCEA", "MadhuInv", "MadhuNoInv"]:
      pp.PT(posterior, besttpars=bestp[pyrat.ret.itemp], pyrat=pyrat,
            filename="{:s}_PT_posterior_profile.png".format(outfile))
    # Contribution or transmittance functions:
    if   pyrat.od.path == "eclipse":
      cf  = pt.cf(pyrat.od.depth, pyrat.atm.press, pyrat.od.B)
      bcf = pt.bandcf(cf, pyrat.obs.bandtrans, pyrat.spec.wn, pyrat.obs.bandidx)
    elif pyrat.od.path == "transit":
      transmittance = pt.transmittance(pyrat.od.depth, pyrat.od.ideep)
      bcf = pt.bandcf(transmittance, pyrat.obs.bandtrans, pyrat.spec.wn,
                      pyrat.obs.bandidx)
    pp.cf(bcf, 1.0/(pyrat.obs.bandwn*pc.um), pyrat.od.path,
          pyrat.atm.press, pyrat.atm.radius,
          pyrat.atm.rtop, filename="{:s}_bestfit_cf.png".format(outfile))

    pyrat.log = log  # Un-mute
    log.msg("\nOutput MCMC posterior results, log, bestfit atmosphere, "
      "and spectrum:\n'{:s}.npz',\n'{:s}',\n'{:s}',\n'{:s}'.\n\n".
      format(outfile, os.path.basename(args.logfile), bestatm, pyrat.outspec))
    log.close()
    return pyrat, bestp

  # Avoid printing to screeen the System-Exit Traceback error:
  except SystemExit:
    return None
