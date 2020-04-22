# Copyright (c) 2016-2020 Patricio Cubillos.
# Pyrat Bay is open-source software under the GNU GPL-2.0 license (see LICENSE).

import os
import multiprocessing as mp
from collections import OrderedDict

import numpy  as np

from .. import constants  as pc
from .. import atmosphere as pa
from .. import tools      as pt
from .. import plots      as pp
from .. import io         as io

from .  import extinction as ex
from .  import crosssec   as cs
from .  import rayleigh   as ray
from .  import clouds     as cl
from .  import alkali     as al
from .  import readatm    as ra
from .  import optdepth   as od
from .  import spectrum   as sp
from .  import objects    as ob
from .  import argum      as ar
from .  import makesample as ms
from .  import voigt      as v
from .  import readlinedb as rl


class Pyrat(object):
  """
  Main Pyrat object.
  """
  def __init__(self, cfile, no_logfile=False, mute=False):
      """
      Parse the command-line arguments into the pyrat object.

      Parameters
      ----------
      cfile: String
          A Pyrat Bay configuration file.
      no_logfile: Bool
          If True, enforce not to write outputs to a log file
          (e.g., to prevent overwritting log of a previous run).
      mute: Bool
          If True, enforce verb to take a value of -1.

      Examples
      --------
      >>> import pyratbay as pb
      >>> # Initialize and execute task:
      >>> pyrat = pb.run('spectrum_transmission.cfg')

      >>> # Initialize only:
      >>> pyrat = pb.Pyrat('spectrum_transmission.cfg')
      >>> # Then, setup internal varible for spectra evaluation:
      >>> pyrat.setup_spectrum()
      """
      # Sub-classes:
      self.spec     = ob.Spectrum()        # Spectrum data
      self.atm      = ob.Atm()             # Modeling atmospheric model
      self.lt       = ob.Linetransition()  # Line-transition data
      self.mol      = ob.Molecules()       # Molecules data
      self.iso      = ob.Isotopes()        # Isotopes data
      self.voigt    = ob.Voigt()           # Voigt profile
      self.ex       = ob.Extinction()      # Extinction-coefficient
      self.cs       = ob.Cross()           # Cross-section extinction
      self.od       = ob.Optdepth()        # Optical depth
      self.cloud    = ob.Cloud()           # Cloud models
      self.rayleigh = ob.Rayleigh()        # Rayleigh models
      self.alkali   = ob.Alkali()          # Alkali opacity models
      self.obs      = ob.Observation()     # Observational data
      self.phy      = ob.Physics()         # System physical parameters
      self.ret      = ob.Retrieval()       # Retrieval variables
      self.timestamps = OrderedDict()

      # Parse config file inputs:
      pt.parse(self, cfile, no_logfile, mute)
      self.inputs.atm = ob.Atm()


  def setup_spectrum(self):
      # Setup time tracker:
      timer = pt.Timer()

      # Check that user input arguments make sense:
      ar.check_spectrum(self)
      self.timestamps['init'] = timer.clock()

      # Initialize wavenumber sampling:
      ms.make_wavenumber(self)
      self.timestamps['wn sample'] = timer.clock()

      # Read the atmospheric file:
      ra.read_atm(self)
      self.timestamps['read atm'] = timer.clock()

      # Read line database:
      rl.read_tli(self)
      self.timestamps['read tli'] = timer.clock()

      # Make atmospheric profiles (pressure, radius, temperature, abundances):
      ms.make_atmprofiles(self)
      self.timestamps['atm sample'] = timer.clock()

      # Setup more observational/retrieval parameters:
      ar.setup(self)

      # Extinction Voigt grid:
      v.voigt(self)
      # Alkali Voigt grid:
      al.init(self)
      self.timestamps['voigt'] = timer.clock()

      # Calculate extinction-coefficient table:
      ex.exttable(self)
      self.timestamps['ext table'] = timer.clock()

      # Read CIA files:
      cs.read(self)
      self.timestamps['read cs'] = timer.clock()


  def run(self, temp=None, abund=None, radius=None):
      """
      Evaluate a Pyrat spectroscopic model

      Parameters
      ----------
      pyrat: A Pyrat instance
      temp: 1D float ndarray
          Updated atmospheric temperature profile in Kelvin, of size nlayers.
      abund: 2D float ndarray
          Updated atmospheric abundances profile by number density, of
          shape [nlayers, nmol].
      radius: 1D float ndarray
          Updated atmospheric altitude profile in cm, of size nlayers.
      """
      timer = pt.Timer()

      # Re-calculate atmospheric properties if required:
      status = ra.reloadatm(self, temp, abund, radius)
      if status == 0:
          return
      # Interpolate CIA absorption:
      cs.interpolate(self)
      self.timestamps['interp cs'] = timer.clock()

      # Calculate cloud and Rayleigh absorption:
      cl.absorption(self)
      ray.absorption(self)
      self.timestamps['cloud+ray'] = timer.clock()

      # Calculate the alkali absorption:
      al.absorption(self)
      self.timestamps['alkali'] = timer.clock()

      # Calculate the optical depth:
      od.opticaldepth(self)
      self.timestamps['odepth'] = timer.clock()

      # Calculate the spectrum:
      sp.spectrum(self)
      self.timestamps['spectrum'] = timer.clock()

      self.log.msg("\nTimestamps (s):\n" +
                   "\n".join("{:10s}: {:10.6f}".format(key,val)
                             for key,val in self.timestamps.items()))

      if len(self.log.warnings) > 0 and self.log.logname is not None:
          # Write all warnings to file:
          wpath, wfile = os.path.split(self.log.logname)
          wfile = f'{wpath}/warnings_{wfile}'
          with open(wfile, 'w') as f:
              f.write(f'Warnings log:\n\n{self.log.sep}\n')
              f.write(f'\n\n{self.log.sep}\n'.join(self.log.warnings))
          # Report it:
          self.log.head(
              f"\n{self.log.sep}"
              f"\n  There were {len(self.log.warnings)} warnings raised.  "
              f"See '{wfile}'."
              f"\n{self.log.sep}")


  def eval(self, params, retmodel=True, verbose=False):
      """
      Fitting routine for MCMC.

      Parameters
      ----------
      params: 1D float iterable
         Array of fitting parameters that define the atmosphere.
      retmodel: Bool
         Flag to include the model spectra in the return.
      verbose: Bool
         Flag to print out if a run failed.

      Returns
      -------
      spectrum: 1D float ndarray
         The output model spectra.  Returned only if retmodel=True.
      bandflux: 1D float ndarray
         The waveband-integrated spectrum values.
      """
      params = np.asarray(params)
      q0 = np.copy(self.atm.qbase)

      if len(params) != self.ret.nparams:
          self.log.warning(
              f'The number of input fitting parameters ({len(params)}) does '
              f'not match\nthe number of required '
              f'parameters ({self.ret.nparams}).')
          return None, None if retmodel else None

      rejectflag = False
      # Update temperature profile if requested:
      if self.ret.itemp is not None:
          temp = self.atm.tmodel(params[self.ret.itemp])
      else:
          temp = self.atm.temp
      # Turn-on reject flag if temperature is out-of-bounds:
      if np.any(temp < self.ret.tlow) or np.any(temp > self.ret.thigh):
          temp[:] = 0.5*(self.ret.tlow + self.ret.thigh)
          rejectflag = True
          if verbose:
              self.log.warning("Input temperature profile runs out of "
                  f"boundaries ({self.ret.tlow:.1f}--{self.ret.thigh:.1f} K)")

      # Update abundance profiles if requested:
      if self.ret.imol is not None:
          q2 = pa.qscale(q0, self.mol.name, self.atm.molmodel,
                         self.atm.molfree, params[self.ret.imol],
                         self.atm.bulk,
                         iscale=self.atm.ifree, ibulk=self.atm.ibulk,
                         bratio=self.atm.bulkratio, invsrat=self.atm.invsrat)
      else:
          q2 = self.atm.q

      # Check abundaces stay within bounds:
      if pa.qcapcheck(q2, self.ret.qcap, self.atm.ibulk):
          rejectflag = True
          if verbose:
              self.log.warning("The sum of trace abundances' fraction exceeds "
                              f"the cap of {self.ret.qcap:.3f}.")

      # Update reference radius if requested:
      if self.ret.irad is not None:
          self.phy.rplanet = params[self.ret.irad][0] * pt.u(self.atm.runits)

      # Update planetary mass if requested:
      if self.ret.imass is not None:
          self.phy.mplanet = params[self.ret.imass][0] * pt.u(self.phy.mpunits)

      # Keep M-g-R0 consistency:
      if self.atm.rmodelname == 'hydro_g': # and self.ret.igrav is None:
          self.phy.gplanet = pc.G * self.phy.mplanet / self.phy.rplanet**2
      #if self.atm.rmodelname == 'hydro_m' and self.ret.igrav is not None:
      #    self.phy.mplanet = self.phy.gplanet * self.phy.rplanet**2 / pc.G

      # Update Rayleigh parameters if requested:
      if self.ret.iray is not None:
          j = 0
          rpars = params[self.ret.iray]
          for rmodel in self.rayleigh.models:
              rmodel.pars = rpars[j:j+rmodel.npars]
              j += rmodel.npars

      # Update cloud parameters if requested:
      if self.ret.icloud is not None:
          j = 0
          pars = params[self.ret.icloud]
          for model in self.cloud.models:
              model.pars = pars[j:j+model.npars]
              j += model.npars

      # Update patchy-cloud fraction if requested:
      if self.ret.ipatchy is not None:
          self.cloud.fpatchy = params[self.ret.ipatchy]

      # Calculate spectrum:
      self.run(temp=temp, abund=q2)

      # Band-integrate spectrum:
      self.obs.bandflux = self.band_integrate()

      # Reject this iteration if there are invalid temperatures or radii:
      if self.obs.bandflux is not None and rejectflag:
          self.obs.bandflux[:] = np.inf

      self.ret.params = params
      if retmodel:
          return self.spec.spectrum, self.obs.bandflux

      return self.obs.bandflux


  def band_integrate(self, spectrum=None):
      """
      Band-integrate transmission spectrum (transit) or planet-to-star
      flux ratio (eclipse) over transmission band passes.
      """
      if self.obs.bandtrans is None:
          return None

      if spectrum is None:
          spectrum = self.spec.spectrum
      specwn   = self.spec.wn
      bandidx  = self.obs.bandidx

      if self.od.path == 'transit':
          bandtrans = self.obs.bandtrans
      elif self.od.path == 'eclipse':
          bandtrans = [btrans/sflux * (self.phy.rplanet/self.phy.rstar)**2
               for btrans, sflux in zip(self.obs.bandtrans, self.obs.starflux)]

      self.obs.bandflux = np.array([np.trapz(spectrum[idx]*btrans, specwn[idx])
                                    for btrans, idx in zip(bandtrans, bandidx)])

      return self.obs.bandflux


  def hydro(self, pressure, temperature, mu, g, mass, p0, r0):
      """
      Hydrostatic-equilibrium driver.
      Depending on self.atm.rmodelname, select between the g=GM/r**2
      (hydro_m) or constant-g (hydro_g) formula to compute
      the hydrostatic-equilibrium radii of the planet layers.

      Parameters
      ----------
      pressure: 1D float ndarray
         Atmospheric pressure for each layer (in barye).
      temperature: 1D float ndarray
         Atmospheric temperature for each layer (in K).
      mu: 1D float ndarray
         Mean molecular mass for each layer (in g mol-1).
      g: Float
         Atmospheric gravity (in cm s-2).
      mass: Float
         Planetary mass (in g).
      p0: Float
         Reference pressure level (in barye) where radius(p0) = r0.
      r0: Float
         Reference radius level (in cm) corresponding to p0.
      """
      if self.atm.rmodelname is None:
          print('No hydrostatic-equilibrium model defined.')
          return None
      # H.E. with  g=GM/r**2:
      elif self.atm.rmodelname == 'hydro_m':
          return pa.hydro_m(pressure, temperature, mu, mass, p0, r0)
      # H.E. with constant g:
      elif self.atm.rmodelname == 'hydro_g':
          return pa.hydro_g(pressure, temperature, mu, g, p0, r0)

  def set_filters(self):
      """
      Set observational variables (pyrat.obs) based on given parameters.
      """
      if self.obs.filters is None:
          return

      bandidx   = []  # Filter wavenumber indices
      starflux  = []  # Interpolated stellar flux
      bandtrans = []  # Normalized interpolated filter transmission
      bandwn    = []  # Band's mean wavenumber
      for filter in self.obs.filters:
          # Read filter wavenumber and transmission curves:
          filterwn, filtertr = io.read_spectrum(filter)
          # Resample the filters into the planet wavenumber array:
          btrans, bidx = pt.resample(filtertr, filterwn, self.spec.wn,
              normalize=True)
          bandidx.append(bidx)
          bandtrans.append(btrans)
          bandwn.append(np.sum(filterwn*filtertr)/np.sum(filtertr))
          if self.phy.starflux is not None:
              starflux.append(self.spec.starflux[bidx])

      # Per-band variables:
      self.obs.bandidx   = bandidx
      self.obs.bandtrans = bandtrans
      self.obs.starflux  = starflux
      self.obs.bandwn    = np.asarray(bandwn)
      self.obs.bandflux  = np.zeros(self.obs.nfilters, np.double)


  def get_ec(self, layer):
      """
      Extract extinction-coefficient contribution (in cm-1) from each
      component of the atmosphere at the requested layer.

      Parameters
      ----------
      layer: Integer
         The index of the atmospheric layer where to extract the EC.

      Returns
      -------
      ec: 2D float ndarray
         An array of shape [ncomponents, nwave] with the EC spectra
         (in cm-1) from each component of the atmosphere.
      label: List of strings
         The names of each atmospheric component that contributed to EC.
      """
      # Allocate outputs:
      ec = np.empty((0, self.spec.nwave))
      label = []
      # Line-by-line extinction coefficient:
      if self.ex.nspec != 0:
          e, lab = ex.get_ec(self, layer)
          ec = np.vstack((ec, e))
          label += lab
      # Cross-section extinction coefficient:
      if self.cs.nfiles != 0:
          e, lab = cs.interpolate(self, layer)
          ec = np.vstack((ec, e))
          label += lab
      # Rayleigh scattering extinction coefficient:
      if self.rayleigh.models != []:
          e, lab = ray.get_ec(self, layer)
          ec = np.vstack((ec, e))
          label += lab
      # Haze/clouds extinction coefficient:
      if self.cloud.models != []:
          e, lab = cl.get_ec(self, layer)
          ec = np.vstack((ec, e))
          label += lab
      # Alkali resonant lines extinction coefficient:
      if self.alkali.models != []:
          e, lab = al.get_ec(self, layer)
          ec = np.vstack((ec, e))
          label += lab
      return ec, label


  def percentile_spectrum(self, nmax=None):
      """Compute spectrum posterior percentiles."""
      if self.ret.posterior is None:
          print('pyrat objec does not have a posterior distribution.')
          return

      nsamples = np.shape(self.ret.posterior)[0]
      draws = np.arange(nsamples)
      if nmax is not None:
          nmax = np.clip(nmax, 0, nsamples)
          draws = np.random.choice(draws, nmax, replace=False)

      # Unique MCMC samples:
      u, uind, uinv = np.unique(self.ret.posterior[draws,0],
          return_index=True, return_inverse=True)
      print('Computing {:d} models.'.format(len(u)))

      # Array of all model parameters (with unique samples)
      posterior = np.repeat([self.ret.params], len(u), axis=0)
      ifree = np.where(self.ret.pstep >0)[0]
      posterior[:,ifree] = self.ret.posterior[uind]
      # Need to keep FILE objects out of pool:
      logfile, self.log.file = self.log.file, None
      verb, self.log.verb = self.log.verb, -1

      with mp.Pool(self.ncpu) as pool:
          models = pool.map(self.eval, posterior)
      models = np.array([model for model, bandm in models])

      self.log.file = logfile
      self.log.verb = verb

      nwave = len(self.spec.wn)
      low1   = np.zeros(nwave)
      low2   = np.zeros(nwave)
      median = np.zeros(nwave)
      high1  = np.zeros(nwave)
      high2  = np.zeros(nwave)
      for i in range(nwave):
          msample = models[uinv,i]
          low2[i]   = np.percentile(msample,  2.275)
          low1[i]   = np.percentile(msample, 15.865)
          median[i] = np.percentile(msample, 50.000)
          high1[i]  = np.percentile(msample, 84.135)
          high2[i]  = np.percentile(msample, 97.725)

      self.ret.spec_median = median
      self.ret.spec_low1 = low1
      self.ret.spec_low2 = low2
      self.ret.spec_high1 = high1
      self.ret.spec_high2 = high2


  def plot_spectrum(self, spec='model', logxticks=None, gaussbin=2.0,
                    yran=None, filename=None):
      """
      Plot spectrum.

      Parameters
      ----------
      spec: String
          Flag indicating which model to plot.  By default plot the
          latest evaulated model (spec='model').  Other options are
          'best' or 'median' to plot the posterior best-fit or median
          model, in which case, the code will plot the 1- and 2-sigma
          boundaries if they have been computed (see
          self.percentile_spectrum).
      logxticks: 1D float ndarray
          If not None, switch the X-axis scale from linear to log, and set
          the X-axis ticks at the locations given by logxticks.
      gaussbin: Integer
          Standard deviation for Gaussian-kernel smoothing (in number
          of samples).
      yran: 1D float ndarray
          Figure's Y-axis boundaries.
      filename: String
          If not None, save figure to filename.

      Returns
      -------
      ax: AxesSubplot instance
          The matplotlib Axes of the figure.
      """
      wavelength = 1.0/(self.spec.wn*pc.um)
      if self.obs.bandwn is not None:
          bandwl = 1.0/(self.obs.bandwn*pc.um)
      else:
          bandwl = None

      if self.obs.bandtrans is not None:
          bandflux = self.band_integrate()
      else:
          bandflux = None

      if logxticks is None:
          logxticks = self.inputs.logxticks
      if yran is None:
          yran = self.inputs.yran

      bounds = None
      if self.ret.spec_low2 is not None and spec != 'model':
          bounds = [self.ret.spec_low2,  self.ret.spec_low1,
                    self.ret.spec_high1, self.ret.spec_high2]

      if spec == 'model':
          label = 'model'
          spectrum = self.spec.spectrum
      elif spec == 'best':
          label = 'best-fit model'
          spectrum = self.ret.spec_best
          bandflux = self.ret.bestbandflux
      elif spec == 'median':
          label = 'median model'
          spectrum = self.ret.spec_median
          bandflux = self.band_integrate(spectrum)
      else:
          print("Invalid 'spec'.  Select from 'model' (default), 'best', "
                "or 'median'.")
          return

      if self.phy.rplanet is None or self.phy.rstar is None:
          rprs = None
      else:
          rprs = self.phy.rplanet/self.phy.rstar

      ax = pp.spectrum(spectrum, wavelength, self.od.path,
          self.obs.data, self.obs.uncert, bandwl, bandflux,
          self.obs.bandtrans, self.obs.bandidx,
          self.spec.starflux, rprs, label, bounds,
          logxticks, gaussbin, yran, filename)
      return ax


  def plot_posterior_pt(self, filename=None):
      """
      Plot posterior distribution of PT profile.

      Parameters
      ----------
      filename: String
          If not None, save figure to filename.

      Returns
      -------
      ax: AxesSubplot instance
          The matplotlib Axes of the figure.
      """
      if self.ret.posterior is None:
          print('pyrat objec does not have a posterior distribution.')
          return

      posterior = self.ret.posterior
      ifree = self.ret.pstep[self.ret.itemp] > 0
      itemp = np.arange(np.sum(ifree))
      if filename is None:
          outfile = os.path.splitext(os.path.basename(self.ret.mcmcfile))[0]
          filename = '{:s}_posterior_PT_profile.png'.format(outfile)
      ax = pp.posterior_pt(posterior[:,itemp], self.atm.tmodel,
          self.ret.params[self.ret.itemp], ifree, self.atm.press,
          self.ret.bestp[self.ret.itemp], filename)
      return ax


  def __str__(self):
      if self.spec.resolution is not None:
         wave = "R={:.1f}".format(self.spec.resolution)
      else:
         wave = "dwn={:.3f} cm-1".format(self.spec.wnstep)

      opacities = []
      if self.ex.nspec != 0:
          for mol in self.ex.species:
              imol = np.where(self.mol.name == mol)[0][0]
              opacities.append(self.mol.name[imol])
      if self.cs.nfiles != 0:
          for molecs in self.cs.molecules:
              if len(molecs) == 2:
                  opacities.append('CIA ' + '-'.join(molecs))
              else:
                  opacities.append(molecs[0])
      for rmodel in self.rayleigh.models:
          opacities.append(rmodel.name)
      for cloud in self.cloud.models:
          opacities.append(cloud.name)
      for alkali in self.alkali.models:
          opacities.append(alkali.mol)

      return ("Pyrat atmospheric model\n"
          "configuration file:  '{:s}'\n"
          "Pressure profile (bar):  {:.2e} -- {:.2e} ({:d} layers)\n"
          "Wavelength range (um):  {:.2f} -- {:.2f} ({:d} samples, {:s})\n"
          "Composition:  {}\n"
          "Opacity sources:  {}".format(
          self.inputs.configfile,
          self.atm.press[ 0]/pc.bar,
          self.atm.press[-1]/pc.bar,
          self.atm.nlayers,
          1.0/(self.spec.wn[-1]*pc.um),
          1.0/(self.spec.wn[ 0]*pc.um),
          self.spec.nwave,
          wave,
          self.mol.name,
          opacities))

