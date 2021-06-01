# Copyright (c) 2021 Patricio Cubillos
# Pyrat Bay is open-source software under the GNU GPL-2.0 license (see LICENSE)

import numpy as np
from numpy.core.fromnumeric import transpose
from scipy.interpolate import interp1d

from .. import constants as pc
from .. import io as io
from .. import spectrum as ps
from ..lib import _trapz as t

import math
from viztracer import log_sparse, get_tracer
import numba
import numba.cuda as cuda


@log_sparse
def spectrum(pyrat):
    """
    Spectrum calculation driver.
    """
    pyrat.log.head('\nCalculate the planetary spectrum.')

    # Initialize the spectrum array:
    pyrat.spec.spectrum = np.empty(pyrat.spec.nwave, np.double)
    if pyrat.cloud.fpatchy is not None:
        pyrat.spec.clear  = np.empty(pyrat.spec.nwave, np.double)
        pyrat.spec.cloudy = np.empty(pyrat.spec.nwave, np.double)

    # Call respective function depending on the geometry:
    if pyrat.od.rt_path in pc.transmission_rt:
        modulation(pyrat)

    elif pyrat.od.rt_path in pc.emission_rt:
        intensity(pyrat)
        flux(pyrat)

    # Print spectrum to file:
    if pyrat.od.rt_path in pc.transmission_rt:
        spec_type = 'transit'
    elif pyrat.od.rt_path in pc.emission_rt:
        spec_type = 'emission'

    io.write_spectrum(
        1.0/pyrat.spec.wn, pyrat.spec.spectrum, pyrat.spec.specfile, spec_type)
    pyrat.log.head('Done.')


@log_sparse
def modulation(pyrat):
    """Calculate modulation spectrum for transit geometry."""
    rtop = pyrat.atm.rtop
    radius = pyrat.atm.radius
    depth = pyrat.od.depth

    # Get Delta radius (and simps' integration variables):
    h = np.ediff1d(radius[rtop:])
    # The integrand:
    integ = (np.exp(-depth[rtop:,:]) * np.expand_dims(radius[rtop:],1))

    if pyrat.cloud.fpatchy is not None:
        h_clear = np.copy(h)
        integ_clear = (
            np.exp(-pyrat.od.depth_clear[rtop:,:]) *
                  np.expand_dims(radius[rtop:],1))

    if 'deck' in (m.name for m in pyrat.cloud.models):
        # Replace (interpolating) last layer with cloud top:
        deck = pyrat.cloud.models[pyrat.cloud.model_names.index('deck')]
        if deck.itop > rtop:
            h[deck.itop-rtop-1] = deck.rsurf - radius[deck.itop-1]
            integ[deck.itop-rtop] = interp1d(
                radius[rtop:], integ, axis=0)(deck.rsurf)

    # Number of layers for integration at each wavelength:
    nlayers = pyrat.od.ideep - rtop + 1
    spectrum = t.trapz2D(integ, h, nlayers-1)
    pyrat.spec.spectrum = (radius[rtop]**2 + 2*spectrum) / pyrat.phy.rstar**2

    if pyrat.cloud.fpatchy is not None:
        nlayers = pyrat.od.ideep_clear - rtop + 1
        pyrat.spec.clear = t.trapz2D(integ_clear, h_clear, nlayers-1)

        pyrat.spec.clear = ((radius[rtop]**2 + 2*pyrat.spec.clear)
                             / pyrat.phy.rstar**2)
        pyrat.spec.cloudy = pyrat.spec.spectrum
        pyrat.spec.spectrum = (   pyrat.cloud.fpatchy  * pyrat.spec.cloudy +
                               (1-pyrat.cloud.fpatchy) * pyrat.spec.clear  )

    if pyrat.spec.specfile is not None:
        specfile = f": '{pyrat.spec.specfile}'"
    else:
        specfile = ""
    pyrat.log.head(f"Computed transmission spectrum{specfile}.", indent=2)


@cuda.jit(device=True)
def tdiff(dtau, depth, mu, top, last, wave):
    for i in range(0, int(last - top)):
        val1 = math.exp(-depth[top + i + 1, wave] / mu)
        val2 = math.exp(-depth[top + i, wave] / mu) 
        dtau[i] =  val1 - val2


@cuda.jit(device=True)
def itrapz(bbody, dtau, top, last, wave_ind):
    res = np.float32(0)

    for i in range(0, int(last - top)):
        res += dtau[i] * (bbody[top + i + 1, wave_ind] + bbody[top + i, wave_ind])
    #print(0.5 * res)
    return 0.5  * res


@cuda.jit
def t_intensity_cuda(depth, ideep, bbody, mu, rtop, return_arr):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x

    wave_ind = tx + ty * bw
    kvalue = cuda.threadIdx.y

    dtau = cuda.local.array((100,), dtype=numba.types.float32)

    if wave_ind < 8001 and kvalue < 5:

        last = int(ideep[wave_ind])
        tau_max = depth[last, wave_ind]
        
        if last - rtop == 1:
            return_arr[kvalue, wave_ind] = bbody[last, wave_ind]
        else:
            # integration
            tdiff(dtau, depth, mu[kvalue], rtop, last, wave_ind)
            return_arr[kvalue, wave_ind] = bbody[last, wave_ind] * math.exp(-tau_max / mu[kvalue]) - itrapz(bbody, dtau, rtop, last, wave_ind)


@log_sparse
def t_intensity_cuda_wrapper(depth, ideep, bbody, mu, rtop):
    alts, nwave = depth.shape
    ntheta = len(mu)
    #dtau = cuda.device_array((5, 8001, 100,), dtype=np.float64)
    return_arr = cuda.device_array((5, 8001), dtype=np.float64)
    print("cuda wrapper")
    t_intensity_cuda[64, (128, 8)](depth.astype(np.float32), ideep.astype(np.float32), bbody.astype(np.float32), mu.astype(np.float32), rtop, return_arr)

    return return_arr.copy_to_host().astype(np.float64)

@log_sparse
def intensity(pyrat):
    """
    Calculate the intensity spectrum [units] for eclipse geometry.
    """
    spec = pyrat.spec
    pyrat.log.msg('Computing intensity spectrum.', indent=2)
    if spec.quadrature is not None:
        spec.raygrid = np.arccos(np.sqrt(spec.qnodes))

    # Allocate intensity array:
    spec.nangles = len(spec.raygrid)
    spec.intensity = np.empty((spec.nangles, spec.nwave), np.double)

    # Calculate the Planck Emission:
    with get_tracer().log_event("planck_emission"):
        pyrat.od.B = np.zeros((pyrat.atm.nlayers, spec.nwave), np.double)
        ps.blackbody_wn_2D(spec.wn, pyrat.atm.temp, pyrat.od.B, pyrat.od.ideep)

    if 'deck' in (m.name for m in pyrat.cloud.models):
        with get_tracer().log_event("deck"):
            deck = pyrat.cloud.models[pyrat.cloud.model_names.index('deck')]
            pyrat.od.B[deck.itop] = ps.blackbody_wn(pyrat.spec.wn, deck.tsurf)

    # Plane-parallel radiative-transfer intensity integration:
    with get_tracer().log_event("t.intensity"):
        spec.intensity = t.intensity(
            pyrat.od.depth, pyrat.od.ideep, pyrat.od.B, np.cos(spec.raygrid),
            pyrat.atm.rtop)

        _intensity = t_intensity_cuda_wrapper(pyrat.od.depth, pyrat.od.ideep, pyrat.od.B, np.cos(spec.raygrid), pyrat.atm.rtop)
        print(spec.intensity)
        print(_intensity)

@log_sparse
def flux(pyrat):
    """
    Calculate the hemisphere-integrated flux spectrum [units] for eclipse
    geometry.
    """
    spec = pyrat.spec
    # Calculate the projected area:
    boundaries = np.linspace(0, 0.5*np.pi, spec.nangles+1)
    boundaries[1:spec.nangles] = 0.5 * (spec.raygrid[:-1] + spec.raygrid[1:])
    area = np.pi * (np.sin(boundaries[1:])**2 - np.sin(boundaries[:-1])**2)

    if spec.quadrature is not None:
        area = spec.qweights * np.pi
    # Weight-sum the intensities to get the flux:
    spec.spectrum[:] = np.sum(spec.intensity * np.expand_dims(area,1), axis=0)
    pyrat.log.head(f"Computed emission spectrum: '{spec.specfile}'.", indent=2)
