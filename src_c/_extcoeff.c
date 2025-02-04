// Copyright (c) 2021 Patricio Cubillos
// Pyrat Bay is open-source software under the GNU GPL-2.0 license (see LICENSE)

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdarg.h>

#include "ind.h"
#include "constants.h"
#include "utils.h"


PyDoc_STRVAR(extinction__doc__,
"Calculate the extinction-coefficient (cm-1, if add==1) or the    \n\
opacity per molecule (cm2 molecule-1), at a given pressure        \n\
and temperature, over a wavenumber range.                         \n\
                                                                  \n\
Parameters                                                        \n\
----------                                                        \n\
ext: 2D float ndarray                                             \n\
    Output extinction coefficient or opacity [nextinct,nwave].    \n\
profile: 1D float ndarray                                         \n\
    Array of Voigt profiles.                                      \n\
psize: 2D integer ndarray                                         \n\
    Profiles half-size.                                           \n\
pindex: 2D integer ndarray                                        \n\
    Index where each profile starts.                              \n\
lorentz: 1D Float ndarray                                         \n\
    Sample of Lorentz HWHMs.                                      \n\
doppler: 1D Float ndarray                                         \n\
    Sample of Doppler HWHMs.                                      \n\
wn: 1D Float ndarray                                              \n\
    Spectrum wavenumber array [nwave] (cm-1).                     \n\
own: 1D Float ndarray                                             \n\
    Oversampled wavenumber array (cm-1).                          \n\
divisors: 1D integer ndarray                                      \n\
    Integer divisors for oversampling factor.                     \n\
moldensity: 1D Float ndarray                                      \n\
    Number density for each species [nmol] (molecules cm-3).      \n\
molq: 1D Float ndarray                                            \n\
    Atmospheric species mole mixing ratio [nmol].                 \n\
molrad: 1D Float ndarray                                          \n\
    Atmospheric species collision radius [nmol] (Angstrom).       \n\
molmass: 1D Float ndarray                                         \n\
    Atmospheric species mass [nmol] (gr mol-1).                   \n\
isoimol: 1D Float ndarray                                         \n\
    Index of atmospheric molecule for each isotope [niso].        \n\
isomass: 1D Float ndarray                                         \n\
    Isotopes mass [niso] (g mol-1).                               \n\
isoratio: 1D Float ndarray                                        \n\
    Isotopes abundance ratio [niso].                              \n\
isoz: 1D Float ndarray                                            \n\
    Isotopes partition function [niso].                           \n\
isoiext: 1D Float ndarray                                         \n\
    Index of molecule (in ext array) for each isotope [niso].     \n\
lwn: 1D Float ndarray                                             \n\
    Line-transition wavenumber [nlines] (cm-1).                   \n\
elow: 1D Float ndarray                                            \n\
    Line-transition lower-state energy [nlines] (cm-1).           \n\
gf: 1D Float ndarray                                              \n\
    Line-transition oscillator strength [nlines].                 \n\
lID: 1D integer ndarray                                           \n\
    Line-transition isotope ID [nlines].                          \n\
cutoff: Float                                                     \n\
    Voigt profile cutoff (cm-1).                                  \n\
ethresh: Float                                                    \n\
    Extinction-coefficient threshold factor.                      \n\
pressure: Float                                                   \n\
    Atmospheric-layer pressure (barye).                           \n\
temp: Float                                                       \n\
    Atmospheric-layer temperature (K).                            \n\
verb: Integer                                                     \n\
    Verbosity level.                                              \n\
add: Integer                                                      \n\
    If add=1 calculate the total extinction coefficient (in cm-1) \n\
    for this layer, if add=0 calculate the opacity per species    \n\
    (in cm2 molecule-1) in the layer.                             \n\
resolution: Integer                                               \n\
    Flag, if True perform a linear interpolation of the extinction\n\
    spectrum into the constant-resolution output spectrum.        \n\
    Otherwise, resample into the constant-delta-wavenumber        \n\
    output spectrum.                                              \n\
                                                                  \n\
Uncredited developers                                             \n\
---------------------                                             \n\
Patricio Rojo  U. Cornell.");

static PyObject *extinction(PyObject *self, PyObject *args){
  PyArrayObject *ext,
      *profile, *psize, *pindex,
      *lorentz, *doppler,
      *wn, *own, *divisors,
      *moldensity, *molq, *molrad, *molmass,
      *isoimol, *isomass, *isoratio,
      *isoz, *isoiext,
      *lwn, *lID, *elow, *gf;

  long nmol, niso, nlines, nextinct, ndivs,
      onwn, dnwn,
      minj, maxj,
      nLor, nDop;
  int iown, idwn, offset, subw,
      imol, ofactor, iprof,
      nadd=0, nskip=0, neval=0,
      verb, add=0, resolution;
  int mincut, maxcut;
  int i, j, iext, ln;
  long jj;
  double pressure, temp, collision_diameter, density, minwidth=1e5, vwidth,
      cutoff, ethresh, florentz, fdoppler, wnstep, ownstep, dwnstep,
      wavn, next_wn, k;
  double *alphal, *alphad, *kmax, **ktmp, *kprop;
  int *idop, *ilor;

  /* Load inputs: */
  if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOOOOOOOOOOddddi|ii",
          &ext,
          &profile, &psize, &pindex, &lorentz, &doppler,
          &wn, &own, &divisors,
          &moldensity, &molq, &molrad, &molmass,
          &isoimol, &isomass, &isoratio, &isoz, &isoiext,
          &lwn, &elow, &gf, &lID,
          &cutoff, &ethresh, &pressure, &temp,
          &verb, &add, &resolution))
      return NULL;

  /* Sizes: */
  nLor   = PyArray_DIM(lorentz, 0);  /* Lorentz widths */
  nDop   = PyArray_DIM(doppler, 0);  /* Doppler widths */
  nmol   = PyArray_DIM(molmass, 0);  /* species */
  niso   = PyArray_DIM(isomass, 0);  /* isotopes */
  ndivs  = PyArray_DIM(divisors, 0);  /* divisors of osamp */
  onwn   = PyArray_DIM(own, 0);  /* fine-wavenumber samples */
  nlines = PyArray_DIM(lwn, 0);  /* line transitions */
  nextinct = PyArray_DIM(ext, 0);  /* extinction-coefficient species */

  if (add)
      nextinct = 1;

  /* Constant factors for line widths: */
  fdoppler = sqrt(2*KB*temp/AMU) * SQRTLN2 / LS;
  florentz = sqrt(2*KB*temp/PI/AMU) / LS;

  /* Allocate alpha Lorentz and Doppler arrays: */
  alphal = (double *)malloc(niso * sizeof(double));
  alphad = (double *)malloc(niso * sizeof(double));

  /* Allocate width indices array: */
  idop = (int *)malloc(niso * sizeof(int));
  ilor = (int *)malloc(niso * sizeof(int));

  /* Allocate line strength per transition: */
  kprop = (double *)malloc(nlines * sizeof(double));
  /* Array to hold maximum line strength: */
  kmax = (double *)calloc(nextinct, sizeof(double));

  ktmp = (double **)malloc(nextinct * sizeof(double *));
  ktmp[0] = (double  *)calloc(nextinct*onwn, sizeof(double));
  for (i=1; i<nextinct; i++)
      ktmp[i] = ktmp[0] + onwn*i;

  /* Calculate the isotopes' widths: */
  for (i=0; i<niso; i++){
      imol = INDi(isoimol, i);
      /* Lorentz profile width: */
      alphal[i] = 0.0;
      for (j=0; j<nmol; j++){
          collision_diameter = INDd(molrad, imol) + INDd(molrad, j);
          density = INDd(molq,j) * pressure / (KB*temp);
          alphal[i] += density * collision_diameter * collision_diameter *
                       sqrt(1/INDd(isomass,i) + 1/INDd(molmass,j));
          if (i==-1)
              printf("j:%d,  %.6e,  %.6e,  %.6e\n", j,
                     density, collision_diameter, INDd(molmass,j));
      }
      alphal[i] *= florentz;

      /* Doppler profile width (divided by central wavenumber): */
      alphad[i] = fdoppler / sqrt(INDd(isomass,i));
      if (i <= 0 && verb > 6){
          printf("    Lorentz: %.3e cm-1, Doppler: %.3e cm-1.\n",
                 alphal[i], alphad[i]*INDd(own,0));
      }
      /* Estimate the Voigt FWHM: */
      vwidth = 0.5346*alphal[i] + sqrt(pow(alphal[i], 2)*0.2166    +
                                       pow(alphad[i]*INDd(own,0),2) );
      minwidth = fmin(minwidth, vwidth);

      /* Search for aDop and aLor indices for alphal[i] and alphad[i]: */
      idop[i] = binsearchapprox(doppler, alphad[i]*INDd(own,0), 0, (int)nDop-1);
      ilor[i] = binsearchapprox(lorentz, alphal[i], 0, (int)nLor-1);
  }

  wnstep  = INDd(wn, 1) - INDd(wn, 0);
  ownstep = INDd(own,1) - INDd(own,0);
  /* Set the wavenumber sampling resolution: */
  /* Have at least two samples across the minimum FWHM: */
  for (i=1; i<ndivs; i++)
      if (INDi(divisors,i)*ownstep >= 0.5 * minwidth){
          break;
      }
  ofactor = INDi(divisors,(i-1)); /* Dynamic-sampling oversampling factor */
  dwnstep = ownstep * ofactor; /* Dynamic-sampling grid stepsize */
  dnwn = 1 + (onwn-1) / ofactor; /* Number of dynamic-sampling values */
  if (verb > 6){
      printf("    Dynamic-sampling grid interval: %.4e "
             "(factor:%i, minwidth:%.3e)\n", dwnstep, ofactor, minwidth);
      printf("    Number of dynamic-sampling values: %ld\n", dnwn);
  }

  /* Find the maximum line-strength per molecule: */
  for (ln=0; ln<nlines; ln++){
      /* Wavelength, isotope index, and species index of line: */
      wavn = INDd(lwn, ln);
      i = INDi(lID, ln);
      iext = INDi(isoiext, i);
      if (add)  /* Co-add extinction coefficient */
          iext = 0;

      /* If this line falls beyond limits, skip to next line transition: */
      if ((wavn < INDd(own,0)) || (wavn > INDd(own,(onwn-1))))
          continue;

      /* Calculate the line strength divided by the molecular abundance: */
      kprop[ln] = k = INDd(isoratio,i)       *  /* Density fraction */
          SIGCTE * INDd(gf,ln)               *  /* Constant * gf */
          exp(-EXPCTE*INDd(elow,ln) / temp)  *  /* Level population */
          (1-exp(-EXPCTE*wavn/temp))         /  /* Induced emission */
          INDd(isoz,i);                         /* Partition function */
      /* Check if this is the maximum k: */
      kmax[iext] = fmax(kmax[iext], k);
  }

  /* Compute the extinction-coefficient for each species: */
  for (ln=0; ln<nlines; ln++){
      wavn = INDd(lwn, ln);
      i = INDi(lID, ln);
      iext = INDi(isoiext, i);
      if (add)
          iext = 0;

      if ((wavn < INDd(own,0)) || (wavn > INDd(own,(onwn-1))))
          continue;

      /* Line strength: */
      k = kprop[ln];

      /* Index of closest oversampled wavenumber: */
      iown = (wavn - INDd(own,0))/ownstep;
      if (fabs(wavn - INDd(own,(iown+1))) < fabs(wavn - INDd(own,iown)))
          iown++;

      /* Check if the next line falls on the same sampling index: */
      while (ln+1 != nlines && INDi(lID, (ln+1)) == i
             && INDd(lwn,(ln+1)) <= INDd(own,(onwn-1))){
          next_wn = INDd(lwn, (ln+1));
          if (fabs(next_wn - INDd(own,iown)) < ownstep){
              nadd++;
              ln++;
              /* Add the contribution from this line into the opacity: */
              k += kprop[ln];
          }
          else
              break;
      }

      /* Skip weakly contributing lines: */
      if (k < ethresh * kmax[iext]){
          nskip++;
          continue;
      }

      /* Co-add opacity into extinction coefficient: */
      if (add)
          k *= INDd(moldensity, (INDi(isoimol,i)));

      /* Index of closest (but not larger than) dynamic-sampling wavenumber: */
      idwn = (wavn - INDd(own,0))/dwnstep;

      /* Calculate index for Doppler width: */
      idop[i] = pyramidsearch(doppler, alphad[i]*wavn, idop[i], (int)nDop-1);

      /* Sub-sampling offset between center of line and dyn-sampled wn: */
      subw = iown - idwn*ofactor;
      /* Offset between the profile and the wavenumber-array indices: */
      offset = ofactor*idwn - IND2i(psize, ilor[i], idop[i]) + subw;
      /* Range that contributes to the opacity: */
      /* Set the lower and upper indices of the profile to be used: */
      minj = idwn - (IND2i(psize, ilor[i], idop[i]) - subw) / ofactor;
      maxj = idwn + (IND2i(psize, ilor[i], idop[i]) + subw) / ofactor;
      if (minj < 0)
          minj = 0;
      if (maxj > dnwn)
          maxj = dnwn;

      /* Fixed Voigt cutoff: */
      if (cutoff>0.0){
          mincut = (int)(idwn - cutoff/dwnstep);
          if (mincut > minj) minj = mincut;
          maxcut = (int)(idwn + cutoff/dwnstep);
          if (maxcut < maxj) maxj = maxcut;
      }

      /* Add the contribution from this line to the opacity spectrum: */
      iprof = IND2i(pindex, ilor[i], idop[i]);
      jj = iprof + ofactor*minj - offset;
      for (j=(int)minj; j<maxj; j++){
          ktmp[iext][j] += k * INDd(profile, jj);
          jj += ofactor;
      }
      neval++;
  }

  if (verb > 5){
      printf("    Number of co-added lines:     %8i  (%5.2f%%)\n",
             nadd,  nadd*100.0/nlines);
      printf("    Number of skipped profiles:   %8i  (%5.2f%%)\n",
             nskip, nskip*100.0/nlines);
      printf("    Number of evaluated profiles: %8i  (%5.2f%%)\n",
             neval, neval*100.0/nlines);
  }

  /* Interpolate ktmp to constant-R output spectrum: */
  if (resolution == 1){
      for (iext=0; iext<nextinct; iext++){
          linterp(ktmp, ext, INDd(wn,0), dwnstep, wn, iext);
      }
  }
  /* Resample ktmp to the final sampling size (constant delta-wavenumber): */
  else{
      for (iext=0; iext<nextinct; iext++){
          resample(ktmp, ext, (int)dnwn, (int)round(wnstep/ownstep/ofactor), iext);
      }
  }

  /* Free the no-longer used memory: */
  free(alphal);
  free(alphad);
  free(kmax);
  free(idop);
  free(ilor);
  free(ktmp[0]);
  free(ktmp);
  free(kprop);

  return Py_BuildValue("i", 1);
}


PyDoc_STRVAR(interp_ec__doc__,
"Interpolate extinction coefficient from etable at temperature\n\
values.                                                    \n\
                                                           \n\
Parameters                                                 \n\
----------                                                 \n\
extinction: 1D float ndarray                               \n\
    Extinction coefficient array [nwave] to calculate.     \n\
etable: 3D float ndarray                                   \n\
    Tabulated extinction coefficient [nmol, ntemp, nwave]. \n\
ttable: 1D float ndarray                                   \n\
    Tabulated temperature array [ntemp].                   \n\
temperature: Float                                         \n\
    Atmospheric layer temperature.                         \n\
density: 1D float ndarray                                  \n\
    Density of etable species in the atmospheric layer [nmol].");

static PyObject *interp_ec(PyObject *self, PyObject *args){
    PyArrayObject *extinction, *etable, *ttable, *density;
    int tlo, thi;
    long nwave, nmol, ntemp;
    int i, j;
    double ext, temperature;

    /* Load inputs: */
    if (!PyArg_ParseTuple(args, "OOOdO",
            &extinction, &etable, &ttable, &temperature, &density))
        return NULL;

    nmol  = PyArray_DIM(etable, 0);  /* species samples       */
    ntemp = PyArray_DIM(etable, 1);  /* temperature samples   */
    nwave = PyArray_DIM(etable, 2);  /* spectral samples      */

    /* Find index of grid-temperature immediately lower than temperature: */
    tlo = binsearchapprox(ttable, temperature, 0, (int)ntemp-1);
    if (temperature < INDd(ttable,tlo) || tlo == ntemp-1){
        tlo--;
    }
    thi = tlo + 1;

    /* Add contribution from each molecule: */
    for (j=0; j<nmol; j++){
        for (i=0; i<nwave; i++){
            /* Linear interpolation of the extinction coefficient: */
            ext = (IND3d(etable,j,tlo,i) * (INDd(ttable,thi) - temperature) +
                   IND3d(etable,j,thi,i) * (temperature - INDd(ttable,tlo)) ) /
                  (INDd(ttable,thi) - INDd(ttable,tlo));

            INDd(extinction, i) += INDd(density,j) * ext;
        }
    }

    return Py_BuildValue("i", 1);
}


/* The module doc string: */
PyDoc_STRVAR(extcoeff__doc__,
    "Python wrapper for the extinction-coefficient calculation.");

/* A list of all the methods defined by this module: */
static PyMethodDef extcoeff_methods[] = {
    {"extinction", extinction, METH_VARARGS, extinction__doc__},
    {"interp_ec",  interp_ec,  METH_VARARGS, interp_ec__doc__},
    {NULL,         NULL,       0,            NULL} /* sentinel */
};


/* Module definition for Python 3: */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_extcoeff",
    extcoeff__doc__,
    -1,
    extcoeff_methods
};

/* When Python 3 imports a C module named 'X' it loads the module */
/* then looks for a method named "PyInit_"+X and calls it.        */
PyObject *PyInit__extcoeff (void) {
    PyObject *module = PyModule_Create(&moduledef);
    import_array();
    return module;
}
