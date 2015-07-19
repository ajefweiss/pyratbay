# ****************************** START LICENSE ******************************
# ******************************* END LICENSE ******************************

import sys
import os
import numpy as np

import ptools     as pt
import pconstants as pc
from driver import dbdriver

import struct, time

class voplez(dbdriver):
  """
  Download the linelist from:

  """
  def __init__(self, dbfile, pffile):
    """
    Initializer.
    """
    super(voplez, self).__init__(dbfile, pffile)

    # Database name:
    self.name = "Bertrand Plez VO"
    # Isotopic names:
    self.isotopes = ["16"]  # I'm Using AFGL naming convention
    # Isotopic masses (amu):
    self.mass     = [66.941]
    # Isotopic abundance ratio:
    self.isoratio = [1.0]

    # Molecule name:
    self.molecule = "VO"

    # Partition-function polynomial coefficients:
    # (from communication with B. Pelz):
    self.PFcoeffs = np.array([[ 6.62090157e+02, -4.03350494e+02,
                                9.82836218e+01, -1.18526504e+01,
                                7.08429905e-01, -1.67235124e-02]])

    # Other utilities:
    self.recsize  = 53  # Record length
    self.recwnpos = 33  # Wavenumber position in record
    self.recelpos = 44  # Elow       position in record
    self.recgfpos = 21  # gf         position in record
    self.recwnlen = 10  # Record lengths
    self.recwnend = 43  # Record lengths
    self.recelend = 50
    self.recgfend = 32


  def readwl(self, dbfile, irec):
    """
    Extract the wavelength from record irec.

    Parameters:
    -----------
    dbfile: File pointer
       Pointer to file being read.
    irec: Integer
       Index of record to read.

    Returns:
    --------
    wl: Float
       The wavelength for record irec.
    """
    # Set pointer at required record:
    dbfile.seek(irec*self.recsize + self.recwnpos)
    # Read record (wavenumber in cm-1):
    wave = float(dbfile.read(self.recwnlen))
    # Convert to wavelength (micron) and return:
    return 1.0/(wave*pc.MTC)


  def dbread(self, iwn, fwn, verbose, *args):
    """
    Read the B. Plez VO database between the wavelengths iwl and fwl.
 
    Parameters:
    -----------
    iwn: Scalar
       Initial wavenumber limit (in cm-1).
    fwn: Scalar
       Final wavenumber limit (in cm-1).
    verbose: Integer
       Verbosity threshold.
    args:
       Additional arguments, not needed for voplez.
 
    Returns:
    --------
    wnumber: 1D float ndarray
      Line-transition central wavelength (cm-1).
    gf: 1D float ndarray
      gf value (unitless).
    elow: 1D float ndarray
      Lower-state energe (cm-1).
    isoID: 2D integer ndarray
      Isotope index (1, 2, 3, ...).

    Notes:
    ------
    The VO data is sorted monotonically increasing in wavelength (cm-1).
    """

    # Open the file:
    data = open(self.dbfile, "r")
    # Get the total number of transitions:
    data.seek(0, 2)
    nlines   = data.tell() / self.recsize

    # Get database limiting wavelengths:
    minwl = self.readwl(data,        0)  # Lowest  wavelength in DB
    maxwl = self.readwl(data, nlines-1)  # Highest wavelength in DB

    # Conver input wavenumber to database units (wavelength):
    iwl = 1.0 / (fwn * pc.MTC)
    fwl = 1.0 / (iwn * pc.MTC)

    # Find the record index for iwl and fwl:
    if iwl > minwl:
      istart = self.binsearch(data, iwl, 0,     nlines, 0)
    else:
      istart = 0

    if fwl < maxwl:
      istop = self.binsearch(data, fwl, istart, nlines, 1)
    else:
      istop = nlines-1

    # Number of records to read:
    nread = istop - istart + 1

    # Store data in two arrays for doubles and integers:
    wnumber = np.zeros(nread, np.double)
    gf      = np.zeros(nread, np.double)
    elow    = np.zeros(nread, np.double)
    isoID   = np.zeros(nread, int)
 
    pt.msg(verbose, "Beginning to read Plez VO database, between "
                    "records {:d} and {:d}.".format(istart, istop))

    interval = (istop - istart)/10  # Check-point interval

    i = 0  # Record index counter
    while (i < nread):
      # Read a record:
      data.seek((istart+i) * self.recsize)
      line = data.read(self.recsize)
      # Store values:
      wnumber[i] = float(line[self.recwnpos:self.recwnend])
      gf     [i] = float(line[self.recgfpos:self.recgfend])
      elow   [i] = float(line[self.recelpos:self.recelend])

      # Print a checkpoint statement every 10% interval:
      if verbose > 1:
        if (i % interval) == 0.0  and  i != 0:
          pt.msg(verbose-1,"Checkpoint {:5.1f}%".format(10.*i/interval), 2)
          pt.msg(verbose-2,"Wavenumber: {:8.2f} cm-1   Wavelength: {:6.3f} um\n"
                          "Elow:     {:.4e} cm-1   gf: {:.4e}   Iso ID: {:2d}".
                             format(wnumber[i], 1.0/(wnumber[i]*pc.MTC),
                                    elow[i]*pc.eV2Kayser, gf[i], isoID[i]), 4)
      i += 1

    # Convert Elow from eV to cm-1:
    elow[:] = elow * pc.eV2Kayser

    pt.msg(verbose, "Done.\n")
    data.close()

    return wnumber, gf, elow, isoID
