# ****************************** START LICENSE ******************************
# ******************************* END LICENSE ******************************

import sys
import os
import numpy as np

# Directory of db_hitran.py:
DBHdir = os.path.dirname(os.path.realpath(__file__))
# Add path to ctips source code:
sys.path.append(DBHdir + "/../ctips/lib")
import ctips as ct


class dbdriver(object):
  def __init__(self, dbfile, pffile):
    self.dbfile = dbfile
    self.pffile = pffile


  def getpf(self, verbose=0):
    """
    Compute partition function for specified source.
    """
    # Calculate the partition-function from the CTIPS module:
    if self.pffile == "ctips":
      # Number of isotopes:
      niso = len(self.isotopes)

      # Array of temperatures (TIPS range of temperature is 70K to 3000K):
      temp = np.arange(70.0, 3000.1, 10.0)
      ntemp = len(temp)
      # Output array for table of Temperature and PF values:
      PF = np.zeros((niso, ntemp), np.double)

      molID = np.repeat(int(self.molID), ntemp)
      # Calculate the partition function for each isotope:
      for i in np.arange(niso):
        isoID = np.repeat(int(self.isotopes[i]), ntemp)
        PF[i] = ct.tips(molID, isoID, temp)/self.gi[i]
      return temp, PF

    # Use polynomial expression:
    elif self.pffile == "poly":
      # Temperature array:
      Temp = np.arange(1000.0, 7001.0, 50.0)
      Ntemp = len(Temp)

      # Number of isotopes:
      Niso = len(self.isotopes)

      # Intialize PF array:
      PF = np.zeros((Niso, Ntemp), np.double)

      # Calculate log(PF) at each temperature and isotope:
      for j in np.arange(Niso):
        for i in np.arange(Ntemp):
          # Formula from Irwin 1981, ApJS 45, 621 (equation #2):
          PF[j,i] = (self.PFcoeffs[j,0]                      +
                     self.PFcoeffs[j,1]* np.log(Temp[i])     +
                     self.PFcoeffs[j,2]*(np.log(Temp[i]))**2 +
                     self.PFcoeffs[j,3]*(np.log(Temp[i]))**3 +
                     self.PFcoeffs[j,4]*(np.log(Temp[i]))**4 +
                     self.PFcoeffs[j,5]*(np.log(Temp[i]))**5 )
      # Get the exponential of log(PF):
      PF = np.exp(PF)

      return Temp, PF

    # Extract the partition-function from the tabulated file:
    else:
      return self.readpf()


  def dbread(self, iwl, fwl, verbose):
    """
      Read linelist values for specific database type.

      Parameters:
      -----------
      dbtype: String
    """
    pass

  def readwl(self, dbfile, irec):
    """
      Read the wavelength parameter as given in the database
    """
    pass

  def binsearch(self, dbfile, wavelength, ilo, ihi, searchup=True):
    """
    Do a binary search of record position in File dbfile that has wavelength iwl

    Parameters:
    -----------
    dbfile: File object
       File where to search.
    wavelength: Scalar
       Target wavelength (as given in each specific database).
    ilo: Integer
       Index of smallest record to search.
    ihi: Integer
       Index of largerst record to search.
    dbtype: String
       Database type: ['ps', 'hit']
    searchup: Boolean
       Search up (True) or down (False) the records for duplicate results
       after the binary search.

    Returns:
    --------
    Index of record with

    Modification History:
    ---------------------
    2013        madison   Initial implementation.
    2014-03-05  patricio  Added documentation, updated Madison's code, and
                          included searchup parameter
                                                    pcubillos@fulbrightmail.org
    """
    # Wavelength of record:
    rec_wl = 0

    # Start binary search:
    while ihi - ilo > 1:
      # Middle record index:
      irec = (ihi + ilo)/2

      # Read wavelength, depending on linelist format:
      rec_wl = self.readwl(dbfile, irec)
      # Update search limits:
      if rec_wl > wavelength:
        ihi = irec
      else:
        ilo = irec

    # Search up/or down if there are multiple records with the same wavelength:
    while (self.readwl(dbfile, irec) == rec_wl):
      if searchup:
        irec += 1
      else:
        irec -= 1

    # Move file pointer to begining:
    dbfile.seek(0)

    #print("Found wavelength %d at position %d."%(rec_wl, ihi))
    return irec

  def readpf(self):
    """
    Extract the partition-function and temperature from file.

    Returns:
    --------
    temp: 1D float ndarray
       Array with temperature sample.
    PF:   2D float ndarray
       The partition function data for each isotope at each temperature.
    """
    # Open-read file:
    f = open(self.pffile, "r")
    lines = f.readlines()
    f.close()

    # Number of header lines (to skip when reading the tabulated data):
    nskip = 0
    while True:
      line = lines[nskip].strip()
      # Skip blank/empty lines:
      if line == "" or line.startswith('#'):
        pass
      # Read isotopes:
      elif line == "@ISOTOPES":
        isotopes = np.asarray(lines[nskip+1].strip().split())
      # Stop when the tabulated data begins:
      if line == "@DATA":
        nskip += 1
        break
      nskip += 1

    # Number of isotopes:
    niso = len(isotopes)
    # Number of temperature samples:
    ntemp = len(lines) - nskip

    # Allocate output arrays:
    temp = np.zeros(ntemp, np.double)
    PF   = np.zeros((niso, ntemp), np.double)

    # Read the data:
    for i in np.arange(ntemp):
      info = lines[nskip+i].strip().split()
      temp[i] = info[0]
      PF[:,i] = info[1:]

    return temp, PF
