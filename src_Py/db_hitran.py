# ****************************** START LICENSE ******************************
# ******************************* END LICENSE ******************************

import os
import numpy as np

import ptools     as pt
import pconstants as pc
from db_driver import dbdriver

# Directory of db_hitran.py:
DBHdir = os.path.dirname(os.path.realpath(__file__))

class hitran(dbdriver):
  def __init__(self, dbfile, pffile):
    """
    Initialize the basic database info.

    Parameters:
    -----------
    dbfile: String
       File with the Database info as given from HITRAN.
    pffile: String
       File with the partition function.
    """
    super(hitran, self).__init__(dbfile, pffile)

    self.recsize   =   0 # Record length (will be set in self.dbread())
    self.recisopos =   2 # Isotope        position in record
    self.recwnpos  =   3 # Wavenumber     position in record
    self.reclinpos =  15 # Line intensity position in record
    self.recApos   =  25 # Einstein coef  position in record
    self.recairpos =  35 # Air broadening position in record
    self.recelpos  =  45 # Low Energy     position in record
    self.recg2pos  = 155 # Low stat weight position in record
    self.recmollen =   2 # Molecule   record length
    self.recwnlen  =  12 # Wavenumber record length
    self.reclinend =  25 # Line intensity end position
    self.recelend  =  55 # Low Energy     end position

    self.molID = self.getMolec()
    # Get info from HITRAN configuration file:
    self.molecule, self.isotopes, self.mass, self.isoratio, self.gi = \
                                                    self.getHITinfo()
    # Database name:
    self.name = "HITRAN " + self.molecule


  def readwl(self, dbfile, irec):
    """
    Read wavelength parameter from irec record in dbfile database.

    Parameters:
    -----------
    dbfile: File object
       File where to extract the wavelength.
    irec: Integer
       Index of record.

    Returns:
    --------
    rec_wl: Float
       Wavelength value at record irec, as given in dbfile database.
    """
    # Set pointer at required wavenumber record:
    dbfile.seek(irec*self.recsize + self.recwnpos)
    # Read:
    wavenumber = dbfile.read(self.recwnlen)
    # Convert to float:
    rec_wl = float(wavenumber)

    return rec_wl


  def getMolec(self):
    """
    Get the HITRAN molecule index
    """
    # Open file and read first two characters:
    data = open(self.dbfile, "r")
    molID  = data.read(self.recmollen)
    data.close()
    # Set database name:
    return molID #self.molname[molID-1]


  def getHITinfo(self):
    """
    Get HITRAN info from configuration file.

    Returns:
    --------
    molname:  Molecule's name
    isotopes: Isotopes names
    mass:     Isotopes mass
    isoratio: Isotopic abundance ratio
    gi:       State-independent statistical weight
    """
    # Read HITRAN configuration file from inputs folder:
    hfile = open(DBHdir + '/../inputs/hitran.dat', 'r')
    lines = hfile.readlines()
    hfile.close()

    isotopes = []
    mass     = []
    isoratio = []
    gi       = []

    # Get values for our molecule:
    for i in np.arange(len(lines)):
      if lines[i][0:2] == self.molID:
        line = lines[i].split()
        molname  = line[1]
        gi.      append(  int(line[3]))
        isotopes.append(      line[2] )
        isoratio.append(float(line[4]))
        mass.    append(float(line[5]))

    return molname, isotopes, mass, isoratio, gi


  def dbread(self, iwn, fwn, verbose, *args):
    """
    Read a HITRAN or HITEMP database (dbfile) between wavenumbers iwn and fwn.

    Parameters:
    -----------
    dbfile: String
       A HITRAN or HITEMP database filename.
    iwn: Float
       Initial wavenumber limit (in cm-1).
    fwn: Float
       Final wavenumber limit (in cm-1).
    verbose: Integer
       Verbosity threshold.
    pffile: String
       Partition function filename.

    Returns:
    --------
    wnumber: 1D ndarray (double)
      Line-transition central wavenumber (cm-1).
    gf: 1D ndarray (double)
      gf value (unitless).
    elow: 1D ndarray (double)
      Lower-state energy (cm-1).
    isoID: 2D ndarray (integer)
      Isotope index (1, 2, 3, ...).

    Notes:
    ------
    - The HITRAN data is provided in ASCII format.
    - The line transitions are sorted in increasing wavenumber (cm-1) order.
    """

    # Open HITRAN file for reading:
    data = open(self.dbfile, "r")

    # Read first line to get the record size:
    data.seek(0)
    line = data.readline()
    self.recsize = len(line)

    # Get Total number of transitions in file:
    data.seek(0, 2)
    nlines   = data.tell() / self.recsize

    # Find the record index for iwn and fwn:
    istart = self.binsearch(data, iwn, 0,      nlines-1, 0)
    istop  = self.binsearch(data, fwn, istart, nlines-1, 1)

    # Number of records to read:
    nread = istop - istart + 1

    # Allocate arrays for values to extract:
    wnumber = np.zeros(nread, np.double)
    gf      = np.zeros(nread, np.double)
    elow    = np.zeros(nread, np.double)
    isoID   = np.zeros(nread,       int)
    A21     = np.zeros(nread, np.double)  # Einstein A coefficient
    g2      = np.zeros(nread, np.double)  # Lower statistical weight

    pt.msg(verbose, "Starting to read HITRAN database between "
                    "records {:d} and {:d}.".format(istart, istop))
    interval = (istop - istart)/10  # Check-point interval

    i = 0  # Stored record index
    while (i < nread):
      # Read a record:
      data.seek((istart+i) * self.recsize)
      line = data.read(self.recsize)
      # Extract values:
      isoID  [i] = float(line[self.recisopos:self.recwnpos ])
      wnumber[i] = float(line[self.recwnpos: self.reclinpos])
      elow   [i] = float(line[self.recelpos: self.recelend ])
      A21    [i] = float(line[self.recApos:  self.recairpos])
      g2     [i] = float(line[self.recg2pos: self.recsize  ])
      # Print a checkpoint statement every 10% interval:
      if verbose > 1:
        if (i % interval) == 0.0  and  i != 0:
          gfval = A21[i]*g2[i]*pc.C1/(8.0*np.pi*pc.c)/wnumber[i]**2.0
          pt.msg(verbose-1,"Checkpoint {:5.1f}%".format(10.*i/interval), 2)
          pt.msg(verbose-2,"Wavenumber: {:8.2f} cm-1   Wavelength: {:6.3f} um\n"
                          "Elow:     {:.4e} cm-1   gf: {:.4e}   Iso ID: {:2d}".
                             format(wnumber[i], 1.0/(wnumber[i]*pc.MTC),
                                    elow[i], gfval, (isoID[i]-1)%10), 4)
      i += 1


    # Set isotopic index to start counting from 0:
    isoID -= 1
    isoID[np.where(isoID < 0)] = 9 # 10th isotope had index 0 --> 10-1=9

    # Calculate gf (Equation (36) of Simekova 2006):
    gf = A21 * g2 * pc.C1 / (8.0 * np.pi * pc.c) / wnumber**2.0

    data.close()
    pt.msg(verbose, "Done.\n")

    # Remove lines with unknown Elow (see Rothman et al. 1996):
    igood = np.where(elow > 0)
    return wnumber[igood], gf[igood], elow[igood], isoID[igood]
