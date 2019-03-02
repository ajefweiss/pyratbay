# Copyright (c) 2016-2019 Patricio Cubillos and contributors.
# Pyrat Bay is currently proprietary software (see LICENSE).

__all__ = ["parray", "defaultp", "getparam",
           "binsearch", "pprint", "divisors", "u", "unpack",
           "ifirst", "ilast",
           "isfile", "addarg", "path", "wrap",
           "make_tea", "clock", "get_exomol_mol",
           "pf_exomol"]

import os
import sys
import re
import struct
import time
import numbers
import textwrap
import itertools
if sys.version_info.major == 3:
    import configparser
else:
    import ConfigParser as configparser

import numpy as np

from .. import constants as pc
from .. import io        as io

sys.path.append(pc.ROOT + "pyratbay/lib/")
import _indices
sys.path.append(pc.ROOT + "modules/MCcubed/")
import MCcubed.utils as mu


def parray(string):
  """
  Convert a string containin a list of white-space-separated (and/or
  newline-separated) values into a numpy array
  """
  if string == 'None':
    return None
  try:    # If they can be converted into doubles, do it:
    return np.asarray(string.split(), np.double)
  except: # Else, return a string array:
    return string.split()


def binsearch(dbfile, wavelength, rec0, nrec, upper=True):
  """
  Do a binary search in TLI dbfile for record that has wavelength iwl

  Parameters
  ----------
  dbfile: File object
     TLI file where to search.
  wavelength: Scalar
     Target wavelength.
  rec0: Integer
     Position of first record.
  nrec: Integer
     Number of records.
  upper: Boolean
     Wavelength is an upper boundary, return the index of value smaller
     than wavelength.

  Returns
  -------
  Index of record with ...
  """
  # Wavelength of record:
  ilo = 0
  ihi = nrec

  # Start binary search:
  while ihi - ilo > 1:
    # Middle record index:
    irec = (ihi + ilo)//2

    # Read wavelength from TLI file record:
    dbfile.seek(irec*pc.dreclen + rec0, 0)
    rec_wl = struct.unpack('d', dbfile.read(8))[0]
    # Update search limits:
    if rec_wl > wavelength:
      ihi = irec
    else:
      ilo = irec
  #print("Binary found:      %.7f (%d)"%(rec_wl, irec))

  # Sequential search:
  if rec_wl < wavelength:
    jump =  1 # Move up
  else:
    jump = -1 # Move down

  dbfile.seek(irec*pc.dreclen + rec0, 0)
  next_wl = struct.unpack('d', dbfile.read(8))[0]
  while (np.sign(next_wl-wavelength) == np.sign(rec_wl-wavelength)):
    irec += jump
    # Check for boundaries:
    if (irec < 0 or irec > nrec):
      return np.clip(irec, 0, nrec)
    dbfile.seek(irec*pc.dreclen + rec0, 0)
    rec_wl  = next_wl
    next_wl = struct.unpack('d', dbfile.read(8))[0]
  #print("Sequential found:  %.7f"%next_wl)

  # Return the index withing the boundaries:
  return irec - (jump+upper)//2


def pprint(array, precision=3, fmt=None):
  """
  Pretty print a Numpy array.  Set desired precision and format, and
  remove line break from the string output.

  Parameters
  ----------
  array: 1D ndarray
     Array to be pretty printed.
  precision: Integer
     Precision for floating point values.
  fmt: format
     Numeric format.
  """
  default_prec = np.get_printoptions().get('precision')
  np.set_printoptions(precision=precision)

  # Pretty array is a copy of array:
  parray = np.copy(array)
  if fmt is not None:
    parray = np.asarray(array, fmt)

  # Convert to string and remove line-breaks:
  sarray = str(array).replace("\n", "")
  np.set_printoptions(precision=default_prec)
  return sarray


def divisors(number):
  """
  Find all the integer divisors of number.
  """
  divs = []
  for i in np.arange(1, number/2+1):
    if number % i == 0:
      divs.append(i)
  divs.append(number)
  return np.asarray(divs, np.int)


def unpack(file, n, dtype):
  """
  Wrapper for struct unpack.

  Parameters
  ----------
  file: File object
     File object to read from.
  n: Integer
     Number of elements to read from file.
  dtype: String
     Data type of the bytes read.

  Returns:
  --------
  output: Scalar, tuple, or string
     If dtype is 's' return the string.
     If there is a single element to read, return the scalar value.
     Else, return a tuple with the elements read.
  """
  # Compute the reading format:
  fmt  = "{:d}{:s}".format(n, dtype)
  # Calculate the number of bytes to read:
  size = struct.calcsize(fmt)
  # Read:
  output = struct.unpack(fmt, file.read(size))
  # Return:
  if (n == 1) or (dtype == "s"):
    return output[0]
  else:
    return output


def u(units, log=None):
  """
  Get the conversion factor (to the CGS system) for units.

  Parameters
  ----------
  units: String
     Name of units
  """
  # Accept only valid units:
  if units not in pc.validunits:
    # Throw error:
    print("Units name '{:s}' does not exist.".format(units))
    sys.exit(0)
  factor = getattr(pc, units)
  return factor


def defaultp(param, default, msg, log):
  """
  Return param if not None, else, return default and print the
  corresponding warning message.

  Parameters
  ----------
  param:  Any
     Input parameter value.
  default: Any
     Default parameter value.
  msg: String
     Printed message if param is None.
  log: Log object
     Screen-output log handler.
  """
  if param is None:
      log.warning(msg.format(default))
      return default
  return param


def getparam(param, units, log=None, integer=False):
  """
  Read a parameter that may or may not have units included.
  If it doesn't, default to the 'units' input argument.

  Parameters
  ----------
  param: String
     The parameter name.
  units: String
     The default units for the parameter.
  log: Log object
     Screen-output log handler.
  """
  if param is None:
    return None

  if log is None:
    log = mu.Log(logname=None)

  # Return if it is a numeric value:
  if isinstance(param, numbers.Number):
    if units not in pc.validunits:
      log.error("Units name '{:s}' does not exist.".format(units))

    return param * u(units)

  # Split the parameter if it has a white-space:
  par = param.split()

  # If the parameter contains units, read the units:
  if len(par) == 2:
    units = par[1]
    if units not in pc.validunits:
      log.error("Units name '{:s}' does not exist.".format(units))

  # Get the value of the parameter:
  try:
    value = np.float(par[0])
  except:
    log.error("Invalid parameter format for '{:s}'.  param must be a float "
              "or integer.  If it contains units, it must be blank-space "
              "separated.".format(param), lev=-3)

  # Apply the units:
  value *= u(units)

  if integer:
    return int(value)

  return value


def ifirst(data, default_ret=-1):
  """
  Get the first index where data is True or 1.

  Parameters
  ----------
  data: 1D bool/integer iterable
      An array of bools or integers.
  default_ret: Integer
      Default returned value when no value in data is True or 1.

  Returns
  -------
  first: integer
     First index where data == True or 1.  Return default_ret otherwise.

  Examples
  --------
  >>> import pyratbay.tools as pt
  >>> import numpy as np
  >>> print(pt.ifirst([1,0,0]))
  0
  >>> print(pt.ifirst(np.arange(5)>2.5))
  3
  >>> print(pt.ifirst([False, True, True]))
  1
  >>> print(pt.ifirst([False, False, False]))
  -1
  >>> print(pt.ifirst([False, False, False], default_ret=0))
  0
  """
  return _indices.ifirst(np.asarray(data, np.int), default_ret)


def ilast(data, default_ret=-1):
  """
  Get the last index where data is 1 or True.

  Parameters
  ----------
  data: 1D bool/integer iterable
      An array of bools or integers.
  default_ret: Integer
      Default returned value when no value in data is True or 1.

  Returns
  -------
  last: integer
     Last index where data == 1 or True.  Return default_ret otherwise.

  Examples
  --------
  >>> import pyratbay.tools as pt
  >>> import numpy as np
  >>> print(pt.ilast([1,0,0]))
  0
  >>> print(pt.ilast(np.arange(5)<2.5))
  2
  >>> print(pt.ilast([False, True, True]))
  2
  >>> print(pt.ilast([False, False, False]))
  -1
  >>> print(pt.ilast([False, False, False], default_ret=0))
  0
  """
  return _indices.ilast(np.asarray(data, np.int), default_ret)


def isfile(path):
  """
  Check whether a path is a regular file.

  Parameters
  ----------
  path:  String
    Path to check.

  Returns
  -------
  status: Integer
    If path is None, return -1.
    If path is not a regular file, return 0.
    If path is a regular file, return 1.
  """
  # None exception:
  if path is None:
    return -1

  # Regular file or not:
  return os.path.isfile(path)


def addarg(variable, pgroup, type, default, help):
  """
  Wrapper of the parser argument group.

  Parameters
  ----------
  pgroup: _ArgumentGroup
     A parser's argument group.
  variable: String
     The variable name.
  type: Callable
     Type of the variable.
  default: Any
     Default value of the variable.
  help: String
     Brief description of the argument.
  """
  pgroup.add_argument("--{:s}".format(variable), dest=variable,
                      action="store", type=type, default=default, help=help)

def path(filename):
    """
    Ensure file names have non-null path

    Parameters
    ----------
    filename: String
        A file name.

    Examples
    --------
    >>> import pyratbay.tools as pt
    >>> print(pt.path('file.txt'))
    ./file.txt
    >>> print(pt.path('./file.txt'))
    ./file.txt
    >>> print(pt.path('/home/user/file.txt'))
    /home/user/file.txt
    """
    if filename is None:
        return None
    path, filename = os.path.split(filename)
    if path == '':
        path = '.'
    return '{:s}/{:s}'.format(path, filename)


def wrap(outlist, text, indent=0, si=None):
    """
    Wrap input text, store it into outlist list.

    Parameters
    ----------
    outlist: List
        List where to append the wrapped text.
    text: String
        Text to wrap.
    indent: Integer
        Number of spaces to indent the first line.
    si: Integer
        Number of spaces to indent subsequent lines.  If None, use
        use same indentation as indent.

    Examples
    --------
    >>> import pyratbay.tools as pt
    >>> info = []
    >>> pt.wrap(info, "Pyrat atmospheric model\n")
    >>> pt.wrap(info, "Pressure = 1.0 bar\nTemperature = 1000.0 K", indent=2)
    >>> print("\n".join(info))
    Pyrat atmospheric model
      Pressure = 1.0 bar
      Temperature = 1000.0 K
    """
    indspace = " "*indent
    if si is None:
        sindspace = indspace
    else:
        sindspace = " "*si
    lines = text.splitlines()
    for line in lines:
        outlist.append(textwrap.fill(line, break_long_words=False,
                                     initial_indent=indspace,
                                     subsequent_indent=sindspace, width=80))


def make_tea(cfile=None, maxiter=100, save_headers=False, save_outputs=False,
             doprint=False, times=False, location_TEA=None, abun_file=None,
             location_out="./TEA"):
  """
  Make a TEA configuration file.

  Parameters
  ----------
  cfile: String
      Input configuration file to get arguments for TEA config file.
  """
  if location_TEA is None:
      location_TEA = os.path.realpath(pc.ROOT + "modules/TEA/")

  # Open New Config parser:
  config = configparser.SafeConfigParser()
  config.add_section('TEA')
  config.set("TEA", "maxiter",      str(maxiter))
  config.set("TEA", "save_headers", str(save_headers))
  config.set("TEA", "save_outputs", str(save_outputs))
  config.set("TEA", "doprint",      str(doprint))
  config.set("TEA", "times",        str(times))
  config.set("TEA", "location_TEA", str(location_TEA))
  config.set("TEA", "location_out", str(location_out))
  config.set("TEA", "abun_file",    str(abun_file))

  # Override with input Config parser values:
  if cfile is not None:
      input_config = configparser.ConfigParser()
      input_config.read([cfile])

      keys = ["maxiter", "save_headers", "save_outputs", "doprint",
              "times", "location_TEA", "abun_file", "location_out"]
      # Set TEA default arguments:
      for i in np.arange(len(keys)):
          if input_config.has_option("PBAY", keys[i]):
              config.set("TEA", keys[i], input_config.get("PBAY", keys[i]))

  # For completion:
  config.add_section('PRE-ATM')
  config.set("PRE-ATM", "PT_file",        "None")
  config.set("PRE-ATM", "pre_atm_name",   "None")
  config.set("PRE-ATM", "input_elem",     "None")
  config.set("PRE-ATM", "output_species", "None")

  # Write TEA configuration file:
  with open("TEA.cfg", 'w') as configfile:
      config.write(configfile)


def clock(t0=time.time()):
  """
  Timer generator yields the time (in seconds) since the last call.
  """
  while True:
      tnew = time.time()
      delta = tnew - t0
      t0 = tnew
      yield delta


def get_exomol_mol(dbfile):
  """
  Parse an exomol file to extract the molecule and isotope name.

  Parameters
  ----------
  dbfile: String
      An exomol line-list file (must follow ExoMol naming convention).

  Returns
  -------
  molecule: String
      Name of the molecule.
  isotope: String
      Name of the isotope (See Tennyson et al. 2016, jmosp, 327).

  Examples
  --------
  >>> import pyratbay.tools as pt
  >>> filenames = [
  >>>     '1H2-16O__POKAZATEL__00400-00500.trans.bz2',
  >>>     '1H-2H-16O__VTT__00250-00500.trans.bz2',
  >>>     '12C-16O2__HITEMP.pf',
  >>>     '12C-16O-18O__Zak.par',
  >>>     '12C-1H4__YT10to10__01100-01200.trans.bz2',
  >>>     '12C-1H3-2H__MockName__01100-01200.trans.bz2'
  >>>    ]
  >>> for db in filenames:
  >>>     print(pt.get_exomol_mol(db))
  ('H2O', '116')
  ('H2O', '126')
  ('CO2', '266')
  ('CO2', '268')
  ('CH4', '21111')
  ('CH4', '21112')
  """
  atoms = os.path.split(dbfile)[1].split("_")[0].split("-")
  elements = []
  isotope  = ""
  for atom in atoms:
      match = re.match(r"([0-9]+)([a-z]+)([0-9]*)", atom, re.I)
      N = 1 if match.group(3) == "" else int(match.group(3))
      elements += N * [match.group(2)]
      isotope  += match.group(1)[-1:] * N

  composition = [list(g[1]) for g in itertools.groupby(elements)]
  molecule = "".join([c[0] + str(len(c))*(len(c)>1)
                      for c in composition])

  return molecule, isotope


def pf_exomol(pf_files):
  """
  Re-format Exomol partition-function files for use with Pyrat Bay.

  Parameters
  ----------
  pf_files: String or List of strings
     Input Exomol partition-function filenames.  If there are
     multiple isotopes, all of them must correspond to the same
     molecule.

  Examples
  --------
  >>> import pyratbay.tools as pt
  >>> # A single file:
  >>> pt.pf_exomol('14N-1H3__MockBYTe.pf')
  Written partition-function file:
    'PF_Exomol_NH3.dat'
  for molecule NH3, with isotopes ['4111'],
  and temperature range 1--1600 K.

  >>> # Multiple files (isotopes) for a molecule:
  >>> pt.pf_exomol(['14N-1H3__MockBYTe.pf', '15N-1H3__MockBYTe-15.pf'])
  Warning: Length of PF files do not match.  Zero-padding the shorter array(s).
  Written partition-function file:
    'PF_Exomol_NH3.dat'
  for molecule NH3, with isotopes ['4111', '5111'],
  and temperature range 1--2000 K.
  """
  # Put into list if necessary:
  if isinstance(pf_files, str):
      pf_files = [pf_files]

  # Read and extract data from files:
  isotopes = []
  data, temps = [], []
  molecule = ""
  nt = 0
  for pf_file in pf_files:
      # Get info from file name:
      mol, iso = get_exomol_mol(pf_file)

      # Check all files correspond to the same molecule.
      if molecule == "":
          molecule = mol
      elif molecule != mol:
          raise ValueError("All files must correspond to the same molecule.")

      isotopes.append(iso)
      # Read data:
      temp, z = np.loadtxt(pf_file).T
      data.append(z)
      temps.append(temp)

  # Number of isotopes:
  niso = len(isotopes)
  # Check temp sampling:
  minlen = min(len(temp) for temp in temps)
  maxlen = max(len(temp) for temp in temps)
  ntemp = maxlen
  if minlen != maxlen:
      for temp in temps:
          if np.any(temp[0:minlen] - temps[0][0:minlen] != 0):
              raise ValueError(
                  "Temperature sampling in PF files are not compatible.")
      print('Warning: Length of PF files do not match.  Zero-padding the '
            'shorter array(s).')

  pf = np.zeros((niso, ntemp), np.double)
  for i,z in enumerate(data):
      pf[i, 0:len(z)] = z
      if len(z) == maxlen:
          temp = temps[i]

  # Write output file:
  file_out = "PF_Exomol_{:s}.dat".format(molecule)
  header = ("# This file incorporates the tabulated {:s} partition-function "
            "data\n# from Exomol\n\n".format(molecule))
  io.write_pf(file_out, pf, isotopes, temp, header)

  print("\nWritten partition-function file:\n  '{:s}'\nfor molecule {:s}, "
        "with isotopes {},\nand temperature range {:.0f}--{:.0f} K.".
        format(file_out, molecule, isotopes, temp[0], temp[-1]))
