# Copyright (c) 2016-2021 Patricio Cubillos.
# Pyrat Bay is open-source software under the GNU GPL-2.0 license (see LICENSE).

from .blackbody import *
from .kurucz import *
#from .marcs import *
#from .phoenix import *
from .spec_tools import *
from .contribution_funcs import *

__all__ = ( blackbody.__all__
          + kurucz.__all__
          + spec_tools.__all__
          + contribution_funcs.__all__
          )


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)
