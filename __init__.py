#import pkgutil

#__all__ = []
#for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
#    __all__.append(module_name)
#    _module = loader.find_module(module_name).load_module(module_name)
#    globals()[module_name] = _module

from .actflowcomp import *
from . import connectivity_estimation
from . import dependencies
from . import infotransfermapping
from . import pipelines
from . import preprocessing
from . import simulations
from . import tools
