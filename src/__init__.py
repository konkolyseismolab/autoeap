import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__

from autoeap.autoeapcore import createlightcurve
from autoeap.autoeapcore import autoeap_from_commandline
