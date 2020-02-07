import logging

logging.basicConfig(format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO)

from .merf import MERF
from .utils import MERFDataGenerator
from . import crown_ensemble
from . import outliers
from . import read
from . import resample
from . import transform
from . import write

# Version of the merf package
__version__ = "0.3.0"
