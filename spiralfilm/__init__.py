# spiral_film/__init__.py

# Import main classes and functions from the relevant modules
from .core import FilmCore
from .config import FilmConfig
from .embed import FilmEmbed, FilmEmbedConfig
from .utils import TextCutter
from .errors import MaxRetriesExceededError, ContentFilterError

# Specify the version of the package
__version__ = "0.2.4"
