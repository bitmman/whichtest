"""whichtest: design → test → assumption checks → run → cross-check."""
from .registry import available, run
from .result import TestResult
from .spec import Design, Spec, SpecError, Variables
from .taxonomy import Taxonomy, TaxonomyError, load_taxonomy

__all__ = [
    "run",
    "available",
    "Spec",
    "Design",
    "Variables",
    "SpecError",
    "TestResult",
    "Taxonomy",
    "TaxonomyError",
    "load_taxonomy",
]
__version__ = "0.1.0"
