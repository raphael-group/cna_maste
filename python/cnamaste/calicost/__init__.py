"""Everything cnamaste takes from CalicoST, in one place and under cnamaste's words.

CalicoST is a dependency, pinned at `c1abcae` by `[tool.uv.sources]`, not a tree of copies (T- #44): what
`ab_decode.decoder: calicost` and `hmm.initializer: calicost` claim is that the run does what CalicoST does, and
a copy is that claim only until CalicoST moves.

Every name it takes is imported here, `as` the word cnamaste uses for the concept, so the modules that call them
read in one vocabulary and a reader has one file to look at for what the dependency supplies:

- `fixdiploid_copy_numbers` -- the diploid decoder (`ab_decode.decode`, `adapters.calicost_fixdiploid`);
- `gmm_start` -- the GMM start (`initialize.genomic.get_calicost_states`);
- `filter_diffexp` -- the differential-expression gene filter, CalicoST's three-class one, which
  `normal_spot.filter_normal_diffexp` is a faster form of rather than a caller of (T- #47);
- `filter_diffexp_two_class` -- its two-class sibling, which no cnamaste stage uses and which is here so that the
  one the stage follows is named beside the one it is not.

Imports alone, made with the environment held as it was. A function written here is a function CalicoST does not have, and belongs in the module whose job it
is. Two submodules, imported by name and not re-exported: `cnamaste.calicost.read`, readers of CalicoST's run
directories, and `cnamaste.calicost.adapters`, CalicoST's calls adapted to cnamaste's.
"""

from __future__ import annotations

import os
from unittest import mock

# NB CalicoST's modules set MKL_NUM_THREADS, OPENBLAS_NUM_THREADS and OMP_NUM_THREADS to "1" at import
#    (utils_distribution_fitting); the environment is restored as it was once they have loaded.
with mock.patch.dict(os.environ):
    from calicost.find_integer_copynumber import (
        hill_climbing_integer_copynumber_fixdiploid as fixdiploid_copy_numbers,
    )
    from calicost.utils_hmm import initialization_by_gmm as gmm_start
    from calicost.utils_IO import (
        filter_de_genes as filter_diffexp_two_class,
        filter_de_genes_tri as filter_diffexp,
    )

__all__ = [
    "filter_diffexp",
    "filter_diffexp_two_class",
    "fixdiploid_copy_numbers",
    "gmm_start",
]
