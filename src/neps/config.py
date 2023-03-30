from __future__ import annotations

import metahyper
from neps.search_spaces.search_space import SearchSpace



class ConfigResult(metahyper.Config.Result[SearchSpace]):
    """Result of a configuration evaluation.

    This class exists purely to put a type annotation on the `config` attribute.
    """
