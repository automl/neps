from __future__ import annotations

import rich

import neps

space = neps.SearchSpace(
        {
            "a": neps.Integer(0, 10),
            "b": neps.Categorical(["a", "b", "c"]),
            "c": neps.Float(1e-5, 1e0, log=True, prior=1e-3),
            }
        )

rich.print(space)
