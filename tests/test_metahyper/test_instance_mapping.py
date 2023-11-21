
from metahyper.utils import instance_from_map
from metahyper.exceptions import MissingDependencyError

import pytest


@pytest.mark.parametrize(
    "err", [
        MissingDependencyError(libname="neps", dep="test_dep_1", install_group="group_1", cause=ImportError()),
        MissingDependencyError(libname="neps", dep="test_dep_2", install_group=None, cause=ImportError())
    ]
)
@pytest.mark.metahyper
def test_missing_dependancy_gets_flagged(err: MissingDependencyError) -> None:
    with pytest.raises(MissingDependencyError, match=err.dep):
        instance_from_map(mapping={}, request=err)

