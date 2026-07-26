"""Package-level smoke tests."""

import pilotnet


def test_package_imports() -> None:
    assert pilotnet.__doc__
