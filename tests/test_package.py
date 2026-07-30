"""Package-level smoke tests."""

import pilotnet


def test_package_exports_model() -> None:
    assert pilotnet.PilotNet
