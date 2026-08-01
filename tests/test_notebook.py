"""Structural checks for the standalone learner notebook."""

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "pilotnet_walkthrough.ipynb"


def test_notebook_is_valid_json_and_has_no_package_imports() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert notebook["nbformat"] == 4
    assert "from pilotnet" not in code
    assert "import pilotnet" not in code
    compile(code, str(NOTEBOOK_PATH), "exec")
