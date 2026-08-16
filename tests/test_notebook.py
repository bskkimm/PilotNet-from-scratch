"""Structural checks for the learner notebook."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "pilotnet_walkthrough.ipynb"


def test_notebook_rebuilds_the_model_and_imports_production_training() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert notebook["nbformat"] == 4
    assert "from pilotnet.models" not in code
    assert "from pilotnet.engine import evaluate, train_epoch" in code
    assert "from pilotnet.tracking import MlflowTracker" in code
    compile(code, str(NOTEBOOK_PATH), "exec")
