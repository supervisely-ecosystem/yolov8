"""Source level guards for the /auto_train handler.

Both bugs this pins down were invisible until the endpoint was actually called, and the
handler cannot be imported in a test: `train/src/main.py` builds the whole GUI and talks
to the instance at import time. So the invariants are checked on the parsed source, which
is enough to keep the two mistakes from coming back.

Run with:  python -m pytest tests/test_auto_train_invariants.py -v
"""

import ast
from pathlib import Path

import pytest

MAIN = Path(__file__).parents[1] / "train" / "src" / "main.py"


@pytest.fixture(scope="module")
def tree():
    return ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))


def functions(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def test_devices_is_declared_global_wherever_it_is_assigned(tree):
    """`devices` is module level: assigning it inside a function without `global` makes it
    local, so reading it first raises UnboundLocalError. /auto_train did exactly that and
    every API driven training died before it started."""
    offenders = []
    for func in functions(tree):
        assigns = {
            target.id
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if "devices" not in assigns:
            continue
        globals_declared = {
            name
            for node in ast.walk(func)
            if isinstance(node, ast.Global)
            for name in node.names
        }
        if "devices" not in globals_declared:
            offenders.append(f"{func.name} at line {func.lineno}")

    assert not offenders, f"assign devices without declaring it global: {offenders}"


def test_single_gpu_branch_parses_devices_not_device(tree):
    """The API branch had `int(device)` where the GUI branch has `int(devices)`, which
    would raise as soon as a single GPU was selected."""
    bad = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "device"
        ):
            bad.append(node.lineno)

    assert not bad, f"int(device) instead of int(devices) at lines {bad}"


def test_both_upload_sites_share_one_progress_callback(tree):
    """The callback used to be copy pasted into the GUI and the API path, and the same bug
    had to be fixed twice. It lives in src/upload_progress.py now."""
    source = MAIN.read_text(encoding="utf-8")
    assert "from src.upload_progress import make_upload_monitor" in source
    assert source.count("make_upload_monitor(progress, artifacts_pbar)") == 2
    assert "def upload_monitor(" not in source
