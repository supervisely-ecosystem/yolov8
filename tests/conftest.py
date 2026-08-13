import sys
from pathlib import Path

# tests import "src.*" modules the same way the serve app itself does
# (see tests/deploy_and_infer.py), so serve/ needs to be on sys.path.
serve_src_path = str(Path(__file__).parents[1] / "serve")
if serve_src_path not in sys.path:
    sys.path.append(serve_src_path)
