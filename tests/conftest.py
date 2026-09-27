"""Make `import gems` work without installing the package.

The tests are run as `python -m pytest tests/` from the repo root; the package
lives in src/ (src-layout) and is deliberately not pip-installed so that the repo
needs no build step on the runner.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
