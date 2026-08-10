"""Make the repo and the vendored feature library importable during tests.

``libs/bullsense_features`` goes first so the in-repo ``hft_features`` wins over
any stale editable install of the old out-of-tree package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

for _path in (_ROOT / "libs" / "bullsense_features", _ROOT):
    _sp = str(_path)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
