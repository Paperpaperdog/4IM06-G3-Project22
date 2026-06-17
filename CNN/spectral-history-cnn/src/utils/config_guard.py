"""Block accidental training on archived v1 configs."""

from __future__ import annotations

import os
import sys

_LEGACY_MARKERS = ("/legacy/", "v1_final64", "v1_fourier")


def reject_legacy_config_path(config_path: str) -> None:
    normalized = str(config_path).replace("\\", "/")
    if not any(marker in normalized for marker in _LEGACY_MARKERS):
        return
    if os.environ.get("ALLOW_LEGACY_CONFIG") == "1":
        print(f"WARN: using legacy config (ALLOW_LEGACY_CONFIG=1): {config_path}", file=sys.stderr)
        return
    print(
        "ERROR: refusing legacy experiment config.\n"
        f"  path: {config_path}\n"
        "  n6 main configs: configs/size_sweep/n6_poscnn_size{32,64,96,128}.yaml\n"
        "  see docs/EXPERIMENT_RUNBOOK.md\n"
        "  to override: ALLOW_LEGACY_CONFIG=1",
        file=sys.stderr,
    )
    raise SystemExit(1)
