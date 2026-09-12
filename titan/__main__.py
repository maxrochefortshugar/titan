"""``python -m titan`` runs the same entry point as the installed script."""

from __future__ import annotations

import sys

from titan.cli import main

if __name__ == "__main__":
    sys.exit(main())
