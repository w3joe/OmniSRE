#!/usr/bin/env python3
"""Generate Dataset A (Cold-Start SFT) and Dataset B (Tier-1 RLVR + Held-Out Eval)."""

from sre_rl.cli import main_build_datasets

if __name__ == "__main__":
    main_build_datasets()
