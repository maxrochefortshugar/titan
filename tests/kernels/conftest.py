# SPDX-License-Identifier: MIT
"""Fixtures for the kernel tests.

The measures themselves live in ``exactness.py`` next door, so a test imports
them by name rather than reaching into a conftest.

Shapes here are small on purpose: the workbench may be using the GPU, so
nothing allocates more than a few hundred MB, and the real Flash-Next shapes
live behind ``-m slow``.
"""

from __future__ import annotations

import mlx.core as mx
import pytest


@pytest.fixture(scope="session", autouse=True)
def _require_metal():
    if not mx.metal.is_available():
        pytest.skip("kernel tests need a Metal device")


@pytest.fixture()
def seeded():
    """Deterministic inputs. Returns the seeding function so a test can reseed
    between cases without importing mx itself."""
    def seed(value: int = 0):
        mx.random.seed(value)
    seed()
    return seed
