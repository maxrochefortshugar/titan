"""Engine layer: scheduling and generation logic. Imports ``titan.core`` only.

The engine is where the decode cycle lives. It holds no device arrays and makes
no framework calls; everything it touches is a port. That is what lets the whole
scheduler run against fake adapters in a unit test, at speed, with no model.
"""

from titan.engine import admission, decode_cycle, engine, scheduler
from titan.engine.engine import TitanEngine
from titan.engine.scheduler import EngineLoop

__all__ = [
    "admission",
    "decode_cycle",
    "engine",
    "scheduler",
    "EngineLoop",
    "TitanEngine",
]
