"""Cache adapters: the two-tier store and the prefix policy over it.

``store.py`` implements :class:`~titan.core.ports.KVStateStore`: a hot RAM tier
sized in gigabytes and an SSD tier, with a background writer and a bounded
stall. ``prefix.py`` implements :class:`~titan.core.ports.PrefixCache`: block
hashing, the two grids, chunk planning and the store decision.
"""

from titan.adapters.cache import prefix, store

__all__ = ["prefix", "store"]
