"""Cache adapters: the two-tier store and the prefix policy over it.

``store.py`` implements :class:`~titan.core.ports.KVStateStore`: a hot RAM tier
sized in gigabytes and an SSD tier, with a background writer and a bounded
stall. ``prefix.py`` implements :class:`~titan.core.ports.PrefixCache`: block
hashing, the two grids, chunk planning and the store decision. ``format.py``
holds the chain hash, the compatibility signature and the on-disk record;
``codec.py`` is the one seam where device state becomes bytes, and the only
place in this package that an mlx array is allowed anywhere near.

Building the pair takes three lines, and ``titan.config.wiring`` is where they
belong:

```
signature = CacheSignature(model_name, layer_layout, block_tokens=512,
                           snapshot_dtype="fp32")
store = TwoTierStateStore(signature, ssd_dir=cfg.cache.ssd_dir, ...)
cache = BlockPrefixCache(store, codec, block_tokens=512, snapshot_grid=2048)
```
"""

from titan.adapters.cache import codec, format, prefix, store
from titan.adapters.cache.codec import StateCodec
from titan.adapters.cache.format import CacheSignature, chain_hash, snapshot_id_for
from titan.adapters.cache.prefix import BlockPrefixCache, PrefixLease, PrefixStats
from titan.adapters.cache.store import StoreStats, TwoTierStateStore

__all__ = [
    "codec",
    "format",
    "prefix",
    "store",
    "BlockPrefixCache",
    "CacheSignature",
    "PrefixLease",
    "PrefixStats",
    "StateCodec",
    "StoreStats",
    "TwoTierStateStore",
    "chain_hash",
    "snapshot_id_for",
]
