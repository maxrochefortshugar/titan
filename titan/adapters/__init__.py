"""Adapters implement the core's ports. Nothing in the core imports these.

    titan.adapters.mlx      model backend, tokenizer, template, n-gram reader
    titan.adapters.cache    KV state store and prefix cache policy

Each adapter is replaceable by a fake in tests, and the fakes are the reason the
scheduler can be tested without a model.
"""
