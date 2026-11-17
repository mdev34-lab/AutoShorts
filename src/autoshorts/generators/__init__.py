"""Generator registry.

Generator classes are imported lazily so that running one video type never
imports the modules of another. In particular the explainer CLI path no
longer loads the Shopee product pipeline (generators/product.py and
modules/product/*), which is a separate responsibility with its own CLI
entry point.
"""

import importlib

_GENERATOR_SPECS = {
    "explainer": ("autoshorts.generators.explainer", "ExplainerGenerator"),
    "product": ("autoshorts.generators.product", "ProductGenerator"),
}


class _LazyVideoTypes:
    """Dict-like registry that imports a generator class on first access."""

    def __getitem__(self, key: str) -> type:
        if key not in _GENERATOR_SPECS:
            raise KeyError(key)
        module_name, class_name = _GENERATOR_SPECS[key]
        return getattr(importlib.import_module(module_name), class_name)

    def __contains__(self, key: object) -> bool:
        return key in _GENERATOR_SPECS

    def __iter__(self):
        return iter(_GENERATOR_SPECS)

    def __len__(self) -> int:
        return len(_GENERATOR_SPECS)

    def keys(self):
        return _GENERATOR_SPECS.keys()


VIDEO_TYPES = _LazyVideoTypes()


def __getattr__(name: str):
    """Backward-compatible ``from autoshorts.generators import ExplainerGenerator``."""
    for module_name, class_name in _GENERATOR_SPECS.values():
        if name == class_name:
            return getattr(importlib.import_module(module_name), class_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
