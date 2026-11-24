"""Project components, loaded lazily so data preparation does not import PyTorch."""
from importlib import import_module

_EXPORTS = {
    "BaselineConfig": ("config", "BaselineConfig"),
    "load_config": ("config", "load_config"),
    "ProteinSequenceSplit": ("data", "ProteinSequenceSplit"),
    "StandardDataModule": ("data", "StandardDataModule"),
    "build_model": ("factories", "build_model"),
    "build_optimizer": ("factories", "build_optimizer"),
    "BaselinePipeline": ("pipeline", "BaselinePipeline"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(f"project1.{module}"), symbol)
    globals()[name] = value
    return value
