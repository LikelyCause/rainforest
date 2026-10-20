"""Registered transformers; see registry.py."""
from .builtin import BUILTINS, ENCODING_SETTINGS
from .registry import BuildContext, TransformSpec, get, param_problems, params_for, register, specs

for _spec in BUILTINS:
    register(_spec)
