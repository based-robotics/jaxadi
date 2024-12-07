from casadi import Function
from typing import Any
from collections.abc import Callable

from ._declare import declare
from ._graph import translate as graph_translate
from ._expand import translate as expand_translate
from ._compile import compile as compile_fn
from ._preprocess import densify


def convert(casadi_fn: Function, translate=None, compile=False) -> Callable[..., Any]:
    """
    Convert given casadi function into python
    callable based on JAX backend, optionally
    the function will be AOT compiled

    :param casadi_fn (casadi.Function): CasADi function to convert
    :param compile (bool): Whether to AOT compile function
    :return (Callable[..., Any]): Resulting python function
    """
    if translate is None:
        translate = graph_translate

    casadi_fn = densify(casadi_fn)

    jax_str = translate(casadi_fn)
    jax_fn = declare(jax_str)

    if compile:
        compile_fn(jax_fn, casadi_fn)

    return jax_fn
