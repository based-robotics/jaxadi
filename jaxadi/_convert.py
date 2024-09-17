from casadi import Function
from typing import Any
from collections.abc import Callable

from ._declare import declare
from ._translate import translate
from ._compile import compile as compile_fn


def convert(casadi_fn: Function, compile=False, num_threads=1) -> Callable[..., Any]:
    """
    Convert given casadi function into python
    callable based on JAX backend, optionally
    the function will be AOT compiled

    :param casadi_fn (casadi.Function): CasADi function to convert
    :param compile (bool): Whether to AOT compile function
    :return (Callable[..., Any]): Resulting python function
    """
    jax_str = translate(casadi_fn, num_threads=num_threads)
    jax_fn = declare(jax_str)

    if compile:
        compile_fn(jax_fn, casadi_fn)

    return jax_fn
