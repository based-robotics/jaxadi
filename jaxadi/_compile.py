from collections.abc import Callable
from typing import Any

import casadi
import jax
import jax.numpy as jnp


def lower(f: Callable[..., Any], cf: casadi.Function) -> jax._src.stages.Lowered:
    """
    Lower the given casadi function

    :param f (Callable[..., Any]): The JAX function to lower
    :param cf (casadi.Function): The corresponding CasADi function
    :return (jax._src.stage.Lowered): The lowered function
    """
    jax_input_structs = []
    for i in range(cf.n_in()):
        # Determine the shape based on CasADi's sparsity
        sparsity = cf.sparsity_in(i)
        shape = sparsity.shape

        if jax.config.jax_enable_x64:
            dtype = jnp.dtype("float64")
        else:
            dtype = jnp.dtype("float32")

        jax_input_structs.append(jax.ShapeDtypeStruct(shape, dtype))

    # Pre-compile the JAX function
    jax_func = jax.jit(f)
    hlo = jax_func.lower(*jax_input_structs)
    return hlo


def compile(f: Callable[..., Any], cf: casadi.Function) -> Callable[..., Any]:
    """
    AOT JIT function compilation

    :param f (Callable[..., Any]): The JAX function to cache.
    :param cf (casadi.Function): The corresponding CasADi function.
    :return (Callable[..., Any]): The compiled JAX
    """

    compiled_function = lower(f, cf).compile()
    return compiled_function
