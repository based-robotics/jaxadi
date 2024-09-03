from typing import Callable, Any
import jax.numpy as jnp


def declare(f: str) -> Callable[..., Any]:
    """
    Return local scope function
    based on string definition
    """
    local_vars = {}
    exec(f, globals(), local_vars)
    func_name = next(iter(local_vars))
    return local_vars[func_name]
