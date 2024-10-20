import casadi as cs
import jax.numpy as jnp
import numpy as np

from jaxadi import convert


def test_different_shapes():
    x = cs.SX.sym("x", 2, 3)
    y = cs.SX.sym("y", 3, 2)
    casadi_fn = cs.Function("myfunc", [x, y], [x @ y])

    jax_fn = convert(casadi_fn, compile=True)

    in1 = jnp.array(np.random.randn(2, 3))
    in2 = jnp.array(np.random.randn(3, 2))

    jax_fn(in1, in2)
