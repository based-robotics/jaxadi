import casadi as cs
import jax
import jax.numpy as jnp
import numpy as np

from jaxadi import convert


def test_1d_flat_behaviour():
    key = jax.random.key(0)
    x = cs.SX.sym("x", 2)
    Ax = cs.SX.sym("Ax", 3)

    Ax[0] = x[0] + x[1]
    Ax[1] = -x[0]
    Ax[2] = -x[1]

    cs_Ax = cs.Function("cs_Ax", [x], [Ax], ["x"], ["Ax"])

    jax_Ax = convert(cs_Ax)
    x = jax.random.uniform(key, (4, 2))

    # VMAP should fail if dimensionality
    # is incompatible with the translation
    Ax = jax.vmap(jax_Ax)(x)


def test_1d_non_flat_behaviour():
    key = jax.random.key(0)
    x = cs.SX.sym("x", 2)
    Ax = cs.SX.sym("Ax", 3)

    Ax[0] = x[0] + x[1]
    Ax[1] = -x[0]
    Ax[2] = -x[1]

    cs_Ax = cs.Function("cs_Ax", [x], [Ax], ["x"], ["Ax"])

    jax_Ax = convert(cs_Ax)
    x = jax.random.uniform(key, (4, 2, 1))
    # VMAP should fail if dimensionality
    # is incompatible with the translation
    Ax = jax.vmap(jax_Ax)(x)


def test_different_shapes():
    x = cs.SX.sym("x", 2, 3)
    y = cs.SX.sym("y", 3, 2)
    casadi_fn = cs.Function("myfunc", [x, y], [x @ y])

    jax_fn = convert(casadi_fn, compile=True)

    in1 = jnp.array(np.random.randn(2, 3))
    in2 = jnp.array(np.random.randn(3, 2))

    jax_fn(in1, in2)
