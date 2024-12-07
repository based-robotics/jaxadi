import casadi as cs
import jax
import jax.numpy as jnp
import numpy as np

from jaxadi import convert, expand_translate, graph_translate


def test_1d_flat_vmap_graph():
    """
    Test the input compatibility with
    flat 1D arrays on VMAP operation
    """
    key = jax.random.key(0)
    x = cs.SX.sym("x", 2)
    Ax = cs.SX.sym("Ax", 3)

    Ax[0] = x[0] + x[1]
    Ax[1] = -x[0]
    Ax[2] = -x[1]

    cs_Ax = cs.Function("cs_Ax", [x], [Ax], ["x"], ["Ax"])

    jax_Ax = convert(cs_Ax, translate=graph_translate)
    x = jax.random.uniform(key, (4, 2))

    # VMAP should fail if dimensionality
    # is incompatible with the translation
    Ax = jax.vmap(jax_Ax)(x)


def test_1d_flat_vmap_expand():
    """
    Test the input compatibility with
    flat 1D arrays on VMAP operation
    """
    key = jax.random.key(0)
    x = cs.SX.sym("x", 2)
    Ax = cs.SX.sym("Ax", 3)

    Ax[0] = x[0] + x[1]
    Ax[1] = -x[0]
    Ax[2] = -x[1]

    cs_Ax = cs.Function("cs_Ax", [x], [Ax], ["x"], ["Ax"])

    jax_Ax = convert(cs_Ax, translate=expand_translate)
    x = jax.random.uniform(key, (4, 2))

    # VMAP should fail if dimensionality
    # is incompatible with the translation
    Ax = jax.vmap(jax_Ax)(x)


def test_1d_non_flat_behaviour_graph():
    """
    Test the input compatibility with
    non-flat 1D arrays on VMAP operation
    """
    key = jax.random.key(0)
    x = cs.SX.sym("x", 2)
    Ax = cs.SX.sym("Ax", 3)

    Ax[0] = x[0] + x[1]
    Ax[1] = -x[0]
    Ax[2] = -x[1]

    cs_Ax = cs.Function("cs_Ax", [x], [Ax], ["x"], ["Ax"])

    jax_Ax = convert(cs_Ax, translate=graph_translate)
    x = jax.random.uniform(key, (4, 2, 1))
    # VMAP should fail if dimensionality
    # is incompatible with the translation
    Ax = jax.vmap(jax_Ax)(x)


def test_1d_non_flat_behaviour_expand():
    """
    Test the input compatibility with
    non-flat 1D arrays on VMAP operation
    """
    key = jax.random.key(0)
    x = cs.SX.sym("x", 2)
    Ax = cs.SX.sym("Ax", 3)

    Ax[0] = x[0] + x[1]
    Ax[1] = -x[0]
    Ax[2] = -x[1]

    cs_Ax = cs.Function("cs_Ax", [x], [Ax], ["x"], ["Ax"])

    jax_Ax = convert(cs_Ax, translate=expand_translate)
    x = jax.random.uniform(key, (4, 2, 1))
    # VMAP should fail if dimensionality
    # is incompatible with the translation
    Ax = jax.vmap(jax_Ax)(x)


def test_different_shapes_graph():
    """
    Try MIMO with different shapes
    """
    x = cs.SX.sym("x", 2, 3)
    y = cs.SX.sym("y", 3, 2)
    z = cs.SX.sym("y", 3, 1)
    w = cs.SX.sym("y", 3)
    casadi_fn = cs.Function("myfunc", [x, y, z, w], [x @ y, z.T @ w])

    jax_fn = convert(casadi_fn, translate=graph_translate)

    in1 = jnp.array(np.random.randn(2, 3))
    in2 = jnp.array(np.random.randn(3, 2))
    in3 = jnp.array(np.random.randn(3, 1))
    in4 = jnp.array(np.random.randn(3))

    jax_fn(in1, in2, in3, in4)


def test_different_shapes_expand():
    """
    Try MIMO with different shapes
    """
    x = cs.SX.sym("x", 2, 3)
    y = cs.SX.sym("y", 3, 2)
    z = cs.SX.sym("y", 3, 1)
    w = cs.SX.sym("y", 3)
    casadi_fn = cs.Function("myfunc", [x, y, z, w], [x @ y, z.T @ w])

    jax_fn = convert(casadi_fn, translate=expand_translate)

    in1 = jnp.array(np.random.randn(2, 3))
    in2 = jnp.array(np.random.randn(3, 2))
    in3 = jnp.array(np.random.randn(3, 1))
    in4 = jnp.array(np.random.randn(3))

    jax_fn(in1, in2, in3, in4)
