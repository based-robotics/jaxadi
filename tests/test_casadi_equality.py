import casadi as ca
import jax.numpy as jnp
import numpy as np
import pytest

from jaxadi import convert, translate

# Set a fixed seed for reproducibility
np.random.seed(42)


def compare_results(casadi_f, jax_f, *inputs):
    casadi_res = casadi_f(*inputs)
    inputs = list(inputs)
    jax_res = jax_f(*inputs)
    # Jax returns list always
    # Casadi returns list only in
    # multiple output case
    if len(jax_res) == 1:
        casadi_res = [casadi_res]
    try:
        for i in range(len(jax_res)):
            r1 = jnp.array(jax_res[i])
            r2 = jnp.array(casadi_res[i].toarray())
            assert np.allclose(r1, r2, rtol=1e-2, atol=1e-8)
    except Exception as e:
        pytest.fail(f"Comparison failed: {e}")


def test_simo_trig():
    x = ca.SX.sym("x", 1, 1)
    casadi_f = ca.Function("simo_trig", [x], [ca.sin(x), ca.cos(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(1, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_simo_poly():
    x = ca.SX.sym("x", 1, 1)
    casadi_f = ca.Function("simo_poly", [x], [x**2, x**3, ca.sqrt(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(1, 1)
    x_val = np.abs(x_val)  # Ensure positive for sqrt
    compare_results(casadi_f, jax_f, x_val)


def test_simo_matrix():
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("simo_matrix", [x], [ca.trace(x), ca.det(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_miso_add():
    x = ca.SX.sym("x", 3, 1)
    y = ca.SX.sym("y", 3, 1)
    casadi_f = ca.Function("miso_add", [x, y], [x + y])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 1)
    y_val = np.random.randn(3, 1)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_miso_multiply():
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("miso_multiply", [x, y], [x * y])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_miso_combined():
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    z = ca.SX.sym("z", 3, 3)
    casadi_f = ca.Function("miso_combined", [x, y, z], [ca.mtimes(x, y) + z])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    z_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val, z_val)


def test_mimo_arith():
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("mimo_arith", [x, y], [x + y, x - y])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_mimo_trig():
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("mimo_trig", [x, y], [ca.sin(x), ca.cos(y)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_mimo_complex():
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    z = ca.SX.sym("z", 3, 3)
    casadi_f = ca.Function("mimo_complex", [x, y, z], [ca.mtimes(x, y), ca.inv(z), x + z])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    z_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val, z_val)


def test_sin():
    x = ca.SX.sym("x", 1, 1)
    casadi_f = ca.Function("sin", [x], [ca.sin(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(1, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_cos():
    x = ca.SX.sym("x", 1, 1)
    casadi_f = ca.Function("cos", [x], [ca.cos(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(1, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_mtimes():
    x = ca.SX.sym("x", 2, 2)
    y = ca.SX.sym("y", 2, 2)
    casadi_f = ca.Function("mtimes", [x, y], [x @ y])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(2, 2)
    y_val = np.random.randn(2, 2)
    # x_val = np.array([[1, 1], [2, 2]])
    # y_val = np.array([[2, 2], [2, 2]])
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_inv():
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("inv", [x], [ca.inv(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_norm_2():
    x = ca.SX.sym("x", 3, 1)
    casadi_f = ca.Function("norm_2", [x], [ca.norm_2(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_sum1():
    x = ca.SX.sym("x", 2, 2)
    casadi_f = ca.Function("sum1", [x], [ca.sum1(x)])
    x_val = np.random.randn(2, 2)
    jax_f = convert(casadi_f)
    compare_results(casadi_f, jax_f, x_val)


def test_dot():
    x = ca.SX.sym("x", 3, 1)
    y = ca.SX.sym("y", 3, 1)
    casadi_f = ca.Function("dot", [x, y], [ca.dot(x, y)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 1)
    y_val = np.random.randn(3, 1)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_transpose():
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("transpose", [x], [ca.transpose(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_add():
    x = ca.SX.sym("x", 5, 5)
    y = ca.SX.sym("y", 5, 5)
    casadi_f = ca.Function("add", [x, y], [x + y])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(5, 5)
    y_val = np.random.randn(5, 5)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_multiply():
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("multiply", [x, y], [x * y])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_combined():
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("combined", [x, y], [ca.mtimes(x, y) + ca.inv(x)])
    jax_f = convert(casadi_f)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)
