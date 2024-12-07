import casadi as ca
import jax.numpy as jnp
import numpy as np
import pytest

from jaxadi import convert

from jaxadi import graph_translate
from jaxadi import expand_translate

# Set a fixed seed for reproducibility
np.random.seed(42)


def compare_results(casadi_f, jax_f, *inputs):
    """
    Compare results of evaluations of the
    original casadi function and jax translation
    """
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


def test_simo_trig_graph():
    """
    SIMO trigonometry test
    """
    x = ca.SX.sym("x", 1)
    casadi_f = ca.Function("simo_trig", [x], [ca.sin(x), ca.cos(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(1)
    compare_results(casadi_f, jax_f, x_val)


def test_simo_trig_expand():
    """
    SIMO trigonometry test
    """
    x = ca.SX.sym("x", 1)
    casadi_f = ca.Function("simo_trig", [x], [ca.sin(x), ca.cos(x)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(1)
    compare_results(casadi_f, jax_f, x_val)


def test_all_zeros_graph():
    """
    Input multiplied by matrix with all zeros
    """
    X = ca.SX.sym("x", 2)
    A = np.zeros((2, 2))
    Y = ca.jacobian(A @ X, X)

    casadi_f = ca.Function("foo", [X], [Y])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(2, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_all_zeros_expand():
    """
    Input multiplied by matrix with all zeros
    """
    X = ca.SX.sym("x", 2)
    A = np.zeros((2, 2))
    Y = ca.jacobian(A @ X, X)

    casadi_f = ca.Function("foo", [X], [Y])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(2, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_structural_zeros_graph():
    """
    Input contains some structural zeros
    """
    X = ca.SX.sym("x", 2)
    A = np.ones((2, 2))
    A[1, :] = 0.0
    Y = ca.jacobian(A @ X, X)

    casadi_f = ca.Function("foo", [X], [ca.densify(Y)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(2, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_structural_zeros_expand():
    """
    Input contains some structural zeros
    """
    X = ca.SX.sym("x", 2)
    A = np.ones((2, 2))
    A[1, :] = 0.0
    Y = ca.jacobian(A @ X, X)

    casadi_f = ca.Function("foo", [X], [ca.densify(Y)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(2, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_simo_poly_graph():
    """
    SIMO polynomials test
    """
    x = ca.SX.sym("x", 1)
    casadi_f = ca.Function("simo_poly", [x], [x**2, x**3, ca.sqrt(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(1)
    x_val = np.abs(x_val)  # Ensure positive for sqrt
    compare_results(casadi_f, jax_f, x_val)


def test_simo_poly_expand():
    """
    SIMO polynomials test
    """
    x = ca.SX.sym("x", 1)
    casadi_f = ca.Function("simo_poly", [x], [x**2, x**3, ca.sqrt(x)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(1)
    x_val = np.abs(x_val)  # Ensure positive for sqrt
    compare_results(casadi_f, jax_f, x_val)


def test_simo_matrix_graph():
    """
    SIMO matrix operations
    """
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("simo_matrix", [x], [ca.trace(x), ca.det(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_simo_matrix_expand():
    """
    SIMO matrix operations
    """
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("simo_matrix", [x], [ca.trace(x), ca.det(x)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_miso_add_graph():
    """
    MISO addition operation
    """
    x = ca.SX.sym("x", 3, 1)
    y = ca.SX.sym("y", 3, 1)
    casadi_f = ca.Function("miso_add", [x, y], [x + y])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 1)
    y_val = np.random.randn(3, 1)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_miso_add_expand():
    """
    MISO addition operation
    """
    x = ca.SX.sym("x", 3, 1)
    y = ca.SX.sym("y", 3, 1)
    casadi_f = ca.Function("miso_add", [x, y], [x + y])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 1)
    y_val = np.random.randn(3, 1)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_miso_multiply_graph():
    """
    MISO matrix multiplication
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("miso_multiply", [x, y], [x * y])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_miso_multiply_expand():
    """
    MISO matrix multiplication
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("miso_multiply", [x, y], [x * y])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_miso_combined_graph():
    """
    MISO matrix multiplication and addition
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    z = ca.SX.sym("z", 3, 3)
    casadi_f = ca.Function("miso_combined", [x, y, z], [ca.mtimes(x, y) + z])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    z_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val, z_val)


def test_miso_combined_expand():
    """
    MISO matrix multiplication and addition
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    z = ca.SX.sym("z", 3, 3)
    casadi_f = ca.Function("miso_combined", [x, y, z], [ca.mtimes(x, y) + z])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    z_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val, z_val)


def test_mimo_arith_graph():
    """
    MIMO arithmetics test
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("mimo_arith", [x, y], [x + y, x - y])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_mimo_arith_expand():
    """
    MIMO arithmetics test
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("mimo_arith", [x, y], [x + y, x - y])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_mimo_trig_graph():
    """
    MIMO trigonometry test
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("mimo_trig", [x, y], [ca.sin(x), ca.cos(y)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_mimo_trig_expand():
    """
    MIMO trigonometry test
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("mimo_trig", [x, y], [ca.sin(x), ca.cos(y)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_mimo_mtinv_graph():
    """
    MIMO combination of multiplication, inversion and addition
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    z = ca.SX.sym("z", 3, 3)
    casadi_f = ca.Function("mimo_complex", [x, y, z], [ca.mtimes(x, y), ca.inv(z), x + z])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    z_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val, z_val)


def test_mimo_mtinv_expand():
    """
    MIMO combination of multiplication, inversion and addition
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    z = ca.SX.sym("z", 3, 3)
    casadi_f = ca.Function("mimo_complex", [x, y, z], [ca.mtimes(x, y), ca.inv(z), x + z])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    z_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val, z_val)


def test_sin_graph():
    """
    Test sin operation
    """
    x = ca.SX.sym("x", 1, 1)
    casadi_f = ca.Function("sin", [x], [ca.sin(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(1, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_sin_expand():
    """
    Test sin operation
    """
    x = ca.SX.sym("x", 1, 1)
    casadi_f = ca.Function("sin", [x], [ca.sin(x)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(1, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_cos_graph():
    """
    Test cosine operation
    """
    x = ca.SX.sym("x", 1, 1)
    casadi_f = ca.Function("cos", [x], [ca.cos(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(1, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_cos_expand():
    """
    Test cosine operation
    """
    x = ca.SX.sym("x", 1, 1)
    casadi_f = ca.Function("cos", [x], [ca.cos(x)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(1, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_mtimes_graph():
    """
    Test matrix product
    """
    x = ca.SX.sym("x", 2, 2)
    y = ca.SX.sym("y", 2, 2)
    casadi_f = ca.Function("mtimes", [x, y], [x @ y])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(2, 2)
    y_val = np.random.randn(2, 2)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_mtimes_expand():
    """
    Test matrix product
    """
    x = ca.SX.sym("x", 2, 2)
    y = ca.SX.sym("y", 2, 2)
    casadi_f = ca.Function("mtimes", [x, y], [x @ y])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(2, 2)
    y_val = np.random.randn(2, 2)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_inv_graph():
    """
    Test matrix inversion
    """
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("inv", [x], [ca.inv(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_inv_expand():
    """
    Test matrix inversion
    """
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("inv", [x], [ca.inv(x)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_norm_2_graph():
    """
    Test second norm
    """
    x = ca.SX.sym("x", 3, 1)
    casadi_f = ca.Function("norm_2", [x], [ca.norm_2(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_norm_2_expand():
    """
    Test second norm
    """
    x = ca.SX.sym("x", 3, 1)
    casadi_f = ca.Function("norm_2", [x], [ca.norm_2(x)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 1)
    compare_results(casadi_f, jax_f, x_val)


def test_sum1_graph():
    """
    Test sum1
    """
    x = ca.SX.sym("x", 2, 2)
    casadi_f = ca.Function("sum1", [x], [ca.sum1(x)])
    x_val = np.random.randn(2, 2)
    jax_f = convert(casadi_f, translate=graph_translate)
    compare_results(casadi_f, jax_f, x_val)


def test_sum1_expand():
    """
    Test sum1
    """
    x = ca.SX.sym("x", 2, 2)
    casadi_f = ca.Function("sum1", [x], [ca.sum1(x)])
    x_val = np.random.randn(2, 2)
    jax_f = convert(casadi_f, translate=expand_translate)
    compare_results(casadi_f, jax_f, x_val)


def test_dot_graph():
    """
    Test dot product
    """
    x = ca.SX.sym("x", 3)
    y = ca.SX.sym("y", 3)
    casadi_f = ca.Function("dot", [x, y], [ca.dot(x, y)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3)
    y_val = np.random.randn(3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_dot_expand():
    """
    Test dot product
    """
    x = ca.SX.sym("x", 3)
    y = ca.SX.sym("y", 3)
    casadi_f = ca.Function("dot", [x, y], [ca.dot(x, y)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3)
    y_val = np.random.randn(3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_transpose_graph():
    """
    Test transposition
    """
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("transpose", [x], [ca.transpose(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_transpose_expand():
    """
    Test transposition
    """
    x = ca.SX.sym("x", 3, 3)
    casadi_f = ca.Function("transpose", [x], [ca.transpose(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val)


def test_add_graph():
    """
    Test addition
    """
    x = ca.SX.sym("x", 5, 5)
    y = ca.SX.sym("y", 5, 5)
    casadi_f = ca.Function("add", [x, y], [x + y])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(5, 5)
    y_val = np.random.randn(5, 5)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_add_expand():
    """
    Test addition
    """
    x = ca.SX.sym("x", 5, 5)
    y = ca.SX.sym("y", 5, 5)
    casadi_f = ca.Function("add", [x, y], [x + y])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(5, 5)
    y_val = np.random.randn(5, 5)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_multiply_graph():
    """
    Test multiplication
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("multiply", [x, y], [x * y])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_multiply_expand():
    """
    Test multiplication
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("multiply", [x, y], [x * y])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_combined_graph():
    """
    MISO matrix product added with inversion
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("combined", [x, y], [ca.mtimes(x, y) + ca.inv(x)])
    jax_f = convert(casadi_f, translate=graph_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    compare_results(casadi_f, jax_f, x_val, y_val)


def test_combined_expand():
    """
    MISO matrix product added with inversion
    """
    x = ca.SX.sym("x", 3, 3)
    y = ca.SX.sym("y", 3, 3)
    casadi_f = ca.Function("combined", [x, y], [ca.mtimes(x, y) + ca.inv(x)])
    jax_f = convert(casadi_f, translate=expand_translate)
    x_val = np.random.randn(3, 3)
    y_val = np.random.randn(3, 3)
    print(expand_translate(casadi_f))
    compare_results(casadi_f, jax_f, x_val, y_val)
