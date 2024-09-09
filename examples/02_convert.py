"""
This example shows how to convert a matrix multiplication function defined
in CasADi to a JAX-compatible function using the jaxadi library.
It demonstrates defining a function in CasADi, converting it to a JAX function,
and running the compiled function with random input matrices.
"""

import casadi as cs

from jaxadi import convert

# define input variables for the function
x = cs.SX.sym("x", 10, 10)
y = cs.SX.sym("y", 10, 10)
casadi_fn = cs.Function("myfunc", [x, y], [x @ y])

# define jax function from casadi one
jax_fn = convert(casadi_fn, compile=True)

# Run compiled function
jax_fn(cs.np.random.rand(10, 10), cs.np.random.rand(10, 10))
