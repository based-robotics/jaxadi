import casadi as cs

from jaxadi import lower, declare
from jaxadi import graph_translate as translate

# define input variables for the function
x = cs.SX.sym("x", 2, 1)
y = cs.SX.sym("y", 2, 1)
casadi_fn = cs.Function("myfunc", [x, y], [x.T @ y])

print("Signature of the CasADi function:")
print(casadi_fn)

# define jax function from casadi one
jax_fn = declare(translate(casadi_fn))

print(translate(casadi_fn))

print("Lowered JAX function:")
print(lower(jax_fn, casadi_fn).as_text())
