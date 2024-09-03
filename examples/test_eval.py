import jax.numpy as jnp

from codegen import evaluate_cf
from jaxadi import compile
import casadi as cs

result = evaluate_cf(jnp.array([1.0, 2.0, 3.0]).reshape(3, 1), jnp.array([[2.0]]))
print(result)

x = cs.SX.sym("x", 3)
y = cs.SX.sym("y", 1)
casadi_function = cs.Function("cf", [x, y], [x**2 + y**2 + 1])
print(casadi_function)

something = compile(evaluate_cf, casadi_function)

print(evaluate_cf(jnp.array([1.0, 2.0, 3.0]).reshape(3, 1), jnp.array([[2.0]])))
print(evaluate_cf(jnp.array([1.0, 2.0, 3.0]).reshape(3, 1), jnp.array([[2.0]])))
print(evaluate_cf(jnp.array([1.0, 2.0, 3.0]).reshape(3, 1), jnp.array([[2.0]])))
print(evaluate_cf(jnp.array([1.0, 2.0, 3.0]).reshape(3, 1), jnp.array([[2.0]])))
