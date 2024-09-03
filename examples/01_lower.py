import casadi as cs

from jaxadi import lower, translate, declare

# define input variables for the function
x = cs.SX.sym("x", 10, 10)
y = cs.SX.sym("y", 10, 10)
casadi_fn = cs.Function("myfunc", [x, y], [x @ y])

print("Signature of the CasADi function:")
print(casadi_fn)

# define jax function from casadi one
jax_fn = declare(translate(casadi_fn))

print("Lowered JAX function:")
print(lower(jax_fn, casadi_fn).as_text())
