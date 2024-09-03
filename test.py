
import jax.numpy as jnp
import casadi as ca
from jaxadi import compile, declare, lower

# Example Usage:
# Define a CasADi function with known inputs
x = ca.SX.sym('x', 10)
y = ca.SX.sym('y', 10)
casadi_function = ca.Function('cf', [x, y], [x**2 + y**2 + 1])

# Define a corresponding JAX fn str
jax_fn = """
def jax_function(x, y):
    print('cached')
    return x**2 + y**2 + 1"""
jax_function = declare(jax_fn)

# To hlo function
hlo_jax = lower(jax_function, casadi_function)
print(hlo_jax.as_text())

# Cache and compile the JAX function
compiled_jax_function = compile(jax_function, casadi_function)

# Now you can use compiled_jax_function with actual JAX arrays
x_jax = jnp.ones(10, dtype=jnp.float32)
y_jax = jnp.ones(10, dtype=jnp.float32)
print('Call after compilation')
result = compiled_jax_function(x_jax, y_jax)
result = compiled_jax_function(x_jax, y_jax)
result = compiled_jax_function(x_jax, y_jax)

print(result)
