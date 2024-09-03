import jax.numpy as jnp
import casadi as ca
from jaxadi import compile, declare, lower

# Example Usage:
# Define a CasADi function with known inputs
x = ca.SX.sym("x", 10, 10)
y = ca.SX.sym("y", 10)
casadi_function = ca.Function("cf", [x, y], [x**2 + y**2 + 1])

print(casadi_function)

func = casadi_function

n_instr = func.n_instructions()
n_in = func.n_in()  # number of arguments in the function
n_out = func.n_out()  # number of outputs in the function
nnz_in = [func.size_in(i) for i in range(n_in)]  # get the shape of each input
nnz_out = [func.size_out(i) for i in range(n_out)]  # get the shape of each output
n_w = func.sz_w()

print(f"nnz_in: {nnz_in}")
print(f"nnz_out: {nnz_out}")
print(f"n_w: {n_w}")

exit()

# Define a corresponding JAX fn str
jax_fn = """
def evaluate_cf(inputs):
    outputs = [jnp.zeros(nnz_out_i) for nnz_out_i in [10]]
    work = jnp.zeros(3)

    work = work.at[0].set(inputs[0][0])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[1].set(inputs[1][0])
    work = work.at[1].set(work[1] * work[1])
    work = work.at[0].set(work[0] + work[1])
    work = work.at[1].set(1.0000000000000000)
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[0].set(work[0])
    work = work.at[0].set(inputs[0][1])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][1])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[1].set(work[0])
    work = work.at[0].set(inputs[0][2])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][2])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[2].set(work[0])
    work = work.at[0].set(inputs[0][3])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][3])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[3].set(work[0])
    work = work.at[0].set(inputs[0][4])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][4])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[4].set(work[0])
    work = work.at[0].set(inputs[0][5])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][5])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[5].set(work[0])
    work = work.at[0].set(inputs[0][6])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][6])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[6].set(work[0])
    work = work.at[0].set(inputs[0][7])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][7])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[7].set(work[0])
    work = work.at[0].set(inputs[0][8])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][8])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[8].set(work[0])
    work = work.at[0].set(inputs[0][9])
    work = work.at[0].set(work[0] * work[0])
    work = work.at[2].set(inputs[1][9])
    work = work.at[2].set(work[2] * work[2])
    work = work.at[0].set(work[0] + work[2])
    work = work.at[0].set(work[0] + work[1])
    outputs[0] = outputs[0].at[9].set(work[0])
    return outputs
"""
jax_function = declare(jax_fn)

# To hlo function
hlo_jax = lower(jax_function, casadi_function)
print(hlo_jax.as_text())

# Cache and compile the JAX function
compiled_jax_function = compile(jax_function, casadi_function)

# Now you can use compiled_jax_function with actual JAX arrays
x_jax = jnp.ones(10, dtype=jnp.float32)
y_jax = jnp.ones(10, dtype=jnp.float32)
print("Call after compilation")
result = compiled_jax_function([x_jax, y_jax])
# result = compiled_jax_function(x_jax, y_jax)
# result = compiled_jax_function(x_jax, y_jax)

print(result)
