"""
This script compares the performance between sequential calls of CasADi
and its converted JAX counterpart. The comparison is made between a sequential
evaluation using CasADi and a vectorized evaluation using JAX for the forward
kinematics of a Panda robot's end-effector.

Disclaimer: Performance results may vary depending on hardware configuration and system load.
For optimal performance, consider installing CUDA-enabled JAX.
"""

import timeit
import jax
import casadi as ca
import jax.numpy as jnp
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from robot_descriptions.panda_description import URDF_PATH
from jaxadi import convert
from jaxadi import graph_translate as translate

# Load the Panda robot model
model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()

# Transfer to casadi model
cmodel = cpin.Model(model)
cdata = cmodel.createData()

# Define the joint configuration
q = ca.SX.sym("q", model.nq)

# Compute the forward kinematics
cpin.framesForwardKinematics(cmodel, cdata, q)

# Create casadi function of forward kinematics for end-effector
omf = cdata.oMf[model.getFrameId("panda_hand_tcp")]
fk = ca.Function("fk", [q], [omf.translation])

# translate the casadi function to jax
print(translate(fk, add_import=True, add_jit=True))

jax_fn = convert(fk, compile=True)


# Function to generate random inputs
def generate_random_inputs(N):
    return np.random.rand(N, model.nq)


# Casadi: Sequential Evaluation
def casadi_sequential_evaluation(q_vals):
    return [fk(q) for q in q_vals]


# JAX: Vectorized Evaluation
jax_fn_vectorized = jax.jit(jax.vmap(jax_fn))  # Vectorize the function

# Evaluate the function performance for a batch
N_test = 1000  # Small number for initial test
q_vals_test = generate_random_inputs(N_test)
jax_q_vals_test = jnp.array(q_vals_test).reshape(N_test, model.nq, 1)  # Create a batch of 1000

print(f"Casadi sequential evaluation ({N_test} times):")
casadi_results_test = np.array(casadi_sequential_evaluation(q_vals_test))[:, :, 0]
print(f"First result: {casadi_results_test[0]}")
print(f"Last result: {casadi_results_test[-1]}")
print(f"Shape: {casadi_results_test.shape}")
print(f"\nJAX vectorized evaluation ({N_test} times):")
jax_results_test = np.array(jax_fn_vectorized(jax_q_vals_test))[0, :, :, 0]
print(f"First result: {jax_results_test[0]}")
print(f"Last result: {jax_results_test[-1]}")
print(f"Shape: {jax_results_test.shape}")

print("\nVerifying initial batch results:")
print("Results match:", np.allclose(casadi_results_test, jax_results_test, atol=1e-6))

# Warm-up call for JAX
print("\nPerforming warm-up call for JAX...")
_ = jax_fn_vectorized(jax_q_vals_test)
print("Warm-up call completed.")

# Performance comparison
print("\nPerformance comparison:")
N = int(1e5)  # Number of evaluations for performance test

# call with same dimensions as target input to avoid re-compiling
q_vals = generate_random_inputs(N)
jax_q_vals = jnp.array(q_vals).reshape(N, model.nq, 1)
np.array(jax_fn_vectorized(jax_q_vals))[0, :, :, 0]

# Generate new random inputs for performance comparison
q_vals = generate_random_inputs(N)
jax_q_vals = jnp.array(q_vals).reshape(N, model.nq, 1)

print(f"Casadi sequential evaluation ({N} times):")
casadi_time = timeit.timeit(lambda: np.array(casadi_sequential_evaluation(q_vals))[:, :, 0], number=1)
print(f"Time: {casadi_time:.4f} seconds")

print(f"\nJAX vectorized evaluation ({N} times):")
jax_time = timeit.timeit(lambda: np.array(jax_fn_vectorized(jax_q_vals))[0, :, :, 0], number=1)
print(f"Time: {jax_time:.4f} seconds")

print(f"\nSpeedup factor: {casadi_time / jax_time:.2f}x")

# Verify results
print("\nVerifying performance test results:")
casadi_results = np.array(casadi_sequential_evaluation(q_vals[:100]))[:, :, 0]
jax_results = np.array(jax_fn_vectorized(jax_q_vals[:100]))[0, :, :, 0]
print("First 100 results match:", np.allclose(casadi_results, jax_results, atol=1e-6))
