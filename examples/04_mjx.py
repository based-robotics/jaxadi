import timeit
import jax
import casadi as ca
import jax.numpy as jnp
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from robot_descriptions.iiwa14_mj_description import MJCF_PATH
from jaxadi import convert, translate
import mujoco
import mujoco.mjx as mjx

# iiwa_mj_description

# Load the Panda robot model
model = pin.buildModelFromMJCF(MJCF_PATH)
data = model.createData()

# Transfer to casadi model
cmodel = cpin.Model(model)
cdata = cmodel.createData()

# Define the joint configuration
q = ca.SX.sym("q", model.nq)

# Compute the forward kinematics
cpin.framesForwardKinematics(cmodel, cdata, q)

# Create casadi function of forward kinematics for end-effector
omf = cdata.oMf[model.getFrameId("link7")]
fk = ca.Function("fk", [q], [omf.translation])

# translate the casadi function to jax
# print(translate(fk, add_import=True, add_jit=True))

jax_fn = convert(fk, compile=True)

# Function to generate random inputs
def generate_random_inputs(N):
    return np.random.rand(N, model.nq)

# Casadi: Sequential Evaluation
def casadi_sequential_evaluation(q_vals):
    return [fk(q) for q in q_vals]

# JAX: Vectorized Evaluation
jax_fn_vectorized = jax.jit(jax.vmap(jax_fn))  # Vectorize the function

# Create MJX forward kinematics
mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
mj_model.opt.integrator = 1
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)

@jax.jit
def mjx_fk(joint_pos):
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(qpos=joint_pos)
    mjx_data = mjx.fwd_position(mjx_model, mjx_data)
    return mjx_data.xpos[-1]  # Assuming the last body is the end-effector

mjx_fn_vectorized = jax.jit(jax.vmap(mjx_fk))  # Corrected: JIT-compiled vectorized function

# Evaluate the function performance for a batch
N_test = 100  # Small number for initial test
q_vals_test = generate_random_inputs(N_test)
jax_q_vals_test = jnp.array(q_vals_test).reshape(N_test, model.nq, 1)  # Create a batch of 100
mjx_q_vals_test = jnp.array(q_vals_test)

print(f"Casadi evaluation (batch of {N_test}):")
casadi_results_test = np.array(casadi_sequential_evaluation(q_vals_test))[:,:,0]
print(f"First result: {casadi_results_test[0]}")
print(f"Last result: {casadi_results_test[-1]}")
print(f"Shape: {casadi_results_test.shape}")

print(f"\nJAX evaluation (batch of {N_test}):")
jax_results_test = np.array(jax_fn_vectorized(jax_q_vals_test))[0,:,:,0]
print(f"First result: {jax_results_test[0]}")
print(f"Last result: {jax_results_test[-1]}")
print(f"Shape: {jax_results_test.shape}")

print(f"\nMJX evaluation (batch of {N_test}):")
mjx_results_test = np.array(mjx_fn_vectorized(mjx_q_vals_test))
print(f"First result: {mjx_results_test[0]}")
print(f"Last result: {mjx_results_test[-1]}")
print(f"Shape: {mjx_results_test.shape}")

print("\nVerifying initial batch results:")
print("JAX and Casadi results match:", np.allclose(casadi_results_test, jax_results_test, atol=1e-6))
print("MJX and Casadi results match:", np.allclose(casadi_results_test, mjx_results_test, atol=1e-6))


# Performance comparison
print("\nPerformance comparison:")
N = int(1e5)  # Number of evaluations for performance test

# call with same dimensions as target input to avoid re-compiling
q_vals = generate_random_inputs(N)
jax_q_vals = jnp.array(q_vals).reshape(N, model.nq, 1)
mjx_q_vals = jnp.array(q_vals)
np.array(jax_fn_vectorized(jax_q_vals))
np.array(mjx_fn_vectorized(mjx_q_vals))

# Generate new random inputs for performance comparison
q_vals = generate_random_inputs(N)
jax_q_vals = jnp.array(q_vals).reshape(N, model.nq, 1)
mjx_q_vals = jnp.array(q_vals)


print(f"Casadi sequential evaluation ({N} times):")
casadi_time = timeit.timeit(lambda: np.array(casadi_sequential_evaluation(q_vals))[:,:,0], number=1)
print(f"Time: {casadi_time:.4f} seconds")

print(f"\nJAX vectorized evaluation ({N} times):")
jax_time = timeit.timeit(lambda: np.array(jax_fn_vectorized(jax_q_vals))[0,:,:,0], number=1)
print(f"Time: {jax_time:.4f} seconds")

print(f"\nMJX vectorized evaluation ({N} times):")
mjx_time = timeit.timeit(lambda: np.array(mjx_fn_vectorized(mjx_q_vals)), number=1)
print(f"Time: {mjx_time:.4f} seconds")

print(f"\nSpeedup factors:")
print(f"Generated JAX vs Casadi: {casadi_time / jax_time:.2f}x")
print(f"MJX vs Casadi: {casadi_time / mjx_time:.2f}x")
print(f"MJX vs Generated JAX: {jax_time / mjx_time:.2f}x")

# Verify results
print("\nVerifying performance test results:")
casadi_results = np.array(casadi_sequential_evaluation(q_vals[:10]))[:,:,0]
jax_results = np.array(jax_fn_vectorized(jax_q_vals[:10]))[0,:,:,0]
mjx_results = np.array(mjx_fn_vectorized(mjx_q_vals[:10]))
print("First 10 JAX and Casadi results match:", np.allclose(casadi_results, jax_results, atol=1e-6))
print("First 10 MJX and Casadi results match:", np.allclose(casadi_results, mjx_results, atol=1e-6))
