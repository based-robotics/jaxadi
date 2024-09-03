import timeit
import jax
import casadi as ca
import jax.numpy as jnp
import pinocchio as pin
import pinocchio.casadi as cpin
from robot_descriptions.panda_description import URDF_PATH

from jaxadi import convert, translate

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

# Evaluate the function performance
q_val = ca.np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0, 0])
jax_q_val = jnp.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0], [0]])

print("Casadi evaluation:")
print(fk(q_val))
print("JAX evaluation:")
print(jax_fn(jax_q_val))

# pwease do not run, it will take a lot of time
# print("Performance comparison:")
# print("Casadi evaluation:")
# print(timeit.timeit(lambda: fk(q_val), number=100))
#
# print("JAX evaluation:")
# print(timeit.timeit(lambda: jax_fn(jax_q_val), number=100))


# Second part
# Casadi: Sequential Evaluation
N = int(1e7)


def casadi_sequential_evaluation():
    for _ in range(N):
        fk(q_val)


# JAX: Vectorized Evaluation using vmap
jax_q_vals = jnp.tile(jax_q_val, (N, 1, 1))  # Create a batch of 100 inputs
print(jax_q_vals.shape)
jax_fn_vectorized = jax.vmap(jax_fn, in_axes=(1,), out_axes=1)  # Vectorize the function

# Performance comparison
print("Performance comparison:")
print(f"Casadi sequential evaluation ({N} times):")
print(timeit.timeit(casadi_sequential_evaluation, number=1))

print("JAX vectorized evaluation using vmap:")
print(timeit.timeit(lambda: jax_fn_vectorized(jax_q_vals), number=1))
