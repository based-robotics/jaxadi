import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from robot_descriptions.panda_description import URDF_PATH
import jax.numpy as jnp

from jaxadi import translate, convert

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
import timeit

q_val = ca.np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0, 0])
jax_q_val = jnp.array(q_val)

print("Casadi evaluation:")
print(fk(q_val))
print("JAX evaluation:")
print(jax_fn(jax_q_val))

print("Performance comparison:")
print("Casadi evaluation:")
print(timeit.timeit(lambda: fk(q_val), number=100))

print("JAX evaluation:")
print(timeit.timeit(lambda: jax_fn(jax_q_val), number=100))
