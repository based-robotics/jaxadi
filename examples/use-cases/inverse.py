import timeit

from jax import jit, vmap
import casadi as cs
import jax.numpy as jnp
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from robot_descriptions import panda_mj_description

from jaxadi import convert

model = pin.buildModelFromMJCF(panda_mj_description.MJCF_PATH)
cmodel = cpin.Model(model)
cdata = cmodel.createData()

q = cs.SX.sym("q", model.nq)
dq = cs.SX.sym("dq", model.nv)
cpin.computeAllTerms(cmodel, cdata, q, dq)
M = cdata.M

# print(M)
# print(cs.inv(M))

fn = cs.Function("fn", [q, dq], [cs.inv(M)])
jax_fn = convert(fn)
vjax_fn = jit(vmap(jax_fn, in_axes=(0, 0)))
num_batches = 100000
# Warmup
vjax_fn(jnp.ones((model.nq, num_batches)), jnp.ones((model.nv, num_batches)))
num_runs = 100000

# Compare times
print("CasADi:")
print(timeit.timeit(lambda: fn(np.ones(model.nq), np.ones(model.nv)), number=num_runs))
print("JAX-ADi:")
print(
    timeit.timeit(
        lambda: vjax_fn(jnp.ones((model.nq, num_batches)), jnp.ones((model.nv, num_batches))),
        number=num_runs // num_batches,
    )
)
