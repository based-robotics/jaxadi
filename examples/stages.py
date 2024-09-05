import casadi as cs
import jax
import jax.numpy as jnp

from jaxadi import translate

# define input variables for the function
x = cs.SX.sym("x", 2, 2)
y = cs.SX.sym("y", 2, 2)
z = cs.SX.sym("z", 2, 2)
casadi_function = cs.Function("mimo_complex", [x, y, z], [cs.mtimes(x, y), cs.inv(z), x + z])

print("Signature of the CasADi function:")
print(casadi_function)

print("Translated JAX function:")
# secure add_import and add_jit to True to get the complete code
print(translate(casadi_function, add_import=True, add_jit=True))

jx = jnp.array([[1, 1], [2, 2]])
jy = jnp.array([[2, 2], [3, 3]])
jz = jnp.array([[2, 2], [3, 3]])


def evaluate_mimo_complex(*args):
    inputs = args
    outputs = [jnp.zeros(out) for out in [(2, 2), (2, 2), (2, 2)]]
    work = jnp.zeros((10, 1))

    work = work.at[jnp.array([0, 1, 3, 4, 6, 8])].set(
        jnp.array(
            [
                jnp.array([inputs[0][0, 0]]),
                jnp.array([inputs[1][0, 0]]),
                jnp.array([inputs[0][0, 1]]),
                jnp.array([inputs[1][1, 0]]),
                jnp.array([inputs[1][1, 1]]),
                jnp.array([inputs[2][1, 0]]),
            ]
        ).reshape(-1, 1)
    )
    work = work.at[jnp.array([2, 5, 7])].set(
        jnp.array([work[0] * work[1], work[3] * work[4], work[3] * work[6]]).reshape(-1, 1)
    )
    work = work.at[jnp.array([2])].set(jnp.array([work[2] + work[5]]).reshape(-1, 1))
    outputs[0] = outputs[0].at[0, 0].set(work[2][0])
    work = work.at[jnp.array([2, 5])].set(
        jnp.array([jnp.array([inputs[0][1, 0]]), jnp.array([inputs[0][1, 1]])]).reshape(-1, 1)
    )
    work = work.at[jnp.array([1, 4, 6])].set(
        jnp.array([work[2] * work[1], work[5] * work[4], work[5] * work[6]]).reshape(-1, 1)
    )
    work = work.at[jnp.array([1])].set(jnp.array([work[1] + work[4]]).reshape(-1, 1))
    outputs[0] = outputs[0].at[1, 0].set(work[1][0])
    work = work.at[jnp.array([1])].set(jnp.array([jnp.array([inputs[1][0, 1]])]).reshape(-1, 1))
    work = work.at[jnp.array([4])].set(jnp.array([work[0] * work[1]]).reshape(-1, 1))
    work = work.at[jnp.array([4, 1])].set(jnp.array([work[4] + work[7], work[2] * work[1]]).reshape(-1, 1))
    outputs[0] = outputs[0].at[0, 1].set(work[4][0])
    work = work.at[jnp.array([1, 7])].set(jnp.array([work[1] + work[6], jnp.array([inputs[2][0, 1]])]).reshape(-1, 1))
    outputs[0] = outputs[0].at[1, 1].set(work[1][0])
    work = work.at[jnp.array([1, 6, 9, 3])].set(
        jnp.array(
            [jnp.array([inputs[2][1, 1]]), jnp.array([inputs[2][0, 0]]), work[7] * work[8], work[3] + work[7]]
        ).reshape(-1, 1)
    )
    work = work.at[jnp.array([4, 0, 5])].set(
        jnp.array([work[6] * work[1], work[0] + work[6], work[5] + work[1]]).reshape(-1, 1)
    )
    work = work.at[jnp.array([4, 2])].set(jnp.array([work[4] - work[9], work[0][0]]).reshape(-1, 1))
    work = work.at[jnp.array([9, 2])].set(jnp.array([work[1] / work[4], work[2] + work[8]]).reshape(-1, 1))
    outputs[1] = outputs[1].at[0, 0].set(work[9][0])
    work = work.at[jnp.array([9, 2])].set(jnp.array([work[8] / work[4], work[2][0]]).reshape(-1, 1))
    work = work.at[jnp.array([9, 2])].set(jnp.array([-work[9], work[3][0]]).reshape(-1, 1))
    outputs[1] = outputs[1].at[1, 0].set(work[9][0])
    work = work.at[jnp.array([9, 2])].set(jnp.array([work[7] / work[4], work[5][0]]).reshape(-1, 1))
    work = work.at[jnp.array([9, 4])].set(jnp.array([-work[9], work[6] / work[4]]).reshape(-1, 1))
    outputs[1] = outputs[1].at[0, 1].set(work[9][0])
    outputs[1] = outputs[1].at[1, 1].set(work[4][0])
    return outputs


print(evaluate_mimo_complex(jx, jy, jz))
