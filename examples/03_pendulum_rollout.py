import timeit
import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
from jaxadi import convert

# Static parameters
dt = 0.02
g = 9.81  # Acceleration due to gravity
L = 1.0  # Length of the pendulum
b = 0.1  # Damping coefficient
I = 1.0  # Moment of inertia
# Test parameters
batch_size = 4096
timesteps = 100


# Define the uncontrolled pendulum model in CasADi
def casadi_pendulum_model():
    state = ca.SX.sym("state", 2)
    theta, omega = state[0], state[1]

    theta_dot = omega
    omega_dot = (-b * omega - (g / L) * ca.sin(theta)) / I

    next_theta = theta + theta_dot * dt
    next_omega = omega + omega_dot * dt

    next_state = ca.vertcat(next_theta, next_omega)
    return ca.Function("pendulum_model", [state], [next_state])


# Create CasADi function
casadi_model = casadi_pendulum_model()

# Convert CasADi function to JAX
jax_model = convert(casadi_model, compile=True)


# Function to generate random inputs
def generate_random_inputs(batch_size):
    return np.random.uniform(-np.pi, np.pi, (batch_size, 2))


# CasADi: Sequential Evaluation
def casadi_sequential_rollout(initial_states):
    batch_size = initial_states.shape[0]
    rollout_states = np.zeros((timesteps + 1, batch_size, 2))

    rollout_states[0] = initial_states
    for t in range(1, timesteps + 1):
        rollout_states[t] = np.array([casadi_model(state).full().flatten() for state in rollout_states[t - 1]])

    return rollout_states


# JAX: Vectorized Evaluation
@jax.jit
def jax_vectorized_rollout(initial_states):
    def single_step(state):
        return jnp.array(jax_model(state)).reshape(
            2,
        )

    def scan_fn(carry, _):
        next_state = jax.vmap(single_step)(carry)
        return next_state, next_state

    _, rollout_states = jax.lax.scan(scan_fn, initial_states, None, length=timesteps)
    return jnp.concatenate([initial_states[None, ...], rollout_states], axis=0)


# Generate random inputs
initial_states = generate_random_inputs(batch_size)

# Warm-up call for JAX
print("Performing warm-up call for JAX...")
_ = jax_vectorized_rollout(initial_states)
print("Warm-up call completed.")
# Performance comparison
print("\nPerformance comparison:")
# Generate new random inputs
initial_states = generate_random_inputs(batch_size)

print(f"CasADi sequential rollout ({batch_size} pendulums, {timesteps} timesteps):")
casadi_time = timeit.timeit(lambda: casadi_sequential_rollout(initial_states), number=1)
print(f"Time: {casadi_time:.4f} seconds")

print(f"\nJAX vectorized rollout ({batch_size} pendulums, {timesteps} timesteps):")
jax_time = timeit.timeit(lambda: np.array(jax_vectorized_rollout(initial_states)), number=1)
print(f"Time: {jax_time:.4f} seconds")

print(f"\nSpeedup factor: {casadi_time / jax_time:.2f}x")

# Verify results
print("\nVerifying results:")
casadi_results = casadi_sequential_rollout(initial_states[:10])
jax_results = np.array(jax_vectorized_rollout(initial_states[:10]))

print("First 10 rollouts match:", np.allclose(casadi_results, jax_results, atol=1e-4))
