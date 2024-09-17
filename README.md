<!-- # JaxADi -->

[![CI](https://img.shields.io/github/actions/workflow/status/based-robotics/jaxadi/build.yaml?branch=master)](https://github.com/based-robotics/jaxadi/actions)
[![PyPI version](https://img.shields.io/pypi/v/jaxadi?color=blue)](https://pypi.org/project/jaxadi/)
[![PyPI downloads](https://img.shields.io/pypi/dm/jaxadi?color=blue)](https://pypistats.org/packages/jaxadi)

<p align="center">
  <!-- Placeholder for a cool logo -->
  <a href="https://based-robotics.github.io/jaxadi/">
  <img src="https://github.com/based-robotics/jaxadi/blob/master/_assets/_logo.png?raw=true" alt="JAXADI Logo" width="400"/></a>
</p>

**JaxADi** is a powerful Python library designed to bridge the gap between `casadi.Function` and JAX-compatible functions. By leveraging the strengths of both CasADi and JAX, JAXADI opens up exciting opportunities for building highly efficient, batchable code that can be executed seamlessly across CPUs, GPUs, and TPUs.

JAXADI can be particularly useful in scenarios involving:

- Robotics simulations
- Optimal control problems
- Machine learning models with complex dynamics

## Installation

You can install JAXADI using pip:

<!-- Change once it will be realeased -->

```bash
pip install jaxadi
```

For a complete environment setup for examples, we recommend using Conda/Mamba:

```bash
mamba env create -f environment.yml
```

## Usage

JAXADI provides a simple and intuitive API:

```python
import casadi as cs
import numpy as np
from jaxadi import translate, convert
from jax import numpy as jnp

x = cs.SX.sym("x", 2, 2)
y = cs.SX.sym("y", 2, 2)
# Define a complex nonlinear function
z = x @ y  # Matrix multiplication
z_squared = z * z  # Element-wise squaring
z_sin = cs.sin(z)  # Element-wise sine
result = z_squared + z_sin  # Element-wise addition
# Create the CasADi function
casadi_fn = cs.Function("complex_nonlinear_func", [x, y], [result])
# Get JAX-compatible function string representation
jax_fn_string = translate(casadi_fn)
print(jax_fn_string)
# Define JAX function from CasADi one
jax_fn = convert(casadi_fn, compile=True)
# Run compiled function
input_x = jnp.array(np.random.rand(2, 2))
input_y = jnp.array(np.random.rand(2, 2))
output = jax_fn(input_x, input_y)

```

<strong>Note:</strong> For now translation does not support functions with very
large number of operations, due to the translation implementation. Secret component of
translation is work-tree expansion, which might lead to large overhead in number of
symbols. We are working on finding the compromise in both speed and extensive
functions support.

## Examples

JAXADI comes with several examples to help you get started:

1. [Basic Translation](examples/00_translate.py): Learn how to translate CasADi functions to JAX.

2. [Lowering Operations](examples/01_lower.py): Understand the lowering process in JaxADi.

3. [Function Conversion](examples/02_convert.py): See how to fully convert CasADi functions to JAX.

4. [Pendulum Rollout](examples/03_pendulum_rollout.py): Batched rollout of the nonlinear passive nonlinear pendulum

5. [Pinocchio Integration](examples/04_pinocchio.py): Explore how to convert Pinocchio-based CasADi functions to JAX.

6. [MJX Comparison](examples/05_mjx.py): Compare the transformed Pinnocchio forward kinematics with one provided by Mujoco MJX

> **Note**: To run the Pinocchio and MJX examples, ensure you have them properly installed in your environment.

## Performance Benchmarks

![speedup](https://github.com/based-robotics/jaxadi/blob/master/docs/static/images/speedup_ratio.png?raw=true)

The process of benchmarking and evaluating the performance of Jaxadi is described in the [benchmarks](benchmarks/README.md) directory.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## Citation

If you use JaxADi in your research, please cite it as follows:

```bibtex
@misc{jaxadi2024,
  title = {JaxADi: Bridging CasADi and JAX for Efficient Numerical Computing},
  author = {Alentev, Igor and Kozlov, Lev and Nedelchev, Simeon},
  year = {2024},
  url = {https://github.com/based-robotics/jaxadi},
  note = {Accessed: [Insert Access Date]}
}
```

## Acknowledgements

This project draws inspiration from [cusadi](https://github.com/se-hwan/cusadi), with a focus on simplicity and JAX integration.

## Contact

For questions, issues, or suggestions, please [open an issue](https://github.com/based-robotics/jaxadi/issues) on our GitHub repository.

We hope JAXADI empowers your numerical computing and optimization tasks! Happy coding!
