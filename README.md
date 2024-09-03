# jaxadi

**jaxadi** is a Python library designed to translate `casadi.Function` into `jax`-compatible functions.

## Installation

```bash
pip3 install -e .

mamba env create -f environment.yml
```

## Usage

```python
from jaxadi import translate, convert, lower

translate(casadi_function) # get jax-compatible function string representation
lower(jax_fn, casadi_fn) # get jax.lower function which is compatible with different platforms for exporting
convert(casad_fn) # get possibly compiled jax function from casadi function
```

## Examples

1. Translate - [example](examples/00_translate.py)
2. Lower - [example](examples/01_lower.py)
3. Convert - [example](examples/02_convert.py)
4. Pinocchio casadi -> jax - [example](examples/03_pinocchio.py)

## References

This work is heavily inspired from [cusadi](https://github.com/se-hwan/cusadi), but simplier to support and focused on jax.
