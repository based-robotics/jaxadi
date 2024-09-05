"""
This example demonstrates how to define a mathematical function using CasADi
and translate it into JAX-compatible code using the jaxadi library.
It shows the function's signature in CasADi and its equivalent representation
in JAX, enabling the use of JAX's automatic differentiation and optimization
capabilities for the same function.
"""

import casadi as cs

from jaxadi import translate, legacy_translate

# define input variables for the function
x = cs.SX.sym("x", 30, 30)
y = cs.SX.sym("y", 30, 30)
casadi_function = cs.Function("myfunc", [x, y], [x @ y])

print("Signature of the CasADi function:")
print(casadi_function)

print("Translated JAX function:")
# secure add_import and add_jit to True to get the complete code
print("Legacy translation:")
tr2 = legacy_translate(casadi_function)
print("New translation:")
tr1 = translate(casadi_function)

nl1 = len(tr1.split("\n"))
nl2 = len(tr2.split("\n"))

print(f"New implementation is {nl1 / nl2}x of old in length")
print(f"Total number of lines: New {nl1}, Old {nl2}")
