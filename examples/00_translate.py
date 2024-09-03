"""
This example demonstrates how to define a mathematical function using CasADi
and translate it into JAX-compatible code using the jaxadi library.
It shows the function's signature in CasADi and its equivalent representation
in JAX, enabling the use of JAX's automatic differentiation and optimization
capabilities for the same function.
"""

import casadi as cs

from jaxadi import translate

# define input variables for the function
x = cs.SX.sym("x", 3)
y = cs.SX.sym("y", 1)
casadi_function = cs.Function("myfunc", [x, y], [x**2 + y**2 - 1])

print("Signature of the CasADi function:")
print(casadi_function)

print("Translated JAX function:")
for cg_str in translate(casadi_function):
    print(cg_str)
