import casadi as cs
from jaxadi import translate

# x = cs.SX.sym("x", 2)

# f = cs.Function("f", [x], [x[1]**2, x[0], x[1] + x[0]])
x = cs.SX.sym("x", 3)
y = cs.SX.sym("y", 1)
casadi_function = cs.Function("cf", [x, y], [x**2 + y**2 + 1])
print(casadi_function)


# * Write codegen to file
with open("codegen.py", "w") as codegen_file:
    for cg_str in translate(casadi_function):
        codegen_file.write(cg_str)
