import casadi as cs
from jaxadi import transcribe

x = cs.SX.sym("x")

f = cs.Function("f", [x], [x**2])


# * Write codegen to file
with open("codegen.py", "w") as codegen_file:
    for cg_str in transcribe(f):
        codegen_file.write(cg_str)
