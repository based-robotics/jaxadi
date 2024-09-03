import casadi as cs

from jaxadi import translate

# define input variables for the function
x = cs.SX.sym("x", 3)
y = cs.SX.sym("y", 1)
casadi_function = cs.Function("myfunc", [x, y], [x**2 + y**2 - 1])

print("Signature of the casadi function:")
print(casadi_function)

print("Transcribed code:")
for cg_str in translate(casadi_function):
    print(cg_str)
