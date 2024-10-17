import casadi as cs
from visast import generate, visualise

from jaxadi import translate, legacy_translate
import jax

A = cs.SX.sym("A", 2, 2)
q = cs.SX.sym("q", 2)

fn = cs.Function("large_mult", [A, q], [q.T @ A @ q])

jax_fn_equiv = """
@jax.jit
def jax_fn_equiv(A, q):
    return q.T @ A @ q.T
"""

jax_fn_text = translate(fn)
jax_fn_text_legacy = legacy_translate(fn)
jax_fn_ast = generate.fromString(jax_fn_text)
jax_fn_ast_ideal = generate.fromString(jax_fn_equiv)
jax_fn_ast_legacy = generate.fromString(jax_fn_text_legacy)
visualise.graph(jax_fn_ast)
visualise.graph(jax_fn_ast_legacy)
visualise.graph(jax_fn_ast_ideal)
