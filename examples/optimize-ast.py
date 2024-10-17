import ast
import numpy as np


class MatrixOptimizer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        # Detect if we are working with multiplications and additions
        if isinstance(node.op, ast.Add):
            left = self.visit(node.left)
            right = self.visit(node.right)
            # If both sides are multiplications, return a matrix operation
            if isinstance(left, ast.BinOp) and isinstance(right, ast.BinOp):
                return self.merge_to_matrix(node)
        return self.generic_visit(node)

    def merge_to_matrix(self, node):
        # Replace detected subtree with a matrix expression
        q = np.array([1, 2])  # Example q vector
        A = np.array([[1, 2], [3, 4]])  # Example A matrix
        result = q.T @ A @ q
        print(f"Compressed to: {result}")
        return ast.Constant(value=result)


# Example code AST (simulated distributed computation)
code = """
w0 = q[0] * A[0][0] + q[1] * A[0][1]
w1 = q[0] * A[1][0] + q[1] * A[1][1]
result = w0 * q[0] + w1 * q[1]
"""
tree = ast.parse(code)

# Optimize AST
optimizer = MatrixOptimizer()
optimized_tree = optimizer.visit(tree)

# Display the optimized code
optimized_code = ast.unparse(optimized_tree)
print(optimized_code)
