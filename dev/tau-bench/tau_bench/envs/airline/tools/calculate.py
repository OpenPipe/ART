import ast
import operator

def safe_eval(expression):
    """Safely evaluate mathematical expressions"""
    # Whitelist of allowed operations
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def eval_node(node):
        if isinstance(node, ast.Num):  # Numbers
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = eval_node(node.left)
            right = eval_node(node.right)
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = eval_node(node.operand)
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
            return op(operand)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        return eval_node(tree.body)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid expression: {e}")

def calculate(expression):
    """Calculate the result of a mathematical expression safely"""
    try:
        result = safe_eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
