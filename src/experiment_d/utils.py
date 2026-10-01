import re
import sys
from fractions import Fraction


def answers_match(predicted, expected):
    """
    Compare choice letters exactly, numeric answers via fractions, 
    and algebraic/complex answers via symbolic math (SymPy).
    """
    if predicted is None or expected is None:
        return False
    left, right = str(predicted).strip(), str(expected).strip()
    
    # 1. Exact string match (handles A/B/C/D choices perfectly)
    if left.casefold() == right.casefold():
        return True

    def clean_numeric(value):
        return value.replace(',', '').replace('$', '').strip()

    # 2. Numeric / Fraction evaluation (handles 95% of GSM8K)
    try:
        def as_fraction(value):
            value = clean_numeric(value)
            mixed = re.fullmatch(r'([+-]?\d+)\s+(\d+)\s*/\s*(\d+)', value)
            if mixed:
                whole, numerator, denominator = map(int, mixed.groups())
                sign = -1 if whole < 0 else 1
                return Fraction(whole) + sign * Fraction(numerator, denominator)
            return Fraction(value)
            
        if as_fraction(left) == as_fraction(right):
            return True
    except (ValueError, ZeroDivisionError):
        pass

    # 3. Symbolic Math evaluation (Critical for MATH-500)
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
        
        # Basic heuristic LaTeX cleanup to make it SymPy-friendly
        def clean_latex(val):
            val = re.sub(r'\\text\{.*?\}', '', val)
            val = val.replace('\\$', '').replace('$', '')
            val = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', r'((\1)/(\2))', val)
            val = val.replace('{', '(').replace('}', ')')
            val = val.replace('^', '**')
            val = val.replace('\\sqrt', 'sqrt')
            val = val.replace('\\pi', 'pi')
            val = val.replace('\\cdot', '*')
            val = val.replace('\\times', '*')
            val = val.replace('\\%', '/100')
            return val
            
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        
        expr_left = parse_expr(clean_latex(left), transformations=transformations, evaluate=False)
        expr_right = parse_expr(clean_latex(right), transformations=transformations, evaluate=False)
        
        diff = sympy.simplify(expr_left - expr_right)
        if diff == 0:
            return True
    except ImportError:
        print("Warning: sympy not installed. Symbolic math fallback disabled.", file=sys.stderr)
    except Exception:
        pass # Malformed expression or unsupported format, continue
        
    return False
