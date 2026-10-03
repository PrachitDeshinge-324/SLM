import re
import signal
import sys
import threading
from fractions import Fraction
from functools import lru_cache

_SYMPY_TIMEOUT_SEC = 2


class _Timeout(Exception):
    pass


def normalize_answer(value) -> str:
    """
    Canonical string for an extracted answer, used both as the voting key and as a
    fast-path for equality. Collapses '18' / '18.00' / '$18' and
    '\\dfrac{1}{2}' / '\\frac12' style variants so identical answers share one vote.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return text

    text = text.replace('\\$', '').replace('$', '')
    text = re.sub(r'\\text\{\s*([^{}]*)\}', r'\1', text)  # keep text content, drop wrapper
    text = re.sub(r'\\(?:left|right|!|,|;|:|quad|qquad)', '', text)
    text = text.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    text = text.replace('^\\circ', '').replace('^{\\circ}', '').replace('°', '')
    text = text.replace('\\%', '').replace('%', '')
    text = re.sub(r'\\frac\s*(\d)\s*(\d)', r'\\frac{\1}{\2}', text)  # \frac12 -> \frac{1}{2}
    text = re.sub(r'\s+', '', text).rstrip('.')

    # Pure number -> canonical Fraction string ('18.00' -> '18', '0.5' -> '1/2').
    candidate = text.replace(',', '')
    try:
        frac = Fraction(candidate)
        return str(frac)
    except (ValueError, ZeroDivisionError):
        pass
    m = re.fullmatch(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', text)
    if m and int(m.group(2)) != 0:
        return str(Fraction(int(m.group(1)), int(m.group(2))))
    return text.casefold() if len(text) == 1 else text


def _split_top_level(body: str):
    parts, depth, cur = [], 0, ''
    for ch in body:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts


def _as_sequence(text: str):
    """('(', [elements], ')') for tuples/intervals like (3,\\frac{\\pi}{2}), else None."""
    if len(text) >= 2 and text[0] in '([' and text[-1] in ')]':
        parts = _split_top_level(text[1:-1])
        if len(parts) > 1:
            return text[0] + text[-1], parts
    return None


def _symbolic_equal(left: str, right: str) -> bool:
    """SymPy equality with a hard timeout so one pathological expression cannot hang a run."""
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor,
        )
    except ImportError:
        print("Warning: sympy not installed. Symbolic math fallback disabled.", file=sys.stderr)
        return False

    def clean_latex(val):
        val = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', r'((\1)/(\2))', val)
        val = val.replace('{', '(').replace('}', ')')
        val = val.replace('^', '**')
        val = val.replace('\\sqrt', 'sqrt').replace('\\pi', 'pi')
        val = val.replace('\\cdot', '*').replace('\\times', '*')
        return val

    use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, 'setitimer')

    def _raise(*_):
        raise _Timeout()

    old = None
    try:
        if use_alarm:
            old = signal.signal(signal.SIGALRM, _raise)
            signal.setitimer(signal.ITIMER_REAL, _SYMPY_TIMEOUT_SEC)
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        a = parse_expr(clean_latex(left), transformations=transformations, evaluate=False)
        b = parse_expr(clean_latex(right), transformations=transformations, evaluate=False)
        return sympy.simplify(a - b) == 0
    except BaseException:  # malformed/unsupported/timeout
        return False
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


@lru_cache(maxsize=200_000)
def _match_cached(predicted: str, expected: str) -> bool:
    left, right = normalize_answer(predicted), normalize_answer(expected)

    # 1. Exact match after normalization (choice letters, numbers, simple LaTeX)
    if left.casefold() == right.casefold():
        return True

    # 2. Tuples / intervals: compare bracket type and each element
    seq_l, seq_r = _as_sequence(left), _as_sequence(right)
    if seq_l and seq_r:
        (br_l, parts_l), (br_r, parts_r) = seq_l, seq_r
        return (br_l == br_r and len(parts_l) == len(parts_r)
                and all(_match_cached(a, b) for a, b in zip(parts_l, parts_r)))
    if seq_l or seq_r:
        return False

    # 3. Mixed numbers ('1 1/2') on the raw strings
    def as_fraction(value):
        value = value.replace(',', '').replace('$', '').strip()
        mixed = re.fullmatch(r'([+-]?\d+)\s+(\d+)\s*/\s*(\d+)', value)
        if mixed:
            whole, numerator, denominator = map(int, mixed.groups())
            sign = -1 if whole < 0 else 1
            return Fraction(whole) + sign * Fraction(numerator, denominator)
        return Fraction(value)
    try:
        if as_fraction(predicted) == as_fraction(expected):
            return True
    except (ValueError, ZeroDivisionError):
        pass

    # 4. Symbolic equality only if both sides look like math expressions
    if re.search(r'[\\^()a-zA-Z]', left + right) and len(left) < 200 and len(right) < 200:
        return _symbolic_equal(left, right)
    return False


def answers_match(predicted, expected):
    """
    Compare choice letters exactly, numeric answers via fractions,
    tuples element-wise, and algebraic/complex answers via symbolic math (SymPy,
    with a timeout).
    """
    if predicted is None or expected is None:
        return False
    return _match_cached(str(predicted).strip(), str(expected).strip())
