"""Tests for program_execution: extracting and running the solve() programs of the 'program' prompt variant."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_d.program_execution import execute_many, execute_response, extract_program, format_number


def fenced(body):
    return f"Here is the code:\n```python\n{body}\n```\nDone."


def status(text, **kw):
    return execute_response(text, **kw)["status"]


# extraction: last fenced block with solve(); unfenced or cut-off answers from 'def solve' on
assert extract_program("no code here") is None
assert extract_program("```python\nx = 1\n```") is None
two = "```python\ndef solve():\n    return 1\n```\nbetter:\n```python\ndef solve():\n    return 2\n```"
assert execute_response(two)["answer"] == "2"
assert execute_response("Sure.\ndef solve():\n    a = 3\n    return a * 4")["answer"] == "12"      # unfenced
assert execute_response("1. code:\n    def solve():\n        return 5\n")["answer"] == "5"          # indented

# numbers come back in the prose extractor's spelling
assert format_number(18.0) == "18" and format_number(17.9999999) == "18" and format_number(18.5) == "18.5"
assert format_number(7) == "7" and format_number(0.1 + 0.2) == "0.3" and format_number(-2.25) == "-2.25"
assert execute_response(fenced("def solve():\n    return 48 + 48 / 2"))["answer"] == "72"
assert execute_response(fenced("def solve():\n    return 10 ** 30"))["answer"] == str(10 ** 30)       # big ints exact
assert execute_response(fenced("import math\ndef solve():\n    return math.floor(7.5)"))["answer"] == "7"
assert execute_response(fenced("from fractions import Fraction\ndef solve():\n    return Fraction(3, 2)"))["answer"] == "1.5"
assert execute_response(fenced("def solve():\n    return '1,250'"))["answer"] == "1250"
assert execute_response(fenced("def solve():\n    print('noise 99')\n    return 4"))["answer"] == "4"   # prints ignored

# failures are classified, never a number
assert status("nothing") == "no_code"
assert status(fenced("def solve(:\n    return 1")) == "syntax_error"
assert status(fenced("def solve():\n    return 1 / 0")) == "runtime_error"
assert status(fenced("def solve():\n    return undefined_name")) == "runtime_error"
assert status(fenced("def helper():\n    return 1")) == "no_code"
assert status("```python\ndef solve():\n    pass\nsolve = 3\n```") == "runtime_error"                # not callable
for blocked in ("import os\ndef solve():\n    return 1",
                "def solve():\n    open('x.txt', 'w').write('a')\n    return 1",
                "def solve():\n    return eval('1+1')",
                "import subprocess\ndef solve():\n    return 1"):
    assert status(fenced(blocked)) == "runtime_error", blocked
for bad in ("return [1, 2]", "return None", "return True", "return float('nan')", "return 'many'"):
    assert status(fenced(f"def solve():\n    {bad}")) == "non_numeric", bad

start = time.time()
assert status(fenced("def solve():\n    while True:\n        pass"), timeout=0.5) == "timeout"
assert time.time() - start < 5

# many responses keep their order
texts = [fenced(f"def solve():\n    return {i} * 2") for i in range(12)] + ["no code"]
results = execute_many(texts, workers=4)
assert [r["answer"] for r in results[:12]] == [str(2 * i) for i in range(12)] and results[12]["status"] == "no_code"

print("All program_execution tests passed!")
