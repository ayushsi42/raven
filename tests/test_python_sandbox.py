"""Security and functional tests for `PythonSandbox`.

These tests exist because `PythonSandbox` previously ran LLM-generated
code with `exec()` against the real interpreter `__builtins__`: no
allowlist, no timeout. That meant a malformed or adversarial model
completion could read/write arbitrary files, shell out, or hang a
pipeline stage forever. This file proves the hardened sandbox (a)
still supports the numeric/data-munging work the analysis pipeline
actually needs, and (b) blocks or times out the concrete attack shapes
called out in the roadmap.
"""
from __future__ import annotations

import time

import pytest

from hypothesis_agent.orchestration.langgraph_pipeline import SAFE_BUILTINS
from hypothesis_agent.orchestration.python_sandbox import (
    PythonSandbox,
    SandboxTimeoutError,
)


# ---------------------------------------------------------------------------
# (a) Allowed operations still work
# ---------------------------------------------------------------------------


def test_basic_numeric_analysis_still_works() -> None:
    """Arithmetic, comprehensions, and stdlib helpers the pipeline needs."""

    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    code = """
values = [1.0, 2.0, 3.0, 4.0, 5.0]
total = sum(values)
average = total / len(values)
squared = [v ** 2 for v in values]
print(f"total={total} average={average} max={max(squared)}")
"""
    output = sandbox.run(code)
    assert "total=15.0" in output
    assert "average=3.0" in output
    assert "max=25.0" in output


def test_pandas_style_tabular_work_via_preamble_helpers() -> None:
    """Mimics the real pipeline: trusted preamble imports json/statistics,
    binds loader/writer helpers, then untrusted code consumes them."""

    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    preamble = """
import json
import statistics

records = [
    {"date": "2024-01-31", "close": 100.0},
    {"date": "2024-02-29", "close": 102.0},
    {"date": "2024-03-31", "close": 104.5},
]

def load_records():
    return records
"""
    sandbox.run_trusted(preamble)

    code = """
rows = load_records()
closes = [row["close"] for row in rows]
mean_close = statistics.mean(closes)
growth = (closes[-1] - closes[0]) / closes[0]
result = {"mean_close": mean_close, "growth": round(growth, 4)}
print("RESULT::" + json.dumps(result))
"""
    output = sandbox.run(code)
    assert "RESULT::" in output
    assert '"mean_close": 102.16666666666667' in output


def test_run_safe_reports_success_with_no_error() -> None:
    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    stdout, stderr, error = sandbox.run_safe("print('ok')")
    assert stdout.strip() == "ok"
    assert stderr == ""
    assert error is None


def test_state_persists_across_calls() -> None:
    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    sandbox.run("x = 41")
    output = sandbox.run("print(x + 1)")
    assert output.strip() == "42"


def test_run_trusted_can_import_for_setup_but_run_cannot() -> None:
    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    # Trusted setup code may import stdlib modules.
    sandbox.run_trusted("import math\nSQRT2 = math.sqrt(2)")
    output = sandbox.run("print(round(SQRT2, 3))")
    assert output.strip() == "1.414"


# ---------------------------------------------------------------------------
# (b) Malicious / dangerous operations are blocked
# ---------------------------------------------------------------------------


def test_dunder_import_os_system_is_blocked() -> None:
    """The classic sandbox-escape one-liner must not execute.

    `__import__` itself is present (guarded) so that legitimate
    `import json`/`import pandas` in generated code still works, but
    `os` is not on the allowlist, so this must still raise.
    """

    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    with pytest.raises(ImportError):
        sandbox.run("__import__('os').system('echo pwned')")


def test_import_statement_is_blocked_for_disallowed_modules() -> None:
    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    with pytest.raises(ImportError):
        sandbox.run("import os\nos.system('echo pwned')")
    with pytest.raises(ImportError):
        sandbox.run("import subprocess")
    with pytest.raises(ImportError):
        sandbox.run("import sys")


def test_import_statement_is_allowed_for_analysis_modules() -> None:
    """Real LLM-generated code routinely does `import json`/`import pandas
    as pd` at the top of its own snippet even though those names are
    already bound by the trusted preamble; this must keep working."""

    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    output = sandbox.run(
        "import json\nimport pandas as pd\n"
        "df = pd.DataFrame({'x': [1, 2, 3]})\n"
        "print('RESULT::' + json.dumps({'sum': int(df['x'].sum())}))"
    )
    assert '"sum": 6' in output


def test_open_builtin_is_blocked() -> None:
    """Filesystem access via the bare `open` builtin must be unavailable."""

    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    stdout, stderr, error = sandbox.run_safe("open('/etc/passwd').read()")
    assert error is not None
    assert "NameError" in error


def test_eval_builtin_is_blocked() -> None:
    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    stdout, stderr, error = sandbox.run_safe("eval('1+1')")
    assert error is not None
    assert "NameError" in error


def test_exec_builtin_is_blocked() -> None:
    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    stdout, stderr, error = sandbox.run_safe("exec('x = 1')")
    assert error is not None
    assert "NameError" in error


def test_compile_builtin_is_blocked() -> None:
    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    stdout, stderr, error = sandbox.run_safe("compile('1', '<s>', 'eval')")
    assert error is not None
    assert "NameError" in error


def test_network_module_unreachable_without_import() -> None:
    """With `import` blocked, `socket`/`urllib`/etc. can never be reached."""

    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS)
    stdout, stderr, error = sandbox.run_safe(
        "import socket\nsocket.socket().connect(('example.com', 80))"
    )
    assert error is not None
    # `import` compiles to a call to the (absent) `__import__` builtin.
    assert "ImportError" in error or "NameError" in error


#: A runaway loop used to exercise the timeout path. Deliberately finite
#: (rather than `while True: pass`) so the daemon thread the sandbox
#: abandons on timeout (Python cannot forcibly kill a running thread)
#: finishes on its own after a few seconds instead of spinning forever
#: and stealing CPU from the rest of the test suite.
_RUNAWAY_LOOP = "i = 0\nwhile i < 30_000_000:\n    i += 1\n"


def test_infinite_loop_times_out() -> None:
    """A runaway LLM-generated loop must not hang the pipeline forever."""

    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS, timeout=0.3)
    started = time.monotonic()
    with pytest.raises(SandboxTimeoutError):
        sandbox.run(_RUNAWAY_LOOP)
    elapsed = time.monotonic() - started
    # Should abort close to the configured timeout, not wait for the loop.
    assert elapsed < 3.0


def test_infinite_loop_times_out_via_run_safe() -> None:
    sandbox = PythonSandbox(allowed_builtins=SAFE_BUILTINS, timeout=0.3)
    stdout, stderr, error = sandbox.run_safe(_RUNAWAY_LOOP)
    assert error is not None
    assert "time limit" in error.lower() or "timeout" in error.lower()


def test_default_sandbox_uses_restricted_builtins_when_unspecified() -> None:
    """Security must be the default, not opt-in: `PythonSandbox()` with no
    args must not fall back to real interpreter builtins."""

    sandbox = PythonSandbox()
    with pytest.raises(Exception):
        sandbox.run("open('/etc/passwd')")
