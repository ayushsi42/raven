"""Flexible Python code execution sandbox for analysis tasks.

Security model
---------------
`PythonSandbox` is used to execute *LLM-generated* Python code (the
"hybrid analysis" REPL loop in ``langgraph_pipeline.py``). That code is
untrusted: it is model output, not something a developer wrote, so it
must not be able to touch the filesystem, the network, or the process
in ways the pipeline didn't explicitly sanction.

By default the sandbox therefore executes with a restricted
``__builtins__`` allowlist (see :data:`DEFAULT_SAFE_BUILTINS`) instead
of the real interpreter builtins. `open`, `eval`, `exec`, `compile`,
`input`, and friends are simply absent from that allowlist, so they all
raise ``NameError`` rather than running. This is defense-in-depth, not
a hard security boundary (pure-Python objects can still be walked via
attribute access), but it closes off the obvious escape hatches an LLM
completion might reach for.

`__import__` is not removed outright: in practice, LLM-generated
analysis code routinely writes `import json` / `import pandas as pd`
at the top of its own snippet (even when those names are already bound
by the trusted preamble), and simply raising on any `import` broke the
real pipeline in end-to-end testing. Instead `__import__` is replaced
with a *guarded* import (see :data:`DEFAULT_ALLOWED_IMPORT_MODULES` /
:func:`make_guarded_import`) that only allows a fixed allowlist of
pure-computation/data modules; `import os`, `import sys`,
`import subprocess`, `import socket`, etc. still raise `ImportError`.

Execution also runs under a wall-clock timeout (`timeout`, seconds) so
a runaway/infinite loop in generated code cannot hang a pipeline stage
forever. The timeout is enforced by running the code in a daemon
thread and giving up (raising `SandboxTimeoutError`) if it hasn't
finished in time; the orphaned thread is abandoned (Python has no safe
way to kill a running thread) but, being a daemon thread, it will not
block process shutdown.

Trusted setup code (e.g. the developer-authored preamble that wires up
helper functions like `load_json_artifact`) can be run with full
interpreter builtins via :meth:`run_trusted`. That method must never be
used with model-generated content.
"""
from __future__ import annotations

import io
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, Optional, Tuple

#: Wall-clock seconds allowed for a single `run`/`run_safe` call before
#: execution is abandoned and `SandboxTimeoutError` is raised.
DEFAULT_TIMEOUT_SECONDS = 10.0

#: Modules the sandboxed `import` statement is allowed to reach. This is
#: the pipeline's actual analysis toolkit (pure computation + pandas for
#: tabular data), not a general-purpose import allowance: `os`, `sys`,
#: `subprocess`, `socket`, `shutil`, `pathlib`, etc. are all absent, so
#: `import os` (and therefore `os.system(...)`) still raises `ImportError`.
#:
#: Residual risk, noted deliberately rather than swept under the rug:
#: `pandas` ships file/URL readers (`read_csv`, `read_json`,
#: `read_parquet`, ...) that perform their own I/O internally rather
#: than going through this sandbox's (absent) `open` builtin, so
#: allowing `import pandas` does reopen a narrower path for generated
#: code to read arbitrary local files (or, if the network happens to be
#: reachable, remote URLs). It is allowed anyway because (a) pandas is a
#: core project dependency the analysis prompt is written around and
#: real generated code relies on it (confirmed via live end-to-end
#: testing against the real OpenAI model), and (b) it is a materially
#: smaller risk than the arbitrary-code-execution baseline this sandbox
#: replaces. Fully closing this would require running analysis code in
#: a separate OS-level sandboxed process — tracked as further hardening,
#: not attempted here.
DEFAULT_ALLOWED_IMPORT_MODULES = frozenset(
    {
        "json",
        "math",
        "statistics",
        "re",
        "datetime",
        "decimal",
        "itertools",
        "functools",
        "collections",
        "pandas",
    }
)


def make_guarded_import(allowed_modules: frozenset[str] | set[str]):
    """Build a restricted `__import__` that only permits `allowed_modules`.

    Submodule imports (`import collections.abc`) and `from` imports are
    checked against the top-level package name, matching how Python's
    `import` statement itself resolves `level`/`fromlist`.
    """

    def _guarded_import(
        name: str,
        globals: Optional[Dict[str, Any]] = None,
        locals: Optional[Dict[str, Any]] = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        root = name.split(".", 1)[0]
        if root not in allowed_modules:
            raise ImportError(
                f"Import of module '{name}' is not permitted in the analysis sandbox"
            )
        # `__import__` here resolves to the REAL builtin via this
        # function's own module globals (python_sandbox.py), not the
        # restricted sandbox globals it is installed into.
        return __import__(name, globals, locals, fromlist, level)

    return _guarded_import


#: Builtins allowlist used by default for untrusted (LLM-generated) code.
#: `open`, `eval`, `exec`, `compile`, `input`, and any other name that
#: would grant filesystem, network, or interpreter-escape capability are
#: simply absent. `__import__` is present but guarded (see
#: `make_guarded_import`) rather than absent — see module docstring.
DEFAULT_SAFE_BUILTINS: Dict[str, Any] = {
    "__import__": make_guarded_import(DEFAULT_ALLOWED_IMPORT_MODULES),
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "setattr": setattr,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    # Exception types generated code may reasonably need to catch/raise.
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError,
    "NotImplementedError": NotImplementedError,
}


class SandboxTimeoutError(RuntimeError):
    """Raised when sandboxed code exceeds its execution time budget."""


class PythonSandbox:
    """Execute Python code in an isolated namespace with captured output.

    Unlike the langchain PythonREPL, this sandbox:
    - Allows multi-function code naturally
    - Supports classes and complex control flow
    - Captures both stdout and stderr properly
    - Maintains state across multiple exec() calls

    Security-relevant behavior (see module docstring for detail):
    - `run`/`run_safe` execute with a restricted builtins allowlist by
      default (`allowed_builtins`, falling back to
      `DEFAULT_SAFE_BUILTINS`), not the real interpreter builtins.
    - `run`/`run_safe` enforce a wall-clock `timeout`.
    - `run_trusted` is the explicit escape hatch for developer-authored
      setup code that needs real builtins (e.g. to `import` modules);
      it must never be handed model-generated code.
    """

    def __init__(
        self,
        globals_dict: Dict[str, Any] | None = None,
        *,
        allowed_builtins: Dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the sandbox with an optional globals dictionary.

        Args:
            globals_dict: Initial global namespace. If it already
                contains an explicit ``__builtins__`` entry, that entry
                is respected as-is (the caller has opted out of the
                default restriction). Otherwise the restricted
                allowlist is installed.
            allowed_builtins: Builtins allowlist to use for untrusted
                execution (`run`/`run_safe`). Defaults to
                `DEFAULT_SAFE_BUILTINS` when not provided.
            timeout: Wall-clock seconds allowed per `run`/`run_safe`
                call before `SandboxTimeoutError` is raised.
        """
        self._globals: Dict[str, Any] = dict(globals_dict) if globals_dict else {}
        self._restricted_builtins: Dict[str, Any] = (
            dict(allowed_builtins) if allowed_builtins is not None else dict(DEFAULT_SAFE_BUILTINS)
        )
        self.timeout = timeout
        if "__builtins__" not in self._globals:
            self._globals["__builtins__"] = self._restricted_builtins

    @property
    def globals(self) -> Dict[str, Any]:
        """Access the current global namespace."""
        return self._globals

    def _exec_with_timeout(self, code: str, target_globals: Dict[str, Any]) -> None:
        """Run `code` against `target_globals`, aborting after `self.timeout`.

        Runs the exec() call on a background daemon thread so a
        CPU-bound infinite loop cannot hang the caller forever. Note
        this is a *soft* timeout: Python offers no safe way to forcibly
        kill a running thread, so a runaway loop keeps consuming CPU in
        the background after we give up on it. It is a daemon thread,
        so it will not prevent process shutdown.
        """
        compiled = compile(code, "<sandbox>", "exec")
        outcome: list[BaseException] = []

        def _target() -> None:
            try:
                exec(compiled, target_globals)
            except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
                outcome.append(exc)

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            raise SandboxTimeoutError(
                f"Sandboxed code exceeded the {self.timeout}s execution time limit and was aborted"
            )
        if outcome:
            raise outcome[0]

    def run(self, code: str) -> str:
        """Execute untrusted code and return captured stdout.

        Runs with the restricted builtins allowlist and under the
        configured timeout.

        Args:
            code: Python code to execute.

        Returns:
            Captured stdout as a string.

        Raises:
            Exception: Re-raises any exception from the executed code.
            SandboxTimeoutError: If execution exceeds `self.timeout`.
        """
        stdout_capture = io.StringIO()

        with redirect_stdout(stdout_capture):
            self._exec_with_timeout(code, self._globals)

        return stdout_capture.getvalue()

    def run_safe(self, code: str) -> Tuple[str, str, Optional[str]]:
        """Execute untrusted code and capture output without raising exceptions.

        Runs with the restricted builtins allowlist and under the
        configured timeout.

        Args:
            code: Python code to execute.

        Returns:
            Tuple of (stdout, stderr, error_message).
            error_message is None if execution succeeded.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        error_msg: Optional[str] = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                self._exec_with_timeout(code, self._globals)
        except SandboxTimeoutError as exc:
            error_msg = str(exc)
            stderr_capture.write(error_msg)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            # Also capture the traceback for debugging
            stderr_capture.write(traceback.format_exc())

        return stdout_capture.getvalue(), stderr_capture.getvalue(), error_msg

    def run_trusted(self, code: str) -> str:
        """Execute developer-authored setup code with full interpreter builtins.

        This is the explicit escape hatch used for the sandbox
        "preamble" (imports, helper function definitions) that the
        pipeline itself writes. It must NEVER be called with
        model-generated code — doing so defeats the sandboxing this
        class exists to provide.

        Still runs under the configured timeout.

        Args:
            code: Trusted Python source to execute.

        Returns:
            Captured stdout as a string.
        """
        stdout_capture = io.StringIO()
        previous_builtins = self._globals.get("__builtins__")
        self._globals["__builtins__"] = __builtins__
        try:
            with redirect_stdout(stdout_capture):
                self._exec_with_timeout(code, self._globals)
        finally:
            self._globals["__builtins__"] = previous_builtins
        return stdout_capture.getvalue()

    def set(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox namespace."""
        self._globals[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Get a variable from the sandbox namespace."""
        return self._globals.get(name, default)

    def reset(self) -> None:
        """Reset the sandbox to a clean state (restricted builtins restored)."""
        self._globals.clear()
        self._globals["__builtins__"] = self._restricted_builtins
