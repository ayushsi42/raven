"""Flexible Python code execution sandbox for analysis tasks."""
from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional, Tuple


class PythonSandbox:
    """Execute Python code in an isolated namespace with captured output.
    
    Unlike the langchain PythonREPL, this sandbox:
    - Allows multi-function code naturally
    - Supports classes and complex control flow
    - Captures both stdout and stderr properly
    - Maintains state across multiple exec() calls
    """

    def __init__(self, globals_dict: Dict[str, Any] | None = None) -> None:
        """Initialize the sandbox with an optional globals dictionary.
        
        Args:
            globals_dict: Initial global namespace. If None, starts with safe builtins.
        """
        self._globals: Dict[str, Any] = globals_dict or {}
        # Ensure essential builtins are available
        if "__builtins__" not in self._globals:
            self._globals["__builtins__"] = __builtins__

    @property
    def globals(self) -> Dict[str, Any]:
        """Access the current global namespace."""
        return self._globals

    def run(self, code: str) -> str:
        """Execute code and return captured stdout.
        
        Args:
            code: Python code to execute.
            
        Returns:
            Captured stdout as a string.
            
        Raises:
            Exception: Re-raises any exception from the executed code.
        """
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            exec(code, self._globals)
        
        return stdout_capture.getvalue()

    def run_safe(self, code: str) -> Tuple[str, str, Optional[str]]:
        """Execute code and capture output without raising exceptions.
        
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
                exec(code, self._globals)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            # Also capture the traceback for debugging
            stderr_capture.write(traceback.format_exc())
        
        return stdout_capture.getvalue(), stderr_capture.getvalue(), error_msg

    def set(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox namespace."""
        self._globals[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        """Get a variable from the sandbox namespace."""
        return self._globals.get(name, default)

    def reset(self) -> None:
        """Reset the sandbox to a clean state."""
        self._globals.clear()
        self._globals["__builtins__"] = __builtins__
