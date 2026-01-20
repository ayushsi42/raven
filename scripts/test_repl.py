"""Minimal runner that mimics the pipeline's code execution loop using PythonSandbox."""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from openai import OpenAI

from hypothesis_agent.config import get_settings
from hypothesis_agent.orchestration.python_sandbox import PythonSandbox


RESULT_PREFIX = "RESULT::"
ARTIFACT_ROOT = Path("./data/artifacts").resolve()
WORKFLOW_ID = "standalone-repl"


def extract_python_code(raw_text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def prepare_analysis_preamble(workflow_dir: Path) -> str:
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir_str = str(workflow_dir.resolve())
    return textwrap.dedent(
        f"""
        import json
        from pathlib import Path

        RESULT_PREFIX = {RESULT_PREFIX!r}
        ARTIFACT_OUTPUT_DIR = Path({workflow_dir_str!r})
        ARTIFACT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        def load_json_artifact(slug: str):
            raise KeyError(f"No artifacts available in standalone runner: {{slug}}")

        def write_json_artifact(name: str, payload):
            target = ARTIFACT_OUTPUT_DIR / f"{{name}}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return str(target.resolve())
        """
    )


def run_generated_code(code_text: str) -> Tuple[str, str, Optional[dict], Optional[str], str]:
    cleaned_code = extract_python_code(code_text)
    sandbox = PythonSandbox()
    workflow_dir = ARTIFACT_ROOT / WORKFLOW_ID

    try:
        sandbox.run(prepare_analysis_preamble(workflow_dir))
    except Exception as exc:
        feedback = f"Failed to initialise analysis runtime: {exc}"
        return "", feedback, None, feedback, cleaned_code

    stdout_text = ""
    stderr_text = ""
    feedback: Optional[str] = None
    result_payload: Optional[dict] = None

    try:
        stdout_text = sandbox.run(cleaned_code)
    except Exception as exc:
        stderr_text = str(exc)
        feedback = stderr_text or "Python sandbox execution raised an exception"

    result_line: Optional[str] = None
    for line in stdout_text.splitlines():
        if line.startswith(RESULT_PREFIX):
            result_line = line[len(RESULT_PREFIX) :].strip()
            break

    if result_line is not None:
        try:
            result_payload = json.loads(result_line)
            feedback = None
        except json.JSONDecodeError as exc:
            feedback = f"Failed to parse RESULT payload: {exc}"
    elif feedback is None:
        combined_error = stderr_text.strip()
        feedback = combined_error or "No RESULT:: line emitted in stdout"

    return stdout_text, stderr_text, result_payload, feedback, cleaned_code


def _collect_user_input() -> str:
    print("Describe the analysis or computation you want (type END on a blank line to finish):")
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def generate_code_from_prompt(prompt: str) -> str:
    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key is not configured; set OPENAI_API_KEY.")

    model = settings.openai_model or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You translate natural-language analysis requests into executable Python code. "
        "Return only Python wrapped in a code fence. You can define multiple functions, classes, "
        "and use any control flow you need. The code must import json if needed and "
        'print the final structured result using print("RESULT::" + json.dumps(result, default=str)).'
    )
    user_prompt = prompt.strip()
    if not user_prompt:
        raise RuntimeError("Prompt is empty; nothing to generate.")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise RuntimeError("Model returned an empty response.")
    return content


def main() -> int:
    prompt = "Write a Python program with two functions where one calls the other. The first function should calculate factorial and the second should use it to compute combinations (n choose k). Test with n=10, k=3."
    try:
        generated_code = generate_code_from_prompt(prompt)
    except Exception as exc:
        print(f"Failed to generate code: {exc}", file=sys.stderr)
        return 1

    stdout_text, stderr_text, result_payload, feedback, cleaned_code = run_generated_code(generated_code)

    print("\n=== RAW PROMPT ===")
    print(prompt.strip() or "<empty>")
    print("\n=== MODEL OUTPUT ===")
    print(generated_code.strip())

    print("\n=== CLEANED CODE ===")
    print(cleaned_code or "<empty>")
    print("\n=== STDOUT ===")
    print(stdout_text.rstrip())
    print("\n=== STDERR ===")
    print(stderr_text.rstrip())
    print("\n=== RESULT PAYLOAD ===")
    print(json.dumps(result_payload, indent=2, ensure_ascii=False) if result_payload is not None else "null")
    print("\n=== FEEDBACK ===")
    print(feedback or "null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())