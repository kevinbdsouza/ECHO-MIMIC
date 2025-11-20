"""Utilities for writing, running, and auto-fixing heuristic code blocks."""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from .command import CommandOutputCapture
from .fix import fix_with_model

Fixer = Callable[[str, str], str]


@contextmanager
def pushd(path: Path):
    """Temporarily change the working directory."""
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def copy_input_template(
    workdir: Path,
    *,
    source_name: str = "input_cp.geojson",
    target_name: str = "input.geojson",
) -> None:
    """Ensure ``target_name`` exists by copying from ``source_name`` when available."""
    source = workdir / source_name
    target = workdir / target_name
    if source.exists():
        shutil.copyfile(source, target)


def write_code(workdir: Path, filename: str, code: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    script_path = workdir / filename
    script_path.write_text(code)
    return script_path


def validate_python_code(
    code: str,
    *,
    workdir: Path,
    capture: CommandOutputCapture,
    fixer: Optional[Fixer] = None,
    script_name: str = "temp_fix.py",
    max_attempts: int = 2,
    copy_template: bool = True,
    input_template: str = "scenario_cp.json",
    input_target: str = "scenario.json",
    pre_run: Optional[Callable[[Path], None]] = None,
) -> Optional[str]:
    """Run ``code`` inside ``workdir`` and optionally fix failures via ``fixer``.

    Returns the working source when execution succeeds, otherwise ``None``.
    """
    attempt_code = code

    for attempt in range(max_attempts):
        write_code(workdir, script_name, attempt_code)
        if copy_template:
            copy_input_template(workdir, source_name=input_template, target_name=input_target)
        if pre_run:
            pre_run(workdir)

        with pushd(workdir):
            exit_code, _, stderr = capture.run_python_script(script_name)

        if exit_code == 0:
            return attempt_code
        if fixer is None:
            break
        attempt_code = fixer(attempt_code, stderr)
    return None


def make_code_validator(
    *,
    workdir: Path,
    capture: CommandOutputCapture,
    fix_model,
    rate_limiter,
    default_script: str = "temp_fix.py",
    default_attempts: int = 2,
):
    """Build a reusable validator closure with shared execution context."""

    def _validator(
        source: str,
        *,
        script_name: Optional[str] = None,
        max_attempts: Optional[int] = None,
        pre_run: Optional[Callable[[Path], None]] = None,
    ) -> str:
        script = script_name or default_script
        attempts = max_attempts or default_attempts

        def _fixer(code: str, trace: str) -> str:
            return fix_with_model(
                fix_model,
                code,
                trace,
                rate_limiter=rate_limiter,
            )

        validated = validate_python_code(
            source,
            workdir=workdir,
            capture=capture,
            fixer=_fixer,
            script_name=script,
            max_attempts=attempts,
            pre_run=pre_run,
        )
        return validated if validated is not None else source

    return _validator


__all__ = [
    "CommandOutputCapture",
    "copy_input_template",
    "make_code_validator",
    "pushd",
    "validate_python_code",
    "write_code",
]
