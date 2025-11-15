"""Command execution helpers shared across experiments."""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


class CommandOutputCapture:
    """Wrapper around ``subprocess`` that records output to timestamped logs."""

    def __init__(self, log_dir: Path | str = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("command_capture")
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"command_run_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

    def run_command(self, command: str | Iterable[str], *, capture_stderr: bool = True):
        """Execute ``command`` and return ``(code, stdout, stderr)``."""
        shell = isinstance(command, str)
        if not shell:
            command = list(command)

        self.logger.info("Running command: %s", command)

        with subprocess.Popen(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else None,
            universal_newlines=True,
        ) as process:
            stdout, stderr = process.communicate()
            return_code = process.returncode

        if return_code:
            self.logger.info("Command returned with code: %s", return_code)
            if stdout:
                self.logger.debug("STDOUT:\n%s", stdout)
            if stderr:
                self.logger.debug("STDERR:\n%s", stderr)

        return return_code, stdout, stderr

    def run_python_script(self, script_path: str, args: Optional[Iterable[str]] = None):
        """Run ``python script_path [args...]`` and capture output."""
        args = list(args or [])
        cmd = [sys.executable, script_path, *args]
        return self.run_command(cmd, capture_stderr=True)


__all__ = ["CommandOutputCapture"]
