from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def memory_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def machine_info() -> dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    try:
        output = subprocess.check_output(
            ["wmic", "cpu", "get", "name", "/value"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        values = [
            line.split("=", 1)[1].strip()
            for line in output.splitlines()
            if line.startswith("Name=")
        ]
        if values:
            cpu = values[0]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import psutil

        total_ram = int(psutil.virtual_memory().total)
    except ImportError:
        total_ram = None
    return {
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "release": platform.release(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "total_ram_bytes": total_ram,
        "gpu_used": False,
        "dependencies": {
            name: package_version(name)
            for name in [
                "remem-ai",
                "numpy",
                "usearch",
                "sentence-transformers",
                "transformers",
                "torch",
                "datasets",
                "pandas",
                "psutil",
                "matplotlib",
            ]
        },
    }
