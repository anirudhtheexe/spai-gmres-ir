"""Locate CUDA runtime DLLs without assuming a user's installation path."""
import os
from pathlib import Path
import shutil


def cuda_environment():
    environment = os.environ.copy()
    roots = []
    if environment.get("CUDA_PATH"):
        roots.append(Path(environment["CUDA_PATH"]))
    compiler = shutil.which("nvcc")
    if compiler:
        roots.append(Path(compiler).resolve().parent.parent)
    folders = [str(root / suffix) for root in roots for suffix in ("bin", "bin/x64")
               if (root / suffix).is_dir()]
    environment["PATH"] = os.pathsep.join(folders + [environment.get("PATH", "")])
    environment["CUBLAS_EMULATE_DOUBLE_PRECISION"] = "0"
    return environment
