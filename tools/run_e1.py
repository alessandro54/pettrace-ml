"""Bootstrap E1 outside the host kernel. Only the standard library is required."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

RELEASE = "0.2.2"
PYTHON = "3.11.15"
UV = "0.11.22"
BASE = f"https://github.com/alessandro54/pettrace-ml/releases/download/v{RELEASE}"


def clean_environment(work):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "JUPYTER", "IPYTHON", "PIP_", "UV_"))
           and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}
    # Clearing IPYTHONDIR alone falls back to Colab's default profile under the user's home.
    env.update(IPYTHONDIR=str(work / ".ipython"), JUPYTER_NO_CONFIG="1",
               PYTHONNOUSERSITE="1", PYTHONHASHSEED="0", MPLBACKEND="Agg",
               HF_HUB_DISABLE_PROGRESS_BARS="1", MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR="false",
               MLFLOW_DISABLE_AGENT_HINT="1")
    return env


def download(name, destination, inventory):
    with urlopen(f"{BASE}/{name}", timeout=60) as response:
        contents = response.read()
    if hashlib.sha256(contents).hexdigest() != inventory[name]:
        raise ValueError(f"Release checksum mismatch: {name}")
    destination.write_bytes(contents)


def bootstrap_uv(directory, env):
    subprocess.run([sys.executable, "-m", "pip", "--isolated", "install", "--quiet", "--no-deps",
                    "--ignore-installed", "--prefix", str(directory), f"uv=={UV}"], env=env, check=True)
    # System Python can put prefix scripts in local/bin instead of bin (Colab/Debian).
    binaries = [path for path in directory.rglob("uv") if path.is_file() and path.parent.name == "bin"]
    if len(binaries) != 1:
        raise RuntimeError(f"Expected one installed uv executable under {directory}; found {binaries}")
    return str(binaries[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path, help="New directory, or existing one for --resume")
    parser.add_argument("--env-file", type=Path, help="Local credentials file; never copied into reports")
    parser.add_argument("--resume", action="store_true", help="Retry publication only, without repeating inference")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    env = clean_environment(work)
    python = work / ".venv/bin/python"
    if not args.resume:
        work.mkdir(parents=True, exist_ok=False)
        with urlopen(f"{BASE}/SHA256SUMS", timeout=60) as response:
            sums = response.read()
        (work / "SHA256SUMS").write_bytes(sums)
        inventory = dict((name, digest) for digest, name in
                         (line.split("  ", 1) for line in sums.decode().splitlines()))
        for name in ("requirements-evaluation.txt", "e1_clip_zero_shot.ipynb", "release.json"):
            download(name, work / name, inventory)
        print(f"Preparing Python {PYTHON} + PetTrace {RELEASE} in {work}", flush=True)
        uv = bootstrap_uv(work / ".bootstrap", env)
        env["UV_PYTHON_INSTALL_DIR"] = str(work / ".python")
        subprocess.run([uv, "--no-config", "venv", "--managed-python", "--python", PYTHON,
                        str(work / ".venv")], env=env, check=True)
        subprocess.run([uv, "--no-config", "pip", "sync", "--python", str(python), "--require-hashes",
                        str(work / "requirements-evaluation.txt")], env=env, check=True)
        subprocess.run([uv, "--no-config", "pip", "check", "--python", str(python)], env=env, check=True)
    elif not python.is_file():
        raise SystemExit("The isolated environment is missing; resume requires the original work directory.")
    command = [str(python), "-I", "-m", "pettrace.experiments.e1", "--work-dir", str(work)]
    if args.env_file:
        command.extend(["--env-file", str(args.env_file.resolve())])
    if args.resume:
        command.append("--resume")
    subprocess.run(command, cwd=work, env=env, check=True)


if __name__ == "__main__":
    main()
