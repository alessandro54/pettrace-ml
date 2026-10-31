"""Build package distributions with provenance and checksums for a GitHub Release."""

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    def run(*args):
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()

    if run("git", "status", "--porcelain"):
        raise SystemExit("Commit the reviewed changes before building a release.")
    run("uv", "lock", "--check")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    output = ROOT / "dist" / f"v{version}"
    output.mkdir(parents=True, exist_ok=False)
    build = {
        "version": version, "git_sha": run("git", "rev-parse", "HEAD"), "git_dirty": False,
        "uv_lock_sha256": sha256(ROOT / "uv.lock"),
        "reference_python": (ROOT / ".python-version").read_text().strip(),
        "uv": run("uv", "--version"),
    }
    build_path = ROOT / "src/pettrace/_build.json"
    try:
        build_path.write_text(json.dumps(build, indent=2) + "\n")
        subprocess.run(["uv", "build", "--out-dir", str(output)], cwd=ROOT, check=True)
    finally:
        build_path.unlink(missing_ok=True)
    inventory = {path.name: sha256(path) for path in sorted(output.iterdir()) if not path.name.startswith(".")}
    (output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in inventory.items()))
    print(output)


if __name__ == "__main__":
    main()
