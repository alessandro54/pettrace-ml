"""Install the reviewed source bundle in a disposable Colab runtime."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    if not (root / 'src/pettrace/training/dinov2.py').is_file():
        raise FileNotFoundError('Extract the complete DINO source bundle first')
    # Colab owns its kernel and Jupyter client versions.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e',
                    str(root) + '[training,tracking,notebooks]'], check=True)


if __name__ == '__main__':
    main()
