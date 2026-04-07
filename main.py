"""
ANN-Bench entry point.

Commands:
    python main.py benchmark   -- run full benchmark comparison
    python main.py scaling     -- run scaling experiment
    python main.py ui          -- launch Streamlit UI
"""

import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "benchmark"

    if cmd == "benchmark":
        from src.benchmark.run import run
        synthetic = "--synthetic" in sys.argv
        run(use_synthetic=synthetic)

    elif cmd == "scaling":
        from src.benchmark.scaling import run_scaling
        synthetic = "--synthetic" in sys.argv
        run_scaling(use_synthetic=synthetic)

    elif cmd == "ui":
        import subprocess
        subprocess.run(["streamlit", "run", "src/ui/app.py"])

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
