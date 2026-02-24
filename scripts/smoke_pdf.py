import sys
from pathlib import Path

# Ensure the project root is on sys.path so "from src..." imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.orchestrator import run_once


def main() -> None:
    transcript = "Claire hid the pizza crust and a pizza monster came!"
    pdf_path = run_once(transcript)
    print(pdf_path)


if __name__ == "__main__":
    main()
