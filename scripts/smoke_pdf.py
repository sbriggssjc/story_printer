from src.pipeline.orchestrator import run_once


def main() -> None:
    transcript = "Claire hid the pizza crust and a pizza monster came!"
    pdf_path = run_once(transcript)
    print(pdf_path)


if __name__ == "__main__":
    main()
