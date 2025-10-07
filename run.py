import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from src.main import main as app_main
    except Exception as exc:
        print(f"Failed to import application entrypoint: {exc}")
        return 1

    return app_main() if callable(app_main) else 0


if __name__ == "__main__":
    raise SystemExit(main())


