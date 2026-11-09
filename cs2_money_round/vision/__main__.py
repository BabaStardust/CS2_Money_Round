"""uv run python -m cs2_money_round.vision <screenshot.png>  → JSON auf stdout."""

import sys

from dotenv import load_dotenv

from .readers import read_screenshot


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Aufruf: python -m cs2_money_round.vision <screenshot.png>")
    load_dotenv(".env")
    print(read_screenshot(sys.argv[1]).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
