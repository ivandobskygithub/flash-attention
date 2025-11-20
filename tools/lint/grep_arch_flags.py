import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

PATTERNS = [re.compile(r"sm_120"), re.compile(r"compute_120")]


def main() -> int:
    hits = 0
    for path in ROOT.rglob("*setup.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PATTERNS:
            if pattern.search(text):
                hits += 1
    if hits == 0:
        print("missing sm_120 flags in setup files")
        return 1
    print("arch flag check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
