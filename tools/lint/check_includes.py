import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
INCLUDE_RE = re.compile(r'#include\s+"([^"]+)"')


def main() -> int:
    missing = []
    for path in (*ROOT.rglob("*.h"), *ROOT.rglob("*.hpp"), *ROOT.rglob("*.cu")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in INCLUDE_RE.finditer(text):
            include_path = match.group(1)
            candidate = path.parent / include_path
            if not candidate.exists():
                missing.append(f"Missing include {include_path} referenced from {path}")
    if missing:
        for line in missing:
            print(line)
        return 1
    print("include check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
