import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGETS = [ROOT / "hopper"]

BRACKETS = {"{": "}", "[": "]", "(": ")"}


def check_file(path: pathlib.Path) -> list[str]:
    stack: list[str] = []
    problems: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for idx, ch in enumerate(text):
        if ch in BRACKETS:
            stack.append(BRACKETS[ch])
        elif ch in BRACKETS.values():
            if not stack or stack[-1] != ch:
                problems.append(f"{path}: unexpected token '{ch}' at offset {idx}")
            elif stack:
                stack.pop()
    if stack:
        problems.append(f"{path}: unterminated tokens {stack}")
    return problems


def main() -> int:
    errors: list[str] = []
    for target in TARGETS:
        for path in target.rglob("*.cu"):
            errors.extend(check_file(path))
    if errors:
        for line in errors:
            print(line)
        return 1
    print("token check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
