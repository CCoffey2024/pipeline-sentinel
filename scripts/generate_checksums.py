from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SHA256SUMS.txt for release artifacts")
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args()

    directory = args.directory.resolve()
    artifacts = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    if not artifacts:
        raise SystemExit(f"No release artifacts found in {directory}")

    output = directory / "SHA256SUMS.txt"
    output.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
