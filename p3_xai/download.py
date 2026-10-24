"""Download pinned public assets or verify existing copies against recorded SHA-256 hashes."""

import argparse
import json
import urllib.request
from pathlib import Path

from .utils import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path("configs/p3_assets.json").read_text())
    for asset in manifest["assets"]:
        path = Path(asset["path"])
        if path.exists() and sha256_file(path) == asset["sha256"]:
            print(f"Verified {path}", flush=True)
            continue
        if args.verify_only:
            raise ValueError(f"Missing or mismatched asset: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        print(f"Downloading {path}", flush=True)
        with urllib.request.urlopen(asset["url"], timeout=120) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if sha256_file(temporary) != asset["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}; source may have changed. Do not silently update the snapshot.")
        temporary.replace(path)


if __name__ == "__main__":
    main()
