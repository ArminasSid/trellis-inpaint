import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Copy transforms.json files into a sketches output directory."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/HSSD"),
        help="Base output directory (default: datasets/HSSD)",
    )
    args = parser.parse_args()

    # Source and destination roots
    src_root = args.output_dir / "renders_dmg"
    dst_root = args.output_dir / "renders_dmg_sketches"

    # Find all transforms.json files recursively
    json_files = list(src_root.rglob("transforms.json"))

    print(f"Found {len(json_files)} transforms.json files")

    for src_json in json_files:
        # Relative path from source root
        rel_path = src_json.relative_to(src_root)

        # Destination path
        dst_json = dst_root / rel_path
        dst_json.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_json, dst_json)

        print(f"Copied: {src_json} -> {dst_json}")


if __name__ == "__main__":
    main()