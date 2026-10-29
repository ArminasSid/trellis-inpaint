import argparse
from pathlib import Path
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Composite grayscale sketches onto RGBA render images."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/HSSD"),
        help="Base dataset directory (default: datasets/HSSD)",
    )
    args = parser.parse_args()

    # Base directories
    renders_root = args.output_dir / "renders_dmg"
    sketches_root = args.output_dir / "sketches"
    output_root = args.output_dir / "renders_dmg_sketches"

    # Find all render images recursively
    render_paths = list(renders_root.rglob("*.png"))

    print(f"Found {len(render_paths)} render images")

    for render_path in render_paths:
        rel_path = render_path.relative_to(renders_root)

        sketch_path = sketches_root / rel_path
        output_path = output_root / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not sketch_path.exists():
            print(f"Missing sketch: {sketch_path}")
            continue

        try:
            # Load render WITH alpha preserved
            render_img = Image.open(render_path).convert("RGBA")

            # Load grayscale sketch
            sketch_img = Image.open(sketch_path).convert("L")

            # Resize sketch if needed
            if render_img.size != sketch_img.size:
                sketch_img = sketch_img.resize(render_img.size, Image.BILINEAR)

            render = np.array(render_img)  # H x W x 4
            sketch = np.array(sketch_img)  # H x W

            result = render.copy()

            # Where sketch exists
            sketch_mask = sketch > 0

            # Apply sketch grayscale into RGB
            for c in range(3):
                result[..., c][sketch_mask] = sketch[sketch_mask]

            # REMOVE transparency where sketch exists
            result[..., 3][sketch_mask] = 255

            # Save
            Image.fromarray(result, mode="RGBA").save(output_path)

            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Failed processing {render_path}: {e}")


if __name__ == "__main__":
    main()