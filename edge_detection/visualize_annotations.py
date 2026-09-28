#!/usr/bin/env python3
"""
Visualize Annotated Cube Corners
================================
Visualizes annotated corner coordinates on Rubik's cube images across the
test, training, and validation datasets in `data/`.

Draws filled circles of radius 20 (default) in green at each of the 4 annotated
corners, matching the visualization style in experiment evaluations.

Usage:
    # Save visualizations for all splits (train, validation, test) to visualizations/
    python visualize_annotations.py

    # Interactively view images on screen (with OpenCV window)
    python visualize_annotations.py --show

    # Visualize only a specific split (e.g. validation)
    python visualize_annotations.py --split validation

    # View only on screen without saving to disk
    python visualize_annotations.py --show --no-save

    # Customize output directory, radius, or limit
    python visualize_annotations.py --output-dir my_vis --radius 20 --limit 10
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import pandas as pd


# Corner column pairs in dataset CSV files
CORNER_COLUMNS: List[Tuple[str, str]] = [
    ("top_left_x", "top_left_y"),
    ("top_right_x", "top_right_y"),
    ("bottom_left_x", "bottom_left_y"),
    ("bottom_right_x", "bottom_right_y"),
]

# Split definitions: split_key -> (directory_name, csv_filename)
SPLIT_CONFIG: Dict[str, Tuple[str, str]] = {
    "train": ("edges_train", "edges_train.csv"),
    "validation": ("edges_validation", "edges_validation.csv"),
    "test": ("edges_test", "edges_test.csv"),
}


def resolve_data_dir(custom_path: Optional[str] = None) -> Path:
    """Find and return the valid data directory."""
    if custom_path:
        p = Path(custom_path)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Specified data directory does not exist: {custom_path}")

    # Search common candidate locations
    candidates = [
        Path("data"),
        Path("edge_detection/data"),
        Path(__file__).resolve().parent / "data",
    ]
    for c in candidates:
        if c.is_dir() and (c / "edges_train").is_dir():
            return c.resolve()

    return Path("data").resolve()


def draw_annotated_corners(
    image: cv2.typing.MatLike,
    row: pd.Series,
    radius: int = 20,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = -1,
) -> cv2.typing.MatLike:
    """
    Draws circles at the 4 annotated corner points on an image.

    Args:
        image: BGR image as numpy array.
        row: Pandas Series containing normalized coordinate columns.
        radius: Circle radius in pixels (default: 20).
        color: BGR tuple for circle color (default: (0, 255, 0) Green).
        thickness: Line thickness (-1 for filled circle).

    Returns:
        Annotated copy of the image.
    """
    vis_img = image.copy()
    h, w = vis_img.shape[:2]

    for x_col, y_col in CORNER_COLUMNS:
        if x_col not in row or y_col not in row:
            continue
        x_norm = float(row[x_col])
        y_norm = float(row[y_col])

        # Convert normalized coordinates [0, 1] to pixel coordinates
        x_px = int(round(x_norm * w))
        y_px = int(round(y_norm * h))

        cv2.circle(vis_img, (x_px, y_px), radius, color, thickness)

    return vis_img


def interactive_viewer(
    samples: List[Tuple[str, str, cv2.typing.MatLike]],
    window_name: str = "Annotated Corners Viewer",
) -> None:
    """
    Interactive window to browse through visualized images.

    Controls:
        - Next: Space / Right Arrow / 'n' / 'd'
        - Prev: Left Arrow / 'p' / 'a' / Backspace
        - Quit: 'q' / ESC
    """
    if not samples:
        print("No samples available to view.")
        return

    idx = 0
    total = len(samples)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n--- Interactive Viewer Controls ---")
    print("  [Next]     : Space, Right Arrow, 'n', or 'd'")
    print("  [Previous] : Left Arrow, 'p', 'a', or Backspace")
    print("  [Quit]     : 'q' or ESC")
    print("-----------------------------------\n")

    while 0 <= idx < total:
        split_name, img_name, vis_img = samples[idx]
        title_text = f"[{idx + 1}/{total}] ({split_name}) {img_name}"
        cv2.setWindowTitle(window_name, f"{window_name} - {title_text}")

        # Display image
        cv2.imshow(window_name, vis_img)

        # Wait for key press
        key = cv2.waitKey(0)

        # Handle window close button (X)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # 'q' (113), ESC (27)
        if key in (ord('q'), ord('Q'), 27):
            break
        # Next image: Space (32), 'n' (110), 'd' (100), Right Arrow (83 or 65363 or 0)
        elif key in (32, ord('n'), ord('d'), ord('N'), ord('D'), 83, 65363):
            if idx < total - 1:
                idx += 1
            else:
                print("Reached last image.")
        # Previous image: 'p' (112), 'a' (97), Backspace (8), Left Arrow (81 or 65361)
        elif key in (ord('p'), ord('a'), ord('P'), ord('A'), 8, 81, 65361):
            if idx > 0:
                idx -= 1
            else:
                print("Already at first image.")
        else:
            # Default advance on any other key
            if idx < total - 1:
                idx += 1
            else:
                break

    cv2.destroyAllWindows()


def process_dataset(
    data_dir: Path | str,
    splits_to_process: List[str],
    output_dir: Optional[Path | str] = None,
    save_images: bool = True,
    show_images: bool = False,
    radius: int = 20,
    limit: Optional[int] = None,
) -> None:
    """
    Loads images and corner annotations for each split, draws corners, and saves/displays.
    """
    data_dir = Path(data_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)

    total_processed = 0
    samples_for_viewer: List[Tuple[str, str, cv2.typing.MatLike]] = []

    print(f"Data directory: {data_dir}")
    if save_images and output_dir:
        print(f"Output directory: {output_dir}")
    print(f"Splits to process: {', '.join(splits_to_process)}")
    print(f"Corner circle radius: {radius} px\n")

    for split_key in splits_to_process:
        dir_name, csv_name = SPLIT_CONFIG[split_key]
        split_dir = data_dir / dir_name
        csv_path = split_dir / csv_name

        if not split_dir.is_dir():
            print(f"[{split_key}] Directory not found: {split_dir}, skipping.")
            continue
        if not csv_path.is_file():
            print(f"[{split_key}] CSV annotations not found: {csv_path}, skipping.")
            continue

        df = pd.read_csv(csv_path)
        count = len(df)
        if limit is not None and limit > 0:
            df = df.iloc[:limit]
            print(f"[{split_key}] Processing {len(df)} of {count} images (limit applied)...")
        else:
            print(f"[{split_key}] Processing {len(df)} images...")

        if save_images and output_dir:
            save_split_dir = output_dir / dir_name
            save_split_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_split_dir = None

        split_saved = 0
        for _, row in df.iterrows():
            img_name = str(row["image_name"])
            img_path = split_dir / img_name

            if not img_path.is_file():
                print(f"  Warning: image file not found: {img_path}")
                continue

            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                print(f"  Warning: failed to read image: {img_path}")
                continue

            # Draw circles of specified radius on annotated corners
            vis_img = draw_annotated_corners(img_cv, row, radius=radius)

            # Save to disk
            if save_images and save_split_dir:
                save_path = save_split_dir / img_name
                cv2.imwrite(str(save_path), vis_img)
                split_saved += 1

            # Collect for interactive viewing
            if show_images:
                samples_for_viewer.append((split_key, img_name, vis_img))

            total_processed += 1

        if save_images and save_split_dir:
            print(f"  Saved {split_saved} visualized images to: {save_split_dir}")

    print(f"\nDone! Processed a total of {total_processed} images.")

    # Launch interactive viewer if requested
    if show_images:
        interactive_viewer(samples_for_viewer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize annotated corners on Rubik's cube images in data/."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the data/ folder (default: automatically detected)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualized images (default: 'visualizations')",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "val", "validation", "test"],
        default="all",
        help="Dataset split to visualize: 'train', 'validation' ('val'), 'test', or 'all' (default: all)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=20,
        help="Radius of circle drawn on each corner (default: 20 px)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive GUI window to view images sequentially",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save visualized images to disk (useful with --show)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images processed per split (e.g. for quick inspection)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine splits to run
    if args.split in ("validation", "val"):
        splits = ["validation"]
    elif args.split == "train":
        splits = ["train"]
    elif args.split == "test":
        splits = ["test"]
    else:
        splits = ["train", "validation", "test"]

    try:
        data_path = resolve_data_dir(args.data_dir)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output_dir) if not args.no_save else None

    process_dataset(
        data_dir=data_path,
        splits_to_process=splits,
        output_dir=output_path,
        save_images=not args.no_save,
        show_images=args.show,
        radius=args.radius,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
