"""
Tests for visualize_annotations.py
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import cv2

from visualize_annotations import (
    draw_annotated_corners,
    resolve_data_dir,
    process_dataset,
    CORNER_COLUMNS,
    SPLIT_CONFIG,
)


class TestVisualizeAnnotations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_draw_annotated_corners(self):
        # Create a blank white image (1000 x 500) -> h=1000, w=500
        h, w = 1000, 500
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        row = pd.Series({
            "top_left_x": 0.1,
            "top_left_y": 0.2,
            "top_right_x": 0.8,
            "top_right_y": 0.2,
            "bottom_left_x": 0.1,
            "bottom_left_y": 0.9,
            "bottom_right_x": 0.8,
            "bottom_right_y": 0.9,
        })

        radius = 20
        color = (0, 255, 0)
        vis_img = draw_annotated_corners(img, row, radius=radius, color=color)

        # Original image must not be mutated
        self.assertTrue((img == 255).all())

        # Check that centers of all 4 corners are green
        for x_col, y_col in CORNER_COLUMNS:
            x_px = int(round(row[x_col] * w))
            y_px = int(round(row[y_col] * h))
            self.assertEqual(list(vis_img[y_px, x_px]), [0, 255, 0])

            # Check inside circle
            self.assertEqual(list(vis_img[y_px, x_px + radius - 2]), [0, 255, 0])
            # Check outside circle
            self.assertEqual(list(vis_img[y_px, x_px + radius + 2]), [255, 255, 255])

    def test_resolve_data_dir(self):
        # Should resolve to the workspace's data directory
        data_dir = resolve_data_dir()
        self.assertTrue(data_dir.is_dir())
        self.assertTrue((data_dir / "edges_train").is_dir())

    def test_custom_radius_and_color(self):
        h, w = 400, 400
        img = np.zeros((h, w, 3), dtype=np.uint8)
        row = pd.Series({
            "top_left_x": 0.5,
            "top_left_y": 0.5,
        })
        vis = draw_annotated_corners(img, row, radius=10, color=(0, 0, 255))
        # Center should be red (BGR: [0, 0, 255])
        self.assertEqual(list(vis[200, 200]), [0, 0, 255])
        # Distance 9 should be red
        self.assertEqual(list(vis[200, 209]), [0, 0, 255])
        # Distance 12 should be black (0, 0, 0)
        self.assertEqual(list(vis[200, 212]), [0, 0, 0])

    def test_missing_column_handled_gracefully(self):
        h, w = 100, 100
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Empty series
        row = pd.Series({})
        vis = draw_annotated_corners(img, row, radius=5)
        # No crash, returns unmodified copy
        self.assertTrue((vis == 0).all())

    def test_process_dataset_end_to_end(self):
        out_dir = os.path.join(self.temp_dir, "test_vis_out")
        data_dir = resolve_data_dir()

        process_dataset(
            data_dir=data_dir,
            splits_to_process=["test"],
            output_dir=out_dir,
            save_images=True,
            show_images=False,
            radius=20,
            limit=2,
        )

        test_out_dir = os.path.join(out_dir, "edges_test")
        self.assertTrue(os.path.isdir(test_out_dir))
        files = [f for f in os.listdir(test_out_dir) if f.endswith(".jpg")]
        self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
