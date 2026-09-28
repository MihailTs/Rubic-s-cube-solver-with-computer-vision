"""
Tests for augment.py
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image

import augment


class TestAugment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_path = augment.PATH
        augment.PATH = self.temp_dir

        # Create a test synthetic image with an identifiable pattern
        # Size: 200 x 200
        self.img_size = (200, 200)
        img_arr = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        # Mark 4 corners with distinct pixel colors
        # top_left (20, 30) -> Red
        img_arr[30, 20] = [255, 0, 0]
        # top_right (170, 30) -> Green
        img_arr[30, 170] = [0, 255, 0]
        # bottom_left (20, 160) -> Blue
        img_arr[160, 20] = [0, 0, 255]
        # bottom_right (170, 160) -> Yellow
        img_arr[160, 170] = [255, 255, 0]

        self.test_img_name = "test_sample.jpg"
        test_img_path = os.path.join(self.temp_dir, self.test_img_name)
        Image.fromarray(img_arr).save(test_img_path)

        self.sample_row = pd.Series({
            "image_name": self.test_img_name,
            "top_left_x": 20 / 200,
            "top_left_y": 30 / 200,
            "top_right_x": 170 / 200,
            "top_right_y": 30 / 200,
            "bottom_left_x": 20 / 200,
            "bottom_left_y": 160 / 200,
            "bottom_right_x": 170 / 200,
            "bottom_right_y": 160 / 200,
        })

    def tearDown(self):
        augment.PATH = self.orig_path
        shutil.rmtree(self.temp_dir)

    def test_independent_file_naming(self):
        """Test that generated names are independent and do not cascade."""
        # Force all augmentations to trigger
        res = augment.augment(self.sample_row)
        generated_names = list(res["image_name"])

        # Check each name starts with its own prefix and ends with base name
        for name in generated_names:
            prefix = name.split("_")[0]
            self.assertIn(prefix, ["ns", "hf", "vf", "r90", "r180", "r270"])
            # Ensure name does NOT contain cascaded prefixes like 'hf_vf_ns_'
            self.assertEqual(name, f"{prefix}_{self.test_img_name}")

    def test_horizontal_flip_coordinates(self):
        """Test horizontal flip geometry."""
        res = augment.augment(self.sample_row)
        hf_row = res[res["image_name"] == f"hf_{self.test_img_name}"]
        if not hf_row.empty:
            r = hf_row.iloc[0]
            # Old top_right (x=170/200, y=30/200) flipped becomes top_left (x=30/200, y=30/200)
            self.assertAlmostEqual(r["top_left_x"], 1.0 - 170 / 200)
            self.assertAlmostEqual(r["top_left_y"], 30 / 200)

            # Old top_left (x=20/200, y=30/200) flipped becomes top_right (x=180/200, y=30/200)
            self.assertAlmostEqual(r["top_right_x"], 1.0 - 20 / 200)
            self.assertAlmostEqual(r["top_right_y"], 30 / 200)

            # Old bottom_right becomes bottom_left
            self.assertAlmostEqual(r["bottom_left_x"], 1.0 - 170 / 200)
            self.assertAlmostEqual(r["bottom_left_y"], 160 / 200)

            # Old bottom_left becomes bottom_right
            self.assertAlmostEqual(r["bottom_right_x"], 1.0 - 20 / 200)
            self.assertAlmostEqual(r["bottom_right_y"], 160 / 200)

    def test_vertical_flip_coordinates(self):
        """Test vertical flip geometry."""
        res = augment.augment(self.sample_row)
        vf_row = res[res["image_name"] == f"vf_{self.test_img_name}"]
        if not vf_row.empty:
            r = vf_row.iloc[0]
            # Old bottom_left (x=20/200, y=160/200) flipped becomes top_left (x=20/200, y=40/200)
            self.assertAlmostEqual(r["top_left_x"], 20 / 200)
            self.assertAlmostEqual(r["top_left_y"], 1.0 - 160 / 200)

            # Old bottom_right becomes top_right
            self.assertAlmostEqual(r["top_right_x"], 170 / 200)
            self.assertAlmostEqual(r["top_right_y"], 1.0 - 160 / 200)

            # Old top_left becomes bottom_left
            self.assertAlmostEqual(r["bottom_left_x"], 20 / 200)
            self.assertAlmostEqual(r["bottom_left_y"], 1.0 - 30 / 200)

            # Old top_right becomes bottom_right
            self.assertAlmostEqual(r["bottom_right_x"], 170 / 200)
            self.assertAlmostEqual(r["bottom_right_y"], 1.0 - 30 / 200)


if __name__ == "__main__":
    unittest.main()
