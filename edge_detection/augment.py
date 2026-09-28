import os
import random
from pathlib import Path
import pandas as pd
from skimage.util import random_noise
import numpy as np
from PIL import Image

# Path to raw dataset
if Path("data_raw").exists():
    PATH = "data_raw"
elif Path("edge_detection/data_raw").exists():
    PATH = "edge_detection/data_raw"
else:
    PATH = "data_raw"


def augment(x):
    # print(x)
    noise = random.random()  # gaussian noise
    horizontal_flip = random.random()
    vertical_flip = random.random()
    rotate_90 = random.random()  # 90-degree rotation
    rotate_180 = random.random()  # 180-degree rotation
    rotate_270 = random.random()  # 270-degree rotation

    base_name = x["image_name"]

    new_rows = pd.DataFrame(columns=[
        "image_name",
        "top_left_x",
        "top_left_y",
        "top_right_x",
        "top_right_y",
        "bottom_left_x",
        "bottom_left_y",
        "bottom_right_x",
        "bottom_right_y",
    ])

    img_path = os.path.join(PATH, base_name)
    pil_img = Image.open(img_path)
    exif_data = pil_img.info.get('exif', b'')
    img = np.array(pil_img)

    # 1. Gaussian Noise
    if noise < 0.7:
        noisy_img = random_noise(img, mode='gaussian', var=0.01)
        noisy_img = (noisy_img * 255).astype(np.uint8)

        new_name = 'ns_' + base_name
        output_img = Image.fromarray(noisy_img)
        output_img.save(os.path.join(PATH, new_name), exif=exif_data)

        new_rows.loc[len(new_rows)] = {
            "image_name": new_name,
            "top_left_x": x["top_left_x"],
            "top_left_y": x["top_left_y"],
            "top_right_x": x["top_right_x"],
            "top_right_y": x["top_right_y"],
            "bottom_left_x": x["bottom_left_x"],
            "bottom_left_y": x["bottom_left_y"],
            "bottom_right_x": x["bottom_right_x"],
            "bottom_right_y": x["bottom_right_y"],
        }

    # 2. Horizontal Flip (mirrors left <-> right across vertical axis)
    if horizontal_flip < 0.9:
        flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        new_name = 'hf_' + base_name
        flipped.save(os.path.join(PATH, new_name), exif=exif_data)

        new_rows.loc[len(new_rows)] = {
            "image_name": new_name,
            "top_left_x": x["top_left_x"],
            "top_left_y": 1 - x["top_left_y"],
            "top_right_x": x["top_right_x"],
            "top_right_y": 1 - x["top_right_y"],
            "bottom_left_x": x["bottom_left_x"],
            "bottom_left_y": 1 - x["bottom_left_y"],
            "bottom_right_x": x["bottom_right_x"],
            "bottom_right_y": 1 - x["bottom_right_y"],
        }

    # 3. Vertical Flip (mirrors top <-> bottom across horizontal axis)
    if vertical_flip < 0.9:
        flipped = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        new_name = 'vf_' + base_name
        flipped.save(os.path.join(PATH, new_name), exif=exif_data)

        new_rows.loc[len(new_rows)] = {
            "image_name": new_name,
            "top_left_x": 1 - x["top_left_x"],
            "top_left_y": x["top_left_y"],
            "top_right_x": 1 - x["top_right_x"],
            "top_right_y": x["top_right_y"],
            "bottom_left_x": 1 - x["bottom_left_x"],
            "bottom_left_y": x["bottom_left_y"],
            "bottom_right_x": 1 - x["bottom_right_x"],
            "bottom_right_y": x["bottom_right_y"],
        }

    # 4. Rotate 90 degrees CCW
    if rotate_90 < 0.9:
        flipped = pil_img.transpose(Image.ROTATE_90)
        new_name = 'r90_' + base_name
        flipped.save(os.path.join(PATH, new_name), exif=exif_data)

        new_rows.loc[len(new_rows)] = {
            "image_name": new_name,
            "top_left_x": x["top_right_y"],
            "top_left_y": 1 - x["top_right_x"],
            "top_right_x": x["bottom_right_y"],
            "top_right_y": 1 - x["bottom_right_x"],
            "bottom_left_x": x["top_left_y"],
            "bottom_left_y": 1 - x["top_left_x"],
            "bottom_right_x": x["bottom_left_y"],
            "bottom_right_y": 1 - x["bottom_left_x"],
        }

    # 5. Rotate 180 degrees
    if rotate_180 < 0.9:
        flipped = pil_img.transpose(Image.ROTATE_180)
        new_name = 'r180_' + base_name
        flipped.save(os.path.join(PATH, new_name), exif=exif_data)

        new_rows.loc[len(new_rows)] = {
            "image_name": new_name,
            "top_left_x": 1 - x["bottom_right_x"],
            "top_left_y": 1 - x["bottom_right_y"],
            "top_right_x": 1 - x["bottom_left_x"],
            "top_right_y": 1 - x["bottom_left_y"],
            "bottom_left_x": 1 - x["top_right_x"],
            "bottom_left_y": 1 - x["top_right_y"],
            "bottom_right_x": 1 - x["top_left_x"],
            "bottom_right_y": 1 - x["top_left_y"],
        }

    # 6. Rotate 270 degrees CCW
    if rotate_270 < 0.9:
        flipped = pil_img.transpose(Image.ROTATE_270)
        new_name = 'r270_' + base_name
        flipped.save(os.path.join(PATH, new_name), exif=exif_data)

        new_rows.loc[len(new_rows)] = {
            "image_name": new_name,
            "top_left_x": 1 - x["bottom_left_y"],
            "top_left_y": x["bottom_left_x"],
            "top_right_x": 1 - x["top_left_y"],
            "top_right_y": x["top_left_x"],
            "bottom_left_x": 1 - x["bottom_right_y"],
            "bottom_left_y": x["bottom_right_x"],
            "bottom_right_x": 1 - x["top_right_y"],
            "bottom_right_y": x["top_right_x"],
        }

    return new_rows


def main():
    image_files = [
        f
        for f in sorted(os.listdir(PATH))
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".jfif"))
    ]

    csv_path = os.path.join(PATH, 'cube_corners.csv')
    df = pd.read_csv(csv_path)
    augmented_rows = df.apply(augment, axis=1)

    df = pd.concat([df] + [r for r in augmented_rows if len(r) > 0], ignore_index=True)

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
