import os
import shutil
import random
from pathlib import Path
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# Configuration
if Path("data_raw").exists():
    data_dir_raw = Path("data_raw")
    data_dir = Path("data")
else:
    data_dir_raw = Path("edge_detection/data_raw")
    data_dir = Path("edge_detection/data")

train_dir = data_dir / "edges_train"
test_dir = data_dir / "edges_test"
val_dir = data_dir / "edges_validation"

for split_dir in [train_dir, test_dir, val_dir]:
    split_dir.mkdir(parents=True, exist_ok=True)

# Distribution percentages
train_pct = 0.7
test_pct = 0.15
val_pct = 0.15

df_images = pd.read_csv(data_dir_raw / "cube_corners.csv", index_col=False)
df_images = df_images.sample(frac=1).reset_index(drop=True)
images = df_images["image_name"].tolist()

if not images:
    print(f"No images found in {data_dir_raw}...")
else:
    # Calculate split indices
    total = len(images)
    train_count = int(total * train_pct)
    test_count = int(total * test_pct)

    # Split images
    train_images = images[:train_count]
    df_train = df_images.iloc[:train_count]

    test_images = images[train_count:train_count + test_count]
    df_test = df_images.iloc[train_count: train_count + test_count]

    val_images = images[train_count + test_count:]
    df_val = df_images.iloc[train_count + test_count:]

    # Copy files to respective directories
    for img in train_images:
        src = data_dir_raw / img
        dst = train_dir / img
        shutil.copy2(src, dst)
    df_train.to_csv(train_dir / "edges_train.csv", index=None)

    for img in test_images:
        src = data_dir_raw / img
        dst = test_dir / img
        shutil.copy2(src, dst)
    df_test.to_csv(test_dir / "edges_test.csv", index=None)

    for img in val_images:
        src = data_dir_raw / img
        dst = val_dir / img
        shutil.copy2(src, dst)
    df_val.to_csv(val_dir / "edges_validation.csv", index=None)

    print("\nDistribution complete!")
    print(f"Train set: {train_dir}")
    print(f"Test set: {test_dir}")
    print(f"Validation set: {val_dir}")