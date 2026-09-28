import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

# Configuration
data_dir = Path("data")
color_dirs = ["blue_samples", "white_samples", "red_samples", "green_samples", 
              "orange_samples", "yellow_samples"]
train_dir = data_dir / "colors_train"
test_dir = data_dir / "colors_test"
val_dir = data_dir / "colors_validation"

# Create output directories if they don't exist
for split_dir in [train_dir, test_dir, val_dir]:
    split_dir.mkdir(parents=True, exist_ok=True)
    # Create color subdirectories in each split
    for color in color_dirs:
        (split_dir / color).mkdir(exist_ok=True)

# Distribution percentages
train_pct = 0.8
test_pct = 0.1
val_pct = 0.1

# Process each color directory
for color in color_dirs:
    color_path = data_dir / color
    
    if not color_path.exists():
        print(f"Warning: {color_path} not found, skipping...")
        continue
    
    # Get all image files
    images = [f for f in os.listdir(color_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    if not images:
        print(f"No images found in {color_path}, skipping...")
        continue
    
    # Shuffle images
    random.shuffle(images)
    
    # Calculate split indices
    total = len(images)
    train_count = int(total * train_pct)
    test_count = int(total * test_pct)
    
    # Split images
    train_images = images[:train_count]
    test_images = images[train_count:train_count + test_count]
    val_images = images[train_count + test_count:]
    
    # Copy files to respective directories
    for img in train_images:
        src = color_path / img
        dst = train_dir / color / img
        shutil.copy2(src, dst)
    
    for img in test_images:
        src = color_path / img
        dst = test_dir / color / img
        shutil.copy2(src, dst)
    
    for img in val_images:
        src = color_path / img
        dst = val_dir / color / img
        shutil.copy2(src, dst)
    
    print(f"{color}: {len(train_images)} train, {len(test_images)} test, {len(val_images)} val")

print("\nDistribution complete!")
print(f"Train set: {train_dir}")
print(f"Test set: {test_dir}")
print(f"Validation set: {val_dir}")