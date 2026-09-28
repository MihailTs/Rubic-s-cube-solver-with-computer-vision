import os
import sys
import time
import argparse
import yaml
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ----------------------------------------------------------------------
# 1. Dataset definition
# ----------------------------------------------------------------------
class CubeCornerDataset(Dataset):
    """
    Custom Dataset for loading cube images and their corner coordinate annotations.
    """
    def __init__(self, csv_file, img_dir, img_size=(224, 224), transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV annotation file not found: {csv_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        
        self.target_cols = [
            "top_left_x", "top_left_y",
            "top_right_x", "top_right_y",
            "bottom_left_x", "bottom_left_y",
            "bottom_right_x", "bottom_right_y"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_name"]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image {img_path}: {e}")
            
        original_size = image.size  # (width, height)
        
        # Ground truth normalized coordinates in [0, 1]
        labels = row[self.target_cols].values.astype(np.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels, dtype=torch.float32), img_name, original_size


# ----------------------------------------------------------------------
# 2. Model definition (Custom lightweight CNN)
# ----------------------------------------------------------------------
class CustomCNN(nn.Module):
    """
    A lightweight, robust custom convolutional network for regression tasks.
    Uses AdaptiveAvgPool2d to remain resolution-independent.
    """
    def __init__(self, num_classes=8):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 -> 64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 -> 32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 -> 16
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()  # Restricts predictions strictly to [0, 1] for coordinates
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


# ----------------------------------------------------------------------
# 3. Model Factory
# ----------------------------------------------------------------------
def get_model(architecture_name, pretrained=True, num_classes=8):
    """
    Factory function to instantiate models.
    Supports 'custom_cnn', 'resnet18', 'resnet34', and 'mobilenet_v3'.
    """
    if architecture_name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
        
    elif architecture_name == "resnet18":
        try:
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                model = models.resnet18(weights=None)
        except AttributeError:
            # Fallback for older torchvision versions
            model = models.resnet18(pretrained=pretrained)
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )
        return model
        
    elif architecture_name == "resnet34":
        try:
            if pretrained:
                model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            else:
                model = models.resnet34(weights=None)
        except AttributeError:
            model = models.resnet34(pretrained=pretrained)
            
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )
        return model
        
    elif architecture_name == "mobilenet_v3":
        try:
            if pretrained:
                model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            else:
                model = models.mobilenet_v3_small(weights=None)
        except AttributeError:
            model = models.mobilenet_v3_small(pretrained=pretrained)
            
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )
        return model
        
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")


# ----------------------------------------------------------------------
# 4. Helpers for Optimizers, Schedulers, and Losses
# ----------------------------------------------------------------------
def get_optimizer(model, opt_name, lr, weight_decay):
    opt_name = opt_name.lower()
    lr = float(lr)
    weight_decay = float(weight_decay)
    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def get_scheduler(optimizer, sched_name, epochs):
    if not sched_name or sched_name.lower() == "none":
        return None
    sched_name = sched_name.lower()
    if sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")

def get_loss_fn(loss_name):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# ----------------------------------------------------------------------
# 5. Metrics & Visualizations
# ----------------------------------------------------------------------
def calculate_pixel_error(preds, targets, original_sizes):
    """
    Calculates the average Euclidean distance (in pixels) for the 4 corners
    across a batch of images.
    preds: Tensor of shape (B, 8) in [0, 1]
    targets: Tensor of shape (B, 8) in [0, 1]
    original_sizes: Tuple of (widths, heights), where each is a tensor/list of size B
    """
    B = preds.shape[0]
    preds_coords = preds.view(B, 4, 2).detach().cpu().numpy()
    targets_coords = targets.view(B, 4, 2).detach().cpu().numpy()
    
    widths, heights = original_sizes
    widths = widths.numpy() if torch.is_tensor(widths) else np.array(widths)
    heights = heights.numpy() if torch.is_tensor(heights) else np.array(heights)
    
    batch_errors = []
    for i in range(B):
        w = widths[i]
        h = heights[i]
        
        p_px = preds_coords[i] * np.array([w, h])
        t_px = targets_coords[i] * np.array([w, h])
        
        distances = np.sqrt(np.sum((p_px - t_px) ** 2, axis=1))
        batch_errors.append(np.mean(distances))
        
    return np.mean(batch_errors)


def save_sample_visualizations(model, val_loader, device, output_dir, num_samples=8):
    """
    Overlays ground-truth corners (Green) and predictions (Red) on test/validation
    images, saving them to the experiment's visualizations directory.
    """
    model.eval()
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    samples_saved = 0
    with torch.no_grad():
        for images, targets, img_names, (widths, heights) in val_loader:
            images = images.to(device)
            preds = model(images)
            
            preds_coords = preds.view(-1, 4, 2).cpu().numpy()
            targets_coords = targets.view(-1, 4, 2).cpu().numpy()
            
            for i in range(images.shape[0]):
                if samples_saved >= num_samples:
                    return
                
                img_name = img_names[i]
                img_path = os.path.join(val_loader.dataset.img_dir, img_name)
                
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    continue
                
                h, w = img_cv.shape[:2]
                
                # Plot ground truth in Green
                for pt in targets_coords[i]:
                    x_px = int(pt[0] * w)
                    y_px = int(pt[1] * h)
                    cv2.circle(img_cv, (x_px, y_px), 20, (0, 255, 0), -1)
                
                # Plot predictions in Red
                for pt in preds_coords[i]:
                    x_px = int(pt[0] * w)
                    y_px = int(pt[1] * h)
                    cv2.circle(img_cv, (x_px, y_px), 20, (0, 0, 255), -1)
                
                # Save the image
                save_path = os.path.join(vis_dir, f"val_{img_name}")
                cv2.imwrite(save_path, img_cv)
                samples_saved += 1


# ----------------------------------------------------------------------
# 6. Core Experiment Runner
# ----------------------------------------------------------------------
def train_experiment(config):
    """
    Loads data, compiles model, runs the training-validation loop,
    plots metrics, and saves results/checkpoints according to config.
    """
    exp_name = config.get("experiment_name", "unnamed_experiment")
    output_dir = os.path.join("experiments", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=========================================")
    print(f" Starting Experiment: {exp_name}")
    print(f"=========================================")
    
    # Save a copy of the configuration in the output directory
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)
        
    # Extract config sections
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    hp_cfg = config.get("hyperparameters", {})
    
    img_size = tuple(data_cfg.get("img_size", [224, 224]))
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Datasets & DataLoaders
    train_dataset = CubeCornerDataset(
        csv_file=data_cfg.get("train_csv"),
        img_dir=data_cfg.get("train_dir"),
        img_size=img_size,
        transform=train_transform
    )
    val_dataset = CubeCornerDataset(
        csv_file=data_cfg.get("val_csv"),
        img_dir=data_cfg.get("val_dir"),
        img_size=img_size,
        transform=val_transform
    )
    
    batch_size = int(hp_cfg.get("batch_size", 16))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model compilation
    model = get_model(
        architecture_name=model_cfg.get("architecture", "resnet18"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 8)
    )
    model = model.to(device)
    
    # Optimizer, loss, scheduler
    loss_fn = get_loss_fn(hp_cfg.get("loss_fn", "mse"))
    optimizer = get_optimizer(
        model=model,
        opt_name=hp_cfg.get("optimizer", "adam"),
        lr=hp_cfg.get("learning_rate", 0.001),
        weight_decay=hp_cfg.get("weight_decay", 1e-4)
    )
    
    epochs = int(hp_cfg.get("epochs", 15))
    scheduler = get_scheduler(optimizer, hp_cfg.get("scheduler", "cosine"), epochs)
    
    # Log files
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_pixel_err": [],
        "val_pixel_err": []
    }
    
    best_val_loss = float("inf")
    
    # Training cycle
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # --- Training Epoch ---
        model.train()
        train_loss = 0.0
        train_pixel_err = 0.0
        
        for images, targets, _, original_sizes in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_pixel_err += calculate_pixel_error(outputs, targets, original_sizes) * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        train_pixel_err = train_pixel_err / len(train_loader.dataset)
        
        # --- Validation Epoch ---
        model.eval()
        val_loss = 0.0
        val_pixel_err = 0.0
        
        with torch.no_grad():
            for images, targets, _, original_sizes in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                
                val_loss += loss.item() * images.size(0)
                val_pixel_err += calculate_pixel_error(outputs, targets, original_sizes) * images.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_pixel_err = val_pixel_err / len(val_loader.dataset)
        
        if scheduler:
            scheduler.step()
            
        elapsed = time.time() - start_time
        
        # Track history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pixel_err"].append(train_pixel_err)
        history["val_pixel_err"].append(val_pixel_err)
        
        # Logging
        lr_curr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
              f"Train MPE: {train_pixel_err:.2f}px | Val MPE: {val_pixel_err:.2f}px | "
              f"LR: {lr_curr:.5f} | {elapsed:.1f}s")
              
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"--> Saved best model checkpoint (Val Loss improved to {best_val_loss:.5f})")
            
    # Save the training metrics history to CSV
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    # Save final plotted curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{exp_name} - Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_pixel_err"], label="Train MPE")
    plt.plot(history["epoch"], history["val_pixel_err"], label="Val MPE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Pixel Error (px)")
    plt.title(f"{exp_name} - Corner Error")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    plt.close()
    
    # Save visual predictions on some validation images for review
    print("Generating validation corner predictions images for visual check...")
    best_model = get_model(
        architecture_name=model_cfg.get("architecture", "resnet18"),
        pretrained=False,
        num_classes=model_cfg.get("num_classes", 8)
    )
    best_model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth"), weights_only=True))
    best_model = best_model.to(device)
    
    save_sample_visualizations(best_model, val_loader, device, output_dir, num_samples=8)
    print(f"Experiment finished successfully. Output directory: '{output_dir}'")
    print(f"Best Validation Loss: {best_val_loss:.5f}\n")
    
    return best_val_loss


# ----------------------------------------------------------------------
# 7. CLI Entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PyTorch Corner Detection Experiment Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to a single YAML configuration file")
    parser.add_argument("--run-all", type=str, default=None, help="Directory containing multiple YAML files to run")
    
    args = parser.parse_args()
    
    if args.config is None and args.run_all is None:
        # Default behavior if nothing specified: look for default config or ask
        default_config_path = "configs/custom_cnn_light.yaml"
        if os.path.exists(default_config_path):
            args.config = default_config_path
            print(f"No config specified. Using default: {default_config_path}")
        else:
            parser.print_help()
            sys.exit(1)
            
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found at {args.config}")
            sys.exit(1)
            
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            
        train_experiment(config)
        
    elif args.run_all:
        if not os.path.isdir(args.run_all):
            print(f"Error: Folder not found at {args.run_all}")
            sys.exit(1)
            
        yaml_files = [
            os.path.join(args.run_all, f)
            for f in sorted(os.listdir(args.run_all))
            if f.endswith((".yaml", ".yml"))
        ]
        
        if not yaml_files:
            print(f"No YAML configuration files found in {args.run_all}")
            sys.exit(1)
            
        print(f"Found {len(yaml_files)} experiment configs in {args.run_all}")
        
        results = {}
        for config_path in yaml_files:
            print(f"\nProcessing configuration: {config_path}")
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                best_val_loss = train_experiment(config)
                results[config["experiment_name"]] = best_val_loss
            except Exception as e:
                print(f"!!! Error running experiment {config_path}: {e}")
                
        print("\n=========================================")
        print(" Summary of Completed Experiments")
        print("=========================================")
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        for name, loss in sorted_results:
            print(f"Experiment: {name:<30} | Best Val Loss: {loss:.5f}")
        print("=========================================")


if __name__ == "__main__":
    main()
