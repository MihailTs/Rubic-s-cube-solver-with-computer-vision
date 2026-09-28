import os
import cv2
import numpy as np
from pathlib import Path

class ImageSampler:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.sample_count = 1470
        self.current_image = None
        self.current_image_path = None

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Get list of image files
        self.image_files = [f for f in sorted(os.listdir(input_folder)) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.jfif'))]
        self.current_index = 0
    
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Extract 20x20 patch centered at click position
            h, w = self.current_image.shape[:2]
            
            # Calculate patch boundaries
            left = max(0, x - 10)
            right = min(w, x + 10)
            top = max(0, y - 10)
            bottom = min(h, y + 10)
            
            # Handle edge cases - pad if necessary
            patch = self.current_image[top:bottom, left:right]
            
            # Ensure patch is exactly 20x20 by padding
            if patch.shape[0] < 20 or patch.shape[1] < 20:
                pad_top = max(0, 5 - y)
                pad_bottom = max(0, (y + 5) - h)
                pad_left = max(0, 5 - x)
                pad_right = max(0, (x + 5) - w)
                
                if len(patch.shape) == 3:
                    patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                                  mode='edge')
                else:  # Grayscale
                    patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                  mode='edge')
            
            # Resize to exactly 20x20 if needed
            patch = cv2.resize(patch, (20, 20))
            
            # Save patch
            self.sample_count += 1
            output_path = os.path.join(self.output_folder, f'sample_{self.sample_count:04d}.png')
            cv2.imwrite(output_path, patch)
            print(f"Saved: {output_path}")

            # Draw circle to mark the place already clicked
            cv2.circle(self.current_image, (x, y), 10, (0, 255, 0), 2)
            cv2.imshow("Image Sampler - Click to sample patches", self.current_image)
    
    def process_images(self):
        if not self.image_files:
            print("No images found in the input folder!")
            return
        
        while self.current_index < len(self.image_files):
            image_path = os.path.join(self.input_folder, self.image_files[self.current_index])
            self.current_image = cv2.imread(image_path)
            self.current_image_path = image_path
            
            if self.current_image is None:
                print(f"Failed to load image: {image_path}")
                self.current_index += 1
                continue
            
            window_name = "Image Sampler - Click to sample patches"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.setMouseCallback(window_name, self.on_mouse)
            cv2.imshow(window_name, self.current_image)
            
            # Wait for window to close or key press
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyWindow(window_name)
                    break
                if cv2.getWindowProperty(window_name, 0) < 0:
                    break
            
            self.current_index += 1
        
        print(f"\nFinished! Total samples collected: {self.sample_count}")

if __name__ == "__main__":
    input_folder = "images/solved/white"
    # <color>_samples - the name of the created folder
    output_folder = "data/white_samples"
    
    sampler = ImageSampler(input_folder, output_folder)
    sampler.process_images()