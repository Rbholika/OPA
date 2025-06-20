import os
import cv2
import numpy as np
import random

# Define input and output folders
input_folder = "input_folder"  # Path to input images
output_folder = "output_folder"  # Path to save attacked images

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to perform one-pixel attack
def one_pixel_attack(image):
    height, width, _ = image.shape  # Get image dimensions
    x, y = random.randint(0, width - 1), random.randint(0, height - 1)  # Select random pixel
    random_color = np.random.randint(0, 256, 3)  # Generate random RGB color
    image[y, x] = random_color  # Modify pixel
    return image

# Process all images in input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)  # Read image
        
        if img is not None:
            attacked_img = one_pixel_attack(img)  # Apply one-pixel attack
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, attacked_img)  # Save attacked image
            print(f"Saved attacked image: {output_path}")
        else:
            print(f"Failed to load {img_path}")

print("One-pixel attack completed on all images.")
