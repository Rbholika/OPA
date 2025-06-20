import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
input_folder = "input_folder"
output_folder = "output_folder"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Data Augmentation Configuration
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=lambda x: cv2.GaussianBlur(x, (3,3), 0)
)

# Load images and apply augmentation
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img = img / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            it = datagen.flow(img, batch_size=1)

            # Save multiple augmented versions
            for i in range(3):  # Generates 3 augmented images per original
                aug_img = next(it)[0]
                aug_img = (aug_img * 255).astype(np.uint8)  # Convert back to image
                output_path = os.path.join(output_folder, f"aug_{i}_{filename}")
                cv2.imwrite(output_path, aug_img)
                print(f"Saved augmented image: {output_path}")
        else:
            print(f"Failed to load {img_path}")

print("Data augmentation completed.")
