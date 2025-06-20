import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

# Paths
input_folder = "input_folder"
output_folder = "output_folder"
adversarial_folder = "adversarial_folder"
results_folder = "results"

# Create necessary folders
os.makedirs(adversarial_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# Function to load images from a folder
def load_images_from_folder(folder):
    images, labels = [], []
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (32, 32))  # Resize for model
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(0)  # Placeholder label
    return np.array(images), np.array(labels)

# Load clean & attacked images
x_clean, y_clean = load_images_from_folder(input_folder)
x_attacked, y_attacked = load_images_from_folder(output_folder)

# Merge datasets
x_train = np.concatenate((x_clean, x_attacked))
y_train = np.concatenate((y_clean, y_attacked))

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Placeholder 10-class output
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Generate Adversarial Examples using FGSM
def generate_adversarial_examples(model, x_data):
    adv_examples = fast_gradient_method(model, x_data, 0.02, np.inf)  # FGSM Attack
    return adv_examples

x_adv = generate_adversarial_examples(model, x_clean)

# Save adversarial images
for i, img in enumerate(x_adv):
    adv_img = (img.numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(adversarial_folder, f"adv_{i}.png"), adv_img)

# Evaluate Model
test_loss, test_acc = model.evaluate(x_clean, y_clean)
test_loss_adv, test_acc_adv = model.evaluate(x_adv, y_clean)

# Save results
with open(os.path.join(results_folder, "accuracy_report.txt"), "w") as f:
    f.write(f"Test Accuracy on Clean Images: {test_acc:.2f}\n")
    f.write(f"Test Accuracy on Adversarial Images: {test_acc_adv:.2f}\n")

print(f"Results saved in {results_folder}")
