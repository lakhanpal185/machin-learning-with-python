#this file is for processing the data which is available on the kaggle (fashion mnist) 


import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

def load_and_save_images(folder_path, csv_path):
    images = []
    labels = []

    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            image_files = [img_name for img_name in os.listdir(class_path)
                           if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_name in image_files:
                img_path = os.path.join(class_path, img_name)
                # Load image in grayscale and resize to a fixed size (e.g., 28x28)
                img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
                img_array = image.img_to_array(img)
                img_array = img_array.flatten().astype(int)  # Flatten for CSV
                images.append(img_array)
                labels.append(class_folder)

    # Convert to numpy array
    images = np.array(images)
    labels = np.array(labels).reshape(-1, 1)

    # Combine images and labels
    print(f"Loaded {len(images)} images from {folder_path}")
    data = np.hstack((images, labels))

    # Save to CSV
    columns = [f'pixel_{i}' for i in range(images.shape[1])] + ['label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_path, index=False)

    print(f"Saved {len(images)} images to {csv_path}")
    return images, labels



folder = os.path.dirname(os.path.abspath(__file__))
csv_output = os.path.join(folder, "output_images.csv")
folder = os.path.join(folder, "train")

load_and_save_images(folder, csv_output)
