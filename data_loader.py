# data_loader.py

import os
import numpy as np
import cv2

def load_data(color_dataset_path, gray_scale_dataset_path, class_folders, size=(64, 64)):
    X, Y = [], []
    for folder in class_folders:
        color_folder = os.path.join(color_dataset_path, folder)
        gray_folder = os.path.join(gray_scale_dataset_path, f"{folder}_gray_scale")

        if not os.path.exists(color_folder) or not os.path.exists(gray_folder):
            print(f"Skipping missing folder: {color_folder} or {gray_folder}")
            continue

        color_files = os.listdir(color_folder)

        for filename in color_files:
            color_path = os.path.join(color_folder, filename)
            gray_path = os.path.join(gray_folder, filename)

            if not os.path.isfile(color_path) or not os.path.isfile(gray_path):
                continue

            color_img = cv2.imread(color_path)
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)

            if color_img is None or gray_img is None:
                continue

            color_img = cv2.resize(color_img, size)
            gray_img = cv2.resize(gray_img, size)
            gray_img = np.expand_dims(gray_img, axis=-1)

            X.append(gray_img / 255.0)
            Y.append(color_img / 255.0)

    return np.array(X), np.array(Y)
