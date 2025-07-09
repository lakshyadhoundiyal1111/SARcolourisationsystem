import os
import cv2
from tqdm import tqdm

# Source and target directories
source_dir = './dataset/EuroSAT'  # or '/kaggle/input/eurosat/dataset/EuroSAT' if on Kaggle
target_dir = './dataset/EuroSAT_gray_scale'

# Create the target root directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Loop over all class folders
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # skip files

    gray_class_name = f"{class_name}_gray_scale"
    gray_class_path = os.path.join(target_dir, gray_class_name)
    os.makedirs(gray_class_path, exist_ok=True)

    print(f"Converting class: {class_name}")

    for img_name in tqdm(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_name)
        gray_img_path = os.path.join(gray_class_path, img_name)

        # Read and convert
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Couldn't read {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(gray_img_path, gray)
