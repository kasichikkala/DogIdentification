import os
import shutil
from sklearn.model_selection import train_test_split


root_dir = r"C:\Users\11312\PycharmProjects\dog_classification\images\Images"

test_dir = os.path.join(root_dir, '..\\test')


os.makedirs(test_dir, exist_ok=True)


for class_folder in os.listdir(root_dir):
    class_path = os.path.join(root_dir, class_folder)

    if not os.path.isdir(class_path):
        continue

    class_img_paths = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.jpg')]


    train_class_img_paths, test_class_img_paths = train_test_split(
        class_img_paths, test_size=0.1, random_state=42, stratify=None
    )

    for img_path in test_class_img_paths:
        class_folder_test = os.path.join(test_dir, class_folder)
        os.makedirs(class_folder_test, exist_ok=True)

        target_path = os.path.join(class_folder_test, os.path.basename(img_path))

        shutil.move(img_path, target_path)

