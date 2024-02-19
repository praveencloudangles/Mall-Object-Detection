import os
import glob
import cv2
import numpy as np
import pickle
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        self.all_image_paths = []
        
        # Get all the image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.images_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.images_path, image_name)

        # Read the image
        image = cv2.imread(image_path)
        # Convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transform:
            sample = self.transform(image=image)
            image = sample['image']

        return image

    def __len__(self):
        return len(self.all_images)

# Define transformations for data augmentation
def get_train_transform():
    return A.Compose([
        A.Resize(256, 256),  # Resize the image
        A.HorizontalFlip(p=0.5),  # Horizontal flip with probability 0.5
        A.RandomRotate90(p=0.5),  # Random rotation by 90 degrees with probability 0.5
        A.Normalize(),  # Normalize the image
        ToTensorV2(),  # Convert image to PyTorch tensor
    ])

# Path to the folder containing your images
images_folder_path = "object_detection_data/train/images"

# Initialize the dataset with your images folder and transformations
train_dataset = CustomDataset(images_path=images_folder_path, transform=get_train_transform())

# Save the dataset object into a pickle file
pickle_file_path = "train_dataset.pkl"
with open(pickle_file_path, 'wb') as f:
    pickle.dump(train_dataset, f)

print("Dataset saved as pickle file:", pickle_file_path)
