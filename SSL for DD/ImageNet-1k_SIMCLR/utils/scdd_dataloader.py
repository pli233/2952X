import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision

class SCDDImageNetDataset(Dataset):
    """
    A custom Dataset class for loading the SCDD-ImageNet dataset.

    The dataset is organized into folders where each folder corresponds to a class (e.g., new000, new001),
    and each folder contains images belonging to that class.

    Args:
        root_dir (string): The root directory containing the dataset folders.
        transform (callable, optional): A transform to be applied to the images.

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns a single image and its corresponding label.
        _prepare_dataset(): Prepares the dataset by iterating through the directory to collect image paths and labels.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # List to store all image file paths
        self.labels = []  # List to store labels corresponding to images
        self.classes = []  # List to store class names (folders)
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Traverse the root directory to gather all image file paths and their corresponding labels.
        The folder name is used as the class label.
        """
        for label_folder in os.listdir(self.root_dir):
            label_folder_path = os.path.join(self.root_dir, label_folder)
            if os.path.isdir(label_folder_path):
                self.classes.append(label_folder)
                for img_file in os.listdir(label_folder_path):
                    if img_file.endswith('.jpg'):  # Process only .jpg files
                        img_path = os.path.join(label_folder_path, img_file)
                        self.image_paths.append(img_path)
                        self.labels.append(label_folder)  # Folder name is the label

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            image (Tensor): Transformed image.
            label (str): Corresponding label (class name).
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # Open the image and convert to RGB

        # Apply the transformation twice to generate two views
        if self.transform:
            image1 = self.transform(image)  # First augmented view
            image2 = self.transform(image)  # Second augmented view

        return (image1, image2), label  # Return both views as a tuple


def load_scdd_train_dataset(data_path=None, batch_size=32, num_workers= 4, ipc=50, num_classes=1000, shuffle=True):
    """
    Loads the SCDD-ImageNet dataset and returns a DataLoader along with dataset statistics.

    Args:
        data_path (str): Path to the dataset directory.
        ipc (int): Number of images per class (default: 50).
        num_classes (int): Number of classes to load (default: 1000).
        batch_size (int): Batch size for the DataLoader (default: 32).

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
        int: Total number of images loaded.
        int: Number of classes.
        int: Images Per Class (IPC).
        :param ipc:
        :param num_classes:
        :param batch_size:
        :param num_workers:
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (ImageNet standard size)
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    dataset = SCDDImageNetDataset(root_dir=data_path, transform=transform)

    # Limit images per class (IPC) if specified
    if ipc:
        limited_image_paths = []
        limited_labels = []
        for class_name in dataset.classes[:num_classes]:  # Limit to first num_classes
            class_images = [img for img, label in zip(dataset.image_paths, dataset.labels) if label == class_name]
            limited_image_paths.extend(class_images[:ipc])
            limited_labels.extend([class_name] * min(len(class_images), ipc))
        dataset.image_paths = limited_image_paths
        dataset.labels = limited_labels

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle= shuffle,
        num_workers= num_workers,
    )

    # Print dataset statistics
    print(f"Total Images Loaded: {len(dataset.image_paths)}")
    print(f"Number of Classes: {len(dataset.classes)}")
    print(f"Images Per Class (IPC): {ipc}")

    return dataloader


def show_image_grid(img):
    """
    Display a batch of images using matplotlib.

    Args:
        img (Tensor): Image batch in tensor form to display.
    """
    img = img.numpy().transpose((1, 2, 0))  # Convert to HWC format
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)  # Unnormalize
    plt.imshow(img)
    plt.axis('off')  # Turn off axis labels
    plt.show()


def test_scdd_data_loading(data_path=None, ipc=50, num_classes=1000, batch_size=32, num_workers= 4):
    """
    Test function to verify if the SCDD dataset is loaded properly and display the first batch of images.

    Args:
        data_path (str): The root path to the dataset.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
        int: Total number of images.
        int: Number of classes.
        int: IPC (Images Per Class).
    """
    pass
