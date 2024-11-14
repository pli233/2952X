import os
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from datetime import datetime
import yaml

# Load the YAML file
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)
    
# Dataset definition
class SCDDImageNetDataset(Dataset):
    """
    A custom Dataset class for loading the SCDD-ImageNet dataset.

    The dataset is organized into folders where each folder corresponds to a class (e.g., new000, new001),
    and each folder contains images belonging to that class.

    Args:
        root_dir (string): The root directory containing the dataset folders.
        transform (callable, optional): A transform to be applied to the images.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = []  # List to store all image file paths
        self.labels = []  # List to store labels corresponding to images
        self.classes = []  # List to store class names (folders)
        
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to 224x224 (commonly used for ImageNet)
                transforms.ToTensor(),  # Convert images to PyTorch tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
            ])  # Convert images to tensors
        else:
            self.transform = transform
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
        
         # Convert the image to a tensor
        image = self.transform(image)
        return image, label
    
    
class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


def train_ImageNet_SimCLR(config):
    # Load settings from the config file
    SCDD_DATASET_PATH = config["data_path"]
    SCDD_BATCH_SIZE = config["batch_size"]
    SCDD_NUM_WORKERS = config["num_workers"]
    SCDD_EPOCHS = config["epochs"]
    SCDD_LEARNING_RATE = config["learning_rate"]
    SCDD_SAVE_PATH = config["save_path"]
    SCDD_Pretrain_Resnet = config["pretrain_resnet"]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not config["disable_cuda"] else 'cpu')

    # Print out all training parameters
    print("Training Parameters:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # Print the device being used
    if DEVICE.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Training on device: {gpu_name}")  # Print GPU name
    else:
        gpu_name = "CPU"
        print("Training on CPU")

    # Create a unique directory for this run based on timestamp and GPU name
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = f"{SCDD_SAVE_PATH}/run_{run_timestamp}_{gpu_name.replace(' ', '_')}"
    os.makedirs(run_directory, exist_ok=True)

    #Save config to the run dir
    config_file_path = os.path.join(run_directory, "config.txt")
    with open(config_file_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"Configuration saved to {config_file_path}")
    
    # Load the model and send to device
    resnet = torchvision.models.resnet18(pretrained=SCDD_Pretrain_Resnet)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = SimCLR(backbone)
    model.to(DEVICE)

    # Prepare the dataset and dataloader
    transform = SimCLRTransform(input_size=224)  # Adjust the input size for ImageNet
    scdd_dataset = SCDDImageNetDataset(root_dir=SCDD_DATASET_PATH, transform=transform)

    scdd_dataloader = DataLoader(
        scdd_dataset,
        batch_size=SCDD_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=SCDD_NUM_WORKERS,
    )

    # Set up the criterion and optimizer
    criterion = NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=SCDD_LEARNING_RATE)

    # Variables to track the best model
    best_loss = float('inf')
    best_checkpoint_path = os.path.join(run_directory, "best_checkpoint.pth")

    # Training Loop
    print("Starting Training")
    for epoch in range(SCDD_EPOCHS):
        model.train()
        total_loss = 0

        # Progress bar setup
        progress_bar = tqdm(enumerate(scdd_dataloader), total=len(scdd_dataloader), desc=f"Epoch {epoch + 1}/{SCDD_EPOCHS}")

        for i, batch in progress_bar:
            # Get batch data
            x0, x1 = batch[0]
            x0 = x0.to(DEVICE)
            x1 = x1.to(DEVICE)

            # Forward pass
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)

            # Backward pass and optimization
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update the progress bar
            avg_loss = total_loss / (i + 1)
            progress_bar.set_postfix(loss=avg_loss)

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{SCDD_EPOCHS}, Average Loss: {avg_loss:.5f}")

        # Save the model checkpoint for the current epoch
        epoch_checkpoint_path = os.path.join(run_directory, f"simclr_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Model saved to {epoch_checkpoint_path}")

        # Save the latest checkpoint (overwrite each epoch)
        latest_checkpoint_path = os.path.join(run_directory, "latest_checkpoint.pth")
        torch.save(model.state_dict(), latest_checkpoint_path)
        print(f"Latest model checkpoint updated at {latest_checkpoint_path}")

        # Check if this is the best model so far and save it separately
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"New best model saved to {best_checkpoint_path}")

if __name__ == "__main__":
    train_ImageNet_SimCLR(config=config)
