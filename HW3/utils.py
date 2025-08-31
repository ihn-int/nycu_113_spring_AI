from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
from typing import List, Tuple
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, images, labels):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

class TestDataset(Dataset):
    def __init__(self, image):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = image

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name
    
# (TODO) Load training dataset from the given path, return images and labels
def load_train_dataset(path: str='data/train/')->Tuple[List, List]:
    images = []
    labels = []
    label_map = {
        "elephant": 0,
        "jaguar": 1,
        "lion": 2,
        "parrot": 3,
        "penguin": 4
    }

    images = []
    labels = []

    for animal in os.listdir(path):
        animal_path = os.path.join(path, animal)
        if os.path.isdir(animal_path) and animal in label_map:
            for img_file in os.listdir(animal_path):
                if img_file.lower().endswith('.jpg'):
                    img_path = os.path.join(animal_path, img_file)
                    images.append(img_path)
                    labels.append(label_map[animal])

    return images, labels

# (TODO) Load testing dataset from the given path, return images
def load_test_dataset(path: str='data/test/')->List:
    images = []
    for img_file in os.listdir(path):
        if img_file.lower().endswith('.jpg'):
            img_path = os.path.join(path, img_file)
            images.append(img_path)

    return images

# (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
#        xlabel: 'Epoch', ylabel: 'Loss'
def plot(train_losses: List, val_losses: List):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    plt.close()

    print("Save the plot to 'loss.png'")
    return