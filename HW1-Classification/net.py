import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class my_network(nn.Module):
    def __init__(self):
        super(my_network, self).__init__()

        self.main = nn.Sequential(
            # input size: (3, 64, 64)
            nn.Conv2d(3, 10, 4, 2, 1, bias=False),  # (10, 32, 32)
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # (10, 12, 12)

            nn.Conv2d(10, 10, 4, 2, 1, bias=False), # (10, 12, 12)
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                     # (10, 4, 4)

            nn.Conv2d(10, 10, 4, 2, 1, bias=False), # (10, 2, 2)
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),

            nn.Flatten(),                           # (40)
            nn.Linear(40, 12),                      # (12)
        )

    def forward(self, x):
        return self.main(x)

class my_dataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        image = Image.open(file_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.labels is not None:
            return image, self.labels[index]
        else:
            return image
