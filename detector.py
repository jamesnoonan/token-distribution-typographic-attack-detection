import torch
import torch.nn as nn
from torch.utils.data import Dataset

from os import listdir
from os.path import isfile, join

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(SimpleModel, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.hidden3 = nn.Linear(hidden_size2, hidden_size3)
        self.output = nn.Linear(hidden_size3, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.output(x)
        return x

# Create Datasets
class CustomTensorDataset(Dataset):
    def __init__(self, tensor_dir, classes, transform=None, target_transform=None):
        self.tensor_dir = tensor_dir
        self.filenames = [str(f) for f in listdir(tensor_dir) if isfile(join(tensor_dir, f))]

        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.filenames)
    
    def __getlabel__(self, filename):
        return 0;

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        tensor = torch.load(self.tensor_dir + filename)
        label = self.__getlabel__(filename)

        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            label = self.target_transform(label)

        return tensor, label
    
class ImageTensorDataset(CustomTensorDataset):
    def __getlabel__(self, filename):
        image_class_text = filename.split("_")[0]
        image_class = torch.tensor(self.classes.index(image_class_text))
        return image_class
    
class TextTensorDataset(CustomTensorDataset):
    def __getlabel__(self, filename):
        text_class_text = filename.split("_")[1]
        text_class = torch.tensor(self.classes.index(text_class_text))
        return text_class