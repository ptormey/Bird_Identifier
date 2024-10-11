import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch import optim
import IPython.display as ipd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import time
import glob
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
class BirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.birds = {}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path)
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        y_label = torch.tensor(int(self.annotations.iloc[index, 0]))
        if self.transform:
            image = self.transform(image)
        self.birds[image] = y_label
        return(image, y_label)

class Net(nn.Module):
    def __init__(self, depth=4, dim_img=224, in_channels=3, start_channels=64):
        super().__init__()
        out_channels = start_channels
        dimension_img = dim_img
        layers = []
        
        for i in range(depth):
            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            dimension_img/= 2
            in_channels = out_channels
            out_channels *= 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(int((dim_img // (2**depth))**2 * out_channels // 2), 525))
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)

net = Net(depth=4, dim_img=224, in_channels=3, start_channels=64)

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the tensor
])

datasetTrain = BirdDataset('archive/birdsTrain.csv', root_dir = 'archive', transform = transform)
trainSamples = DataLoader(dataset = datasetTrain, batch_size = 128, shuffle = True)

datasetTest = BirdDataset('archive/birdsTest.csv', root_dir = 'archive', transform = transform)
testSamples = DataLoader(dataset = datasetTest, batch_size = 128, shuffle = True)

optimizer = optim.Adam(net.parameters(), lr=3e-4)
lossFunc = torch.nn.CrossEntropyLoss()

for i in range(30):
    print(i)
    j = 0
    for inputs, labels in trainSamples:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = lossFunc(outputs, labels)
        loss.backward()
        optimizer.step()
        print(j)
        j+=1
    j = 0
print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in samplesTest:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
