from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = datasets.ImageFolder("./Processed_Data", transform)

trainingSize = int(0.8 * len(dataset))
testingSize = len(dataset) - trainingSize
trainingDataset, testingDataset = torch.utils.data.random_split(dataset, [trainingSize, testingSize])

trainLoader = DataLoader(trainingDataset, batch_size=32, shuffle=True)
testLoader = DataLoader(testingDataset, batch_size=32, shuffle=False)
