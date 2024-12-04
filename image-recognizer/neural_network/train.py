from dataloader import trainLoader
from model import CNNClassifier
import torch
import torch.optim as optim

def train(model, trainLoader, criterion, optimizer, numEpochs):
    model.train()
    for epoch in range(numEpochs):
        totalLoss = 0
        for images, labels in trainLoader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totalLoss += loss.item()
        print(f"{epoch+1}/{numEpochs} - Average Loss: {totalLoss / len(trainLoader)}")

model = CNNClassifier().to("cpu")
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train(model, trainLoader, criterion, optimizer, 20)

torch.save(model.state_dict(), "classifier_weights.pth")