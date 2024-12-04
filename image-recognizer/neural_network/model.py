from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, Flatten

class CNNClassifier(Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = Conv2d(3, 10, 5)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2, 2)

        self.conv2 = Conv2d(10, 10, 5)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2, 2)

        self.flatten = Flatten()
        self.fc1 = Linear(10 * 29 * 29, 256)
        self.relu3 = ReLU()
        self.fc2 = Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x
