import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F


class SensorDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            ds = list()
            '''start = 0 if slice.start is None else slice.start
            stop = len(self.data) if slice.stop is None else slice.stop
            step = 1 if slice.step is None else slice.step'''
            for img in self.data[idx]:
                img1 = self.transform(img).float()
                ds.append((img1, img1))
            return ds
        elif isinstance(idx, int):
            image = self.data[idx]
            #print(idx)
            #print(len(self.data))
            if self.transform:
                image = self.transform(image).float()

            return image, image


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.descriptor = None
        # Encoder
        self.conv1 = nn.Conv2d(1, 1, (6, 4))
        self.conv2 = nn.Conv2d(1, 1, (4, 2))
        self.pool1 = nn.MaxPool2d(5, 4, return_indices=True)
        self.pool2 = nn.MaxPool2d(4, 8, return_indices=True)
        self.encoder = nn.Linear(6, 2)

        # Decoder
        self.ll = nn.Linear(2, 6)
        self.t_pool1 = nn.MaxUnpool2d(4, 8)
        self.t_pool2 = nn.MaxUnpool2d(5, 4)
        self.t_conv1 = nn.ConvTranspose2d(1, 1, (4, 2))
        self.t_conv2 = nn.ConvTranspose2d(1, 1, (6, 4))

    def forward(self, x):
        x = self.conv1(x)
        p1 = x.size()
        x, index1 = self.pool1(x)
        x = F.relu(self.conv2(x))
        p2 = x.size()
        x, index2 = self.pool2(x)
        x = x.view(x.size(0), 6)

        self.descriptor = self.encoder(x)
        #print(self.descriptor)

        x = self.ll(self.descriptor)
        x = x.view(x.size(0), 1, 6, 1)
        x = self.t_pool1(x, index2, p2)
        x = F.relu(self.t_conv1(x))
        x = self.t_pool2(x, index1, p1)
        x = torch.sigmoid(self.t_conv2(x))

        return x

    def getDescriptor(self, input):
        self.forward(input)
        return self.descriptor.detach()[0]

    def saveModel(self, filename):
        torch.save(self.state_dict(), filename)

    def loadModel(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()


model = ConvAutoencoder()
# Loss function
criterion = nn.BCELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

tt = list()
for i in range(100):
    a = np.array(torch.rand((200, 39), dtype=torch.float32).tolist())
    tt.append(a)

ds = SensorDataset(tt)
a = int(len(ds) * 0.8)
train_loader = torch.utils.data.DataLoader(ds[:a], batch_size=32, num_workers=0)
test_loader = torch.utils.data.DataLoader(ds[a:], batch_size=32, num_workers=0)

for i in range(10):
    # monitor training loss
    train_loss = 0.0

    # Training
    for data in train_loader:
        images, _ = data
        optimizer.zero_grad()
        outputs = model(images.float())
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(i, train_loss))


#model.encoder.register_forward_hook(get_activation('encoder'))
x = torch.randn((1,1,200,39))
print(model.getDescriptor(x))

model.saveModel("test.pth")

model1 = ConvAutoencoder()
model1.loadModel("test.pth")

print(model1.getDescriptor(x))
#print(activation['encoder'])