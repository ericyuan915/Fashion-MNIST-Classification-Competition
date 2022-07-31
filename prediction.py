
"""
Implement the testing procedure here. 

Inputs:
    Place the folder named "test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).

"""

import random
from PIL import Image
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from torch.autograd import Variable

modellr = 1e-3
BATCH_SIZE = 128
EPOCHS = 11
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(DEVICE)
# print(torch.cuda.get_device_name(0))
# print('Memory Usage:')
# print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
# print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
jitter_param = 0.4
transform = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(
    #         brightness=jitter_param,
    #         contrast=jitter_param,
    #         saturation=jitter_param,
    #         hue=0.1),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomGrayscale(),
    # transforms.RandomCrop(20),
    # transforms.Resize((28, 28)),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # transforms.RandomVerticalFlip(),
    # transforms.RandomGrayscale(),
    # transforms.RandomCrop(24),
    # transforms.Resize((28,28))
    
])

dataset_train = datasets.ImageFolder('hw4_train', transform)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()
model = torchvision.models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=modellr)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

model.eval()
model.to(DEVICE)

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    # transforms.RandomVerticalFlip()
     # transforms.RandomGrayscale(),
    # transforms.RandomCrop(24),
    # transforms.Resize((28,28))
    ])

f = open("prediction.txt", "w+")
for i in range(0, 10000):
    newStr = 'hw4_test/'+str(i)+'.png'
    img = Image.open(newStr)
    img = transform(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    img = img.expand(-1, 3, -1, -1)
    out = model(img)
    _, pred = torch.max(out.data, 1)
    f.write(classes[pred.data.item()]+'\n')
