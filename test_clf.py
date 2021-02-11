# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:08:14 2021

@author: bmcma

loads the already trained classifier and tests it
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

USE_GPU = True
if USE_GPU:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
    
def fetch_lookup_table():
    '''
    Generates a lookup table for images classes

    Returns
    -------
    lookup_table : dictionary
        Each key is of type int and corresponds to a label. Each entry is of type
        string and is the name of that class

    '''
    categories = open("D:/PyTorch/p0/categories_places365.txt")  # file containing image labels
    lookup_table = {}  # will be used for looking up labels
    for line in categories:
        category_name, category_ix = line.strip().split()
        category_name = category_name[3:]
        lookup_table[int(category_ix)] = category_name
    return lookup_table


def imshow(img):
    '''
    plots image data

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.CenterCrop((300,300)),      # crop the image 
     transforms.Resize((300,300))])         # ensure image is at least 300x300

trainset = torchvision.datasets.Places365(root='D:\PyTorch\p0', split='train-standard',
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                          shuffle=True, num_workers=0)


testset = torchvision.datasets.Places365(root='D:\PyTorch\p0', split='val',
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=0)

class Net(nn.Module):
    '''A class to represent a convolutional neural network.

    ...

    Attributes
    ---------

    Methods
    -------
    forward(torch tensor):
        forward propogates x
    '''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 60, 5)  # in channels, out channels, kernel_size  (1, 3, 500, 500)
        self.pool = nn.MaxPool2d(2, 2)                  #
        self.conv2 = nn.Conv2d(60, 16, 5)               #
        self.fc1 = nn.Linear(16 * 72 * 72, 1000)           
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 500)

    def forward(self, x):
        '''
        defines forward propogation

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        x : TYPE
            DESCRIPTION.

        '''                                          # shape prior to executing line
        x = self.pool(F.relu(self.conv1(x)))         # 1,3,300,300
        x = self.pool(F.relu(self.conv2(x)))         # 1, 60, 148, 148
        x = x.view(-1, 16 * 72 * 72)                   # 1, 16, 72, 72
        x = F.relu(self.fc1(x))                  
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



dataiter = iter(testloader)
images, labels = dataiter.next()

classes = fetch_lookup_table()
# print images
plt.figure()   # opens a new figure
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: \n', '\n'.join('%5s' % classes[labels[j].item()] for j in range(4)))
net = Net()
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: \n', '\n'.join('%5s' % classes[predicted[j].item()] for j in range(4)))

# how network performs on entire dataset
correct = 0
total = 0
tmp_counter = 0
with torch.no_grad():
    for (images, labels) in testloader:
        tmp_counter+=1
        if tmp_counter %100 == 99:
            print("100 test samples passed")
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

N_CLASSES = 365
tmp_counter = 0
# get accuracies of each class
class_correct = list(0. for i in range(N_CLASSES))
class_total = list(0. for i in range(N_CLASSES))
with torch.no_grad():
    for (images, labels) in testloader:
        tmp_counter += 1
        if tmp_counter %100 == 99:
            print("100 test samples passed")
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(N_CLASSES):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    