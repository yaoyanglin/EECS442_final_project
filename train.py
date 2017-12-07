
import numpy as np
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
from PIL import Image
import math

LABEL_DIC = {
    '0' : 0,
    '1' : 1,
    '2' : 2,
    '3' : 3,
    '4' : 4,
    '5' : 5,
    '6' : 6,
    '7' : 7,
    '8' : 8,
    '9' : 9,
    'A' : 10,
    'B' : 11,
    'C' : 12,
    'D' : 13,
    'E' : 14,
    'F' : 15,
    'G' : 16,
    'H' : 17,
    'I' : 18,
    'J' : 19,
    'K' : 20,
    'L' : 21,
    'M' : 22,
    'N' : 23,
    'O' : 24,
    'P' : 25,
    'Q' : 26,
    'R' : 27,
    'S' : 28,
    'T' : 29,
    'U' : 30,
    'V' : 31,
    'W' : 32,
    'X' : 33,
    'Y' : 34,
    'Z' : 35,
    'a' : 36,
    'b' : 37,
    'c' : 38,
    'd' : 39,
    'e' : 40,
    'f' : 41,
    'g' : 42,
    'h' : 43,
    'i' : 44,
    'j' : 45,
    'k' : 46,
    'l' : 47,
    'm' : 48,
    'n' : 49,
    'o' : 50,
    'p' : 51,
    'q' : 52,
    'r' : 53,
    's' : 54,
    't' : 55,
    'u' : 56,
    'v' : 57,
    'w' : 58,
    'x' : 59,
    'y' : 60,
    'z' : 61,
}

class Net(nn.Module):
    def __init__(self):  # in_channel: the channels of input image; n_category: classification categories
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1,
                               padding=2)  # input channel: 3; output channels: 6; kernal size: 5x5; padding =1: keep the image size in the first conv layer
        self.pool1 = nn.MaxPool2d(2, 2)  # 2x2 maxpool layer --> downsampling
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 62)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def to_lower_case(labels):

    labels_np = labels.numpy()
    for i in range(len(labels_np)):
        if labels[i] > 10 and labels[i] < 36:
            labels[i] = labels[i] + 26
    labels = torch.from_numpy(labels_np)
    return labels

class training_set(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        # The output of torchvision datasets are PILImage images of range [0, 1].
        # We transform them to Tensors of normalized range [-1, 1]
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)

# # Load the Data
def load_images(path):
    folder_names = [d for d in os.listdir('./English/Fnt/') if d.startswith('Sample')]
    imgs = []
    labels = []
    for folder in folder_names:
        print("reading file names in " + folder)
        print(path + folder + "/")
        names = [d for d in os.listdir(path + folder + "/") if d.endswith('.png')]

        for i in range(100 , len(names)):
            img = Image.open(path + folder + "/" + names[i]).convert('L')
            img = np.reshape(img, (128, 128, 1))
            imgs.append(img)
            # assign label to the image
            labels.append(int(folder[-2:]) - 1)

    return imgs,labels

# # Load the test Data
def load_test_images(path):
    folder_names = [d for d in os.listdir('./English/Fnt/') if d.startswith('Sample')]
    imgs = []
    labels = []
    for folder in folder_names:
        print("reading file names in " + folder)
        print(path + folder + "/")
        names = [d for d in os.listdir(path + folder + "/") if d.endswith('.png')]

        for i in range(100):
            # img = cv2.imread(PATH + name, 0)
            img = Image.open(path + folder + "/" + names[i]).convert('L')
            img = np.reshape(img, (128, 128, 1))
            imgs.append(img)
            # assign label to the image
            labels.append(int(folder[-2:]) - 1)

    return imgs,labels


def to_lower_case(labels):

    labels_np = labels.numpy()
    for i in range(len(labels_np)):
        if labels[i] > 10 and labels[i] < 36:
            labels[i] = labels[i] + 26
    labels = torch.from_numpy(labels_np)
    return labels



SIZE = 128

PATH = './English/Fnt/'


imgs,labels = load_images(PATH)

imgs_test,labels_test = load_test_images(PATH)


train_data = training_set(imgs, labels)
trainloader = data.DataLoader(train_data, batch_size=4, shuffle=True)

testset = training_set(imgs_test, labels_test)
testloader = data.DataLoader(testset, batch_size=4, shuffle=True)


net = Net()

criterion = nn.CrossEntropyLoss()  # loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD method to do gradiant degradation，learning rate:0.001，momentum:0.9

for epoch in range(20):  # epoch means training iteration, should be more than 20

    running_loss = 0.0


    # enumerate(sequence, [start=0])
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data  # data's structure：[tensor dimension: 4x3x32x32,4 for minibatch, i.e. 4 pictures as a training batch, 3 means the channel of picture, 32x32 is dimension of each channel]

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # loss backpropagation
        optimizer.step()  # Use SGD to update parameters

        running_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


print('Finished Training')


correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    # print outputs.data
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    predicted = to_lower_case(predicted)
    labels = to_lower_case(labels)

    correct += (predicted == labels).sum()



print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))

torch.save(net.state_dict(), 'the_train_model.pkl')

