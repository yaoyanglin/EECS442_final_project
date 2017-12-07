
import numpy as np
import cv2
import math
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

class training_set(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        # The output of torchvision datasets are PILImage images of range [0, 1].
        # We transform them to Tensors of normalized range [-1, 1]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)


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


# preprocessing
def preprocessing(img, name):
    [height, width] = img.shape
    # Thresholding

    mean = np.mean(img)
    ret, thresh = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)

    # if the charactor is white, invert the color of image
    if (thresh[0][0] == 0):
        thresh = cv2.bitwise_not(thresh)

    # use the top left color as the indicator of the boarder
    if (thresh[0][0] == 255):
        color = 255
    else:
        color = 0
    # add broader
    thresh = cv2.copyMakeBorder(thresh, math.ceil(0.15 * height), math.ceil(0.15 * height),
                                (height + 2 * math.ceil(0.15 * height) - width) // 2,
                                (height + 2 * math.ceil(0.15 * height) - width) // 2,
                                cv2.BORDER_CONSTANT, value=color)

    # resize the image
    resized_image = cv2.resize(thresh, (128, 128))

    return resized_image

# # Load the test Data
def load_cropped_test_images(path):
    folder_names = [d for d in os.listdir(path) if d.startswith('Sample')]
    imgs = []
    labels = []
    test = []
    for folder in folder_names:
        print("reading file names in " + folder)
        print(path + folder + "/")
        names = [d for d in os.listdir(path + folder + '/') if d.endswith('.jpg')]

        with open(path + folder + '/' + 'detection.txt', 'r') as f:
            rubbish = f.readline().split()[0]
            rubbish = f.readline().split()[0]
            rubbish = f.readline().split()[0]

            for name in names:
                # assign label to the image
                label = f.readline().split()[0]
                if label != "?":
                    labels.append(LABEL_DIC[label])
                    test.append(label)
                    img = cv2.imread(path + folder + '/' + name, 0)
                    # img = Image.open('./test/1.png_dir/' + name).convert('L')

                    img = preprocessing(img, folder)
                    img = np.reshape(img, (128, 128, 1))
                    imgs.append(img)

    return imgs,labels

def to_lower_case(labels):

    labels_np = labels.numpy()
    for i in range(len(labels_np)):
        if labels[i] > 10 and labels[i] < 36:
            labels[i] = labels[i] + 26
    labels = torch.from_numpy(labels_np)
    return labels




SIZE = 128

PATH = './validation/'

PATH2 = './test/'


if __name__=='__main__':

    imgs_test, labels_test = load_cropped_test_images(PATH2)

    testset = training_set(imgs_test, labels_test)
    testloader = data.DataLoader(testset, batch_size=4, shuffle=True)
    net = Net()
    net.load_state_dict(torch.load('the_train_model.pkl'))

    print(net)

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

    print("correct label is: ", correct)
    print("total label is: ", total)

    correct_f = float(correct)
    total_f = float(total)
    print('Accuracy of the network on the test images: %d %%' % (100 * correct_f / (total_f) * 1.0))



