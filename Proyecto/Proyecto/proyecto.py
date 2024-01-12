# imagenes de: https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types/ # cambiar link

# veamos imagenes:
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import time

IMAGE_SIZE = 224
classes = 2

REBUILD_DATA = True

class DogsVSCats():
    IMG_SIZE = 224  # we are going to reshape the images to 50x50
    DOG_CLASSES = os.listdir("../imgs")
    DOG_CLASSES = ["../imgs/" + name for name in DOG_CLASSES]
    DOG_IDS = [id for id in range(len(DOG_CLASSES))]
    print(DOG_CLASSES)

    #print(pokemon_ids,pokemons_imgs)
    LABELS = dict(zip(DOG_CLASSES, DOG_IDS))
    print(LABELS)

    # dataset balance counter variables
    spicie_count = []
    training_data = []
    id_count = 0

    # We want to iterate through these two directories, grab the images, resize, scale, convert the class to number (cats = 0, dogs = 1),
    # and add them to our training_data.

    # All we're doing so far is iterating through the cats and dogs directories, and looking through all of the images
    # and handle for the images:

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if self.id_count == 111:
                        break

                if "jpg" in f:
                    try:
                        self.id_count += 1
                        path = os.path.join(label, f)
                        img = cv2.imread(path,cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        # just makes one_hot matrix as targets.
                        self.training_data.append(
                            [np.array(img), np.eye(85)[self.LABELS[label]]])
                        # example of np.eye(2)[1] -> [0. 1.]
                    except Exception as e:
                        pass
                        # print(label, f, str(e))
            self.spicie_count.append(self.id_count)
            self.id_count = 0
            
            if len(self.spicie_count) == 2:
                break

        print(self.training_data[0])
        np.random.shuffle(self.training_data)
        # dtype because [image,result] have differents sizes
        np.save("training_data.npy", np.array(
            self.training_data, dtype=object))
        print('n_imgs_classes:', self.spicie_count, len(self.spicie_count))


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()

class Net(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(3,IMAGE_SIZE,IMAGE_SIZE).view(-1,3,IMAGE_SIZE,IMAGE_SIZE)
        self._to_linear = None
        self.convs(x)

        self.drop = nn.Sequential(nn.Dropout(0.2))
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, classes)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

net = Net(classes)
training_data = np.load("training_data.npy", allow_pickle=True)
# device = torch.device("cuda:0")
# print(device)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# set our reural network to our device (you can assing diferent neuran network layers to differents GPUs if you have them)
print(net.to(device))
# net = Net().to(device) # assing a new net to our GPU


# training and optimize: optimizer is going to be Adam and because we are using one hot matix we use the crossentropy

# split the data into X and y and convert it into a tensor


optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

np.random.shuffle(training_data)
print(training_data[0][0].shape)
X = torch.Tensor([i[0] for i in training_data]).view(-1, 3,IMAGE_SIZE, IMAGE_SIZE)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

cv2.imshow("test", training_data[0][0])
cv2.waitKey()

# separate some data into training and 10% for testing
VAL_PCT = 0.1  # lest reserve 10% of our data for validation
# converting it to int because we are goint to slice our data in groups of it so it needs to be a valid index
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X), len(test_X))

import time

MODEL_NAME = f"model-{int(time.time())}" # gives a dynamic model name to just help with things getting messy over time. 

# iterate throught the data using batches and calculate train accuracy

def fwd_pass(X,y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    
    return acc, loss

def test(size=32):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1,3,IMAGE_SIZE,IMAGE_SIZE).to(device), y.to(device),train=False)
    return val_acc, val_loss

def train(net):
    BATCH_SIZE = 25
    EPOCHS = 25

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            # from 0, to the len of x, stepping BACH_SIZE at a time.
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                # print(f"{i}:{i+BATCH_SIZE}")
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                #print(f"Acc: {round(float(acc),2)} Loss: {round(float(loss),4)}")
                # analice training acc and loss and test acc and loss each 10 steps
                if i % 10 == 0:
                    val_acc, val_loss = test(size=BATCH_SIZE)
                    print("ep:", epoch, val_acc, "b", val_loss)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")
train(net)
# torch.save(net.state_dict(), ".")

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)

    ax1.plot(accuracies, label="train_acc")
    ax1.plot(val_accs, label="test_acc")
    ax1.legend(loc=2)
    ax2.plot(losses, label="train_loss")
    ax2.plot(val_losses, label="test_loss")
    ax2.legend(loc=2)
    plt.show()

create_acc_loss_graph(MODEL_NAME)
