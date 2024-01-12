# imagenes de: https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types/

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

IMAGE_SIZE = 100
classes = 102

images_dir = "images"
pokemons_imgs = os.listdir(images_dir)
pokemon_class = [name[:-4] for name in pokemons_imgs]
pokemon_ids = [id for id in range(len(pokemon_class))]

labels = dict(zip(pokemon_class, pokemon_ids))
training_data = []

id = 0

CREATE_DATA = True
if CREATE_DATA:
    for f in tqdm(pokemons_imgs):
        path = os.path.join(images_dir, f)
        img = cv2.imread(path)
        if "jpg" in f:
            img = Image.open(path).convert('L')
            img.save(images_dir + "/" + f"{pokemon_class[id]}.png")
            print(images_dir + "/" + f"{pokemon_class[id]}.png")
            try: 
                os.remove(path) 
                print("% s removed successfully" % path) 
            except OSError as error: 
                print(error) 
                print("File path can not be removed") 
        id+=1

id = 0
img_center = (IMAGE_SIZE//2, IMAGE_SIZE//2)

if CREATE_DATA:
    for f in tqdm(pokemons_imgs):
        path = os.path.join(images_dir, f)
        img = cv2.imread(path)
        if "png" in f:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img_binary = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
            inv_img = cv2.bitwise_not(img_binary)
            img = cv2.resize(inv_img, (IMAGE_SIZE, IMAGE_SIZE))

            output = np.eye(len(pokemon_ids))[id]
            for angle in range(0, 360, 3):
                rotation_matrix = cv2.getRotationMatrix2D(img_center, angle, 1.0)

                # Aplica la matriz de rotación a la imagen
                rotated_img = cv2.warpAffine(img, rotation_matrix, (IMAGE_SIZE, IMAGE_SIZE))

                cv2.imshow("Pokemon", rotated_img)
                cv2.waitKey(1)
                training_data.append([np.array(rotated_img), output])
            
            print(id, pokemon_class[id], np.mean(img))
            id += 1
    print(len(training_data)/360*15)
    np.save("pokemons_data_" + chr(classes) + ".npy", np.array(training_data, dtype=object))

class Net(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3,padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=1)

        x = torch.randn(IMAGE_SIZE,IMAGE_SIZE).view(-1,1,IMAGE_SIZE,IMAGE_SIZE)
        self._to_linear = None
        self.convs(x)

        self.drop = nn.Sequential(nn.Dropout(0.4))
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, classes)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2),stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2),stride=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2),stride=2)

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
training_data = np.load("pokemons_data_" + chr(classes) + ".npy", allow_pickle=True)
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

# you can move all the data to the GPU because in this case its not to much big but you normaly won't and what you do is move the batch data.

# training and optimize: optimizer is going to be Adam and because we are using one hot matix we use the MSE error metric

# split the data into X and y and convert it into a tensor


optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

np.random.shuffle(training_data)
X = torch.Tensor([i[0] for i in training_data]).view(-1, IMAGE_SIZE, IMAGE_SIZE)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

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
    val_acc, val_loss = fwd_pass(X.view(-1,1,IMAGE_SIZE,IMAGE_SIZE).to(device), y.to(device),train=False)
    return val_acc, val_loss

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 50

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            # from 0, to the len of x, stepping BACH_SIZE at a time.
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                # print(f"{i}:{i+BATCH_SIZE}")
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                #print(f"Acc: {round(float(acc),2)} Loss: {round(float(loss),4)}")
                # analice training acc and loss and test acc and loss each 10 steps
            
            val_acc, val_loss = test(size=BATCH_SIZE)
            print("ep:", epoch, val_acc, "b", val_loss)
            f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")
train(net)

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

# Mostrar ejemplos aleatorios después de evaluar todo el conjunto de test
num_examples_to_show = 10
randoms_idxs = np.random.choice(np.arange(len(test_X)), size=num_examples_to_show, replace=False)
ind_plot = 1
for idx in randoms_idxs:
    test_image, test_target = test_X[idx], torch.argmax(test_y[idx])
    predicted_label = torch.argmax(net(test_image.view(-1,1,IMAGE_SIZE,IMAGE_SIZE).to(device)))
    
    real_target = pokemon_class[test_target]
    pred_target = pokemon_class[predicted_label]
    
    plt.subplot(5, 5, ind_plot)
    plt.axis('off')
    plt.imshow(test_image.cpu().numpy(),cmap=plt.cm.gray)
    plt.title(f'Pred: {pred_target}, Real: {real_target}')
    ind_plot += 1

plt.tight_layout()
plt.show()

