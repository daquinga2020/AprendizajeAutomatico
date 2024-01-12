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

IMAGE_SIZE = 50
classes = 102

REBUILD_DATA = True

class DogsVSCats():
    IMG_SIZE = 50  # we are going to reshape the images to 50x50
    DOG_CLASSES = os.listdir("images")
    DOG_CLASSES = ["images/" + name for name in DOG_CLASSES]
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
                            [np.array(img), np.eye(2)[self.LABELS[label]]])
                        # example of np.eye(2)[1] -> [0. 1.]
                    except Exception as e:
                        pass
                        # print(label, f, str(e))
            self.spicie_count.append(self.id_count)
            self.id_count = 0

        print(self.training_data[0])
        np.random.shuffle(self.training_data)
        # dtype because [image,result] have differents sizes
        np.save("training_data.npy", np.array(
            self.training_data, dtype=object))
        print('n_imgs_classes:', self.spicie_count)


if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()
