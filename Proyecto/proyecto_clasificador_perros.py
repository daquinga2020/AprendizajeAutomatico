# Redes convolucionales con pytorch para clasificador de razas de perros

import os
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision.transforms.functional import to_pil_image # Solo para mostrar el ejemplo, borrar
import matplotlib.pyplot as plt
import pickle
from torchvision.datasets import ImageFolder

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

IMAGE_SIZE = 224
CLASSES = 2

# Creacion del dataset
class DogsDataset(Dataset):
    global IMAGE_SIZE
    
    def __init__(self, root_dir, split_ratio=0.8, transforms=None):
        self.transform = transform
        self.dataset = ImageFolder(root=root_dir, transform=transform)

        # Calcula los tamaños de los conjuntos de entrenamiento y prueba
        total_size = len(self.dataset)
        train_size = int(split_ratio * total_size)
        test_size = total_size - train_size

        # Divide el dataset
        self.train_set, self.test_set = random_split(self.dataset, [train_size, test_size])

        # Establece el indicador para determinar si estás accediendo al conjunto de entrenamiento o prueba
        self.is_training = True

    def set_train_mode(self):
        self.is_training = True

    def set_test_mode(self):
        self.is_training = False

    def __len__(self):
        if self.is_training:
            return len(self.train_set)
        else:
            return len(self.test_set)

    def __getitem__(self, idx):
        if self.is_training:
            img, label = self.train_set[idx]
        else:
            img, label = self.test_set[idx]

        return img, label
    
    '''
    id_count = 0
    
    def __init__(self, root_dir, transforms=None, test_size=0.2, random_seed=None):
        self.root_dir = root_dir # directorio donde se encuentran las imagenes
        self.transforms = transforms # transformada que se quiere realizar a la imagen
        
        self.breeds = os.listdir(root_dir) # razas de perros
        self.ids = [id for id in range(len(self.breeds))]
        self.labels = dict(zip(self.breeds, self.ids))
        print(self.labels)
        
        self.test_size = test_size
        self.random_seed = random_seed
        self.train_set, self.test_set = self._split_dataset()

    def __len__(self):
        return len(self.train_set) + len(self.test_set)
    
    def _split_dataset(self):
        total_images = []
        train_images, test_images = [], []

        for class_folder in self.labels:
            class_path = os.path.join(self.root_dir, class_folder)
            
            # Guardar imagenes de la clase "class_folder" si son JPG
            for img_file in os.listdir(class_path):
                if img_file.endswith('.jpg'):
                    img_pth = os.path.join(class_path, img_file)
                    img = cv2.imread(img_pth, cv2.IMREAD_COLOR)
                    
                    # Transformar img a PIL para la transformacion de tensor
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    
                    if self.transforms:
                        img = self.transforms(img)
                    
                    total_images.append([np.array(img), self.labels[class_folder]])

			# Se obtienen los datasets para test y entrenamiento
            train_set, test_set = train_test_split(total_images, test_size=self.test_size, random_state=self.random_seed)
            
   			# Se obtiene el dataset [[[img1, label], class1], [[img2, label], class2], ..., [[img_n, label], class_n]]
            print(class_folder)
            print("TRAIN")
            train_images.extend([[img, class_folder] for img in tqdm(train_set[:3])]) # Quitar [:2]
            print("TEST")
            test_images.extend([[img, class_folder] for img in tqdm(test_set[:3])])
            
            print("")
        
        np.random.shuffle(train_images)
        np.random.shuffle(test_images)

        return np.array(train_images, dtype=object), np.array(test_images, dtype=object)'''
    
    '''def __getitem__(self, idx):
        if idx < len(self.train_set):
            img_path, label = self.train_set[idx]
        else:
            img_path, label = self.test_set[idx - len(self.train_set)]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.transforms:
            image = self.transforms(image)
        
        image = np.array(image)
        
        # Label encoding (assuming folder names are class labels)
        label = self.labels.index(label)

        return image, label'''

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), # Se convierte en un tensor para pytorch y se normalizan los pixeles
])

# Obtener un dataset limpio y correcto
dataset_Dogs = DogsDataset(root_dir='imgs', split_ratio=0.8, transforms=transform)

# Configura la clase para acceder al conjunto de entrenamiento
dataset_Dogs.set_train_mode()

print(dataset_Dogs)

# Para mostrar un ejemplo
# sample_image, sample_label = dataset_Dogs[0]

# Convierte la imagen a formato NumPy
# sample_image_np = to_pil_image(sample_image)
# sample_image_np = cv2.cvtColor(np.array(sample_image_np), cv2.COLOR_RGB2BGR)

# print("", sample_label)
# cv2.imshow('Sample Img', sample_image_np)
# cv2.waitKey(0)
##### Fin para mostrar ejemplo


# Aquí, 'batch_images' es un tensor que contiene un lote de imágenes
# y 'batch_labels' es un tensor que contiene las etiquetas correspondientes.
# Dataloaders
batch_size = 2
train_dataloader = DataLoader(dataset_Dogs, batch_size=batch_size, shuffle=True)

dataset_Dogs.set_test_mode()
print(dataset_Dogs)
test_dataloader = DataLoader(dataset_Dogs, batch_size=len(dataset_Dogs.test_set), shuffle=True)

print(enumerate(train_dataloader))

for i in enumerate(train_dataloader):
    print(i)

def plot_dogs_grid():
    total_samples = 4
    plt.figure(figsize=(30,30))
    
 
plot_dogs_grid()


# Definir la red
class CnnNet(nn.Module):
    def __init__(self, classes):
        super(CnnNet, self).__init__()

        # Definicion de capas para las capas que filtran las caracteristicas
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Conv2d(16, 32, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.Conv2d(64, 128, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),  # Ajustado el tamaño de entrada
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, classes)
        )

        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Aplanar para la capa completamente conectada
        return self.classifier(x)


cnn_net = CnnNet(CLASSES)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
# print(cnn_net.to(device))
# optimizer = optim.Adam(cnn_net.parameters(), lr=0.01)

# Entrenamiento del modelo

# epochs = 10

# for epoch in range(epochs):
#     for images, labels in train_dataloader:
#         optimizer.zero_grad()
#         outputs = cnn_net(images)
#         loss = cnn_net.loss_criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# def train(net):
# 	BATCH_SIZE = 25
# 	EPOCHS = 25

# 	with open("model.log", "a") as f:
# 		for epoch in range(EPOCHS):
# 			# from 0, to the len of x, stepping BACH_SIZE at a time.
# 			for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
# 				# print(f"{i}:{i+BATCH_SIZE}")
# 				batch_X = train_X[i:i+BATCH_SIZE].view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
# 				batch_y = train_y[i:i+BATCH_SIZE]	
# 				batch_X, batch_y = batch_X.to(device), batch_y.to(device)	
# 				acc, loss = fwd_pass(batch_X, batch_y, train=True)	
# 				#print(f"Acc: {round(float(acc),2)} Loss: {round(float(loss),4)}")
# 				# analice training acc and loss and test acc and loss each 10 steps
# 				if i % 10 == 0:
# 					val_acc, val_loss = test(size=BATCH_SIZE)
# 					print("ep:", epoch, val_acc, "b", val_loss)
# 					f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")

# train(cnn_net)