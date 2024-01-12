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

IMAGE_SIZE = 200
CLASSES = 25

# Creacion del dataset
class DogsDataset(Dataset):
    global IMAGE_SIZE
    
    def __init__(self, root_dir, split_ratio=0.8, transforms=None):
        self.transform = transforms
        self.dataset = ImageFolder(root=root_dir, transform=transforms)

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


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), # Se convierte en un tensor para pytorch y se normalizan los pixeles
])

# Obtener un dataset limpio y correcto
dataset_Dogs = DogsDataset(root_dir='imgs', split_ratio=0.8, transforms=transform)

# Aquí, 'batch_images' es un tensor que contiene un lote de imágenes
# y 'batch_labels' es un tensor que contiene las etiquetas correspondientes.
# Dataloaders
batch_size = 15
dataset_Dogs.set_train_mode()

# example_image, example_label = dataset_Dogs[0]

# Convierte la imagen de Tensor a Numpy para visualización
# image_np = np.transpose(example_image.numpy(), (1, 2, 0))

# Visualiza la imagen
# plt.imshow(image_np)
# plt.title(f'Etiqueta: {example_label}')
# plt.show()
train_dataloader = DataLoader(dataset_Dogs, batch_size=batch_size, shuffle=True)
train_dataloader2 = DataLoader(dataset_Dogs, batch_size=len(dataset_Dogs), shuffle=True)

dataset_Dogs.set_test_mode()

test_dataloader = DataLoader(dataset_Dogs, batch_size=len(dataset_Dogs), shuffle=True)


# Definir la red
class CnnNet(nn.Module):
    def __init__(self, classes):
        super(CnnNet, self).__init__()

        # Definicion de capas para las capas que filtran las caracteristicas
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reducción menos agresiva
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calcular el tamaño de entrada correcto para la capa lineal
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),  # Ajustado el tamaño de entrada
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
optimizer = optim.Adam(cnn_net.parameters(), lr=0.001)

# Entrenamiento del modelo
# Listas para almacenar las pérdidas y precisión en entrenamiento y validación
train_losses = []
val_losses = []
ind = 0
epochs = 20

for epoch in range(epochs):
    cnn_net.train()
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = cnn_net(images)
        loss = cnn_net.loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    
    # Evaluación en conjunto de entrenamiento
    cnn_net.eval()
    with torch.no_grad():
        for X_train, Y_train in train_dataloader2:
            outputs = cnn_net(X_train)
            train_loss = cnn_net.loss_criterion(outputs, Y_train)  # Utiliza 'outputs', no 'X_train'

            
# Evaluación en conjunto de validación
    with torch.no_grad():
        for X_test, Y_test in test_dataloader:
            outputs = cnn_net(X_test)
            val_loss = cnn_net.loss_criterion(outputs, Y_test)  # Utiliza 'outputs', no 'X_test'

    # Almacenar pérdidas y precisión en listas
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
# Graficar la evolución de pérdidas y precisión
plt.figure(figsize=(12, 4))

plt.plot(train_losses, label='Entrenamiento')
plt.plot(val_losses, label='Validación')
plt.title('Evolución de las Pérdidas')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# torch.save(cnn_net, "clasificador_perros.pth")
cnn_net.eval()
with torch.no_grad():
    # Suponiendo que 'test_dataloader' es tu DataLoader de prueba
    for inputs, labels in test_dataloader:
        outputs = cnn_net(inputs)
        
# Convierte las predicciones y etiquetas a numpy arrays
predictions = outputs.argmax(dim=1).numpy()
ground_truth = labels.numpy()

# Visualiza algunos ejemplos
for i in range(10):  # Visualiza 5 ejemplos
    plt.subplot(1, 10, i + 1)
    plt.imshow(inputs[i][0])  # Suponiendo imágenes en escala de grises
    plt.title(f'Pred: {predictions[i]}, Actual: {ground_truth[i]}')
    plt.axis('off')

plt.show()
err = 0
for i in range(len(predictions)):  # Visualiza 5 ejemplos
    if predictions[i] != ground_truth[i]:
        err += 1
        
print("ERROR:", err/len(predictions))