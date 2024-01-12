# Redes convolucionales con pytorch para clasificador de razas de perros

import os
# from tqdm import tqdm
# import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F

IMAGE_SIZE = 224

# Creacion del dataset
class DogsDataset(Dataset):
    global IMAGE_SIZE
    
    def __init__(self, root_dir, target_images_per_class = -1, transform=None):
        self.root_dir = root_dir
        self.target_images_per_class = target_images_per_class
        self.transform = transform
        self.classes = os.listdir(root_dir) # Y
        self.data = self._load_dataset()

    def _load_dataset(self):
        dataset = []
        for class_folder in self.classes:
            class_path = os.path.join(self.root_dir, class_folder)
            class_images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
            
            if self.target_images_per_class != -1:
                class_images = class_images[:self.target_images_per_class]

            for img_filename in class_images:
                img_path = os.path.join(class_path, img_filename)
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                label = torch.tensor(self.classes.index(class_folder), dtype=torch.long)
                dataset.append((img, label))
        
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), # Se convierte en un tensor para pytorch y se normalizan los pixeles
])

# Obtener un dataset limpio y correcto
dataset_Dogs = DogsDataset(root_dir='imgs', target_images_per_class=-1, transform=transform)

dataset_train, dataset_temp = train_test_split(dataset_Dogs.data, test_size=0.3, random_state=42)
# Ahora, divide dataset_temp en conjuntos de validación y prueba
dataset_val, dataset_test = train_test_split(dataset_temp, test_size=0.5, random_state=42)


batch_size = 20
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True)


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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calcular el tamaño de entrada correcto para la capa lineal
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),  # Ajustado el tamaño de entrada
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, classes),
        )

        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Aplanar para la capa completamente conectada
        return self.classifier(x)


cnn_net = CnnNet(len(dataset_Dogs.classes))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

cnn_net.to(device)
optimizer = optim.Adam(cnn_net.parameters(), lr=0.0001)


def calculate_accuracy(predictions, targets):
    predicted_labels = predictions.argmax(dim=1)
    correct_predictions = (predicted_labels == targets).sum().item()
    total_samples = targets.size(0)
    accuracy = correct_predictions / total_samples
    return accuracy


# Entrenamiento de la red cnn_net
# Listas para almacenar las pérdidas y precisión en entrenamiento y validación
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

epochs = 50

for epoch in range(epochs):
    cnn_net.train()
    
    for train_images, train_labels in train_dataloader:
        train_images, train_labels = train_images.to(device), train_labels.to(device)
        optimizer.zero_grad()
        train_outputs = cnn_net(train_images)
        train_loss = cnn_net.loss_criterion(train_outputs, train_labels)
        train_loss.backward()
        optimizer.step()

    train_accuracy = calculate_accuracy(train_outputs, train_labels)
    
    # Evaluación en conjunto de entrenamiento
    cnn_net.eval()
    with torch.no_grad():
        for val_images, val_labels in val_dataloader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = cnn_net(val_images)
            val_loss = cnn_net.loss_criterion(val_outputs, val_labels)

        val_accuracy = calculate_accuracy(val_outputs, val_labels)
    
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss.item()}, Training Accuracy: {train_accuracy}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy}')
    
# Graficar la evolución de pérdidas y precisión
fig, axs = plt.subplots(1, 2, figsize=(10, 8))

axs[0].plot(train_losses, label='Entrenamiento')
axs[0].plot(val_losses, label='Validación')
axs[0].set_title('Evolución de las Pérdidas')

axs[1].plot(train_accuracies, label='Entrenamiento')
axs[1].plot(val_accuracies, label='Validación')
axs[1].set_title('Evolución de la Precisión')

# Agregar etiquetas y leyenda si es necesario
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Pérdidas')
axs[0].legend()

axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Precisión')
axs[1].legend()


plt.tight_layout()
plt.show()

# Guardar el modelo completo en un archivo (por ejemplo, 'modelo_completo.pth')
pth2save = 'models/cnn_pruebas_3conv_50epch.pth'
torch.save(cnn_net, pth2save)

# Evaluación del conjunto de testeo
test_loss = 0.0
test_accuracy = 0.0
all_predictions = []
all_targets = []

cnn_net.eval()
with torch.no_grad():
    for test_images, test_labels in test_dataloader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_outputs = cnn_net(test_images)
        test_loss = cnn_net.loss_criterion(test_outputs, test_labels)
        test_loss = test_loss.item()

        test_accuracy = calculate_accuracy(test_outputs, test_labels)

        all_predictions.extend(test_outputs.argmax(dim=1).cpu().numpy())
        all_targets.extend(test_labels.cpu().numpy())


# Calcular la pérdida y la precisión promedio
average_test_loss = test_loss / len(test_dataloader)
average_test_accuracy = test_accuracy / len(test_dataloader)

print(f'Average Test Loss: {average_test_loss:.4f}')
print(f'Average Test Accuracy: {average_test_accuracy * 100:.2f}%')

err = 0
for i in range(len(all_predictions)):
    if all_predictions[i] != all_targets[i]:
        err += 1

print(f'Error: {err/len(all_predictions) * 100:.2f}%')

# Mostrar ejemplos aleatorios después de evaluar todo el conjunto de test
num_examples_to_show = 10
randoms_idxs = np.random.choice(np.arange(len(all_predictions)), size=num_examples_to_show, replace=False)
ind_plot = 1
for idx in randoms_idxs:
    test_image, test_target = test_images[idx], all_targets[idx]
    predicted_label = all_predictions[idx]
    
    real_target = dataset_Dogs.classes[test_target]
    pred_target = dataset_Dogs.classes[predicted_label]
    
    plt.subplot(5, 5, ind_plot)
    plt.axis('off')
    plt.imshow(np.transpose(test_image.cpu().numpy(), (1, 2, 0)))
    plt.title(f'Pred: {pred_target}, Real: {real_target}')
    ind_plot += 1

plt.tight_layout()
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix

print(dataset_Dogs.classes)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Matriz de Confusión")
    plt.xlabel("Clases Predichas")
    plt.ylabel("Clases Verdaderas")
    plt.show()

cm = confusion_matrix(all_targets, all_predictions)
plot_confusion_matrix(cm, list(range(len(dataset_Dogs.classes))))

