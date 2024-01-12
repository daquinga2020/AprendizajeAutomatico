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
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

IMAGE_SIZE = 224
CLASSES = 25

# Creacion del dataset
class DogsDataset(Dataset):
    global IMAGE_SIZE
    
    def __init__(self, root_dir, target_images_per_class = -1, transform=None):
        self.root_dir = root_dir
        self.target_images_per_class = target_images_per_class
        self.transform = transform
        self.classes = os.listdir(root_dir)
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
dataset_Dogs = DogsDataset(root_dir='imgs', target_images_per_class = -1, transform=transform)

# Aquí, 'batch_images' es un tensor que contiene un lote de imágenes
# y 'batch_labels' es un tensor que contiene las etiquetas correspondientes.
# Dataloaders


'''example_image, example_label = dataset_Dogs[6]

# Convierte la imagen de Tensor a Numpy para visualización
image_np = np.transpose(example_image.numpy(), (1, 2, 0))

# Visualiza la imagen
plt.imshow(image_np)
dog_breed = dataset_Dogs.classes[example_label.item()]
plt.title(f'Dog Breed: {dog_breed}')
plt.show()'''

dataset_train, dataset_temp = train_test_split(dataset_Dogs.data, test_size=0.3, random_state=42)

# Ahora, divide X_temp e y_temp en conjuntos de validación y prueba
dataset_val, dataset_test = train_test_split(dataset_temp, test_size=0.2, random_state=42)


batch_size = 15
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calcular el tamaño de entrada correcto para la capa lineal
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 1024),  # Ajustado el tamaño de entrada
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, classes),
        )

        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Aplanar para la capa completamente conectada
        return self.classifier(x)


cnn_net = CnnNet(CLASSES)

'''if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")'''
    
device = torch.device("cpu")
cnn_net.to(device)
optimizer = optim.Adam(cnn_net.parameters(), lr=0.0001)


def calculate_accuracy(predictions, targets):
    predicted_labels = predictions.argmax(dim=1)
    correct_predictions = (predicted_labels == targets).sum().item()
    total_samples = targets.size(0)
    accuracy = correct_predictions / total_samples
    return accuracy


# Entrenamiento del cnn_neto
# Listas para almacenar las pérdidas y precisión en entrenamiento y validación
train_losses = []
val_accuracies = []

epochs = 50

for epoch in range(epochs):
    cnn_net.train()
    for images, labels in train_dataloader:
        # images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_net(images)
        loss = cnn_net.loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    
    val_predictions = []
    val_true_labels = []
    
    # Evaluación en conjunto de entrenamiento
    cnn_net.eval()
    with torch.no_grad():
        for X_val, Y_val in val_dataloader:
            # X_train, Y_train = X_train.to(device), Y_train.to(device)
            val_outputs = cnn_net(X_val)
            train_loss = cnn_net.loss_criterion(val_outputs, Y_val)
            
    #         _, val_preds = torch.max(val_outputs, 1)
    #         val_predictions.extend(val_preds.cpu().numpy())
    #         val_true_labels.extend(Y_val.cpu().numpy())

    # val_accuracy = accuracy_score(val_true_labels, val_predictions)
    val_accuracy = calculate_accuracy(val_outputs, Y_val)
    
    train_losses.append(loss.item())
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Accuracy: {val_accuracy}')
    
# Graficar la evolución de pérdidas y precisión
plt.figure(figsize=(12, 4))

plt.plot(train_losses, label='Entrenamiento')
plt.plot(val_accuracies, label='Validación')
plt.title('Evolución de las Pérdidas y Precisión')
plt.xlabel('Época')
plt.ylabel('Pérdida y Precision')
plt.legend()

plt.tight_layout()
plt.show()


# Evaluación en el conjunto de prueba
test_loss = 0.0
test_accuracy = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = cnn_net(inputs)
        loss = cnn_net.loss_criterion(outputs, targets)
        test_loss += loss.item()

        accuracy = calculate_accuracy(outputs, targets)
        test_accuracy += accuracy

        all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Calcular la pérdida y la precisión promedio
average_test_loss = test_loss / len(test_dataloader)
average_test_accuracy = test_accuracy / len(test_dataloader)

print(f'Average Test Loss: {average_test_loss:.4f}')
print(f'Average Test Accuracy: {average_test_accuracy * 100:.2f}%')

# Función para mostrar una imagen junto con la predicción y la etiqueta real
def show_example(image, prediction, target):
    plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
    plt.title(f'Prediction: {prediction}, Target: {target}')
    plt.show()

# Mostrar ejemplos aleatorios después de evaluar todo el conjunto de prueba
num_examples_to_show = 10
for _ in range(num_examples_to_show):
    idx = np.random.randint(len(all_predictions))
    test_image, test_target = dataset_test[idx]

    predicted_label = all_predictions[idx]
    show_example(test_image, predicted_label.item(), test_target.item())

