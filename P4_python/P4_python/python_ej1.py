import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


datos = pd.read_csv('housing.csv')


# Separar las columnas en entrada (X) y salida (y)
X = datos.iloc[:, :-1].values

# Convertir las salidas de string a números (asumiendo que son binarias)
y = datos.iloc[:, -1].values

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos (opcional, pero a menudo es beneficioso)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir los datos a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # Reshape y a una columna
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # Reshape y a una columna


# Crear un conjunto de datos y un DataLoader con un tamaño de lote de 10
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

# Definir el modelo
class Modelo(nn.Module):
    def __init__(self, input_size):
        super(Modelo, self).__init__()
        # descomentar para apartado 1
        # self.fc1 = nn.Linear(input_size, 4)
        # self.fc2 = nn.Linear(4, 1)
        
        # comentar para apartado 1 e ir descomentando capas para apartado 2
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        # descomentar para apartado 1
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)

        # comentar para apartado 1 e ir descomentando capas para apartado 2         
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

# Instanciar el modelo
input_size = X_train.shape[1]
modelo = Modelo(input_size)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

modelo.to(device)

# Definir la función de pérdida y el optimizador
criterio = nn.MSELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# Listas para almacenar las pérdidas y precisión en entrenamiento y validación
train_losses = []
val_losses = []
# Entrenamiento del modelo
epochs = 100


for epoch in range(epochs):
    modelo.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizador.zero_grad()
        output = modelo(batch_X)
        loss = criterio(output, batch_y)
        loss.backward()
        optimizador.step()

    
    # Evaluación en conjunto de entrenamiento
    modelo.eval()
    with torch.no_grad():
        train_outputs = modelo(X_train_tensor.to(device))
        train_loss = criterio(train_outputs.to(device), y_train_tensor.to(device)) # error cuadratico medio

    # Evaluación en conjunto de validación
    with torch.no_grad():
        val_outputs = modelo(X_test_tensor.to(device))
        val_loss = criterio(val_outputs.to(device), y_test_tensor.to(device))

    # Almacenar pérdidas y precisión en listas
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    print("Epoch:", epoch)

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
