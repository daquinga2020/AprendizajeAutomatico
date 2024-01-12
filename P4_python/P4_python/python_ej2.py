import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Especifica la ruta de tu archivo CSV
ruta_csv = 'sonar.csv'

# Lee el archivo CSV con pandas
datos = pd.read_csv(ruta_csv, header=None)

# Separar las columnas en entrada (X) y salida (y)
X = datos.iloc[:, :-1].values

# Convertir las salidas de string a números (asumiendo que son binarias)
y_str = datos.iloc[:, -1].values
y = [1 if label == 'M' else 0 for label in y_str]

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

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
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Definir el modelo
class Modelo(nn.Module):
    def __init__(self, input_size):
        super(Modelo, self).__init__()
        # descomentar para apartado 1
        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 1)
        # self.sigmoid = nn.Sigmoid()

        # comentar para apartado 1 e ir descomentando capas para apartado 2     
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # descomentar para apartado 1
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        # x = self.sigmoid(x)

        # comentar para apartado 1 e ir descomentando capas para apartado 2     
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Instanciar el modelo
input_size = X_train.shape[1]
modelo = Modelo(input_size)

# Definir la función de pérdida y el optimizador
criterio = nn.BCELoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.1)

# Listas para almacenar las pérdidas y precisión en entrenamiento y validación
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Entrenamiento del modelo
epochs = 250
for epoch in range(epochs):
    modelo.train()
    for batch_X, batch_y in train_loader:
        optimizador.zero_grad()
        output = modelo(batch_X)
        loss = criterio(output, batch_y)
        loss.backward()
        optimizador.step()

    # Evaluación en conjunto de entrenamiento
    modelo.eval()
    with torch.no_grad():
        train_outputs = modelo(X_train_tensor)
        train_loss = criterio(train_outputs, y_train_tensor)
        train_acc = ((train_outputs >= 0.5).float() == y_train_tensor).float().mean().item()

    # Evaluación en conjunto de validación
    with torch.no_grad():
        val_outputs = modelo(X_test_tensor)
        val_loss = criterio(val_outputs, y_test_tensor)
        val_acc = ((val_outputs >= 0.5).float() == y_test_tensor).float().mean().item()

    # Almacenar pérdidas y precisión en listas
    train_losses.append(train_loss.item())
    train_accuracies.append(train_acc)
    val_losses.append(val_loss.item())
    val_accuracies.append(val_acc)
    print("Epoch:", epoch)

# Graficar la evolución de pérdidas y precisión
plt.figure(figsize=(12, 4))

# Pérdidas
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Entrenamiento')
plt.plot(val_losses, label='Validación')
plt.title('Evolución de las Pérdidas')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Precisión
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Entrenamiento')
plt.plot(val_accuracies, label='Validación')
plt.title('Evolución de la Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()
