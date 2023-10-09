import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x1 = np.arange(1,11)
x2 = np.ones(10)

Ye = np.array([1.56, 1.95, 2.44, 3.05, 3.81, 4.77, 5.96, 7.45, 9.31, 11.64])

Xi = np.vstack((x1,x2)).T

print(Xi[:,0], Ye)

mdl = LinearRegression().fit(Xi,Ye)

print("Intercept(w0):", mdl.intercept_)

# Pesos del modelo de regresion lineal
print("Pesos del modelo: ", mdl.coef_)

# Creamos la figura
fig = plt.figure()

# Agregamos un plano 3D
ax = plt.axes(projection='3d')

# plot_wireframe nos permite agregar los datos x, y, z. Por ello 3D
ax.scatter(x1, x2, Ye,  c='b', marker='x')

# ax.plot(x1, x2, Y, c='r')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Ye')
# Mostramos el gráfico
plt.show()
