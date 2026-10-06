import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# Nombre: Wilson Navia
# Apellido 1: Navia
# Apellido 2: Valencia
# Rama: navia_valencia

# 1. CARGA DE DATOS
# Cada fila contiene [inversion_publicitaria, ventas].
data = np.array(
    [
        [1, 3.2],
        [2, 4.8],
        [3, 7.3],
        [4, 8.7],
        [5, 11.1],
        [6, 12.8],
        [7, 15.2],
        [8, 16.7],
    ],
    dtype=float,
)

x = data[:, 0]
y = data[:, 1]

# 2. MATRIZ DE DISEÑO
X = np.column_stack((np.ones(len(x)), x))

print("X:")
print(X)
print("Shape X:", X.shape)
print("Shape y:", y.shape)

# 3. OPERACIONES MATRICIALES
XtX = X.T @ X
Xty = X.T @ y

# 4. ESTIMACIÓN DE PARÁMETROS
beta = np.linalg.inv(XtX) @ Xty

beta_0 = beta[0]
beta_1 = beta[1]

print("Beta 0:", beta_0)
print("Beta 1:", beta_1)

# 5. PREDICCIÓN
x_new = np.array([1, 9])
prediction = x_new @ beta

print("Prediction:", prediction)

# 6. PREDICCIONES DEL DATASET
y_pred = X @ beta

# 7. ERROR
errors = y - y_pred
error_norm = np.linalg.norm(errors)

print("Error vector:")
print(errors)
print("Error norm:")
print(error_norm)

# 8. PREGUNTAS
# 1. Cada fila de X representa una observación: el intercepto y la
#    inversion publicitaria de un caso.
#
# 2. La primera columna contiene unos para incluir beta_0, el intercepto,
#    en el producto matricial.
#
# 3. XtX tiene dimension 2x2: X tiene 2 columnas (2 parametros), por lo
#    que X.T @ X tiene tantas filas y columnas como parametros.
#
# 4. beta_0 representa las ventas estimadas cuando la inversion publicitaria
#    es cero.
#
# 5. beta_1 representa el cambio estimado en ventas por cada unidad adicional
#    de inversion publicitaria.
#
# 6. Una norma del error pequena indica que las predicciones del modelo estan
#    cerca de los valores observados.
#
# 7. X @ beta calcula en una sola operacion matricial la prediccion para cada
#    fila (observacion) de X.
