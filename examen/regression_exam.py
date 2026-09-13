import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre:Eduardo
# Apellido 1:Sarmiento
# Apellido 2:Pradilla
# Rama: sarmiento_pradilla


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.array([[1, 3.2], [2, 4.8], [3,7.3], [4, 8.7], [5, 11.1], [6, 12.8], [7, 15.2], [8, 16.7]])


x = np.array([10, 20, 30, 40, 50])
y = np.array([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7])


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

print("X:")
print(X)

print("Shape X:", X.shape)
print("Shape y:", y.shape)


# 3. OPERACIONES MATRICIALES
# ------------------------------------------------------------

XtX = X.T @ X
Xty = X.T @ y


# 4. ESTIMACIÓN DE PARÁMETROS
# ------------------------------------------------------------

beta = np.linalg.inv(X.T @ X) @ X.T @ y

beta_0 = beta[0]
beta_1 = beta[1]

print("Beta 0:", beta_0)
print("Beta 1:", beta_1)


# 5. PREDICCIÓN
# ------------------------------------------------------------

x_new = np.array([1, 9])

prediction = x_new @ beta

print("Prediction:", prediction)


# 6. PREDICCIONES DEL DATASET
# ------------------------------------------------------------

y_pred =X @ beta

print("Predicciones para el dataset:")
print(y_pred)

# 7. ERROR
# ------------------------------------------------------------

errors = y - y_pred
error_norm = np.linalg.norm(errors)

print("Error vector:")
print(errors)

print("Error norm:")
print(error_norm)


# 8. PREGUNTAS
# ------------------------------------------------------------

# 1. ¿Qué representa cada fila de X?
# Respuesta:

# Cada fila de X representa una observación del data set

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:

# El valor 1 permite incorporar el término independiente b0 dentro del producto matricial

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:

# X tiene dimensión 8 x 2 y X.T tiene dimensión 2 x 8, por lo tanto:
# (2 x 8) @ (8 x 2) = (2 x 2)
# La dimensión de X.T @ X es 2 x 2


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:

#beta_0 representa el valor esperado de sales cuando advertising = 0

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:

# En este caso, indica el aumento promedio de las ventas por cada unidad adicional de publicidad.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:

# Significa que, en conjunto, las predicciones del modelo están relativamente cerca de los valores reales de sales.

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:

#Porque el producto matricial multiplica cada fila de X por el vector de parámetros beta.
# De esta manera, calcula las predicciones de todas las observaciones en una sola operación, esto se conoce como vectorización.