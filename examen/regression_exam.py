import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: juan diego
# Apellido 1: lopez
# Apellido 2: valencia
# Rama: lopez_valencia


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.array([
    [1, 3.2],
    [2, 4.8],
    [3, 7.3],
    [4, 8.7],
    [5, 11.1],
    [6, 12.8],
    [7, 15.2],
    [8, 16.7]
])

x = data[:, 0]
y = data[:, 1]


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

beta = np.linalg.inv(XtX) @ Xty

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

y_pred = X @ beta


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
# Cada fila de X representa una observación del dataset.
# La primera columna corresponde al término independiente
# y la segunda columna corresponde al valor de advertising.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:
# La primera columna de unos permite representar el intercepto
# beta_0 en el modelo lineal.

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:
# X tiene dimensión 8x2, por lo tanto X.T tiene dimensión 2x8.
# Al multiplicar X.T @ X se obtiene una matriz de dimensión 2x2.

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:
# beta_0 representa el valor estimado de sales cuando
# advertising es igual a cero.

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:
# beta_1 representa el cambio estimado en sales por cada
# unidad adicional de advertising.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:
# Significa que las predicciones del modelo están cerca
# de los valores reales del dataset.

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:
# Porque X contiene todas las observaciones y beta contiene
# los parámetros del modelo. La multiplicación matricial
# calcula la predicción correspondiente para cada fila de X.