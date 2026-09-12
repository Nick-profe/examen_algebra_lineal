import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Santiago 
# Apellido 1:Duque 
# Apellido 2:Valencia
# Rama: duque_valencia


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

beta = np.linalg.solve(XtX, Xty)

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
# Cada fila de X representa una observación del conjunto de datos. 
# La primera posición corresponde al término independiente y la segunda 
# corresponde al valor de advertising de esa observación.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:
# La primera columna de unos permite incluir el término independiente 
# beta_0 en el modelo de regresión lineal.  
# De esta manera, cada predicción se calcula como: 
# y = beta_0 + beta_1 * x.

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:
# X tiene dimensión (8, 2), por lo tanto X.T tiene dimensión (2, 8). 
#  Al multiplicar X.T @ X se realiza:  
# (2, 8) @ (8, 2) # y el resultado tiene dimensión (2, 2).

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:
# beta_0 representa el intercepto del modelo de regresión lineal. 
# En este caso su valor es aproximadamente 1.1143 y representa 
# el valor estimado de sales cuando advertising es igual a 0.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:
# beta_1 representa la pendiente del modelo de regresión lineal. 
# En este caso su valor es aproximadamente 1.9690, lo que significa 
# que por cada unidad adicional de advertising, sales aumenta 
# aproximadamente 1.969 unidades.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:
# Significa que, en general, las predicciones obtenidas por el modelo 
#  están cerca de los valores reales del conjunto de datos. 
#  En este caso la norma del error es aproximadamente 0.6283, 
# lo que indica que el modelo presenta un error relativamente pequeño 
#  respecto a los valores observados.

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:
# Porque cada fila de X contiene una observación y beta contiene 
#  los parámetros beta_0 y beta_1 del modelo. 
#  Al realizar el producto matricial X @ beta, cada fila de X se 
#  multiplica por los parámetros correspondientes, obteniendo así 
# todas las predicciones del dataset en una sola operación.