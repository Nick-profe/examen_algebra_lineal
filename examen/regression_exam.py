import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: David Santiago
# Apellido 1: Alderete
# Apellido 2: Patiño
# Rama: alderete_patino

# 1. CARGA DE DATOS
# ------------------------------------------------------------

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
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
# Respuesta: En la matriz X, cada fila representa un punto de datos.
# La primera columna de unos representa el término constante (intercepto) del modelo de regresión, mientras que la segunda columna contiene
#  los valores de la variable independiente x para cada observación.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: La primera columna de unos en la matriz X se incluye para representar el término constante del intercepto
# sin esto, el modelo de regresión lineal no podría estimar un valor de intercepto, lo que limitaría la capacidad del modelo para ajustarse a los datos.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: La dimensión de X.T @ X es (2, 2) porque X tiene una forma de (8, 2), donde 8 es el número de filas, y 2 es el numero de columnas. 
# Al multiplicar X transpuesta (2, 8) por X (8, 2), el resultado es una matriz de (2, 2).


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Beta_0 representa el intercepto de la linea de regresion, es decir el valor de y cuando x es igual a cero. 


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Beta_1 representa la pendiente de la línea de regresión, es decir, el cambio esperado en la variable dependiente y por cada unidad de cambio en la variable independiente x

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Que la norma del error sea pequeña indica que las predicciones del modelo están muy cerca de los valores reales, lo que sugiere un buen ajuste del modelo a los datos


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: La operación X @ beta permite obtener todas las predicciones simultáneamente porque es una multiplicación matricial que combina la matriz de diseño X con el vector de parámetros beta. 
# Cada fila de X representa un punto de datos, y al multiplicar por beta, se calcula la predicción correspondiente para cada punto de datos en una sola operación, generando un vector de predicciones y_pred 
# que contiene todas las predicciones del modelo para los datos de entrada.