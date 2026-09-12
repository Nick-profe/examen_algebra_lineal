import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Leonel Mauricio Reyes Rodriguez
# Apellido 1: Reyes
# Apellido 2: Rodriguez
# Rama: reyes_rodriguez


# 1. CARGA DE DATOS
# ------------------------------------------------------------

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 17.7])

data = np.colum_stack((x, y))

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
# Respuesta: cada fila representa una observacion que contiene el valor de la variable predictora.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: para que se permita incorporar el intercepto en el modelo de regresión lineal, permitiendo así que la ecuación de la recta tenga un término constante.
#

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: 2x2, porque X tiene 8 filas y 2 columnas, entonces al multiplicar X.T (2x8) por X (8x2) se obtiene una matriz de 2x2.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: beta_0 representa el intercepto de la recta de regresión, es decir, el valor estimado de y cuando x es igual a cero.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: beta_1 representa la pendiente de la recta de regresión, es decir, el cambio estimado en y por cada unidad de cambio en x.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que las predicciones del modelo están muy cerca de los valores reales, lo que indica un buen ajuste del modelo a los datos observados.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque X @ beta es una operación matricial que aplica la transformación lineal definida por beta a toda la matriz X, resultando en un vector con todas las predicciones.