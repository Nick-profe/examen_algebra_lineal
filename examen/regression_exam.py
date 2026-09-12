import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Johan
# Apellido 1: Jurado
# Apellido 2: Lloreda
# Rama: jurado_lloreda


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.loadtxt("../data/sales_data.csv", delimiter=",", skiprows=1)

x = data[:, 0]
y = data[:, 1]
print("x:", x)
print("y", y)


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)),x))

print("X:", X)
print("Shape X:", X.shape)
print("Shape y:", y.shape)


# 3. OPERACIONES MATRICIALES
# ------------------------------------------------------------

XtX = X.T @ X
Xty = X.T @ y

print("XtX:", XtX)
print("Shape XtX:", XtX.shape)


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
print("Prediction dataset:", y_pred)


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
# Respuesta: Cada fila de X representa una observación del conjunto de datos, donde la primera posición corresponde a la constante y la segunda al valor de advertising


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: X contiene una primera columna de unos para el termino independiente β0 para incorporarlo dentro de la matriz


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: 


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: