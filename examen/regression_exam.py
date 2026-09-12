import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Maira Alejandra
# Apellido 1: Balanta
# Apellido 2: Peña
# Rama: balanta_pena


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.genfromtxt("data/sales_data.csv", delimiter=",", skip_header=1)

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
# Respuesta: Cada fila representa una observación del problema.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Porque permite incorporar el término independiente beta_0
# dentro del producto matricial X @ beta.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: X tiene dimensión 8x2 y X.T tiene dimensión 2x8.
# Por tanto, (2x8) @ (8x2) produce una matriz de dimensión 2x2.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: beta_0 representa el intercepto del modelo, es decir,
# el valor esperado de y cuando x = 0. En este problema representa
# las ventas estimadas cuando la inversión en publicidad es 0.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: beta_1 representa la pendiente del modelo, es decir,
# cuánto cambian aproximadamente las ventas cuando la inversión
# en publicidad aumenta una unidad.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que, en conjunto, las predicciones están
# relativamente cerca de los valores reales.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque el producto matricial permite calcular
# simultáneamente las predicciones de todas las observaciones.
# Esto es un ejemplo de vectorización.

