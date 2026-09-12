import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: David Santiago
# Apellido 1: Roa
# Apellido 2: Mayor
# Rama: roa_mayor


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.genfromtxt('../data/sales_data.csv', delimiter=',', skip_header=1)

x = data[:, 0]
y = data[:, 1]


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

print("X:")
print(X)

print("Shape X:", X.shape)
print("Shape y:", y.shape )


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
# Respuesta: Cada fila es una observación del dataset (un valor de publicidad, con el 1 al inicio)


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Para que beta_0 se peuda multiplicar y sumar cuando se opera X @ beta


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: Es (2,2), porque X.T tiene forma (2,n) y X tiene forma (n,2), entonces el resultado queda (2,2)


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Es el valor de ventas esperado cuando no hay inversión en publicidad.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Es cuánto suben las ventas por cada unidad extra invertida en publicidad.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Que el modelo predice valores muy parecidos a los datos reales.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque multiplica todas las filas de X por beta al mismo tiempo, sin necesidad de un ciclo.