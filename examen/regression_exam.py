import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Luis Felipe
# Apellido 1: Murillo
# Apellido 2: Matallana
# Rama: murillo_matallana


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.loadtxt("../data/sales_data.csv", delimiter=",", skiprows=1)

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

print("Shape beta:", beta.shape)

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
# Respuesta: Cada fila representa una observación (un registro) del dataset,
# es decir, un valor de inversión en publicidad junto con el 1 del intercepto.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Para poder incorporar el término independiente beta_0 dentro
# del producto matricial X @ beta. Sin esa columna, el modelo estaría
# forzado a pasar por el origen (y = beta_1 * x).

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: X tiene dimensión (8,2), por lo que X.T tiene dimensión (2,8).
# Al multiplicar (2,8) @ (8,2), las dimensiones internas coinciden (8=8)
# y el resultado toma las dimensiones externas, es decir (2,2).

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Es el intercepto del modelo: el valor esperado de ventas
# cuando la inversión en publicidad es 0.

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Es la pendiente del modelo: cuánto aumenta en promedio
# las ventas por cada unidad adicional invertida en publicidad.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que las predicciones del modelo (y_pred) están
# muy cerca de los valores observados (y), es decir, y ≈ y_pred.

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque el producto matricial aplica la combinación lineal
# beta_0 + beta_1*x a cada fila de X al mismo tiempo (vectorización),
# en lugar de calcular cada predicción individualmente en un ciclo.