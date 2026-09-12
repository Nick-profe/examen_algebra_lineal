import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Valeria 
# Apellido 1: Paz
# Apellido 2: Velasquez
# Rama: paz_velasquez


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.loadtxt("data/sales_data.csv", delimiter=",", skiprows=1)
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
# Cada fila de X representa una observación o dato
# correspondiente a un valor de publicidad y su respectiva venta

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:
# La primera columna de 1 permite incluir el término independiente
# beta_0 en el modelo de regresión lineal.

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:

# X tiene dimensión 8x2 y X.T tiene dimensión 2x8.
# Por eso X.T @ X tiene dimensión 2x2

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:
# beta_0 representa las ventas estimadas cuando la inversión en publicidad
# es igual a 0

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:
# representa cuánto aumentan las ventas estimadas por cada
# unidad adicional de inversión en publicidad

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:
# significa que las predicciones del modelo
# están cerca de los valores reales de ventas

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:
# X @ beta permite calcular las predicciones de todas las observaciones
# al mismo tiempo mediante una sola multiplicación matricial.