import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Dairon
# Apellido 1: Rojas
# Apellido 2: Muñoz
# Rama: Rojas_Muñoz


# 1. CARGA DE DATOS
# ------------------------------------------------------------

##data = 

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
# Respuesta: Cada fila representa una observación del dataset.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Para incluir el intercepto beta_0 en el modelo.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: Es 2x2 porque X.T es 2x8 y X es 8x2., 8=8 y queda 2x2.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Representa las ventas estimadas cuando advertising es 0. El intercepto.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Representa cuánto cambian las ventas cuando advertising aumenta una unidad. La pendiente.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que las predicciones están cerca de los valores reales.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque el producto matricial calcula todas las predicciones al mismo tiempo.