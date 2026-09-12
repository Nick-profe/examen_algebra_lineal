import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Santiago
# Apellido 1: Cuellar
# Apellido 2: Andrade
# Rama: cuellar_andrade



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
# Respuesta: Cada fila representa una observación segun la logica de 
# la guia del dataset.



# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Porque permite que el término independiente (beta_0)
# se estime dentro de la misma multiplicación matricial X @ beta.
# Al multiplicar 1 * beta_0, ese valor se suma automáticamente en cada
# predicción, sin necesidad de agregarlo por separado.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: Es una matriz de 2x2. X tiene dimensión (n, 2), donde n es
# el número de observaciones son 8 y 2 es el número de parámetros del
# modelo intercepto y pendiente. Al multiplicar X.T (dimensión 2xn)
# por X dimensión nx2, el resultado es una matriz de 2x2: el número
# de filas y columnas coincide con el número de parámetros.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Es el intercepto del modelo: el valor esperado de "sales"
# cuando "advertising" es igual a 0.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Es la pendiente del modelo: indica cuánto aumenta en
# la media "sales" por cada unidad adicional de "advertising".


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que las predicciones del modelo y_pred están
# muy cerca de los valores reales observados y, es decir, que el
# modelo se ajusta bien a los datos.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque cada fila de X, al multiplicarse por el vector beta,
# calcula automáticamente beta_0*1 + beta_1*x para esa observación.
