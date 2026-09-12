import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Violeta Isabella
# Apellido 1: España
# Apellido 2: Bolaños
# Rama: Espana_Bolanos


# 1. CARGA DE DATOS 
# ------------------------------------------------------------

data = np.loadtxt("data/sales_data.csv", delimiter=",", skiprows=1)

x = data[:, 0]
y = data[:, 1]

# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones_like(x), x))

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

print("Prediction para x=9:", prediction)


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
# Cada valor de la fila de X representa un conjunto de dos datos, donde la primera columna es 1 (para el término independiente) y la segunda columna es el valor de x correspondiente a un valor de "advertising" para este ejemplo.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: 
# La primera columna de unos permite incluir el término independiente.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: 
# X tiene 8 filas y 2 columnas, por lo que X es de dimensión 8 × 2 y X.T es de dimensión 2 × 8. Al multiplicarlas,
# (2 × 8)(8 × 2) = 2 × 2. Por eso la dimensión de X.T @ X tiene dimensión 2 × 2.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: 
# beta_0 representa el intercepto de la recta, es decir, el valor estimado de "sales" cuando "advertising" = 0.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: 
# beta_1 representa la pendiente de la recta de regresión. Indica cuánto cambia el valor estimado de y (sales) por cada unidad que aumenta x (advertising).



# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: 
# Significa que las predicciones del modelo están cerca de los valores reales de y, lo que indica un buen ajuste del modelo a los datos.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: 
# Porque X contiene todas las observaciones y beta contiene los parámetros del modelo. La multiplicación X @ beta calcula la predicción correspondiente a cada fila de X simultáneamente.

