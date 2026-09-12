import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Juan Esteban 
# Apellido 1: Estacio
# Apellido 2: Alomia
# Rama: estacio_alomia


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
# Cada fila representa una observación del dataset, 
# con el término independiente 1 y el valor de advertising.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:
# La columna de unos permite incluir el intercepto beta_0 
# dentro del producto matricial X @ beta.

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:
# X tiene dimensión 8x2 y X.T tiene dimensión 2x8. 
# Por lo tanto, X.T @ X tiene dimensión 2x2.

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:
# beta_0 representa el valor estimado de sales cuando 
# advertising es igual a 0.

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:
# beta_1 representa cuánto aumenta aproximadamente sales 
# por cada unidad adicional de advertising.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:
# Significa que las predicciones del modelo están cerca 
# de los valores reales de sales.

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:
# Porque el producto matricial aplica el modelo a todas 
# las observaciones de X al mismo tiempo mediante vectorización.