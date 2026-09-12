import numpy as np
from pathlib import Path

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre:
# Apellido 1: Ruiz
# Apellido 2: Calero
# Rama: ruiz_calero


# 1. CARGA DE DATOS
# ------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"

data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1)

x = data[:, 0]
y = data[:, 1]


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(x.shape[0]), x))

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

# Ecuación normal: beta = (X^T X)^(-1) X^T y
# Se resuelve el sistema en lugar de invertir explícitamente,
# por ser numéricamente más estable.
beta = np.linalg.solve(XtX, Xty)

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


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: 


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
